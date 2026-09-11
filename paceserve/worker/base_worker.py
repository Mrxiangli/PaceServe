"""A GPU worker class."""

import os
import time
from threading import Event, Thread
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.distributed
import zmq
import copy

from paceserve.config import CacheConfig, ParallelConfig, SystemConfig
from paceserve.core.datatypes.comm_info import CommInfo
from paceserve.core.datatypes.scheduler_output import SchedulerOutputs
from paceserve.core.datatypes.sequence import SamplerOutputs
from paceserve.core.sequence_manager.worker_sequence_manager import WorkerSequenceManager
from paceserve.logger import init_logger
from paceserve.metrics.metrics_store import MetricsStore
from paceserve.model_executor import set_random_seed
from paceserve.model_executor.model_runner import ModelRunner
from paceserve.model_executor.parallel_utils.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
    initialize_model_parallel,
)
from paceserve.utils.threading_utils import exit_on_error, synchronized
from paceserve.kv_transfer.kv_utils import KVhelper
from paceserve.utils.timer_utils import Timer
from paceserve.ai_utils.ai_helper import AIHelper

logger = init_logger(__name__)


_READY_ACK_WAIT_TIME = 1

@dataclass
class KVStatus:
    status: bool
    insertion_time: float
    allocate_time: float


class BaseWorker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        config: SystemConfig,
        local_rank: int,
        rank: int,
        comm_info: CommInfo,
    ) -> None:
        # Not: the cache config is partially initialized at this point, ie. it doesn't have
        # information about the number of blocks, it will get updated after profiling
        self.config = config
        self.local_rank = local_rank
        self.rank = rank
        self.comm_info = comm_info
        comm_info._show(self.local_rank)
        

        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_engine = None
        self.gpu_cache = None
        # Sequence manager also needs number of blocks for initialization
        self.seq_manager = None

        self._verify_parallel_config()
        self.metrics_store = MetricsStore.get_or_create_instance(
            config.replica_config,
            config.model_config,
            config.metrics_config,
        )
        self.kvhelper = KVhelper(self.rank, self.config.model_config, self.config.cache_config, self.config.parallel_config)
        self._init_zmq_sockets()

        self.worker_ready_event = Event()
        self.execution_thread = Thread(target=self._execution_loop, daemon=True)

        # Need to setup the singleton classes here again because the worker
        # runs in a separate process than the instance.
        _ai_helper = AIHelper()
        _ai_helper.initialize(config.model_config, config.scheduler_config, config.batchregression_config)

    def _init_zmq_sockets(self):
        self.zmq_context = zmq.Context()
        self.enqueue_socket = self.zmq_context.socket(zmq.SUB)
        self.enqueue_socket.connect(
            f"tcp://{self.comm_info.engine_ip_address}:{self.comm_info.enqueue_socket_port}"
        )
        self.enqueue_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.output_socket = self.zmq_context.socket(zmq.PUSH)
        self.output_socket.connect(
            f"tcp://{self.comm_info.engine_ip_address}:{self.comm_info.output_socket_port}"
        )

    def _verify_parallel_config(self) -> None:
        assert self.config.parallel_config.pipeline_parallel_size == 1

    @torch.inference_mode()
    @synchronized
    def init_model(self):
        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        os.environ["KINETO_LOG_LEVEL"] = "5"
        os.environ['TORCH_CUDA_ARCH_LIST'] = "8.0"    # A100 Ampere architecture

        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)

        logger.info(f"Worker {self.rank} is using device {self.local_rank}")
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        
        with Timer("Initialize distributed environment - "):
            # Initialize the distributed environment.
            _init_distributed_environment(
                self.config.parallel_config,
                self.rank,
                self.comm_info.distributed_init_method,
            )

        self.tensor_model_parallel_rank = get_tensor_model_parallel_rank()
        self.pipeline_model_parallel_rank = get_pipeline_model_parallel_rank()

        self.is_tensor_parallel_rank_zero = self.tensor_model_parallel_rank == 0
        self.is_first_pipeline_stage = self.pipeline_model_parallel_rank == 0
        self.is_last_pipeline_stage = (
            self.pipeline_model_parallel_rank
            == self.config.parallel_config.pipeline_parallel_size - 1
        )

        logger.info(
            f"Initializing worker {self.rank} on device {self.device}, "
            f"tensor parallel rank {self.tensor_model_parallel_rank} "
            f"and pipeline parallel rank {self.pipeline_model_parallel_rank}."
        )
    
        # Initialize the model.
        set_random_seed(self.config.model_config.seed)
        self.model_runner = ModelRunner(
            self.config,
            self.device,
            self.rank,
        )
        logger.info(f"Model initialized on worker {self.rank}.")
        
        self.kvhelper.add_model_runner(self.model_runner)

    @torch.inference_mode()
    @synchronized
    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        torch.cuda.set_device(self.device)

        self.config.cache_config = cache_config

        self.model_runner.init_kv_cache(cache_config.num_gpu_blocks)

        self.seq_manager = WorkerSequenceManager(
            self.config,
        )

        self.execution_thread.start()

    def wait_till_ready(self) -> None:
        self.worker_ready_event.wait()
        time.sleep(_READY_ACK_WAIT_TIME)

    @synchronized
    def get_model_parallel_ranks(self) -> Tuple[int, int]:
        return self.tensor_model_parallel_rank, self.pipeline_model_parallel_rank

    def on_step_completed(
        self, scheduler_outputs: SchedulerOutputs, sampler_outputs: SamplerOutputs
    ) -> None:
        self.seq_manager.on_step_completed(scheduler_outputs, sampler_outputs)

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_outputs: SchedulerOutputs,
    ) -> Optional[SamplerOutputs]:
        torch.cuda.synchronize()
        batch_stage_start_time = time.monotonic()

        _, seq_metadata_list = self.seq_manager.on_schedule(scheduler_outputs)

        sampler_outputs = self.model_runner.run(
            seq_metadata_list,
        )

        self.on_step_completed(scheduler_outputs, sampler_outputs)

        torch.cuda.synchronize()

        batch_stage_end_time = time.monotonic()

        '''
        #The following code is a little test for cache retrieval and insertion
        tmp_cache = []
        tmp_indices = []
        for each in seq_metadata_list:
            kv_cache, kv_indices = self.kvhelper.get_kv_cache(each.seq, self.seq_manager)
            tmp_cache.append(kv_cache)
            tmp_indices.append(kv_indices)
        tmp_deep = copy.copy(tmp_cache)
        self.kvhelper.reset_kv_cache(self.config.cache_config.num_gpu_blocks)
        self.kvhelper.insert_kv_cache(tmp_deep, tmp_indices)
        '''

        self.metrics_store.on_batch_stage_end(
            seq_metadata_list,
            scheduler_outputs,
            self.tensor_model_parallel_rank,
            self.pipeline_model_parallel_rank,
            batch_stage_start_time,
            batch_stage_end_time,
        )

        return sampler_outputs
    
    #@synchronized
    def _get_kv_cache(self, seq):
        kv_cache, kv_indices = self.kvhelper.get_kv_cache(seq, self.seq_manager)
        # free the memory space on this worker
        self.seq_manager._free_seq(seq.seq_id)
        return (kv_cache, kv_indices, self.local_rank)

    #when premption happens, we do not free the worker side sequence map during
    #memory swap, this will be hanndled by the distserve_preempt_seq function 
    #in base sequence manager
    def _get_kv_cache_nofree(self, seq):
        kv_cache, kv_indices = self.kvhelper.get_kv_cache(seq, self.seq_manager)
        return (kv_cache, kv_indices, self.local_rank)
           
    @torch.inference_mode()
    def _insert_kv_cache(self, seq, kv_obj, device):
        block_len = len(kv_obj.kv_indices)
        cached_content = kv_obj.kv_cache
        kv_block_allocate_start = time.monotonic()
        # allocate space for the sequence on the worker
        if self.seq_manager.block_manager.get_num_free_gpu_blocks() >= self.seq_manager.block_manager.get_num_initial_blocks(seq):
            self.seq_manager.add_seq(seq)
            self.seq_manager.block_manager.allocate(seq)
        else:
            logger.debug(f"the worker cannot allocate -  free blocks: {self.seq_manager.block_manager.get_num_free_gpu_blocks()} required: {self.seq_manager.block_manager.get_num_initial_blocks(seq)}")
            return False
        # return KVStatus(True, 0.0, 0.0)
        kv_indices = self.seq_manager._get_block_table(seq)
        kv_insert_start = time.monotonic()
        # the prefilled seq has the extra logical token, but the kv cache is not allocated in prefill worker for this token, +1 to account for that
        if seq.logical_token_blocks[-1].get_num_empty_slots() == 15 and seq.state._preemption_count == 0:
            assert len(kv_indices) == block_len + ((len(seq.prompt_token_ids) % seq.block_size) == 0), f"kv len: {len(kv_indices)} block: {block_len} seq: {seq}"
        if seq.logical_token_blocks[-1].get_num_empty_slots() == 15 and seq.state._preemption_count == 0:
            status = self.kvhelper.insert_kv_cache(cached_content, kv_indices[:-1], device)
        # For the decode sequence on the decode worker, the kv cache might, or might not allocated for that extra logic token
        # i.e. in the decode scheduler, there are two requests A and B, both did the previous decoding step, in the subsequent scheduling
        # seq A get allocated with the last free block, seq B did not, lead to preemption. In this case, the block table of seq A equals to 
        # the derived blocks from self.seq_manager._get_block_table(seq), which based on seq logical token. However, the block table of seq B does not
        # equal to that derived, because it's not successfulluy allocated, kinda like the case of on prefilled instance.
        if len(kv_indices) == block_len:
            status = self.kvhelper.insert_kv_cache(cached_content, kv_indices, device)
        else:
            status = self.kvhelper.insert_kv_cache(cached_content, kv_indices[:-1], device)
        logger.debug(f"free blocks on worker {self.local_rank} after inserting seq {seq.seq_id}: {self.seq_manager.block_manager.get_num_free_gpu_blocks()}")
        worker_kv_insert = time.monotonic()-kv_insert_start
        worker_kv_allocate = kv_block_allocate_start - kv_insert_start
        return KVStatus(status, worker_kv_insert, worker_kv_allocate)
        
    @exit_on_error
    def _execution_loop(self) -> None:
        torch.cuda.set_device(self.device)

        self.worker_ready_event.set()

        while True:
            step_inputs = self.enqueue_socket.recv_pyobj()

            for new_seq in step_inputs.new_seqs:
                self.seq_manager.add_seq(new_seq)
            
            for aborted_seq in step_inputs.aborted_seqs:
                self.seq_manager._free_seq(aborted_seq.seq_id)
            
            if not step_inputs.scheduler_outputs.is_empty():
                output = self.execute_model(step_inputs.scheduler_outputs)
                if not self.is_tensor_parallel_rank_zero:
                    continue
                self.output_socket.send_pyobj(output)

    @synchronized
    def get_metrics_store(self) -> MetricsStore:
        return self.metrics_store

    @synchronized
    def mark_initial_memory_profiling_done(self):
        self.metrics_store.mark_initial_memory_profiling_done()

    @synchronized
    def reset_metrics(self) -> None:
        self.metrics_store.reset()

    @synchronized
    def start_profiling(self) -> None:
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        self.profiler.__enter__()

    @synchronized
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
    ) -> Tuple[int, int]:
        return self.model_runner.profile_num_available_blocks(
            block_size, gpu_memory_utilization
        )

    @synchronized
    def stop_profiling(self) -> None:
        self.profiler.__exit__(None, None, None)
        self.profiler.export_chrome_trace(
            f"{self.config.replica_config.output_dir}/profiler_trace_rank_{self.rank}.json"
        )


def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: str,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size})."
            )
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel(
        parallel_config.tensor_parallel_size, parallel_config.pipeline_parallel_size
    )
