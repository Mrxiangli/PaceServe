import copy
import math
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple
import ray
import zmq
import sys

from paceserve.config import ModelConfig, SystemConfig
from paceserve.core.datatypes.comm_info import CommInfo
from paceserve.core.datatypes.request_output import RequestOutput
from paceserve.core.datatypes.sampling_params import SamplingParams
from paceserve.core.datatypes.scheduler_output import SchedulerOutputs
from paceserve.core.datatypes.sequence import SamplerOutputs, Sequence, SequenceMetadata
from paceserve.core.datatypes.step_inputs import StepInputs
from paceserve.core.scheduler.scheduler_registry import SchedulerRegistry
from paceserve.core.sequence_manager.engine_sequence_manager import EngineSequenceManager
from paceserve.engine.ray_utils import RayWorker, initialize_cluster, ray
from paceserve.logger import init_logger
from paceserve.metrics.constants import CpuOperationMetrics
from paceserve.metrics.cpu_timer import CpuTimer
from paceserve.metrics.metrics_store import MetricsStore
from paceserve.transformers_utils.tokenizer import get_tokenizer
from paceserve.utils import Counter, get_ip, unset_cuda_visible_devices
from paceserve.utils.threading_utils import synchronized
from paceserve.utils.timer_utils import Timer
from paceserve.core.datatypes.scheduler_metadata import SchedulerMetadata

logger = init_logger(__name__)

_MAX_WORKER_CONCURRENCY = 8

ModelParallelRank = Tuple[int, int]


class BaseLLMEngine:
    """An LLM engine that receives requests and generates texts.

    This is the main class for the Sarathi engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    Args:
        config; System Config: The system configuration for the engine.
    """

    def __init__(
        self,
        config: SystemConfig,
    ) -> None:
        logger.info(
            "Initializing an LLM engine with config: "
            f"model={config.model_config.model!r}, "
            f"dtype={config.model_config.dtype}, "
            f"tensor_parallel_size={config.parallel_config.tensor_parallel_size}, "
            f"pipeline_parallel_size={config.parallel_config.pipeline_parallel_size}, "
            f"seed={config.model_config.seed})"
        )
        # TODO(woosuk): Print more configs in debug mode.

        self.config = config
        self.instance_id = config.replica_config.replica_id 
        self._verify_args()

        self.tokenizer = get_tokenizer(
            config.model_config.model,
            trust_remote_code=config.model_config.trust_remote_code,
            revision=config.model_config.revision,
        )

        self.seq_manager = EngineSequenceManager(self.tokenizer, config)
        self.seq_counter = Counter()

        self.metrics_store = MetricsStore.get_or_create_instance(
            config.replica_config,
            config.model_config,
            config.metrics_config,
        )

        self.worker_map: Dict[ModelParallelRank, int] = {}
        self.cpu_assigned = ray.get_runtime_context().get_assigned_resources()['CPU']

        # Create the parallel GPU workers.
        with Timer("Initialize GPU worker"):
            self._init_workers_ray()

        # Setup ZMQ communication channels
        with Timer("Initialize ZMQ"):
            self._init_zmq_sockets()

        # Profile the memory usage and initialize the cache.
        with Timer("Initialize cache"):
            self._init_cache()

        # Initialize the worker map.
        with Timer("Initialize worker map"):
            self._init_worker_map()

        self.mark_initial_memory_profiling_done()

        # Create the scheduler.
        self.scheduler = SchedulerRegistry.get(
            config.scheduler_config.get_type(),
            config.instance_config._type,
            config.model_config,
            config.scheduler_config,
            config.cache_config,
            config.parallel_config,
            config.batchregression_config
        )
        self.scheduler.instance_id = self.instance_id

        self._scheduler_timer = CpuTimer(CpuOperationMetrics.SCHEDULE)
        self._process_model_outputs_timer = CpuTimer(
            CpuOperationMetrics.PROCESS_MODEL_OUTPUTS
        )

        self.new_seqs: List[Sequence] = []

        logger.debug('Waiting for the workers to be ready...')
        self._run_workers("wait_till_ready")

    def _init_zmq_sockets(self):
        logger.debug('Initializing zmq sockets')
        self.zmq_context = zmq.Context()
        self.enqueue_socket = self.zmq_context.socket(zmq.PUB)
        self.enqueue_socket.bind(f"tcp://*:{self.comm_info.enqueue_socket_port}")
        self.output_socket = self.zmq_context.socket(zmq.PULL)
        self.output_socket.bind(f"tcp://*:{self.comm_info.output_socket_port}")

    def _validate_parallel_config(self) -> None:
        assert self.config.parallel_config.pipeline_parallel_size == 1

    def _get_worker_impl(self):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from paceserve.worker.base_worker import (
            BaseWorker,  # pylint: disable=import-outside-toplevel
        )

        return BaseWorker

    def _init_workers_ray(self, **ray_remote_kwargs):
        resource_mapping = self.config.instance_config.get_resource_mapping(
            self.config.parallel_config.world_size
        )
        logger.info(f"Starting workers with resource mapping: {resource_mapping}")

        self.workers: List[RayWorker] = []

        unset_cuda_visible_devices()

        driver_ip = None
        for rank, (node_ip, _) in enumerate(resource_mapping):
            worker_class = ray.remote(
                num_cpus=1,
                # num_gpus=1, # we don't use ray for managing GPUs
                **ray_remote_kwargs,
            )(RayWorker)

            if node_ip:
                worker_class = worker_class.options(
                    max_concurrency=_MAX_WORKER_CONCURRENCY,
                    num_cpus = self.cpu_assigned//(3*len(resource_mapping)),
                    resources={
                        node_ip: 0.01,
                    },
                )
            else:
                worker_class = worker_class.options(
                    max_concurrency=_MAX_WORKER_CONCURRENCY,
                    num_cpus = self.cpu_assigned//(3*len(resource_mapping)),
                )
            logger.info(f"worker {node_ip} get cpu: {self.cpu_assigned//(3*len(resource_mapping))}")
            if rank == 0:
                if node_ip:
                    # remove node: prefix
                    driver_ip = node_ip.split(":")[1]
                else:
                    driver_ip = get_ip()

            worker = worker_class.remote(self.config.model_config.trust_remote_code)

            self.workers.append(worker)

        self.comm_info = CommInfo(driver_ip)

        # Initialize torch distributed process group for the workers.
        config = copy.deepcopy(self.config)
        config.metrics_config = self.metrics_store.get_config_for_worker()

        worker_impl = self._get_worker_impl()
        
        with Timer("Initialize worker array - "):
            for rank, worker in enumerate(self.workers):
                local_rank = resource_mapping[rank][1]
                promise = worker.init_worker.remote(
                    lambda rank=rank, local_rank=local_rank: worker_impl(
                        config,
                        local_rank,
                        rank,
                        self.comm_info,
                    )
                )
                ray.get(promise)

        with Timer("Initialize model - "):
            self._run_workers(
                "init_model",
                get_all_outputs=True,
            )

    def _verify_args(self) -> None:
        self._validate_parallel_config()
        self.config.model_config.verify_with_parallel_config(
            self.config.parallel_config
        )

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache."""
        logger.debug('Profilling the memory usage')
        # Get the maximum number of blocks that can be allocated on GPU.
        num_gpu_blocks_across_workers = self._run_workers(
            "profile_num_available_blocks",
            get_all_outputs=True,
            block_size=self.config.cache_config.block_size,
            gpu_memory_utilization=self.config.worker_config.gpu_memory_utilization,
        )

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(num_gpu_blocks_across_workers)
        # FIXME(woosuk): Change to debug log.
        logger.info(f"# GPU blocks: {num_gpu_blocks}")

        if num_gpu_blocks <= 0:
            raise ValueError(
                "No available memory for the cache blocks. "
                "Try increasing `gpu_memory_utilization` when "
                "initializing the engine."
            )
        max_blocks_per_request = math.ceil(
            self.config.model_config.max_model_len / self.config.cache_config.block_size
        )
        if num_gpu_blocks < max_blocks_per_request:
            raise ValueError(
                f"Not enough available memory to schedule a request will maximum allowed length {self.config.model_config.max_model_len}. "
                f"Need {max_blocks_per_request}, available {num_gpu_blocks} gpu blocks. "
                f"Try decreasing `max_batch_size`, `max_model_len`."
            )
        self.config.cache_config.num_gpu_blocks = num_gpu_blocks

        # Initialize the cache.
        self._run_workers(
            "init_cache_engine",
            cache_config=self.config.cache_config,
            get_all_outputs=True,
        )

    def _init_worker_map(self) -> None:
        model_parallel_ranks = self._run_workers(
            "get_model_parallel_ranks",
            get_all_outputs=True,
        )

        self.worker_map = {mp_rank: i for i, mp_rank in enumerate(model_parallel_ranks)}

    def _on_step_completed(
        self,
        scheduler_outputs: SchedulerOutputs,
        scheduler_metadata: SchedulerMetadata,
        ignored_seqs: List[SequenceMetadata],
        seq_metadata_list: List[SequenceMetadata],
        sampler_outputs: Optional[SamplerOutputs],
        start_time: float,
        aborted_seq_list: Optional[List[Sequence]] = [],
    ) -> List[RequestOutput]:
        with self._process_model_outputs_timer:
            self.seq_manager.on_step_completed(
                scheduler_outputs,
                sampler_outputs,
            )
            self.scheduler.on_step_completed()

        end_time = time.perf_counter()

        self.metrics_store.on_aborted_requests(aborted_seq_list)
        self.metrics_store.on_batch_end(
            seq_metadata_list=seq_metadata_list,
            scheduler_metadata = scheduler_metadata,
            scheduler_outputs=scheduler_outputs,
            batch_start_time=start_time,
            batch_end_time=end_time,
        )
        _ignored_seqs = ignored_seqs + aborted_seq_list
        all_request_outputs = self.seq_manager.generate_request_outputs(
            _ignored_seqs, seq_metadata_list
        )
        return all_request_outputs

    def on_schedule(self, seq_metadata_list: List[SequenceMetadata],
                    scheduler_metadata: SchedulerMetadata, start_time: float):
        end_time = time.perf_counter()
        remaining_kv_blocks = self.scheduler.block_manager.get_num_free_gpu_blocks()
        scheduler_metadata.remaining_kv_blocks = remaining_kv_blocks
        self.metrics_store.on_schedule(seq_metadata_list, scheduler_metadata, start_time, end_time)

    def get_model_config(self) -> ModelConfig:
        return self.config.model_config

    def add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        seq_id: Optional[str] = None,
        is_in_window: Optional [bool] = False,
    ) -> int:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            seq_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current time.
            is_in_window: Flag to specify if the request is within the
                measurement window or not.
        """
        if arrival_time is None:
            arrival_time = time.monotonic()

        if not seq_id:
            seq_id = str(next(self.seq_counter))

        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)
        
        padding_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else -100

        # Create the sequences.
        block_size = self.config.cache_config.block_size
        eos_token_id = self.tokenizer.eos_token_id

        seq = Sequence(
            seq_id,
            prompt,
            prompt_token_ids,
            block_size,
            eos_token_id,
            arrival_time,
            sampling_params,
            is_in_window,
            padding_token_id,
        )
        # Add the sequence to the scheduler.
        self.seq_manager.add_seq(seq)
        # we create a copy of the seq so that the workers
        # receive an unmodified version of the seq
        # which is unaffected by the engine's actions
        self._append_new_seq(copy.deepcopy(seq))
        self.scheduler.add_seq(seq)
        self.metrics_store.on_request_arrival(seq)
        return seq_id

    @synchronized
    def _append_new_seq(self, seq: Sequence) -> None:
        self.new_seqs.append(seq)

    @synchronized
    def _get_new_seqs(
        self,
    ) -> List[Sequence]:
        new_seqs = self.new_seqs
        self.new_seqs = []
        return new_seqs

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seqs()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()

    def process_scheduler_metadata(self, scheduler_metadata: SchedulerMetadata):
        pass

    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration.
        Then, it executes the model and updates the scheduler with the model outputs.
        Finally, it decodes the sequences and returns the newly generated results.
        """
        start_time = time.perf_counter()
        self.seq_manager.reset_before_schedule()

        aborted_seqs = []
        if self.config.scheduler_config.allow_drop_requests:
            aborted_seqs = self.scheduler.abort_long_waiting_seq()

        with self._scheduler_timer:
            scheduler_outputs, scheduler_metadata = self.scheduler.schedule()
        self.process_scheduler_metadata(scheduler_metadata)

        if scheduler_outputs.is_empty() and len(aborted_seqs) == 0:
            return []
        elif scheduler_outputs.is_empty():
            self._send_step_to_worker(scheduler_outputs, aborted_seqs, output=False)
            self.metrics_store.on_aborted_requests(aborted_seqs)
            output = self.seq_manager.generate_request_outputs(aborted_seqs, [])
            return output

        ignored_seqs, seq_metadata_list = self.seq_manager.on_schedule(
            scheduler_outputs
        )
        self.on_schedule(seq_metadata_list, scheduler_metadata, start_time)
        sampler_outputs = self._send_step_to_worker(scheduler_outputs, aborted_seqs)
        return self._on_step_completed(
            scheduler_outputs,
            scheduler_metadata,
            ignored_seqs,
            seq_metadata_list,
            sampler_outputs,
            start_time,
            aborted_seqs
        )
    
    def _send_step_to_worker(self, scheduler_outputs, aborted_seqs, output=True):
        self.enqueue_socket.send_pyobj(
            StepInputs(
                scheduler_outputs,
                new_seqs=self._get_new_seqs(),
                aborted_seqs = aborted_seqs
            )
        )
        for aborted_seq in aborted_seqs:
            self.seq_manager._free_seq(aborted_seq.seq_id)
        if output:
            return self.output_socket.recv_pyobj() 

    def _run_workers(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        ignore_output: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        for worker in self.workers:
            executor = partial(worker.execute_method.remote, method)

            output = executor(*args, **kwargs)
            all_outputs.append(output)

        if ignore_output:
            return

        while True:
            try:
                all_outputs = ray.get(all_outputs)
                break
            except ray.exceptions.GetTimeoutError:
                time.sleep(0)
                continue

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output

    def _run_worker(
        self,
        model_parallel_rank: ModelParallelRank,
        method: str,
        *args,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        worker = self.workers[self.worker_map[model_parallel_rank]]
        executor = partial(worker.execute_method.remote, method)

        output = executor(*args, **kwargs)

        while True:
            try:
                output = ray.get(output)
                break
            except ray.exceptions.GetTimeoutError:
                time.sleep(0)
                continue

        return output

    def plot_metrics(self) -> None:
        self.metrics_store.plot()

    def pull_worker_metrics(self) -> None:
        worker_metrics = self._run_workers(
            "get_metrics_store",
            get_all_outputs=True,
        )
        for worker_metric in worker_metrics:
            self.metrics_store.merge(worker_metric)

    def mark_initial_memory_profiling_done(self):
        self.metrics_store.mark_initial_memory_profiling_done()
        self._run_workers("mark_initial_memory_profiling_done", get_all_outputs=True)

    def reset_metrics(self) -> None:
        self.scheduler.reset_state()
        self.metrics_store.reset()
        self._run_workers("reset_metrics", get_all_outputs=True)

    def start_profiling(self) -> None:
        self._run_workers("start_profiling")

    def stop_profiling(self) -> None:
        self._run_workers("stop_profiling")

    def get_metric_store(self) -> MetricsStore:
        return self.metrics_store

    def get_finished_prefill_seqs(self):
        return self.seq_manager.get_finished_prefill_seqs()
    
    def get_block_table_size(self, seq: Sequence):
        return len(self.scheduler.get_block_table(seq))

    def allocate_for_kvcache(self, kv_metadata):
        return self.scheduler.allocate_for_kvcache(kv_metadata)

    def add_decode_sequence(self, seq: Sequence):
        self.scheduler.running.append(seq)
        