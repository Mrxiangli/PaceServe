import logging
import ray
import queue
import time
from typing import List

from paceserve.disutil.instance import Instance
from paceserve.core.datatypes.kv_cache_obj import KVobj, KVmetadata
from paceserve.config.config import PrefillSchedulerConfig
from paceserve.utils.threading_utils import synchronized
from paceserve.benchmark.entities import Request
from paceserve.core.datatypes.request_output import RequestOutput
from paceserve.utils.throttled_logger import ThrottledLogger

logger = logging.getLogger(__name__)
t_logger = ThrottledLogger()

logger = logging.getLogger(__name__)


@ray.remote
class PrefillInstance(Instance):

    def __init__(
        self,
        *args, **kwargs) -> None:
        args_list = list(args)  
        config = args_list[2]
        config.scheduler_config = PrefillSchedulerConfig(
            max_batched_tokens=config.scheduler_config.max_batched_tokens,
            max_num_seqs=config.scheduler_config.max_num_seqs,
            allow_drop_requests = config.scheduler_config.allow_drop_requests,
            request_drop_threshold = config.scheduler_config.request_drop_threshold,
            ttft_slo = config.scheduler_config.ttft_slo,
            tbt_slo = config.scheduler_config.tbt_slo
        )
        args_list[2] = config
        super().__init__(*args, **kwargs)
        self.prefill_finished = 0
        self.decode_instance_count = None
        self.kvc_pending_transfer = queue.Queue()

    def warmup(self) -> None:
        dummy_Request = Request(arrived_at=time.monotonic(), num_prefill_tokens=1000, num_decode_tokens=200)
        self.llm_engine.add_request(**self._get_input_params(dummy_Request, float('inf')))
        step_outputs = self.llm_engine.step()
        if step_outputs[0].prefill_finished:
            self._on_step_complete()

        self.llm_engine.reset_metrics()
    
    def add_decode_instances(self, decode_instance_refs):
        self.decode_instance_refs = decode_instance_refs
        self.decode_instance_count = len(decode_instance_refs)
    
    def _run(self) -> None:
        self.start_time = time.monotonic()

        while not self.terminate:
            while self.requests:
                request = self.requests.pop(0)
                seq_id = self.llm_engine.add_request(**self._get_input_params(request, time.monotonic()))
                self.total_req += 1

            step_outputs: List[RequestOutput] = self.llm_engine.step()
            self.num_steps += 1
            self._on_step_complete()

            for output in step_outputs:
                if output.prefill_finished:
                    self.num_processed_requests += 1
                    # comment the following one as this will trigger instance termination
                    # self.finished_seq(output)
                    t_logger.log(
                        key='instance.run',
                        message=f'num_prefilled_requests: {self.num_processed_requests} free blocks: {self.llm_engine.scheduler.block_manager.get_num_free_gpu_blocks()}'
                    )
            self.gpu_max_blocks.set(self.llm_engine.config.cache_config.num_gpu_blocks)
            self.gpu_free_blocks.set(self.llm_engine.scheduler.block_manager.get_num_free_gpu_blocks())
        if self.calculate_rouge:
            self._evaluate_rouge()
        logger.debug('Returning from instance')
        self._run_finish.send.remote()
        self.run_loop_finished = True
    
    def _on_step_complete(self):
        for seq in self.llm_engine.get_finished_prefill_seqs():
            metadata = KVmetadata(seq, self.llm_engine.get_block_table_size(seq))
            metadata._prefill_instance = self.ref
            self.prefill_finished += 1
            metadata.seq.state._kvmeta_left_prefill = time.time()
            metadata.seq.state._kvmeta_left_prefill_monotonic = time.monotonic()
            self.decode_instance_refs[self.prefill_finished%self.decode_instance_count].add_kvc_metadata.remote(metadata)

    # this funcnction is called through remote() by decode instance to retrieve kv cache
    def transfer_kv_cache(self, seq_id):
        seq = self.llm_engine.seq_manager.get_seq(seq_id)
        result = self.llm_engine._run_workers(
                    "_get_kv_cache",
                    get_all_outputs=True,
                    seq=seq
                )
        # potentially race conditions, solved by lock free block with thread lock for now
        kvobj = KVobj(seq.seq_id, result[0][0], result[0][1], time.monotonic(), time.time())
        self.llm_engine.scheduler.block_manager.free(seq)
        self.llm_engine.seq_manager._free_seq(seq.seq_id)
        return kvobj
