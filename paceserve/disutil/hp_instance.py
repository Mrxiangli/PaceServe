import logging
import copy
import time
from queue import Empty, Queue

import ray

from paceserve.disutil.instance import Instance
from paceserve.config.config import HighPrioritySchedulerConfig
from paceserve.utils.throttled_logger import ThrottledLogger

logger = logging.getLogger(__name__)
t_logger = ThrottledLogger()


@ray.remote
class HPInstance(Instance):

    def __init__(
        self,
        *args, **kwargs) -> None:
        args_list = list(args)  
        config = args_list[2]
        config.scheduler_config = HighPrioritySchedulerConfig(
            enforce_max_num_seqs=getattr(config.scheduler_config, "hp_enforce_max_num_seqs", 0),
            max_batched_tokens=config.scheduler_config.max_batched_tokens,
            max_num_seqs=(getattr(config.scheduler_config, "hp_max_num_seqs", None)
                          or config.scheduler_config.max_num_seqs),
            val_func=config.scheduler_config.val_func,
            allow_drop_requests = config.scheduler_config.allow_drop_requests,
            request_drop_threshold = config.scheduler_config.request_drop_threshold,
            ttft_slo = config.scheduler_config.ttft_slo,
            tbt_slo = config.scheduler_config.tbt_slo
        )
        #NOTE: Xiang: the allow drop flag bool not working, usiing int 0, 1 instead for now
        args_list[2] = config
        
        super().__init__(*args, **kwargs)
        self.transfer = 0
        self.config=config
        self._pending_sequences = Queue()

    def _seq_polling_ticket(self,):
        # dynamic shifting the ticket admission threshold, this is to avoid the KV cache memory being 
        # largely being consumed by the controller direct offloading request, leading to queuing delay of LP rescheduled request. 
        ratio = 1 - len(self.llm_engine.scheduler.running)/self.config.scheduler_config.max_num_seqs
        if self._pending_sequences.empty() and (len(self.llm_engine.scheduler.waiting) == 0) and len(self.requests) == 0 \
                and len(self.llm_engine.scheduler.running) < ratio * self.config.scheduler_config.max_num_seqs \
                and self.llm_engine.scheduler.block_manager.get_num_free_gpu_blocks()/self.llm_engine.scheduler.block_manager.num_total_gpu_blocks > 0.1:
            return True
        return False

    def _add_sequence(self, seq_ref):
        seq = ray.get(seq_ref) if isinstance(seq_ref, ray.ObjectRef) else seq_ref
        received_at = time.time()
        if seq is not None:
            # Stamp receipt before queueing so local queue time is not counted
            # as transfer time. The sequence is not yet shared with the engine.
            seq.state.on_seq_arrive()
            seq.state.on_reschedule_end()
        self._pending_sequences.put((seq, received_at))

    def _process_pending_sequences(self):
        # Ray RPC threads only enqueue. Engine state and metrics are updated
        # here, between steps, so admission control cannot overwrite arrivals.
        for _ in range(self._pending_sequences.qsize()):
            try:
                seq, received_at = self._pending_sequences.get_nowait()
            except Empty:
                break
            self.llm_engine.metrics_store.log_hp_event(
                received_at, "URGENT_RECEIVED", int(seq is not None)
            )
            if seq is None:
                continue
            self.llm_engine.seq_manager.add_seq(seq)
            self.llm_engine._append_new_seq(copy.deepcopy(seq))
            self.llm_engine.scheduler.add_seq(seq)
            self.llm_engine.metrics_store.on_request_offload_arrive(seq)
            self.transfer += 1
            t_logger.log(
                key='_add_sequence',
                message=f'hp: total received: {self.transfer}'
            )
