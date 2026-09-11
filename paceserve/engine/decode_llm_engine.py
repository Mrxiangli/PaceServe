from typing import List
from threading import Event, Thread
import time

from paceserve.logger import init_logger
from paceserve.types import InstanceType
from paceserve.core.datatypes.request_output import RequestOutput
from paceserve.config import SystemConfig
from paceserve.core.datatypes.sequence import Sequence
from paceserve.engine.base_llm_engine import BaseLLMEngine
from paceserve.core.datatypes.scheduler_metadata import SchedulerMetadata
from paceserve.core.datatypes.kv_cache_obj import KVobj
from ray.util.queue import Queue


logger = init_logger(__name__)


class DecodeLLMEngine(BaseLLMEngine):
    def __init__(self, config: SystemConfig) -> None:
        super().__init__(config)
        self.config = config
        self.kv_cache_mapper = {}
        self.pending_transfer_queue = Queue()
        self.has_kv_trasnfer_started = False
        self.preempt_seq_kv = dict()
        self.swapped_list = []
        
    # The following function is blocking inside of the step function, potentially affect the overall performance
    # for now, we keep it blocking for ensuring the memory blocks are syned in worker and engine.             
    def process_scheduler_metadata(self, scheduler_metadata):
        # retrieve the kv cache of the preempted sequence and move to DRAM
        while scheduler_metadata.preempted:
            seq = scheduler_metadata.preempted.pop()
            result = self._run_workers(
                "_get_kv_cache_nofree",
                get_all_outputs=True,
                seq=seq
            )
            self.preempt_seq_kv[seq.seq_id] = KVobj(seq.seq_id, result[0][0], result[0][1])  
            logger.debug(f"seq {seq} is preemptied, # logical token: {len(seq.logical_token_blocks)}, empty slots: {seq.logical_token_blocks[-1].get_num_empty_slots()}, physical block released: {len(result[0][1])}")                
        
    
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
        #ensure the worker space is released
        self.scheduler.preemption_this_step = False
        return self._on_step_completed(
            scheduler_outputs,
            scheduler_metadata,
            ignored_seqs,
            seq_metadata_list,
            sampler_outputs,
            start_time,
            aborted_seqs
        )