from typing import List, Optional
from threading import Event, Thread
import time

from paceserve.logger import init_logger
from paceserve.types import InstanceType
from paceserve.core.datatypes.request_output import RequestOutput
from paceserve.config import SystemConfig
from paceserve.core.datatypes.sequence import Sequence
from paceserve.engine.base_llm_engine import BaseLLMEngine
from paceserve.core.datatypes.scheduler_metadata import SchedulerMetadata
from paceserve.core.datatypes.sequence import SamplerOutputs, Sequence, SequenceMetadata
from paceserve.core.datatypes.scheduler_output import SchedulerOutputs
from paceserve.core.datatypes.kv_cache_obj import KVobj


logger = init_logger(__name__)


class PrefillLLMEngine(BaseLLMEngine):
    def __init__(self, config: SystemConfig) -> None:
        super().__init__(config)
        self.has_started_kv_retrival_loop = False
        self.pending_decode = []
        self.kv_cache_retrival_thread = Thread(target=self._kvcache_retrival_loop, daemon=True)
    
    def _kvcache_retrival_loop(self):

        while True: 

            if self.scheduler.complete_prefill:
                seq = self.scheduler.complete_prefill.pop(0)
                result = self._run_workers(
                        "_get_kv_cache",
                        get_all_outputs=True,
                        seq=seq
                    )
                self.pending_decode.put(KVobj(seq, result[0][0], result[0][1]))
    
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