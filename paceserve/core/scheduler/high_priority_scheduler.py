import time
from typing import List
import numpy as np

from paceserve.config import (
    ModelConfig,
    CacheConfig,
    ParallelConfig,
    VllmSchedulerConfig, 
    BatchregressionConfig
)
from paceserve.core.datatypes.scheduler_output import SchedulerOutputs
from paceserve.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from paceserve.core.scheduler.vllm_scheduler import VLLMScheduler
from paceserve.core.datatypes.batch import Batch, PrefillRequest, DecodeRequest
from paceserve.core.datatypes.scheduler_metadata import SchedulerMetadata
from paceserve.logger import init_logger
from paceserve.types import InstanceType
from paceserve.ai_utils.resource_item import BaseItem, BasicValue, \
    ShortestJobFirstValue, EarliestDeadlineFirstValue

logger = init_logger(__name__)

VALUE_FUNCTION_CLASSES = {
    "BasicValue": BasicValue,
    "ShortestJobFirstValue": ShortestJobFirstValue,
    "EarliestDeadlineFirstValue": EarliestDeadlineFirstValue,
}

class HighPriorityScheduler(VLLMScheduler):
    def __init__(
        self,
        engine_type: InstanceType,
        model_config: ModelConfig,
        scheduler_config: VllmSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        batchregression_config: BatchregressionConfig
    ) -> None:
        super().__init__(engine_type, model_config, scheduler_config, cache_config, parallel_config, batchregression_config)
        self.ttft_violating = []
        self.TTFT_SLO = scheduler_config.ttft_slo
        self.VALUE_FUNCTION_CLS = VALUE_FUNCTION_CLASSES.get(self.scheduler_config.val_func, EarliestDeadlineFirstValue)
        
    def _schedule(self) -> SchedulerOutputs:
        # Fix the current time.
        now = time.monotonic()

        ignored_seq_ids: List[str] = []
        preempted_seqs: List[Sequence] = []
        preempted_seq_ids: List[str] = []
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []

        batch = Batch()
        # The total number of sequences on the fly, including the
        # requests in the generation phase.
        num_batched_tokens = 0
        # Optimization: We do not sort the waiting queue since the preempted
        # sequence groups are added to the front and the new sequence groups
        # are added to the back.
        # if self.prefill_policy:
        #     self.waiting= self.prefill_policy.sort_by_priority(now, self.waiting)   
        tmp_len_waiting  = len(self.waiting)
        
        items = [self.VALUE_FUNCTION_CLS.from_sequence(seq, time.monotonic(), self.TTFT_SLO) for seq in self.waiting[0:tmp_len_waiting]]
        for i, item in enumerate(items):
            item.index = i
        sorted_items: List[self.VALUE_FUNCTION_CLS] = sorted(items,
                                                        key=lambda item: item.value,
                                                        reverse=True)
        #rev_sorted_items: List[VALUE_FUNCTION_CLS] = sorted(sorted_items, key=lambda item: item.index)
        tmp_waiting_non_violate = []
        tmp_waiting_violate = []
        for item in sorted_items:
            if item.value != float('inf'):
                tmp_waiting_non_violate.append(item.sequence)
            else:
                tmp_waiting_violate.append(item.sequence)
        self.waiting = tmp_waiting_non_violate + self.waiting[tmp_len_waiting:] + tmp_waiting_violate
        
        while self.waiting:
            seq = self.waiting[0]
            # This is required to handle benchmarking where
            # we set request arrival time ahead of time

            num_prompt_tokens = seq.get_len()
            if not self._check_request_prompt_length(seq):
                ignored_seq_ids.append(seq.seq_id)
                self.waiting.pop(0)
                continue

            # If the sequence group cannot be allocated, stop.
            if not self.block_manager.can_allocate(seq):
                break

            # If the number of batched tokens exceeds the limit, stop.
            if num_batched_tokens + num_prompt_tokens > self.max_num_batched_tokens:
                break

            if len(self.running) + 1 > self.scheduler_config.max_num_seqs:
                if (self.scheduler_config.enforce_max_num_seqs
                        or self.block_manager.get_num_free_gpu_blocks()/self.block_manager.num_total_gpu_blocks < 0.05):
                    break

            seq = self.waiting.pop(0)
            self._allocate(seq)
            batch.add_request(PrefillRequest(seq))
            num_batched_tokens += num_prompt_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(seq)
            )
            self.running.append(seq)

        if scheduled_seq_metadata_list or ignored_seq_ids:
            scheduler_metadata = SchedulerMetadata(
                id = self._iteration_id,
                batch=batch,
                waiting=self.waiting,
            )
            sch_outputs = SchedulerOutputs(
                id=self._iteration_id,
                ignored_seq_ids=ignored_seq_ids,
                preempted_seq_ids=[],
                scheduled_seq_metadata_list=scheduled_seq_metadata_list,
            )
            return sch_outputs, scheduler_metadata

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)

        # Reserve new token slots for the running sequence groups.
        running: List[Sequence] = []

        while self.running:
            seq = self.running.pop(0)

            if not seq.is_paused():
                # The sequence group is already in the RUNNING state.
                running.append(seq)
                continue

            assert seq.prompt_stage_processing_finished

            while not self.block_manager.can_append_slot():
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq = self.running.pop(-1)
                    self._preempt(victim_seq)
                    preempted_seq_ids.append(victim_seq.seq_id)
                    preempted_seqs.append(victim_seq)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq)
                    preempted_seq_ids.append(seq.seq_id)
                    preempted_seqs.append(seq)
                    break
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq)
                running.append(seq)
                scheduled_seq_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(seq)
                )
                batch.add_request(DecodeRequest(seq))
        self.running = running

        scheduler_metadata = SchedulerMetadata(
            id = self._iteration_id,
            batch=batch,
            waiting=self.waiting,
            preempted=preempted_seqs
        )
        sch_outputs = SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=[],
            preempted_seq_ids=preempted_seq_ids,
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
        )
        return sch_outputs, scheduler_metadata
