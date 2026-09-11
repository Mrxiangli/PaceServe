import time
from typing import List

from paceserve.config import (
    CacheConfig, 
    ModelConfig, 
    ParallelConfig, 
    VllmSchedulerConfig, 
    BatchregressionConfig
)
from paceserve.core.block_space_manager.vllm_block_space_manager import (
    VLLMBlockSpaceManager,
)
from paceserve.core.datatypes.scheduler_output import SchedulerOutputs
from paceserve.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from paceserve.core.scheduler.base_scheduler import BaseScheduler
from paceserve.logger import init_logger
from paceserve.types import InstanceType
from paceserve.core.policy import PolicyFactory
from paceserve.core.datatypes.scheduler_metadata import SchedulerMetadata
from paceserve.core.datatypes.batch import Batch, PrefillRequest, DecodeRequest

logger = init_logger(__name__)


class VLLMScheduler(BaseScheduler):

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
        self.engine_type = engine_type
        self.max_num_batched_tokens = self.scheduler_config.get_max_num_batched_tokens(
            self.model_config.max_model_len
        )
        self.prompt_limit = self.max_num_batched_tokens

    def get_block_space_manager_class(self):
        return VLLMBlockSpaceManager

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
        if self.prefill_policy:
            self.waiting= self.prefill_policy.sort_by_priority(now, self.waiting)

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
