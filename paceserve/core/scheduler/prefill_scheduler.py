import time
from typing import List

from paceserve.config import (
    CacheConfig, 
    ModelConfig, 
    ParallelConfig, 
    DistserveSchedulerConfig, 
    BatchregressionConfig
)
from paceserve.core.block_space_manager.distserve_block_space_manager import (
    DistserveBlockSpaceManager,
)
from paceserve.core.datatypes.scheduler_output import SchedulerOutputs
from paceserve.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from paceserve.core.scheduler.base_scheduler import BaseScheduler
from paceserve.logger import init_logger
from paceserve.types import InstanceType
from paceserve.core.datatypes.scheduler_metadata import SchedulerMetadata
from paceserve.core.datatypes.batch import Batch, PrefillRequest, DecodeRequest


logger = init_logger(__name__)


class PrefillScheduler(BaseScheduler):

    def __init__(
        self,
        engine_type: InstanceType,
        model_config: ModelConfig,
        scheduler_config: DistserveSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        batchregression_config: BatchregressionConfig
    ) -> None:
        super().__init__(engine_type, model_config, scheduler_config, cache_config, parallel_config, batchregression_config)
        self.enable_premption = scheduler_config.enable_premption
        self.max_num_batched_tokens = self.scheduler_config.get_max_num_batched_tokens(
            self.model_config.max_model_len
        )
        self.prompt_limit = self.max_num_batched_tokens
        self.engine_type = engine_type
        print(f"engine_type: {self.engine_type}")

    def get_block_space_manager_class(self):
        return DistserveBlockSpaceManager
    # this can be replaced with VLLM
    def _schedule(self) -> SchedulerOutputs:
        # Fix the current time.
        now = time.monotonic()

        ignored_seq_ids: List[str] = []
        preempted_seqs: List[Sequence] = []
        preempted_seq_ids: List[str] = []
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []
        
        prefill_token_list: List[int] = []  # prefilled token up to this scheduling
        chunk_token_list: List[int] = []
        processed_token_list: List[int] = []

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
            prefill_token_list.append(seq.prompt_tokens_processed + len(seq.prompt_token_ids))
            chunk_token_list.append(len(seq.prompt_token_ids))
        
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
            
