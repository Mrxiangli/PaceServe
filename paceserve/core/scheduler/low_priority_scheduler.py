import time
from typing import List
import numpy as np

from paceserve.config import (
    CacheConfig, 
    ModelConfig, 
    ParallelConfig, 
    PaceSchedulerConfig, 
    BatchregressionConfig
)
from paceserve.core.block_space_manager.pace_block_space_manager import (
    PaceBlockSpaceManager,
)
from paceserve.core.datatypes.scheduler_output import SchedulerOutputs
from paceserve.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from paceserve.core.scheduler.base_scheduler import BaseScheduler
from paceserve.logger import init_logger
from paceserve.types import InstanceType
from paceserve.ai_utils.ai_helper import AIHelper
from paceserve.ai_utils.solver import GreedySolver, NonPreemptiveSolver
from paceserve.core.datatypes.batch import Batch, PrefillRequest, DecodeRequest
from paceserve.core.datatypes.scheduler_metadata import SchedulerMetadata
from paceserve.ai_utils.resource_item import BaseItem

logger = init_logger(__name__)

class LowPriorityScheduler(BaseScheduler):

    def __init__(
        self,
        engine_type: InstanceType,
        model_config: ModelConfig,
        scheduler_config: PaceSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        batchregression_config: BatchregressionConfig
    ) -> None:
        super().__init__(engine_type, model_config, scheduler_config, cache_config, parallel_config, batchregression_config)
        self.engine_type = engine_type
        self.max_num_batched_tokens = self.scheduler_config.get_max_num_batched_tokens(
            self.model_config.max_model_len
        )
        self.TTFT_SLO = self.scheduler_config.ttft_slo
        #self.solver = GreedySolver(scheduler_config)
        self.solver = NonPreemptiveSolver(scheduler_config)
        
        self.ai_helper = AIHelper()
        self.ai_helper.initialize(model_config, scheduler_config, batchregression_config)
        self.prempt_count = 0 
        self.full_batch_execution = 1.0
        
        # Hugging Face sets _name_or_path to the local directory when loading
        # downloaded weights. Keep the same allowance for local and Hub names.
        model_name = model_config.hf_config._name_or_path.rstrip('/').rsplit('/', 1)[-1].lower()
        if model_name in ('meta-llama-3.1-8b-instruct', 'llama-3.1-8b-instruct'):
            self.full_batch_execution = 0.7
        elif model_name == 'mistral-7b-v0.1':
            self.full_batch_execution = 0.5
        else:
            self.full_batch_execution = 1.0 
        
        # 1.0: Qwen-14B
        # 0.7: llama3-8B
        # 0.5: mistral-7B

        # This contains sequences that did their partial prefill in a
        # previous batch but were not picked to finish their prefill in any of
        # the succeeding batches.
        self.partial_prefills: List[Sequence] = []

    def get_block_space_manager_class(self):
        return PaceBlockSpaceManager

    def _seq_offloading(self, batch):
        now = time.monotonic()
        seq_to_transfer: List[Sequence] = []
        waiting_counter = len(self.waiting)
        batch_runtime = self.ai_helper.estimate_time(batch)
        tmp_waiting = []
        for i in range(waiting_counter):
            seq = self.waiting.pop()
            # Only offload the requests that have not been scheduled yet
            if (now - seq.arrival_time + max(batch_runtime, seq.state.estimated_prefill_time) + self.full_batch_execution > self.TTFT_SLO) and \
                    (not seq.state.is_scheduled):
                seq_to_transfer.append(seq)
            else:
                tmp_waiting.append(seq)
                
        self.waiting = tmp_waiting + self.waiting[waiting_counter:]
        return seq_to_transfer

    def on_schedule(
        self,
        scheduler_outputs: SchedulerOutputs,
        scheduler_metadata: SchedulerMetadata) -> tuple[SchedulerOutputs, SchedulerMetadata]:
        if self.engine_type != InstanceType.LOW_PRIORITY:
            return scheduler_outputs, scheduler_metadata

        scheduler_metadata.seq_transfer_list = self._seq_offloading(scheduler_metadata.batch)
        return scheduler_outputs, scheduler_metadata 

    def _schedule(self) -> SchedulerOutputs:
        now = time.monotonic()

        running: List[Sequence] = []
        # TODO: We can move ignored_seq_ids in base scheduler.
        # This does not do anything special in any of the children
        ignored_seq_ids: List[str] = []
        preempted_seq_ids: List[str] = []
        preempted_seqs: List[Sequence] = []
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []
        num_batched_tokens: int = 0

        batch = Batch()
        self.running = self.policy.sort_by_priority(now, self.running)

        def _add_seq_to_batch(seq: Sequence, prompt_chunk_len=None, value=None):
            nonlocal batch, num_batched_tokens, scheduled_seq_metadata_list
            # prompt_chunk_len == None if seq in decode
            if prompt_chunk_len is None:
                batch.add_request(DecodeRequest(seq))
                num_batched_tokens += 1
                # For decode requests, the parent creates a new self.running
                # hence we cannot modify self.running
            else:
                batch.add_request(PrefillRequest(seq, chunk_size=prompt_chunk_len, value=value))
                self.running.append(seq)
                num_batched_tokens = prompt_chunk_len
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(seq, prompt_chunk_len)
            )

        # we do not want to stall decoding, adding all exisiting decoding sequence first
        while self.running:
            seq = self.running.pop(0)
            if not seq.is_paused():
                running.append(seq)
                continue

            if not seq.prompt_stage_processing_finished:
                self.partial_prefills.append(seq)
                continue

            # Defer excess sequences without losing their allocated state.
            if len(running) >= self.scheduler_config.max_num_seqs:
                running.append(seq)
                running.extend(self.running)
                break

            while not self.block_manager.can_append_slot():
                self.prempt_count += 1
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq = self.running.pop(-1)
                    self._preempt(victim_seq)
                    preempted_seq_ids.append(victim_seq.seq_id)
                    preempted_seqs.append(victim_seq)
                    logger.info(f'Preempting a request {victim_seq.seq_id}, total requets preemptied {self.prempt_count}')
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    # NOTE: This will move a decode to WAITING. However, we
                    # consider all the requests in the self.waiting to be
                    # prefill and run the solver. In the case when a decode
                    # moves to self.waiting, we should priortize that request
                    # when picking a new sequence from self.waiting or do
                    # something else. RUnning the solver on decode sequence
                    # will potentially cause it to pick partial decode which 
                    # will result in error.
                    self._preempt(seq)
                    preempted_seq_ids.append(seq.seq_id)
                    preempted_seqs.append(seq)
                    logger.info(f'Preempting a request {seq.seq_id}, total requets preemptied {self.prempt_count}')
                    break
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq)
                running.append(seq)
                _add_seq_to_batch(seq)
        # All the sequences in self.running now are the requests doing their decode
        self.running = running

        partial_prefill_counter = len(self.partial_prefills)
        waiting_counter = len(self.waiting)
        # NOTE: consider the cases where sequence being added to the waiting after counter being set
        prefill_seqs: List[Sequence] = self.partial_prefills + self.waiting[:waiting_counter]
        remaining_gpu_blocks = self.block_manager.get_num_free_gpu_blocks()
        # Either don't have enought memory or no prefill to schedule; if there is sequence preemptied, low memory, no point to schedule from waitintg.
        if (len(prefill_seqs) == 0 or remaining_gpu_blocks <= 0 or preempted_seqs
                or len(self.running) >= self.scheduler_config.max_num_seqs
                or len(self.running) >= self.max_num_batched_tokens):
            scheduler_metadata = SchedulerMetadata(
                id=self._iteration_id,
                batch=batch,
                waiting=self.waiting,
                preempted=preempted_seqs,
                partial_prefills=self.partial_prefills,
            )
            sch_outputs = SchedulerOutputs(
                id=self._iteration_id,
                ignored_seq_ids=ignored_seq_ids,
                preempted_seq_ids=preempted_seq_ids,
                scheduled_seq_metadata_list=scheduled_seq_metadata_list,
            )
            return sch_outputs, scheduler_metadata

        remaining_tokens = self.max_num_batched_tokens - len(self.running)
        remaining_batch_size = self.scheduler_config.max_num_seqs - len(self.running)
        items, _remaining_compute,  _num_tokens = self.solver.solve(
            prefill_seqs,
            remaining_gpu_blocks,
            batch_size=remaining_batch_size,
            max_tokens=remaining_tokens
        )
        items: List[BaseItem]

        tmp_partial_prefills: List[Sequence] = []
        # we allocate the memory for the whole sequence when bring from waiting to partial prefill
        for item in items[:partial_prefill_counter]:
            seq = item.sequence
            if item.frac and self.block_manager.can_allocate(seq):
                next_num_prefill_tokens = int((len(seq.prompt_token_ids)- seq.prompt_tokens_processed) * item.frac)
                assert next_num_prefill_tokens != 0, f"seq: {seq} frac: {item.frac}"
                _add_seq_to_batch(seq, prompt_chunk_len=next_num_prefill_tokens, value=item.value)
            else:
                tmp_partial_prefills.append(seq)
            self.partial_prefills = tmp_partial_prefills

        tmp_waiting: List[Sequence] = []
        # sequence are being moved from waiting to running, we need to allocate memory for the entire sequence prfill here
        for item in items[partial_prefill_counter:]:
            seq = item.sequence
            if item.frac:
                if not self._check_request_prompt_length(seq):
                    ignored_seq_ids.append(seq.seq_id)
                    continue

                if not self.block_manager.can_allocate(seq):
                    tmp_waiting.append(seq)
                    continue

                self._allocate(seq)
                next_num_prefill_tokens = int((len(seq.prompt_token_ids)- seq.prompt_tokens_processed) * item.frac)
                assert next_num_prefill_tokens != 0, f"seq: {seq} frac: {item.frac}"
                _add_seq_to_batch(seq, prompt_chunk_len=next_num_prefill_tokens, value=item.value)
            else:
                tmp_waiting.append(seq)
        self.waiting = tmp_waiting + self.waiting[waiting_counter:]

        scheduler_metadata = SchedulerMetadata(
            id = self._iteration_id,
            batch=batch,
            waiting=self.waiting,
            preempted=preempted_seqs,
            partial_prefills=self.partial_prefills,
            items=items,
            remaining_compute=_remaining_compute,
            remaining_tokens=remaining_tokens - _num_tokens,
        )
        sch_outputs = SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=preempted_seq_ids,
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
        )
        return sch_outputs, scheduler_metadata
