import time
import queue
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
from paceserve.core.datatypes.batch import Batch, DecodeRequest
from paceserve.core.datatypes.kv_cache_obj import KVmetadata


logger = init_logger(__name__)


class DecodeScheduler(BaseScheduler):

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
        self.decode_prempted_seqs = []
        self.kvmeta_queue = queue.Queue()
        # hold kvmeta for those requests that are allocated on engine side, waiting to insert kvcache
        self.pending_kv_transfer = queue.Queue()
        # holds requests that are previously preemptied, now reallocated memory on engine
        self.pending_kv_insert = queue.Queue()
        # holds new request with kv cache inserted, can do decoding
        self.pending_decode = queue.Queue() 
        # Use this to check if the current step results preemption
        # since the engine side preemption happens before the worker side,
        # a poentially issue is that engine side release the block, assuming a new
        # sequence can be added, thus initiate a kv insertion on the worker side (whose block has not been released yet) 
        self.preemption_this_step = False

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
        

        batch = Batch()   
        # We first add those request from the prefill isntance
        # that has already finished kv transfer to running queue, 
        while self.pending_decode.qsize()>0:
            seq = self.pending_decode.get()
            self.running.append(seq)
        
        # sorting by time
        self.running = self.policy.sort_by_priority(now, self.running)
        
        aborted_seq_ids: List[str] = []
        running: List[Sequence] = []
        
        # we first get all running sequence allocated
        while self.running:
            # check if reach the maximum number of decode
            if batch.num_decode >= self.scheduler_config.max_num_seqs:
                break
            
            seq = self.running.pop(0)

            if not seq.is_paused():
                # The sequence group is already in the RUNNING state.
                running.append(seq)
                continue

            assert seq.prompt_stage_processing_finished

            while not self.block_manager.can_append_slot():
                if self.running:
                    victim_seq = self.running.pop(-1)
                    # this directly free the block space
                    self._preempt(victim_seq)
                    preempted_seq_ids.append(victim_seq.seq_id)
                    preempted_seqs.append(victim_seq)
                    self.decode_prempted_seqs.append(victim_seq)
                    self.preemption_this_step = True
                else:
                    self._preempt(seq)
                    preempted_seq_ids.append(seq.seq_id)
                    preempted_seqs.append(seq)
                    self.decode_prempted_seqs.append(seq)
                    self.preemption_this_step = True
                    break
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq)
                running.append(seq)
                
                scheduled_seq_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(seq)
                )
                batch.add_request(DecodeRequest(seq))
            
            #processed_token_list.append(seq.prompt_tokens_processed + len(seq.output_token_ids))
        running = running + self.running 
        self.running = running
            
        # we check the preempited sequence to see if we can bring anything back, we follow the order of preemption to do that
        if self.decode_prempted_seqs and not self.preemption_this_step:
            seq = self.decode_prempted_seqs[0]
            can_allocate = self.block_manager.can_allocate(seq)
            if can_allocate:
                seq = self.decode_prempted_seqs.pop(0)
                self._allocate(seq)
                self.pending_kv_insert.put(seq)
        
        # allocate for new sequence from prefill instance, not preempt : wait for worker and engine block space manager sync
        while not self.preemption_this_step and self.kvmeta_queue.qsize() > 0 :
            kv_metadata: KVmetadata = self.kvmeta_queue.queue[0]
            can_allocate = self.allocate_for_kvcache(kv_metadata)
            if can_allocate:
                kv_metadata = self.kvmeta_queue.get()
                self.pending_kv_transfer.put(kv_metadata)
            else:
                break      
        
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
            preempt_recompute = False
        )
        return sch_outputs, scheduler_metadata
    
    def allocate_for_kvcache(self, kv_metadata):
        # if cannot allocate (this shouldn't be the case as the engine should be synced)
        if not self.block_manager.can_allocate(kv_metadata.seq):
            return False

        self._allocate(kv_metadata.seq)
        return True
    
    def _preempt(
        self,
        seq: Sequence,
    ) -> None:
        assert seq.is_executing(), f"seq: {seq.seq_id}, status: {seq.get_status()}"
        self._free_seq(seq)
        logger.debug(f"seq {seq} is preemptied, number of free blocks on engine: {self.block_manager.get_num_free_gpu_blocks()}")                
        # we do not put it back to waiting queue, as the preemptied sequence will be rescheudled byt instance
        # self.waiting.append(seq)

