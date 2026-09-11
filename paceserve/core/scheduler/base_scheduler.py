from abc import ABC, abstractmethod
from typing import List

from paceserve.config import (
    BaseSchedulerConfig, 
    CacheConfig, 
    ModelConfig, 
    ParallelConfig, 
    BatchregressionConfig
)
from paceserve.core.block_space_manager.block_space_manager_registry import (
    BlockSpaceManagerRegistry,
)
from paceserve.core.datatypes.scheduler_output import SchedulerOutputs
from paceserve.core.datatypes.scheduler_metadata import SchedulerMetadata
from paceserve.core.datatypes.sequence import Sequence, SequenceStatus
from paceserve.core.policy import PolicyFactory
from paceserve.logger import init_logger
from paceserve.metrics.metrics_store import MetricsStore
from paceserve.types import InstanceType
from paceserve.utils.throttled_logger import ThrottledLogger

logger = init_logger(__name__)
t_logger = ThrottledLogger()


class BaseScheduler(ABC):

    def __init__(
        self,
        engine_type: InstanceType,
        model_config: ModelConfig,
        scheduler_config: BaseSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        batchregression_config: BatchregressionConfig
    ) -> None:
        self.instance_id = None
        self.metrics_store = MetricsStore.get_instance()
        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.batchregression_config = batchregression_config

        # we maintain this just for logging purposes
        self._iteration_id = -1
        
        # Instantiate the scheduling policy used for running queue.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        
        # Instantiate the scheduling policy for the prefill queue.
        # This serves the same purpose as value function, but simpler
        self.prefill_policy = None
        if scheduler_config.prefill_policy:
            self.prefill_policy = PolicyFactory.get_policy(policy_name=scheduler_config.prefill_policy)
            
        # Create the block space manager.
        self.block_manager = BlockSpaceManagerRegistry.get(
            scheduler_config.get_type(),
            cache_config.block_size,
            cache_config.num_gpu_blocks,
            model_config.max_model_len,
        )
        self.prompt_limit = model_config.max_model_len

        # number of running batches should be less than or equal to the number of pipeline stages
        self.num_running_batches = 0

        # TODO(zhuohan): Use deque instead of list for better performance.
        # Sequence groups in the WAITING state.
        self.waiting: List[Sequence] = []
        # Sequence groups in the RUNNING state.
        self.running: List[Sequence] = []
        
        self.num_aborted_seq: int = 0

    def reset_state(self) -> None:
        self._iteration_id = -1

    def add_seq(self, seq: Sequence) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq)

    def has_unfinished_seqs(self) -> bool:
        return (not self.waiting.empty()) or self.running

    def get_num_unfinished_seqs(self) -> int:
        return self.waiting.qsize() + len(self.running)

    @abstractmethod
    def _schedule(self) -> tuple[SchedulerOutputs, SchedulerMetadata]:
        pass
    
    def abort_long_waiting_seq(self):
        tmp_waiting = []
        abort_list = []
        while self.waiting:
            seq = self.waiting.pop(0)
            if seq.state.waiting_time + seq.state._estimated_prefill_time  > self.scheduler_config.request_drop_threshold:
                abort_list.append(seq)
                self.num_aborted_seq += 1
                seq.set_status(SequenceStatus.FINISHED_ABORTED)
            else:
                tmp_waiting.append(seq)

        t_logger.log(
            key='abort_long_waiting_seq',
            message=f'Total number of aborted sequences={self.num_aborted_seq}'
        )
        self.waiting = tmp_waiting
        return abort_list

    def on_schedule(
        self,
        scheduler_outputs: SchedulerOutputs,
        scheduler_metadata: SchedulerMetadata) -> tuple[SchedulerOutputs, SchedulerMetadata]:
        return scheduler_outputs, scheduler_metadata

    def schedule(self) -> tuple[SchedulerOutputs, SchedulerMetadata]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running and self.waiting.
        self._iteration_id += 1

        if self.num_running_batches >= self.parallel_config.pipeline_parallel_size:
            logger.debug(f'More than parallel config size | {self.num_running_batches=} | {self.parallel_config.pipeline_parallel_size=}')
            return SchedulerOutputs(
                self._iteration_id,
                ignored_seq_ids=[],
                preempted_seq_ids=[],
                scheduled_seq_metadata_list=[],
            ), SchedulerMetadata(self._iteration_id)

        scheduler_outputs, scheduler_metadata = self._schedule()

        if not scheduler_outputs.is_empty():
            self.num_running_batches += 1

        scheduler_outputs, scheduler_metadata = self.on_schedule(scheduler_outputs, scheduler_metadata)
        return scheduler_outputs, scheduler_metadata

    def free_finished_seqs(self) -> None:
        for seq in self.running:
            if seq.is_finished():
                self._free_seq(seq)
        self.running = [seq for seq in self.running if not seq.is_finished()]

    def on_step_completed(self) -> None:
        self.free_finished_seqs()
        self.num_running_batches -= 1
    
    def get_block_table(self, seq: Sequence):
        return self.block_manager.get_block_table(seq)

    def _allocate(self, seq: Sequence) -> None:
        self.block_manager.allocate(seq)

    def _free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def _append_slot(
        self,
        seq: Sequence,
    ) -> None:
        assert seq.is_executing()
        self.block_manager.append_slot(seq)

    def _preempt(
        self,
        seq: Sequence,
    ) -> None:
        assert seq.is_executing(), f"seq: {seq.seq_id}, status: {seq.get_status()}"
        self._free_seq(seq)
        self.waiting.append(seq)

    def _check_request_prompt_length(self, seq: Sequence) -> bool:
        if seq.get_len() > self.prompt_limit:
            logger.warning(
                f"Input prompt ({seq.get_len()} tokens) is too long"
                f" and exceeds limit of {self.prompt_limit}"
            )
            seq.set_status(SequenceStatus.FINISHED_IGNORED)
            return False

        return True
