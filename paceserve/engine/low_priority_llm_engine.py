from typing import List

from paceserve.logger import init_logger
from paceserve.types import InstanceType
from paceserve.config import SystemConfig
from paceserve.core.datatypes.sequence import Sequence
from paceserve.engine.base_llm_engine import BaseLLMEngine
from paceserve.core.datatypes.scheduler_metadata import SchedulerMetadata


logger = init_logger(__name__)


class LowPriorityLLMEngine(BaseLLMEngine):
    def __init__(self, config: SystemConfig) -> None:
        super().__init__(config)
        self.hp_req_queue: List[Sequence] = []

    def process_scheduler_metadata(self, scheduler_metadata: SchedulerMetadata):
        if self.config.instance_config._type is not InstanceType.LOW_PRIORITY:
            return

        for seq in scheduler_metadata.seq_transfer_list:
            seq.state.on_reschedule_start()
            seq.state.on_seq_left()
            seq.state.on_offload()
            self.metrics_store.on_request_offload(seq)
            self.hp_req_queue.append(seq)
            self.seq_manager._free_seq(seq.seq_id)
