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


class DistserveScheduler(BaseScheduler):

    def __init__(
        self,
        engine_type: InstanceType,
        model_config: ModelConfig,
        scheduler_config: DistserveSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        batchregression_config: BatchregressionConfig
    ) -> None:
        raise NotImplementedError