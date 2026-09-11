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

# TODO: Remove this from here because it's defined again
# in resource_item.


class PaceScheduler(BaseScheduler):

    def __init__(
        self,
        engine_type: InstanceType,
        model_config: ModelConfig,
        scheduler_config: PaceSchedulerConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        batchregression_config: BatchregressionConfig
    ) -> None:
        raise NotImplementedError
