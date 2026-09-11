from paceserve.config import SchedulerType
from paceserve.core.scheduler.faster_transformer_scheduler import (
    FasterTransformerScheduler,
)
from paceserve.core.scheduler.orca_scheduler import OrcaScheduler
from paceserve.core.scheduler.sarathi_scheduler import SarathiScheduler
from paceserve.core.scheduler.niyama_scheduler import NiyamaScheduler
from paceserve.core.scheduler.simple_chunking_scheduler import SimpleChunkingScheduler
from paceserve.core.scheduler.vllm_scheduler import VLLMScheduler
from paceserve.core.scheduler.distserve_scheduler import DistserveScheduler
from paceserve.core.scheduler.prefill_scheduler import PrefillScheduler
from paceserve.core.scheduler.decode_scheduler import DecodeScheduler
from paceserve.core.scheduler.pace_scheduler import PaceScheduler
from paceserve.core.scheduler.high_priority_scheduler import HighPriorityScheduler
from paceserve.core.scheduler.low_priority_scheduler import LowPriorityScheduler
from paceserve.utils.base_registry import BaseRegistry


class SchedulerRegistry(BaseRegistry):

    @classmethod
    def get_key_from_str(cls, key_str: str) -> SchedulerType:
        return SchedulerType.from_str(key_str)


SchedulerRegistry.register(SchedulerType.VLLM, VLLMScheduler)
SchedulerRegistry.register(SchedulerType.DIST, DistserveScheduler)
SchedulerRegistry.register(SchedulerType.PREFILL, PrefillScheduler)
SchedulerRegistry.register(SchedulerType.DECODE, DecodeScheduler)
SchedulerRegistry.register(SchedulerType.PACE, PaceScheduler)
SchedulerRegistry.register(SchedulerType.HIGH, HighPriorityScheduler)
SchedulerRegistry.register(SchedulerType.LOW, LowPriorityScheduler)
SchedulerRegistry.register(SchedulerType.ORCA, OrcaScheduler)
SchedulerRegistry.register(SchedulerType.FASTER_TRANSFORMER, FasterTransformerScheduler)
SchedulerRegistry.register(SchedulerType.SARATHI, SarathiScheduler)
SchedulerRegistry.register(SchedulerType.NIYAMA, NiyamaScheduler)
SchedulerRegistry.register(SchedulerType.SIMPLE_CHUNKING, SimpleChunkingScheduler)
