from paceserve.config import SchedulerType
from paceserve.core.block_space_manager.faster_transformer_block_space_manager import (
    FasterTransformerBlockSpaceManager,
)
from paceserve.core.block_space_manager.orca_block_space_manager import (
    OrcaBlockSpaceManager,
)
from paceserve.core.block_space_manager.sarathi_block_space_manager import (
    SarathiBlockSpaceManager,
)
from paceserve.core.block_space_manager.niyama_block_space_manager import (
    NiyamaBlockSpaceManager,
)
from paceserve.core.block_space_manager.simple_chunking_block_space_manager import (
    SimpleChunkingBlockSpaceManager,
)
from paceserve.core.block_space_manager.vllm_block_space_manager import (
    VLLMBlockSpaceManager,
)
from paceserve.core.block_space_manager.distserve_block_space_manager import (
    DistserveBlockSpaceManager,
)
from paceserve.core.block_space_manager.prefill_block_space_manager import (
    PrefillBlockSpaceManager,
)
from paceserve.core.block_space_manager.decode_block_space_manager import (
    DecodeBlockSpaceManager,
)
from paceserve.core.block_space_manager.pace_block_space_manager import (
    PaceBlockSpaceManager,
)
from paceserve.core.block_space_manager.high_priority_block_space_manager import (
    HighPriorityBlockSpaceManager,
)
from paceserve.core.block_space_manager.low_priority_block_space_manager import (
    LowPriorityBlockSpaceManager,
)
from paceserve.utils.base_registry import BaseRegistry


class BlockSpaceManagerRegistry(BaseRegistry):

    @classmethod
    def get_key_from_str(cls, key_str: str) -> SchedulerType:
        return SchedulerType.from_str(key_str)


BlockSpaceManagerRegistry.register(SchedulerType.VLLM, VLLMBlockSpaceManager)
BlockSpaceManagerRegistry.register(SchedulerType.DIST, DistserveBlockSpaceManager)
BlockSpaceManagerRegistry.register(SchedulerType.PREFILL, PrefillBlockSpaceManager)
BlockSpaceManagerRegistry.register(SchedulerType.DECODE, DecodeBlockSpaceManager)
BlockSpaceManagerRegistry.register(SchedulerType.PACE, PaceBlockSpaceManager)
BlockSpaceManagerRegistry.register(SchedulerType.HIGH, HighPriorityBlockSpaceManager)
BlockSpaceManagerRegistry.register(SchedulerType.LOW, LowPriorityBlockSpaceManager)
BlockSpaceManagerRegistry.register(SchedulerType.ORCA, OrcaBlockSpaceManager)
BlockSpaceManagerRegistry.register(
    SchedulerType.FASTER_TRANSFORMER, FasterTransformerBlockSpaceManager
)
BlockSpaceManagerRegistry.register(SchedulerType.SARATHI, SarathiBlockSpaceManager)
BlockSpaceManagerRegistry.register(
    SchedulerType.SIMPLE_CHUNKING, SimpleChunkingBlockSpaceManager
)
