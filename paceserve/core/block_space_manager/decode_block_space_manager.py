from paceserve.core.block_space_manager.base_block_space_manager import (
    BaseBlockSpaceManager,
)
from paceserve.core.datatypes.sequence import Sequence


class DecodeBlockSpaceManager(BaseBlockSpaceManager):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_num_initial_blocks(self, seq: Sequence) -> int:
        return len(seq.logical_token_blocks)
