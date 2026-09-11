from paceserve.model_executor.attention.flashinfer_attention_wrapper import (
    FlashinferAttentionWrapper,
)
from paceserve.model_executor.attention.no_op_attention_wrapper import (
    NoOpAttentionWrapper,
)
from paceserve.model_executor.attention.naive_attention_wrapper import (
    NaiveAttentionWrapper,
)
from paceserve.types import AttentionBackend
from paceserve.utils.base_registry import BaseRegistry


class AttentionBackendRegistry(BaseRegistry):
    pass


AttentionBackendRegistry.register(AttentionBackend.NO_OP, NoOpAttentionWrapper)

AttentionBackendRegistry.register(
    AttentionBackend.FLASHINFER, FlashinferAttentionWrapper
)

AttentionBackendRegistry.register(
    AttentionBackend.NAIVE, NaiveAttentionWrapper
)

