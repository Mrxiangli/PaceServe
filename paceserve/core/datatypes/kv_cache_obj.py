from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch

from paceserve.core.datatypes.sequence import SamplerOutputs, Sequence

@dataclass
class KVobj:
    """related materials for kv transfer objects

    Attributes:
        kv_cache: kv cache related to the seq
        kv_indices: kv indices in the origin worker
    """
    def __init__(self, seq_id, kv_cache: torch.Tensor, kv_indices: List, left_prefill=None, kv_transfer_start=None):
        self.seq_id = seq_id
        self.kv_cache = kv_cache
        self.kv_indices = kv_indices
        self.size = (kv_cache.element_size() * kv_cache.numel()) / (10**6)
        self.left_prefill = left_prefill
        self.kv_transfer_start = kv_transfer_start

@dataclass
class KVmetadata:
    def __init__(self, seq: Sequence, kv_size: int):
        self.seq = seq
        self.kv_size = kv_size
        self.start_time = 0

        self._prefill_instance = None
        self._transfer_ref = None
