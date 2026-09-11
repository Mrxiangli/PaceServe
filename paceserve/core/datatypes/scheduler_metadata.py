from typing import List, Optional
from dataclasses import dataclass, field

from paceserve.core.datatypes.batch import Batch
from paceserve.ai_utils.ai_helper import AIHelper
from paceserve.core.datatypes.sequence import Sequence
from paceserve.ai_utils.resource_item import BaseItem


@dataclass
class SchedulerMetadata:
    id: int
    seq_transfer_list: Optional[List[Sequence]] = field(default_factory=list)  # Sequence transfer list

    # For logging
    batch: Batch = None
    waiting: Optional[List[Sequence]] = field(default_factory=list)
    preempted: Optional[List[Sequence]] = field(default_factory=list)
    partial_prefills: Optional[List[Sequence]] = field(default_factory=list)
    items: Optional[List[BaseItem]] = field(default_factory=list)
    remaining_tokens: Optional[float] = -1.0  # Remaining tokens after greedy solver
    remaining_compute: Optional[float] = -1.0  # Remaining compute after greedy solver

    # For estimating the batch runtime
    prefill_token_list: Optional[List[int]] = field(default_factory=list)  # l_list
    chunk_token_list: Optional[List[int]] = field(default_factory=list)    # c_list
    processed_token_list: Optional[List[int]] = field(default_factory=list)  # l_hat_list
    predict_batch_time: Optional[float] = 0.0  # Estimated batch runtime from regression model
    remaining_kv_blocks: Optional[int] = 0      # reamining kv block at the end of the batch

    def __post_init__(self):
        self.prefill_token_list = self.batch.prefill_processed
        self.chunk_token_list = self.batch.chunk_sizes
        self.processed_token_list = self.batch.total_processed
        self.predict_batch_time = AIHelper().estimate_time(self.batch)
