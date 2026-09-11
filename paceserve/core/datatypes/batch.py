from enum import IntEnum
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from paceserve.core.datatypes.sequence import Sequence


class RequestTypes(IntEnum):
    PREFILL = 0
    DECODE = 1


class ScheduledRequest:
    def __init__(self, _type: RequestTypes, seq: "Sequence"):
        self._type: RequestTypes = _type
        self.seq = seq

    def is_prefill(self):
        return self._type == RequestTypes.PREFILL

    def is_decode(self):
        return self._type == RequestTypes.DECODE

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(seq_id={self.seq.seq_id})"

    def __str__(self) -> str:
        return self.__repr__()


class PrefillRequest(ScheduledRequest):
    def __init__(self, seq: "Sequence",
                 chunk_size: Optional[int]=None,
                 value: Optional[float]=None):
        super().__init__(RequestTypes.PREFILL, seq)
        self.value = value
        # If chunk size is not provided, assume that the whole prefill will be processed
        self.chunk_size = chunk_size or seq.get_prompt_len()

    def __repr__(self):
        return f'PrefillRequest(seq_id={self.seq.seq_id}, chunk_size={self.chunk_size}, value={self.value})'


class DecodeRequest(ScheduledRequest):
    def __init__(self, seq: "Sequence"):
        super().__init__(RequestTypes.DECODE, seq)


ScheduledRequests = List[ScheduledRequest]


# Batch is used for the analytical modeling
# Used to estimate the execution time of a batch
class Batch:
    def __init__(self):
        self.chunk_sizes = [] # number of prefill tokens for a request
        self.prefill_processed = [] # number of prefill tokens that have been processed including the current chunk size
        self.total_processed = [] # total number of tokens that have been processed so far including the current chunk size
        self.num_decode = 0

        # For logging
        self.requests: ScheduledRequests = []
        self.prefill_ids: List[int] = []
        self.decode_ids: List[int] = []

    @property
    def size(self):
        return len(self.requests)

    def sort_requests(self):
        # Prefill comes first
        self.requests = sorted(self.requests, key=lambda x: not x.is_prefill())

    def get_running_requests(self):
        return self.requests

    def is_empty(self):
        return not (self.chunk_sizes or self.prefill_processed or self.total_processed or self.num_decode)

    # Returns true if the batch is a decoding batch.
    # A decoding batch is the one that only contains request in their
    # deocding phase and no request that is in its prefill phase.
    def is_decoding(self):
        return len(self.chunk_sizes) == len(self.prefill_ids) == len(self.prefill_processed) == 0

    def add_request(self, req: ScheduledRequest):
        self.requests.append(req)
        if req.is_prefill():
            self._add_prefill(req)
        else:
            self._add_decode(req)

    def _add_prefill(self, req: PrefillRequest):
        self.chunk_sizes.append(req.chunk_size)
        self.prefill_processed.append(req.seq.prompt_tokens_processed + req.chunk_size)
        self.prefill_ids.append(req.seq.seq_id)

    def _add_decode(self, req: DecodeRequest):
        self.num_decode += 1
        self.total_processed.append(len(req.seq.tokens))
        self.decode_ids.append(req.seq.seq_id)

