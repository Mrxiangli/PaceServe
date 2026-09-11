import logging
from typing import Tuple, List

from paceserve.benchmark.entities.base_entity import BaseEntity

logger = logging.getLogger(__name__)


class Request(BaseEntity):

    def __init__(
        self,
        arrived_at: float,
        num_prefill_tokens: int,
        num_decode_tokens: int,
    ):
        self._id: int = Request.generate_id()
        self._arrived_at = arrived_at
        self._num_prefill_tokens = num_prefill_tokens
        self._num_decode_tokens = num_decode_tokens
        self._in_window = False
        assert num_prefill_tokens > 0
        assert num_decode_tokens > 0

    @property
    def id(self) -> int:
        return self._id

    @property
    def size(self) -> Tuple[int, int]:
        return (self._num_prefill_tokens, self._num_decode_tokens)

    @property
    def arrived_at(self) -> float:
        return self._arrived_at

    @property
    def is_in_window(self):
        return self._in_window

    @property
    def num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    @property
    def num_decode_tokens(self) -> int:
        return self._num_decode_tokens

    @property
    def pd_ratio(self) -> float:
        return self._num_prefill_tokens / self._num_decode_tokens

    @property
    def total_tokens(self) -> int:
        return self._num_prefill_tokens + self._num_decode_tokens

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "arrived_at": self._arrived_at,
            "num_prefill_tokens": self._num_prefill_tokens,
            "num_decode_tokens": self._num_decode_tokens,
        }


class PromptRequest(Request):
    def __init__(self, prompt):
        # for Prompts, all requests come at the same time
        super().__init__(0, len(prompt), 1)
        self.prompt = prompt


class DatasetRequest(Request):
    def __init__(self, text: str, prompt_ids: List[int]):
        prefill = len(text) if text else len(prompt_ids)
        super().__init__(0, prefill, 1)
        self.text = text
        self.prompt_ids = prompt_ids

class QARequest(DatasetRequest):
    def __init__(self, text: str, prompt_ids: List[int], answer: str):
        super().__init__(text, prompt_ids)
        self.answer = answer
