from dataclasses import dataclass
from typing import List, Optional

from paceserve.core.datatypes.sequence import Sequence
from paceserve.core.datatypes.sequence_status import SequenceStatus


@dataclass
class RequestOutput:
    """The output data of a request to the LLM.

    Args:
        seq_id: The unique ID of the request.
        prompt: The prompt string of the request.
        prompt_token_ids: The token IDs of the prompt.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
    """

    seq_id: str
    prompt: str
    prompt_token_ids: List[int]
    text: str
    token_ids: List[int]
    finished: bool
    finish_reason: Optional[str] = None
    nll: float = None
    token_count: int = None
    prefill_finished: Optional[bool] = False

    @classmethod
    def from_seq(cls, seq: Sequence) -> "RequestOutput":
        return cls(
            seq.seq_id,
            seq.prompt,
            seq.prompt_token_ids,
            seq.output_text,
            seq.get_output_token_ids(),
            seq.is_finished(),
            SequenceStatus.get_finished_reason(seq.get_status()),
            seq.state._seq_nll,
            seq.state._seq_token_count,
            len(seq.get_output_token_ids())==1
        )


@dataclass
class PromptResponse:
    prompt: str
    response: str
