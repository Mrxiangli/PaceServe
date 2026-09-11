from typing import List, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from paceserve.config import SystemConfig
from paceserve.core.datatypes.sequence import Sequence
from paceserve.core.sequence_manager.base_sequence_manager import BaseSequenceManager
from paceserve.transformers_utils.tokenizer import detokenize_incrementally


class EngineSequenceManager(BaseSequenceManager):

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        config: SystemConfig,
    ):
        super().__init__(config)
        self.tokenizer = tokenizer
        # This gives a list of sequences that **just** finished in the current step().
        self._prefill_finished_seqs = []

    def _decode_seq(self, seq: Sequence) -> None:
        """Decodes the new token for a sequence."""
        (new_tokens, new_output_text, prefix_offset, read_offset) = (
            detokenize_incrementally(
                self.tokenizer,
                all_input_ids=seq.get_token_ids(),
                prev_tokens=seq.tokens,
                prefix_offset=seq.prefix_offset,
                read_offset=seq.read_offset,
                skip_special_tokens=True,
            )
        )
        if seq.tokens is None:
            seq.tokens = new_tokens
        else:
            seq.tokens.extend(new_tokens)
        seq.prefix_offset = prefix_offset
        seq.read_offset = read_offset
        seq.output_text += new_output_text

    def _on_append_token(self, seq: Sequence) -> None:
        self._decode_seq(seq)
        if seq.only_finished_prefill():
            self._prefill_finished_seqs.append(seq)

    def get_finished_prefill_seqs(self):
        return self._prefill_finished_seqs

    def reset_before_schedule(self):
        self._prefill_finished_seqs = []

    def _get_block_table(self, seq: Sequence) -> List[int]:
        return []

    # when engine preempt, the block space is released by the scheduler directly, 
    # the sequence map is freed here during _on_schedule() call
    def _distserve_preempt_seq(self, seq_id: str) -> None:
        # on the engine side, preemption just manipulate sequence 
        super()._distserve_preempt_seq(seq_id)
        # this one frees the sequence map
        self._free_seq(seq_id)
