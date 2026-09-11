from paceserve.core.datatypes.sequence import Sequence
from paceserve.ai_utils.ai_helper import AIHelper

class BaseItem:
    def __init__(self, sequence: Sequence) -> None:
        self.sequence = sequence
        self.frac = 0.0
        self.index = 0
        self.value = 0.0
        self.block_size = sequence.block_size
        self.remaining_tokens = len(sequence.prompt_token_ids) - sequence.prompt_tokens_processed
        self.prefill_time = sequence.state.estimated_prefill_time

        if not sequence.prompt_processing_finished:
            self.compute, self.memory = AIHelper().calculate_prefill_resources(self.remaining_tokens, self.sequence.prompt_tokens_processed)
        else:
            self.compute, self.memory = AIHelper().calculate_decode_resources(len(self.sequence.tokens))

    @classmethod
    def from_sequence(cls, seq: Sequence, current_time: float, TTFT_SLO: float) :
        item = cls(seq)
        item.value_func(current_time, TTFT_SLO)
        return item

    def value_func(self, current_time: float, TTFT_SLO: float):
        raise NotImplementedError("Subclasses should implement this method.")

    def __str__(self):
        return (
            f'{self.__class__.__name__}('
            f'seq_id={self.sequence.seq_id}, '
            f'value={self.value}, tokens={self.remaining_tokens}, '
            f'estimate_time = {self.prefill_time}, '
            f'frac = {self.frac})'   
        )

    def __repr__(self) -> str:
        return self.__str__()


class BasicValue(BaseItem):
    def value_func(self, current_time: float, TTFT_SLO: float):
        if self.sequence.state._request_reschedule_interval:
            idle_time = self.sequence.state._request_reschedule_interval + (current_time - self.sequence.state._seq_arrive_at) 
        else:
            idle_time = current_time - self.sequence.arrival_time
        if not self.sequence.prompt_processing_finished:
            value = (1+idle_time)/TTFT_SLO *(1/(self.remaining_tokens))
        self.value = value


class ShortestJobFirstValue(BaseItem):
    def value_func(self, current_time: float, TTFT_SLO: float):
        self.value = 1/self.sequence.get_prompt_len()


class EarliestDeadlineFirstValue(BaseItem):
    def value_func(self, current_time: float, TTFT_SLO: float):
        if self.sequence.state._request_reschedule_interval:
            idle_time = self.sequence.state._request_reschedule_interval + (current_time - self.sequence.state._seq_arrive_at) 
        else:
            idle_time = current_time - self.sequence.arrival_time
            
        if TTFT_SLO - (idle_time + self.prefill_time) > 0:
            self.value = 1/(TTFT_SLO - (idle_time + self.prefill_time))
        else:
            self.value = float('inf')
        # NOTE: produce negative value bug when prefill time > TTFT_SLO

# Only picks requests that have the estimated
# runtime of less than a threshold
class BoundedPrefillValue(BaseItem):
    def value_func(self, current_time: float, TTFT_SLO: float):
        # TODO: incomplete
        self.value = 0.0


"""
class NonLinearValue(BaseItem):
    def value_func(self, batch_decode, remaining_KV_block, current_time: float,):
        if not self.request.prefill_completed:
            latency_term = (1+current_time - self.request.arrival_time)/Request.ttft_slo # somewhere 0 to >1
            prompt_length_term = (1/(self.request.remaining_tokens())) # somhere 0 to 1
            batch_decode_term = batch_decode/128    #somewhere 0 to 1
            system_mem_term = max(max(remaining_KV_block/35000, 0.05) - 0.05, 0)
            value = latency_term * prompt_length_term * batch_decode_term*(1-1/(self.request.remaining_tokens()))* system_mem_term
            #Request.ttft_slo - (current_time - self.request.arrival_time + estimated_time)
        self.value = value
"""
