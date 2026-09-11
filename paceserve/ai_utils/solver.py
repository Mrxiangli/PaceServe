import torch
import time
from typing import List
import numpy as np

from paceserve.core.datatypes.sequence import Sequence
from paceserve.ai_utils.resource_item import BaseItem, BasicValue, \
    ShortestJobFirstValue, EarliestDeadlineFirstValue

GPU_CONFIG_DIC = {
    "A100": {
        'flops': 312 * 10**12,
        'memory_bandwidth': 2 * 10**12,
        'hbm': 80
    }
}

VALUE_FUNCTION_CLASSES = {
    "BasicValue": BasicValue,
    "ShortestJobFirstValue": ShortestJobFirstValue,
    "EarliestDeadlineFirstValue": EarliestDeadlineFirstValue,
}

class BaseKnapsackSolver:
    def __init__(self, config):# config is scheduler config
        # Define GPU limits
        # TODO: These parameters are per-seconds. We need to think about this
        self.hardware = torch.cuda.get_device_name(0).split(' ')[1][0:4]
        self.GPU_MEM_BANDWIDTH = GPU_CONFIG_DIC[self.hardware]['memory_bandwidth']
        self.GPU_FLOPS = GPU_CONFIG_DIC[self.hardware]['flops']
        self.TTFT_SLO = config.ttft_slo
        self.VALUE_FUNCTION_CLS = VALUE_FUNCTION_CLASSES.get(config.val_func, EarliestDeadlineFirstValue)

    def solve(self, remaining_blocks: int, seq: List[Sequence], current_time: float):
        raise NotImplemented()

    def _create_items(self, seq: List[Sequence], current_time: float) -> List[BaseItem]:
        raise NotImplemented()


class GreedySolver(BaseKnapsackSolver):
    def _create_items(self, seqs: List[Sequence]):
        return [self.VALUE_FUNCTION_CLS.from_sequence(seq, time.monotonic(), self.TTFT_SLO) for seq in seqs]

    def solve(self, seqs: List[Sequence], remaining_KV_block: int, batch_size: int, max_tokens: int):
        num_tokens = 0
        num_prefills = 0
        remaining_compute = self.GPU_FLOPS
        remaining_kvblock = remaining_KV_block

        items = self._create_items(seqs)
        for i, item in enumerate(items):
            item.index = i
        sorted_items: List[self.VALUE_FUNCTION_CLS] = sorted(items,
                                                        key=lambda item: item.value,
                                                        reverse=True)
        #NOTE: This is hardcoded to reduce the piggy backing decode delay, so we only choose top 4
        # experiment with this value to find a better configuration
        top_k = 4
        ct = 0
        for item in sorted_items:
            ct += 1
            if remaining_compute <= 0 or \
                remaining_kvblock <= 0 or \
                num_prefills >= batch_size or \
                num_tokens >= max_tokens or \
                ct >= top_k:
                break

            if item.compute <= remaining_compute and \
                num_tokens + item.remaining_tokens <= max_tokens and \
                np.ceil(item.remaining_tokens/item.block_size) < remaining_kvblock:
                item.frac = 1
                num_prefills += 1
                remaining_compute -= item.compute
                num_tokens += item.remaining_tokens
                remaining_kvblock -= item.remaining_tokens // item.block_size
            else:
                frac_compute = min(remaining_compute / item.compute, 1)
                frac_tokens = min((max_tokens - num_tokens)/item.remaining_tokens, 1)
                frac_mem =  min(remaining_kvblock / min(1,(item.remaining_tokens // item.block_size)), 1)

                item.frac = min(frac_compute, frac_tokens, frac_mem)
                num_prefills += 1
                num_tokens += np.ceil(item.frac * item.remaining_tokens)
                remaining_compute -= item.frac * item.compute
                remaining_kvblock -= np.ceil(np.ceil(item.frac*item.remaining_tokens)/item.block_size)

        rev_sorted_items: List[self.VALUE_FUNCTION_CLS] = sorted(sorted_items, key=lambda item: item.index)
        return rev_sorted_items, remaining_compute, num_tokens


class NonPreemptiveSolver(BaseKnapsackSolver):
    def _create_items(self, seqs: List[Sequence]):
        return [self.VALUE_FUNCTION_CLS.from_sequence(seq, time.monotonic(), self.TTFT_SLO) for seq in seqs]

    def solve(self, seqs: List[Sequence], remaining_KV_block: int, batch_size: int, max_tokens: int):
        num_tokens = 0
        num_prefills = 0
        remaining_compute = self.GPU_FLOPS
        remaining_kvblock = remaining_KV_block

        items = self._create_items(seqs)
        for i, item in enumerate(items):
            item.index = i
        sorted_items: List[self.VALUE_FUNCTION_CLS] = sorted(items,
                                                        key=lambda item: item.value,
                                                        reverse=True)

        #NOTE: This is hardcoded to reduce the piggy backing decode delay, so we only choose top 4
        # experiment with this value to find a better configuration
        # top_k = 2
        # ct = 0
        for item in sorted_items:
            # ct+=1
            if remaining_compute <= 0 or \
                remaining_kvblock <= 0 or \
                num_prefills >= batch_size or \
                num_tokens >= max_tokens:
                break

            if num_tokens + item.remaining_tokens > max_tokens:
                continue

            item.frac = 1
            num_prefills += 1
            remaining_compute -= item.compute
            num_tokens += item.remaining_tokens
            remaining_kvblock -= np.ceil(item.remaining_tokens / item.block_size)

        rev_sorted_items: List[self.VALUE_FUNCTION_CLS] = sorted(sorted_items, key=lambda item: item.index)
        return rev_sorted_items, remaining_compute, num_tokens
