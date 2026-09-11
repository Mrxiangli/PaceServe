from typing import List, Optional, Tuple
import os
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F

from flashinfer import (
    append_paged_kv_cache,
    get_seq_lens,
    get_batch_indices_positions,
)

from flashinfer import BatchPrefillWithPagedKVCacheWrapper, append_paged_kv_cache
from flashinfer import get_seq_lens, get_batch_indices_positions

from paceserve.config import CacheConfig, ModelConfig, ParallelConfig
from paceserve.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper
from paceserve.core.datatypes.sequence import SequenceMetadata
from paceserve.metrics.constants import OperationMetrics


class NaiveAttentionWrapper(BaseAttentionWrapper):
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        device: torch.device,
    ):
        super().__init__(model_config, parallel_config, cache_config, device)
        self.prefill_attention_map = False
        self.attention_map_dir = "/scratch/gilbreth/li2068/asplos/PaceServe/exp_result/test_prompts/attention_map"
        os.makedirs(self.attention_map_dir,exist_ok=True)

        self.is_metadata_initialized = False
        self.is_profiling_iteration = False
        self.contains_prefill = False
        self.contains_decode = False
        self.num_prefill_tokens = 0
        self.num_total_tokens = 0

        self.append_qo_indptr_tensor = None
        self.append_kv_page_indices_tensor = None
        self.append_kv_page_indptr_tensor = None
        self.append_kv_last_page_len_tensor = None

    def to_int_tensor(self, data: List[int]) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.int32, device="cuda")

    def init_gpu_cache(self, num_gpu_blocks: int) -> None:
        gpu_cache: List[torch.Tensor] = []
        self.num_gpu_blocks = num_gpu_blocks

        for _ in range(self.num_layers):
            gpu_blocks = self.get_cache_block(
                self.num_gpu_blocks, dtype=self.dtype, device="cuda"
            )
            gpu_cache.append(gpu_blocks)

        self.gpu_cache = gpu_cache

    def get_cache_block(self, num_blocks: int, **kwargs) -> torch.Tensor:
        return torch.randn(
            num_blocks,
            2,
            self.block_size, # this is page size
            self.num_kv_heads,
            self.head_dim,
            **kwargs,
        )

    def begin_forward(
        self,
        seq_metadata_list: List[SequenceMetadata],
    ) -> None:
        # The indptr tensor captures the location query tokens in the input tensor.
        # |<---------------------- num_valid_tokens ----------------------------------------------------->|
        # |<--------------- num_prompt_tokens -------------->||<------- num_generation_tokens (M) ------->|
        # |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->||<--generation_0-->|...|<--generation_M-1-->|<--padding-->|
        #
        # Flashinfer calls this layout as a raggedtensor. The indptr tensor captures the start of each
        # sequence in the ragged tensor. The length of the indptr tensor is the number of sequences + 1.
        # We perform both prefill and decode attention in a single call to batched prefill kernel.
        # prefill_qo_indptr: [0, prompt_0, prompt_0 + prompt_1, ..., prompt_0 + ... + prompt_N-1, generation_0, generation_0 + 1, ..., generation_0 + ... + M]
        prefill_qo_indptr: List[int] = [0]
        decode_qo_indptr: List[int] = [0]
        # The kv_page_indices tensor captures the pages of the key-value cache that
        # are assigned to each token in the input tensor. Since there is a variable number
        # of pages assigned to each sequence, a ragged tensor to represent this.
        prefill_kv_page_indices: List[int] = []
        decode_kv_page_indices: List[int] = []
        # the last page might not be full, so we need to keep track of the length of the last page
        prefill_kv_last_page_len: List[int] = []
        decode_kv_last_page_len: List[int] = []
        # Since the prefill_kv_page_indices tensor is a ragged tensor, we also need to keep track of the
        # indptr tensor for the prefill_kv_page_indices tensor. This tensor captures the start of each sequence
        # in the ragged tensor.
        prefill_kv_page_indptr: List[int] = [0]
        decode_kv_page_indptr: List[int] = [0]

        self.is_profiling_iteration = False
        self.is_metadata_initialized = True

        self.contains_prefill = False
        self.contains_decode = False

        for seq_metadata in seq_metadata_list:
            if not seq_metadata.is_prompt:
                continue

            # ONLY used for profiling
            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                # During memory profiling, the block tables are not initialized yet.
                #  We will just skip the attention computation for now.
                return

            self.contains_prefill = True

            prompt_chunk_len = seq_metadata.prompt_chunk_len
            processed_prompt_len = (
                seq_metadata.seq.get_num_prompt_tokens_stage_processed()
            )
            current_total_len = processed_prompt_len + prompt_chunk_len

            # indptr for the prompt tokens in q/o tensor
            prefill_qo_indptr.append(prefill_qo_indptr[-1] + prompt_chunk_len)
            # Compute the kv page indices for the prompt tokens.
            num_blocks_in_use = (
                current_total_len + self.block_size - 1
            ) // self.block_size
            prefill_kv_page_indices.extend(seq_metadata.block_table[:num_blocks_in_use])
            prefill_kv_page_indptr.append(
                prefill_kv_page_indptr[-1] + num_blocks_in_use
            )
            prefill_kv_last_page_len.append(
                current_total_len % self.block_size or self.block_size
            )

        for seq_metadata in seq_metadata_list:
            if seq_metadata.is_prompt:
                continue

            if seq_metadata.block_table is None:
                self.is_profiling_iteration = True
                return

            self.contains_decode = True

            context_len = seq_metadata.seq.get_len()
            # indptr for the prompt tokens in q/o tensor
            decode_qo_indptr.append(decode_qo_indptr[-1] + 1)
            # Compute the kv page indices for the prompt tokens.
            num_blocks_in_use = (context_len + self.block_size - 1) // self.block_size
            decode_kv_page_indices.extend(seq_metadata.block_table[:num_blocks_in_use])
            decode_kv_page_indptr.append(decode_kv_page_indptr[-1] + num_blocks_in_use)
            decode_kv_last_page_len.append(
                context_len % self.block_size or self.block_size
            )

        self.num_prefill_tokens = prefill_qo_indptr[-1]
        self.num_total_tokens = self.num_prefill_tokens + len(decode_qo_indptr) - 1

        self.append_qo_indptr_tensor = self.to_int_tensor(
            prefill_qo_indptr[:-1]
            + [x + prefill_qo_indptr[-1] for x in decode_qo_indptr]
        )
        self.append_kv_page_indices_tensor = self.to_int_tensor(
            prefill_kv_page_indices + decode_kv_page_indices
        )
        self.append_kv_page_indptr_tensor = self.to_int_tensor(
            prefill_kv_page_indptr[:-1]
            + [x + prefill_kv_page_indptr[-1] for x in decode_kv_page_indptr]
        )
        self.append_kv_last_page_len_tensor = self.to_int_tensor(
            prefill_kv_last_page_len + decode_kv_last_page_len
        )
        # padding to 8 to utilize tensor cores, this is giving issues 
        # if  self.append_qo_indptr_tensor[-1] % 8 != 0:
        #     self.append_qo_indptr_tensor[-1] = (self.append_qo_indptr_tensor[-1] +7)//8*8

    def end_forward(self):

        self.is_metadata_initialized = False
    
    def group_query_attention_from_cache(self, query, key, value, softmax_scale=1.0):
        """
        Args:
            query: [T, num_q_heads, head_dim]
            key:   [T, num_kv_heads, head_dim]
            value: [T, num_kv_heads, head_dim]
        Returns:
            output: [T, num_q_heads, head_dim]
        """
        T, num_q_heads, head_dim = query.shape
        T_k, num_kv_heads, _ = key.shape
        group_size = num_q_heads // num_kv_heads
        assert num_q_heads % num_kv_heads == 0
        
        query_g = query.view(T, num_kv_heads, group_size, head_dim)
        key_t = key.permute(1, 2, 0)
        scores = torch.einsum("thgd,hdk->thgk", query_g, key_t) * softmax_scale
        
        if T == 1: #decoding, no causual mask needed
            pass
        else:
            # creat causal mask for so only later token can attend earlier tokens
            causal_mask = torch.arange(T, device=query.device).unsqueeze(1) >= torch.arange(T_k, device=query.device).unsqueeze(0)
            causal_mask = causal_mask.view(T, 1, 1, T_k)
            scores = scores.masked_fill(~causal_mask, float('-inf'))
            
        attn = torch.softmax(scores, dim=-1)
        value_t = value.permute(1, 0, 2)
        output = torch.einsum("thgk,hkd->thgd", attn, value_t)
        T, H_kv, G, K = attn.shape
        attn_flat = attn.permute(1, 2, 0, 3).reshape(H_kv * G, T, K)
        return output.reshape(T, num_q_heads, head_dim), attn_flat
        
    def retrieve_kv_from_paged_kv_cache(
        self,
        batch_idx: int,
        gpu_cache: torch.Tensor,  # (num_blocks, 2, block_size, num_kv_heads, head_dim)
        page_indices: torch.Tensor,  # (num_total_pages,)
        page_indptr: torch.Tensor,   # (num_sequences + 1,)
        last_page_lens: torch.Tensor,  # (num_sequences,)
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            keys:  (seq_len, num_kv_heads, head_dim)
            values: (seq_len, num_kv_heads, head_dim)
        """
        start = page_indptr[batch_idx].item()
        end = page_indptr[batch_idx + 1].item()
        num_pages = end - start

        page_ids = page_indices[start:end]  # (num_pages,)
        last_len = last_page_lens[batch_idx].item()

        # Extract both keys and values
        kv_blocks = gpu_cache[page_ids]  # shape: (num_pages, 2, block_size, num_kv_heads, head_dim)
        key_blocks = kv_blocks[:, 0]     # (num_pages, block_size, num_kv_heads, head_dim)
        value_blocks = kv_blocks[:, 1]   # (num_pages, block_size, num_kv_heads, head_dim)

        if num_pages > 1:
            full_key_blocks = key_blocks[:-1].reshape(-1, *key_blocks.shape[2:])
            last_key_block = key_blocks[-1, :last_len]
            keys = torch.cat([full_key_blocks, last_key_block], dim=0)

            full_value_blocks = value_blocks[:-1].reshape(-1, *value_blocks.shape[2:])
            last_value_block = value_blocks[-1, :last_len]
            values = torch.cat([full_value_blocks, last_value_block], dim=0)
        else:
            keys = key_blocks[0, :last_len]
            values = value_blocks[0, :last_len]

        return keys, values
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_cache_idx: int,
        softmax_scale: float = 1.0,
        layer_id: Optional[int] = None,
    ) -> torch.Tensor:

        assert self.is_metadata_initialized, "Metadata is not initialized."
        if self.is_profiling_iteration:
            return torch.zeros_like(query)

        with self.get_timer(OperationMetrics.ATTN_INPUT_RESHAPE, layer_id):
            query = query.contiguous().view(-1, self.num_q_heads, self.head_dim)
            key = key.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)
            value = value.contiguous().reshape(-1, self.num_kv_heads, self.head_dim)

        output = torch.empty_like(query)

        seq_length = get_seq_lens(self.append_kv_page_indptr_tensor, self.append_kv_last_page_len_tensor, self.block_size)
        self.batch_indices, self.positions = get_batch_indices_positions(self.append_qo_indptr_tensor, seq_length, self.append_qo_indptr_tensor[-1])

        with self.get_timer(OperationMetrics.ATTN_KV_CACHE_SAVE, layer_id):
            append_paged_kv_cache(
                key,
                value,
                self.batch_indices,                     # (0, 0, 0, 1, 1, 1, 1, 1) : req 0 with 3 tks, req 1 with 5 tks 
                self.positions,
                self.gpu_cache[layer_cache_idx],
                self.append_kv_page_indices_tensor,     # page indices for each request
                self.append_kv_page_indptr_tensor,      # number of pages (x, x+2, y, y+4, ...)
                self.append_kv_last_page_len_tensor,    # number of token in the last page
                kv_layout="NHD",
            )
                
        all_kv = [
            self.retrieve_kv_from_paged_kv_cache(
                batch_idx=i,
                gpu_cache=self.gpu_cache[layer_cache_idx],
                page_indices=self.append_kv_page_indices_tensor,
                page_indptr=self.append_kv_page_indptr_tensor,
                last_page_lens=self.append_kv_last_page_len_tensor,
                block_size=self.block_size,
            )
            for i in range(self.append_kv_page_indptr_tensor.shape[0] - 1)
        ]

        with self.get_timer(OperationMetrics.ATTN_PREFILL, layer_id):
            if self.contains_prefill:
                prefill_outputs = []
                attn_blocks = []

                token_cursor = 0  # Running total to locate tokens in global view
                total_T = self.num_prefill_tokens
                H_total = self.num_q_heads
                global_attn = torch.zeros(H_total, total_T, total_T, device=query.device)
                
                for bidx in range(len(all_kv)):
                    # Get token indices in prefill range for this request
                    mask = (self.batch_indices[:self.num_prefill_tokens] == bidx)
                    query_b = query[:self.num_prefill_tokens][mask]  # [T_b, q_heads, head_dim]
                    T_b = query_b.shape[0]

                    if T_b == 0:
                        continue
                    k, v = all_kv[bidx]                       

                    out, attn_flat = self.group_query_attention_from_cache(
                        query=query_b,
                        key=k,
                        value=v,
                        softmax_scale=1.0 / (self.head_dim ** 0.5)
                    ) 
                    prefill_outputs.append((mask, out))
                   
                    global_attn[:, token_cursor:token_cursor+T_b, token_cursor:token_cursor+T_b] = attn_flat
                    token_cursor += T_b

                # Scatter results into the correct positions in output
                for mask, out in prefill_outputs:
                    output[:self.num_prefill_tokens][mask] = out
                
                if self.prefill_attention_map:
                    layer_path = os.path.join(self.attention_map_dir,f'layer_{layer_cache_idx}')
                    os.makedirs(layer_path,exist_ok=True)
                    
                    for i in range(global_attn.shape[0]):
                        plt.figure(figsize=(4, 4))
                        plt.imshow(global_attn[i].cpu().numpy(), cmap='gray', aspect='auto')
                        plt.title(f'Head {i}: {(global_attn[i].cpu().abs() < 1e-4).sum().item()/global_attn[i].cpu().numel()} ')
                        plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(f"{layer_path}/head_{i}.png")
                        plt.close()
                    cols = 8
                    rows = (global_attn.shape[0] + cols - 1) // cols
                    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))

                    for i in range(rows * cols):
                        ax = axes[i // cols, i % cols]
                        if i < global_attn.shape[0]:
                            ax.imshow(global_attn[i].cpu().numpy(), cmap='gray', aspect='auto')
                            ax.set_title(f'Head {i}', fontsize=12)
                        ax.axis('off')

                    plt.tight_layout()
                    plt.savefig(f"{layer_path}/attention_heads_combined.png")
                    plt.close()

        with self.get_timer(OperationMetrics.ATTN_DECODE, layer_id):
            if self.contains_decode:
                start = self.num_prefill_tokens
                end = self.num_total_tokens

                decode_queries = query[start:end]                       # [N_decode, q_heads, head_dim]
                decode_batch_indices = self.batch_indices[start:end]    # [N_decode]

                decode_outputs = torch.empty_like(decode_queries)

                for bidx in range(len(all_kv)):
                    mask = (decode_batch_indices == bidx)               # [N_decode] -> Bool
                    if not mask.any():
                        continue

                    q = decode_queries[mask]                            # [T_b, q_heads, head_dim]
                    k, v = all_kv[bidx]                                 # [T_kv, kv_heads, head_dim]

                    out, attn_flat = self.group_query_attention_from_cache(
                        query=q,
                        key=k,
                        value=v,
                        softmax_scale=1.0 / (self.head_dim ** 0.5)
                    ) 

                    decode_outputs[mask] = out

                output[start:end] = decode_outputs

        with self.get_timer(OperationMetrics.ATTN_OUTPUT_RESHAPE, layer_id):
            output = output.view(-1, self.num_q_heads * self.head_dim)

        return output
