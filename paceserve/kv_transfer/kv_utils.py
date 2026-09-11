from abc import ABC, abstractmethod

import torch

from paceserve.config import CacheConfig, ModelConfig, ParallelConfig


class KVhelper(ABC):
    
    def __init__(
        self,
        rank: int,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig
    ):
        self.rank = rank
        self.num_q_heads = model_config.get_num_q_heads(parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_dim = model_config.get_head_size()
        self.dtype = model_config.dtype
        self.block_size = cache_config.block_size
        self.num_layers = model_config.get_num_layers(parallel_config)
        print(f"kv helper initialized on worker: {self.rank}")
    
    def add_model_runner(self, model_runner):
        self.model_runner = model_runner
    
    def reset_kv_cache(self,num_gpu_blocks):
        for i in range(self.num_layers):
            self.model_runner.attention_backend_wrapper.gpu_cache[i] = torch.zeros(num_gpu_blocks, 2, self.block_size, self.num_kv_heads, self.head_dim, dtype=self.dtype, device="cuda")
    
    def get_kv_cache(self, seq, seq_manager):
        try:
            kv_page_indices = seq_manager._get_block_table(seq)
        except:
            print(f"seq {seq.seq_id} was not allocated any memory")
            kv_page_indices = None
            
        if not kv_page_indices:
            tmp_cache = torch.zeros(self.num_layers, 1, 2, self.block_size, self.num_kv_heads, self.head_dim, dtype=self.dtype)
            return tmp_cache, [0]
        
        seq_cache = torch.zeros(self.num_layers, len(kv_page_indices), 2, self.block_size, self.num_kv_heads, self.head_dim, dtype=self.dtype)
        for i in range(self.num_layers):
            seq_cache[i] = self.model_runner.attention_backend_wrapper.gpu_cache[i][torch.tensor(kv_page_indices),:,:,:,:]
        return seq_cache, kv_page_indices
    
    def insert_kv_cache(self, cached_content, kv_indices, device):
        try: 
            for i in range(self.num_layers):
                self.model_runner.attention_backend_wrapper.gpu_cache[i][torch.tensor(kv_indices),:,:,:,:] = cached_content[i].to(device)
            return True
        except Exception as e:
            print(f"error during insert kv cache: {e}")
            return False   
                
        
        
        
        
        
    
    
    
