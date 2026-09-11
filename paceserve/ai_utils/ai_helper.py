import logging
import joblib
from typing import Tuple

from paceserve.core.datatypes.batch import Batch, PrefillRequest

logger = logging.getLogger(__name__)

GPU_CONFIG_DIC = {
    "A100": {
        'flops': 312 * 10**12,
        'memory_bandwidth': 2 * 10**12,
        'hbm': 80
    }
}

# Types of regression models.
# TODO: Definining it here for convenience.
PREFILL = 'prefill'
DECODE = 'decode'
MIX = 'mix'

def _model_type_to_name(model_type) -> str:
    return f'batch_regression_{model_type}.pkl'


class AIHelper:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AIHelper, cls).__new__(cls)
        return cls._instance

    def initialize(self, model_config, scheduler_config, batchregression_config):
        if hasattr(self, '_initialize'):
            return
        self.scheduler_config = scheduler_config
        self._initialize = True
        self.GPU_MEM_BANDWIDTH = GPU_CONFIG_DIC['A100']['memory_bandwidth']
        self.GPU_FLOPS = GPU_CONFIG_DIC['A100']['flops']
        self.h = model_config.hf_config.hidden_size
        self.s = model_config.hf_config.head_dim if hasattr(model_config.hf_config, "head_dim") else (model_config.hf_config.hidden_size/model_config.hf_config.num_attention_heads)
        self.m = model_config.hf_config.intermediate_size
        self.b = 128 # this is flash attention specific parameter block size
        self.n = model_config.hf_config.num_attention_heads
        self.l = model_config.hf_config.num_hidden_layers
        self.regression_models = self._load_regression_models(batchregression_config.regression_model_path)

    def _load_regression_models(self, path):
        return {
            x: joblib.load(f'{path}/{_model_type_to_name(x)}')
            for x in [PREFILL, DECODE, MIX]
        }

    # Calculate the FLOPS and memory bandwidth needed for a prefill
    # request with the given in put length
    def calculate_prefill_resources(self, input_len, processed):
        if not hasattr(self, '_initialize'):
            raise Exception('AIHelper has not been initiailized yet!')

        if input_len == 0:
            return (0, 0)
        flops_gemm = 2 * input_len * self.h * self.m + 4 * input_len * self.h**2
        flops_attn = (2 * input_len * (input_len+processed) * self.s)
        memory_gemm = (4 * self.h**2 + 2 * self.h * self.m) + 8* (input_len+processed) * self.h + 2 * (input_len+processed) * self.m
        memory_attn =  2 * (input_len+processed) * self.s + 3 * input_len * self.s * ((input_len+processed) / self.b)
        return (flops_attn * self.n * self.l + flops_gemm * self.l, memory_attn * self.n * self.l + memory_gemm * self.l)

    # Calculate the FLOPS and memory bandwidth needed for a decode
    # request where the input_length is the number of tokens that
    # have already been processed (prompt length + decoded tokens)
    def calculate_decode_resources(self, input_length):
        if input_length == 0:
            return (0, 0)
        flops = (4 * self.h**2 + 2 * self.h * self.m) + 2 * input_length * self.s
        memory = (4 * self.h**2 + 8 * self.h + 2 * self.h * self.m + 2 * self.m) + (2 * input_length * self.s + 2 * self.s)
        return (flops * self.n * self.l, memory * self.n * self.l)

    def estimate_prefill_time(self, seq):
        # Only focus on flops because prefill is compute bounded
        # Set the processed to 0 by assuming that need to process the whole prefill
        # flops, _ = self.calculate_prefill_resources(input_length, 0)
        # return flops / GPU_CONFIG_DIC['A100']['flops']
        batch = Batch()
        batch.add_request(PrefillRequest(seq))
        return AIHelper().estimate_time(batch)

    # Calculate the FLOPS and memory bandwidth needed to execute the given batch
    def calculate_resources(self, batch: Batch) -> Tuple[float, float, float, float, float, float, float, float]:
        prefill_attn_flops = prefill_gemm_flops = prefill_attn_memory = prefill_gemm_memory = 0
        decode_attn_flops =  decode_gemm_flops =  decode_attn_memory =  decode_gemm_memory = 0
        ai = 0
        if batch.is_empty():
            return prefill_attn_flops, prefill_gemm_flops, prefill_attn_memory, prefill_gemm_memory, \
                    decode_attn_flops, decode_gemm_flops, decode_attn_memory, decode_gemm_memory

        l_list = batch.prefill_processed
        c_list = batch.chunk_sizes
        l_hat_list = batch.total_processed
        b_d = batch.num_decode

        assert len(l_list) == len(c_list), \
            "l_list and c_list must be of equal length."
            
        prefill_attn_memory  = 0   
        prefill_gemm_memory = 0
        prefill_attn_flops = 0
        prefill_gemm_flops = 0
        for p_hat, c in zip(l_list, c_list):
            prefill_attn_memory += 2 * p_hat * self.s + 3 * c * self.s * (p_hat / self.b)
            prefill_gemm_memory += 8 * p_hat * self.h + 2 * p_hat * self.m
        
        # add the shared weight to prefill if there are prefill chunks
        if c_list:
            prefill_gemm_memory += 4 * self.h**2 + 2 * self.h * self.m
        
        for p_hat, c in zip(l_list, c_list):  
            prefill_attn_flops += 2 * self.s * p_hat * c
            
        prefill_gemm_flops = 2 * sum(c_list) * self.h * self.m + 4 * sum(c_list) * self.h**2

        decode_attn_memory  = 0   
        decode_gemm_memory = 0
        decode_attn_flops = 0
        decode_gemm_flops = 0

        decode_attn_memory += 2 * sum(l_hat_list) * self.s + 2 * self.s
        decode_gemm_memory += 8 * b_d * self.h + 2 * b_d * self.m + 4 * self.h**2 + 2 * self.h * self.m

        for p in l_hat_list:
            decode_attn_flops += 2 * p * self.s

        decode_gemm_flops += 4 * b_d * self.h**2 + 2 * b_d * self.h * self.m
        
        return prefill_attn_flops, prefill_gemm_flops, prefill_attn_memory, prefill_gemm_memory, \
                    decode_attn_flops, decode_gemm_flops, decode_attn_memory, decode_gemm_memory

    def get_batch_compute(self, batch: Batch):
        paf, pgf, pam, pgm, daf, dgf, dam, dgm = self.calculate_resources(batch)
        com = (paf*self.n + pgf + daf*self.n + dgf) * self.l
        mem = (pam*self.n + pgm + dam*self.n + dgm) * self.l
        return com, mem

    # Given the compute and memory of a function, estimates how much time it will take
    # to process that function. Returns time in ms (milliseconds).
    def estimate_time(self, batch: Batch):
        if batch.is_empty():
            return 0

        com, mem = self.get_batch_compute(batch)
        x1 = max(mem / self.GPU_MEM_BANDWIDTH, com / self.GPU_FLOPS)
        x2 = (mem / self.GPU_MEM_BANDWIDTH) + (com / self.GPU_FLOPS)
        if batch.num_decode == 0:
            _model = self.regression_models[PREFILL]
        elif not batch.prefill_processed:
            _model = self.regression_models[DECODE]
        else:
            _model = self.regression_models[MIX]
        predict_time = _model.predict([[x1, x2, com / self.GPU_FLOPS, mem / self.GPU_MEM_BANDWIDTH ]])
        return predict_time[0]
