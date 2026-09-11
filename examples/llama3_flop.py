import torch
from torchinfo import summary
from sarathi.model_executor.models.llama import LlamaForCausalLM, LlamaDecoderLayer
from sarathi.model_executor.parallel_utils.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from sarathi.model_executor.layers.layernorm import RMSNorm
from sarathi.transformers_utils.config import get_config
from sarathi.config import ModelConfig, ReplicaConfig, MetricsConfig
from sarathi.model_executor import get_model
import torch.nn as nn
import torch.distributed as dist
from sarathi.metrics.metrics_store import MetricsStore
from sarathi.model_executor.parallel_utils.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
    initialize_model_parallel,
)
from sarathi.utils import get_ip, get_random_port
from sarathi.model_executor.attention.base_attention_wrapper import BaseAttentionWrapper

torch.distributed.init_process_group(
            backend="nccl",
            world_size=1,
            rank=0,
            init_method=f"tcp://{get_ip()}:{get_random_port()}",
        )
initialize_model_parallel(1, 1)

# doing it layer by layer to avoid attention
custom_cache_dir = "/scratch/gilbreth/li2068/asplos/sarathi-serve/downloaded_model"  # Change this to your desired path
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

model_config = ModelConfig(model = model_name, download_dir = custom_cache_dir)
replica_config = ReplicaConfig()
metrics_config = MetricsConfig()

metrics_store = MetricsStore.get_or_create_instance(
            replica_config,
            model_config,
            metrics_config,
        )

class qkv_proj(nn.Module):
    def __init__(self, input_size, output_size):
        super(qkv_proj, self).__init__()
        self.layer1 = ColumnParallelLinear(input_size, output_size, world_size=1)
        
    def forward(self, x):
        output, _ = self.layer1(x)
        return output
    
class o_proj(nn.Module):
    def __init__(self, input_size, output_size):
        super(o_proj, self).__init__()
        self.layer1 = RowParallelLinear(input_size, output_size, world_size=1)
        
    def forward(self, x):
        output, _ = self.layer1(x)
        return output

class gate_up_proj(nn.Module):
    def __init__(self, input_size, output_size):
        super(gate_up_proj, self).__init__()
        self.layer1 = ColumnParallelLinear(input_size, output_size, world_size=1)
        
    def forward(self, x):
        output, _ = self.layer1(x)
        return output

class down_proj(nn.Module):
    def __init__(self, input_size, output_size):
        super(down_proj, self).__init__()
        self.layer1 = RowParallelLinear(input_size, output_size, world_size=1)
        
    def forward(self, x):
        output, _ = self.layer1(x)
        return output

class rms(nn.Module):
    def __init__(self, input_size):
        super(rms, self).__init__()
        self.layer1 = RMSNorm(input_size)
        
    def forward(self, x):
        output = self.layer1(x)
        return output
    
hidden_size = 4096
intermediate_size = 14336
total_num_heads = 32
total_num_kv_heads = 8
head_dim = 128
num_layers = 32

qkv_model = qkv_proj(hidden_size, (total_num_heads+2*total_num_kv_heads)*head_dim)
o_model = o_proj(total_num_heads * head_dim, hidden_size)
up_model = gate_up_proj(hidden_size, 2 * intermediate_size)
down_model = down_proj(intermediate_size, hidden_size)
rms_model = rms(hidden_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qkv_model.to(device)
o_model.to(device)
up_model.to(device)
down_model.to(device)
rms_model.to(device)

seq_length = 1000
batch_size = 1
qkv_input = torch.randn(seq_length, batch_size, hidden_size).to(device)
o_input = torch.randn(seq_length, batch_size, total_num_heads * head_dim).to(device)
up_input = torch.randn(seq_length, batch_size, hidden_size).to(device)
down_input = torch.randn(seq_length, batch_size, intermediate_size).to(device)
rms_input = torch.randn(seq_length, batch_size, hidden_size).to(device)

# Get summary of the model

qkv_sum = summary(qkv_model, qkv_input.shape).total_mult_adds
o_sum = summary(o_model, o_input.shape).total_mult_adds 
up_sum = summary(up_model, up_input.shape).total_mult_adds
down_sum = summary(down_model, down_input.shape).total_mult_adds
rms_sum = summary(rms_model, rms_input.shape).total_mult_adds

one_decode_layer = (qkv_sum + o_sum + up_sum + down_sum) * batch_size  # ignore attention 
total = one_decode_layer * num_layers
print(f"total flops: {total/10**12}")

flops_gemm = (2 * seq_length * hidden_size * intermediate_size + 4 * seq_length * hidden_size**2)*num_layers
print(f"our estimation: {flops_gemm/10**12}")

