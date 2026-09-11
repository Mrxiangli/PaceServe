import os

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
import ast

from paceserve.config.base_poly_config import BasePolyConfig
from paceserve.config.flat_dataclass import create_flat_dataclass
from paceserve.logger import init_logger
from paceserve.transformers_utils.config import get_config
from paceserve.types import AttentionBackend, ResourceMapping, SchedulerType, ControllerType
from paceserve.utils.hf_utils import get_and_verify_dtype, get_and_verify_max_len

logger = init_logger(__name__)


@dataclass
class ModelConfig:
    model: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "Name or path of the huggingface model to use."},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer."
        },
    )
    download_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory to download and load the weights, default to the default cache directory of huggingface."
        },
    )
    load_format: str = field(
        default="auto",
        metadata={
            "help": "The format of the model weights to load: 'auto', 'pt', 'safetensors', 'npcache', or 'dummy'."
        },
    )
    dtype: str = field(
        default="float16",
        metadata={
            "help": "Data type for model weights and activations. 'auto' will use FP16 for FP32 and FP16 models, and BF16 for BF16 models."
        },
    )
    seed: int = field(default=0, metadata={"help": "Random seed for reproducibility."})
    revision: Optional[str] = field(
        default=None,
        metadata={
            "help": "The specific model version to use. Can be a branch name, tag name, or commit id."
        },
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum length of a sequence (including prompt and output). If None, will be derived from the model."
        },
    )

    def __post_init__(self):
        self.hf_config = get_config(self.model, self.trust_remote_code, self.revision)
        self.dtype = get_and_verify_dtype(self.hf_config, self.dtype)
        self.hf_config.dtype = self.dtype
        self.max_model_len = get_and_verify_max_len(self.hf_config, self.max_model_len)
        self._verify_load_format()

    def _verify_load_format(self) -> None:
        load_format = self.load_format.lower()
        if load_format not in ["auto", "pt", "safetensors", "npcache", "dummy"]:
            raise ValueError(
                f"Unknown load format: {self.load_format}. Must be one of "
                "'auto', 'pt', 'safetensors', 'npcache', or 'dummy'."
            )
        self.load_format = load_format

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_num_attention_heads = self.hf_config.num_attention_heads
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size})."
            )

        total_num_hidden_layers = self.hf_config.num_hidden_layers
        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if total_num_hidden_layers % pipeline_parallel_size != 0:
            raise ValueError(
                f"Total number of hidden layers ({total_num_hidden_layers}) "
                "must be divisible by pipeline parallel size "
                f"({pipeline_parallel_size})."
            )

    def get_hidden_size(self) -> int:
        return self.hf_config.hidden_size

    def get_head_size(self) -> int:
        # FIXME(woosuk): This may not be true for all models.
        return self.hf_config.hidden_size // self.hf_config.num_attention_heads

    def get_num_kv_heads(self, parallel_config: "ParallelConfig") -> int:
        # For GPTBigCode & Falcon:
        # Note: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (
            self.hf_config.model_type in falcon_model_types
            and getattr(self.hf_config, "new_decoder_architecture", False)
        )
        if not new_decoder_arch_falcon and getattr(
            self.hf_config, "multi_query", False
        ):
            # Multi-query attention, only one KV head.
            return 1
        # For Falcon:
        if getattr(self.hf_config, "n_head_kv", None) is not None:
            return self.hf_config.n_head_kv // parallel_config.tensor_parallel_size
        # For Falcon-40b/Falcon-180b:
        if getattr(self.hf_config, "num_kv_heads", None) is not None:
            return self.hf_config.num_kv_heads // parallel_config.tensor_parallel_size
        # For LLaMA-2:
        if getattr(self.hf_config, "num_key_value_heads", None) is not None:
            return (
                self.hf_config.num_key_value_heads
                // parallel_config.tensor_parallel_size
            )
        total_num_attention_heads = self.hf_config.num_attention_heads
        return total_num_attention_heads // parallel_config.tensor_parallel_size

    def get_num_q_heads(self, parallel_config: "ParallelConfig") -> int:
        if getattr(self.hf_config, "num_attention_heads", None) is not None:
            return (
                self.hf_config.num_attention_heads
                // parallel_config.tensor_parallel_size
            )
        raise ValueError("num_attention_heads is not defined in the model config")

    def get_num_layers(self, parallel_config: "ParallelConfig") -> int:
        total_num_hidden_layers = self.hf_config.num_hidden_layers
        return total_num_hidden_layers // parallel_config.pipeline_parallel_size

    def get_total_num_layers(self) -> int:
        return self.hf_config.num_hidden_layers


@dataclass
class CacheConfig:
    block_size: int = field(
        default=16, metadata={"help": "Size of a cache block in number of tokens."}
    )
    num_gpu_blocks: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of GPU blocks for caching. This gets set after profiling."
        },
    )


@dataclass
class ParallelConfig:
    pipeline_parallel_size: int = field(
        default=2, metadata={"help": "Number of pipeline parallel groups."}
    )
    tensor_parallel_size: int = field(
        default=1, metadata={"help": "Number of tensor parallel groups."}
    )

    def __post_init__(self):
        self.world_size = self.pipeline_parallel_size * self.tensor_parallel_size


@dataclass
class BaseSchedulerConfig(BasePolyConfig):
    max_num_seqs: int = field(
        default=128,
        metadata={
            "help": "Maximum number of sequences to be processed in a single iteration (batch size)."
        },
    )

    prefill_policy: str = field(
        default='fcfs',
        metadata={
            "help": "Policy used to sort the prefill request in waiting queue"
        },
    )
    
    allow_drop_requests: int = field(
        default=0, metadata={"help": "Whether enable request dropping"}
    )
    
    request_drop_threshold: float = field(
        default=10, metadata={"help": "how long the will the system wait(queuing delay) before a request is dropped."}
    )
    
    ttft_slo: Optional[float] = field(
        default=1.0, metadata={"help": "time to first token SLO."}
    )
    
    tbt_slo: Optional[float] = field(
        default=0.15, metadata={"help": "time between token SLO."}
    )

    @abstractmethod
    def get_max_num_batched_tokens(self, max_model_len: int):
        pass


@dataclass
class VllmSchedulerConfig(BaseSchedulerConfig):
    max_batched_tokens: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of batched tokens."}
    )

    def get_max_num_batched_tokens(self, max_model_len: int):
        if self.max_batched_tokens:
            return min(self.max_batched_tokens, max_model_len)
        return max_model_len

    @staticmethod
    def get_type():
        return SchedulerType.VLLM


@dataclass
class PaceSchedulerConfig(BaseSchedulerConfig):
    hp_enforce_max_num_seqs: int = field(
        default=0,
        metadata={"help": "Whether to strictly enforce the HP sequence limit."},
    )
    hp_max_num_seqs: Optional[int] = field(
        default=None,
        metadata={"help": "HP sequence limit; defaults to max_num_seqs."},
    )

    max_batched_tokens: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of batched tokens."}
    )

    val_func: str = field(
        default='EarliestDeadlineFirstValue',
        metadata={
            "help": "value function used for reordering in low priority instance"
        },
    )
    
    def get_max_num_batched_tokens(self, max_model_len: int):
        if self.max_batched_tokens:
            return min(self.max_batched_tokens, max_model_len)
        return max_model_len

    @staticmethod
    def get_type():
        return SchedulerType.PACE


@dataclass
class HighPrioritySchedulerConfig(BaseSchedulerConfig):
    enforce_max_num_seqs: int = field(
        default=0,
        metadata={"help": "Whether to strictly enforce max_num_seqs."},
    )

    max_batched_tokens: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of batched tokens."}
    )
    
    val_func: str = field(
        default='EarliestDeadlineFirstValue',
        metadata={
            "help": "value function used for reordering in high priority instance"
        },
    )

    def get_max_num_batched_tokens(self, max_model_len: int):
        if self.max_batched_tokens:
            return min(self.max_batched_tokens, max_model_len)
        return max_model_len

    @staticmethod
    def get_type():
        return SchedulerType.HIGH


@dataclass
class LowPrioritySchedulerConfig(BaseSchedulerConfig):
    max_batched_tokens: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of batched tokens."}
    )
    
    val_func: str = field(
        default='EarliestDeadlineFirstValue',
        metadata={
            "help": "value function used for reordering in low priority instance"
        },
    )

    def get_max_num_batched_tokens(self, max_model_len: int):
        if self.max_batched_tokens:
            return min(self.max_batched_tokens, max_model_len)
        return max_model_len

    @staticmethod
    def get_type():
        return SchedulerType.LOW


@dataclass
class DistserveSchedulerConfig(BaseSchedulerConfig):
    max_batched_tokens: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of batched tokens."}
    )
    enable_premption: bool = field(
        default=False, metadata={"help": "Enable Premption for distserve."}
    )
    def get_max_num_batched_tokens(self, max_model_len: int):
        if self.max_batched_tokens:
            return min(self.max_batched_tokens, max_model_len)
        return max_model_len

    @staticmethod
    def get_type():
        return SchedulerType.DIST

@dataclass
class PrefillSchedulerConfig(BaseSchedulerConfig):
    max_batched_tokens: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of batched tokens."}
    )
    enable_premption: bool = field(
        default=False, metadata={"help": "Enable Premption for distserve."}
    )
    def get_max_num_batched_tokens(self, max_model_len: int):
        if self.max_batched_tokens:
            return min(self.max_batched_tokens, max_model_len)
        return max_model_len

    @staticmethod
    def get_type():
        return SchedulerType.PREFILL

@dataclass
class DecodeSchedulerConfig(BaseSchedulerConfig):
    max_batched_tokens: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of batched tokens."}
    )
    enable_premption: bool = field(
        default=False, metadata={"help": "Enable Premption for distserve."}
    )
    def get_max_num_batched_tokens(self, max_model_len: int):
        if self.max_batched_tokens:
            return min(self.max_batched_tokens, max_model_len)
        return max_model_len

    @staticmethod
    def get_type():
        return SchedulerType.DECODE


@dataclass
class SimpleChunkingSchedulerConfig(BaseSchedulerConfig):
    chunk_size: int = field(
        default=512,
        metadata={"help": "Size of each chunk for simple chunking scheduler."},
    )

    def get_max_num_batched_tokens(self, max_model_len: int):
        return self.chunk_size

    @staticmethod
    def get_type():
        return SchedulerType.SIMPLE_CHUNKING


@dataclass
class OrcaSchedulerConfig(BaseSchedulerConfig):

    def get_max_num_batched_tokens(self, max_model_len: int):
        return self.max_num_seqs * max_model_len

    @staticmethod
    def get_type():
        return SchedulerType.ORCA


@dataclass
class FasterTransformerSchedulerConfig(BaseSchedulerConfig):

    def get_max_num_batched_tokens(self, max_model_len: int):
        return self.max_num_seqs * max_model_len

    @staticmethod
    def get_type():
        return SchedulerType.FASTER_TRANSFORMER


@dataclass
class SarathiSchedulerConfig(BaseSchedulerConfig):
    chunk_size: int = field(
        default=512, metadata={"help": "Size of each chunk for Sarathi scheduler."}
    )
    enable_dynamic_chunking_schedule: bool = field(
        default=False, metadata={"help": "Enable dynamic chunking schedule."}
    )
    low_chunk_size: Optional[int] = field(
        default=None, metadata={"help": "Minimum chunk size for dynamic chunking."}
    )
    high_chunk_size: Optional[int] = field(
        default=None, metadata={"help": "Maximum chunk size for dynamic chunking."}
    )
    chunk_schedule_max_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of tokens for chunk scheduling."},
    )
    chunk_schedule_stages: Optional[int] = field(
        default=None, metadata={"help": "Number of stages for chunk scheduling."}
    )

    def get_max_num_batched_tokens(self, max_model_len: int):
        # Sarathi never schedules more than chunk_size tokens in one iteration.
        if self.enable_dynamic_chunking_schedule:
            return self.high_chunk_size
        else:
            return self.chunk_size

    @staticmethod
    def get_type():
        return SchedulerType.SARATHI
    
@dataclass
class NiyamaSchedulerConfig(BaseSchedulerConfig):
    chunk_size: int = field(
        default=512, metadata={"help": "Size of each chunk for Sarathi scheduler."}
    )
    enable_dynamic_chunking_schedule: bool = field(
        default=False, metadata={"help": "Enable dynamic chunking schedule."}
    )
    low_chunk_size: Optional[int] = field(
        default=None, metadata={"help": "Minimum chunk size for dynamic chunking."}
    )
    high_chunk_size: Optional[int] = field(
        default=None, metadata={"help": "Maximum chunk size for dynamic chunking."}
    )
    chunk_schedule_max_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of tokens for chunk scheduling."},
    )
    chunk_schedule_stages: Optional[int] = field(
        default=None, metadata={"help": "Number of stages for chunk scheduling."}
    )

    def get_max_num_batched_tokens(self, max_model_len: int):
        # Sarathi never schedules more than chunk_size tokens in one iteration.
        if self.enable_dynamic_chunking_schedule:
            return self.high_chunk_size
        else:
            return self.chunk_size

    @staticmethod
    def get_type():
        return SchedulerType.NIYAMA


@dataclass
class MetricsConfig:
    """Metric configuration."""

    write_metrics: bool = field(
        default=True, metadata={"help": "Whether to write metrics."}
    )
    wandb_project: Optional[str] = field(
        default=None, metadata={"help": "Weights & Biases project name."}
    )
    wandb_group: Optional[str] = field(
        default=None, metadata={"help": "Weights & Biases group name."}
    )
    wandb_run_name: Optional[str] = field(
        default=None, metadata={"help": "Weights & Biases run name."}
    )
    wandb_sweep_id: Optional[str] = field(
        default=None, metadata={"help": "Weights & Biases sweep ID."}
    )
    wandb_run_id: Optional[str] = field(
        default=None, metadata={"help": "Weights & Biases run ID."}
    )
    enable_op_level_metrics: bool = field(
        default=False, metadata={"help": "Enable operation-level metrics."}
    )
    enable_cpu_op_level_metrics: bool = field(
        default=False, metadata={"help": "Enable CPU operation-level metrics."}
    )
    enable_chrome_trace: bool = field(
        default=True, metadata={"help": "Enable Chrome tracing."}
    )
    enable_request_outputs: bool = field(
        default=False, metadata={"help": "Enable request outputs."}
    )
    keep_individual_batch_metrics: bool = field(
        default=False, metadata={"help": "Keep individual batch metrics."}
    )


@dataclass
class ReplicaConfig:
    replica_id: int = field(default=0, metadata={"help": "ID of the replica."})
    output_dir: str = field(
        default=".", metadata={"help": "Output directory for the replica."}
    )
    resource_mapping: Optional[ResourceMapping] = field(
        default=None, metadata={"help": "Resource mapping for the replica."}
    )

    def __post_init__(self):
        self.output_dir = f"{self.output_dir}/replica_{self.replica_id}"

    def get_resource_mapping(self, world_size: int):
        if not self.resource_mapping:
            self.resource_mapping = [
                (None, i) for i in range(world_size)  # List of (node_ip, gpu_id)
            ]
        return self.resource_mapping


@dataclass
class WorkerConfig:
    gpu_memory_utilization: float = field(
        default=0.8, metadata={"help": "GPU memory utilization fraction (0.0 to 1.0)."}
    )
    attention_backend: AttentionBackend = field(
        default=AttentionBackend.FLASHINFER,
        metadata={"help": "Backend to use for attention computation."},
    )

    def __post_init__(self):
        self._verify_args()

    def _verify_args(self) -> None:
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}."
            )
            
            
@dataclass # used for take input and also for create LLM engine
class InstanceConfig:
    num_instance: int = field(default=1, metadata={"help": "ID of the replica."})
    # This is for input
    instances: List[str] = field(
        default_factory=lambda: ['mix'],
        metadata={"help": "A list to specify each runner's type"}
    )
    # This is for internal use only
    _type: str = field(
        default='mix',
        metadata={"help": "A list to specify each runner's type"}
    )
    instance_parallel: str = field(
        default="[(1, 1)]",
        metadata={"help": "A list to specify (TP, PP) of each instance"}
    )
    resource_mapping: Optional[ResourceMapping] = field(
        default=None, metadata={"help": "Resource mapping for the replica."}
    )
    
    num_cpus_per_instance: int = field(
        default=24,
        metadata={"help": "Number of CPUs to reserve per instance actor. "
                          "The internal RayWorkers derive their CPU count as "
                          "num_cpus_per_instance // (3 * world_size)."}
    )

    calculate_rouge: int = field(
        default= 0,
        metadata={"help": "enable rouge score calculation per instance"}
    )
    
    def get_resource_mapping(self, world_size: int):
        if not self.resource_mapping:
            self.resource_mapping = [
                (None, i) for i in range(world_size)  # List of (node_ip, gpu_id)
            ]
        return self.resource_mapping
    
    def __post_init__(self):
        self.instance_parallel = ast.literal_eval(self.instance_parallel)
        if not(isinstance(self.instance_parallel, list) and all(isinstance(item, tuple) for item in self.instance_parallel)):
            raise ValueError("Input is not a list of tuples.")

        # Call the argument verification method
        self._verify_args()

    def _verify_args(self) -> None:
        for each in self.instances:
            if each not in ['mix', 'prefill', 'decode', 'low_priority', 'high_priority']:
                raise ValueError(
                    f"Instance type should be one of (mix, prefill, decode, low_priority, high_priority),"
                    f"but got: {each}"
                )


@dataclass
class ControllerConfig:
    controller_type: str = field(
        default='roundrobin',
        metadata={"help": "A list to specify each runner's type"}
    )
    
    use_logits: int = field(
        default=0,
        metadata={"help": "configure model runner return logits for perplexity calculation"}
    )
    
    def __post_init__(self):
        self._verify_args()

    def _verify_args(self) -> None:
       if self.controller_type not in ['roundrobin', 'distserve', 'pace']:
           raise ValueError(f"controller should be one of roundrobin, distserve, pace")
    
    def get_type(self):
        if self.controller_type == 'roundrobin':
            return ControllerType.ROUNDROBIN
        if self.controller_type == 'distserve':
            return ControllerType.DISTSERVE
        if self.controller_type == 'pace':
            return ControllerType.PACE


@dataclass
class BatchregressionConfig:
    regression_model_path: str = field(
        default='',
        metadata={"help": "the path to the directory contains the linear regression model weights for batch time prediction"}
    )
    
    def __post_init__(self):
        self._verify_args()

    def _verify_args(self) -> None:
        pkl_files = [f for f in os.listdir(self.regression_model_path) if f.endswith(".pkl")]
        assert len(pkl_files) == 3, "There should be exactly three .pkl files in the directory."     
        
        
@dataclass
class SystemConfig:
    replica_config: ReplicaConfig = field(default_factory=ReplicaConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    worker_config: WorkerConfig = field(default_factory=WorkerConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    instance_config: InstanceConfig = field(default_factory=InstanceConfig)
    controller_config: ControllerConfig = field(default_factory=ControllerConfig)
    scheduler_config: BaseSchedulerConfig = field(
        default_factory=SarathiSchedulerConfig
    )
    metrics_config: MetricsConfig = field(default_factory=MetricsConfig)
    batchregression_config: BatchregressionConfig = field(default_factory=BatchregressionConfig)


@dataclass
class BaseEndpointConfig(ABC):
    log_level: str = field(default="info", metadata={"help": "Logging level."})
    output_dir: str = field(default="output", metadata={"help": "Output directory."})
    model_config: ModelConfig = field(default_factory=ModelConfig)
    worker_config: WorkerConfig = field(default_factory=WorkerConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    instance_config: InstanceConfig = field(default_factory=InstanceConfig)
    controller_config: ControllerConfig = field(default_factory=ControllerConfig)
    scheduler_config: BaseSchedulerConfig = field(
        default_factory=SarathiSchedulerConfig
    )
    metrics_config: MetricsConfig = field(default_factory=MetricsConfig)
    batchregression_config: BatchregressionConfig = field(default_factory=BatchregressionConfig)

    def __post_init__(self):
        self.output_dir = (
            f"{self.output_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"
        )

    @classmethod
    def create_from_cli_args(cls):
        flat_config = create_flat_dataclass(cls).create_from_cli_args()
        instance = flat_config.reconstruct_original_dataclass()
        instance.__flat_config__ = flat_config
        return instance

    def to_dict(self):
        if not hasattr(self, "__flat_config__"):
            logger.warning("Flat config not found. Returning the original config.")
            return self.__dict__

        return self.__flat_config__.__dict__

    def create_system_config(self, replica_config: ReplicaConfig) -> SystemConfig:
        system_config = SystemConfig(
            replica_config=replica_config,
            worker_config = self.worker_config,
            model_config=self.model_config,
            cache_config=self.cache_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            controller_config=self.controller_config,
            metrics_config=self.metrics_config,
            batchregression_config = self.batchregression_config
        )
        return system_config
    
    def create_instance_system_config(self, replica_config: ReplicaConfig, instance_config: InstanceConfig, parallel_config: ParallelConfig) -> SystemConfig:
        system_config = SystemConfig(
            instance_config=instance_config,
            replica_config=replica_config,
            worker_config = self.worker_config,
            model_config=self.model_config,
            cache_config=self.cache_config,
            parallel_config=parallel_config,
            scheduler_config=self.scheduler_config,
            controller_config=self.controller_config,
            metrics_config=self.metrics_config,
            batchregression_config = self.batchregression_config
        )
        return system_config
