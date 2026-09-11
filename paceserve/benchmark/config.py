from dataclasses import dataclass, field
from typing import Optional

from paceserve.config import BaseEndpointConfig
from paceserve.config.base_poly_config import BasePolyConfig
from paceserve.config.flat_dataclass import create_flat_dataclass
from paceserve.logger import init_logger
from paceserve.types import (
    ReplicaResourceMapping,
    RequestGeneratorType,
    RequestIntervalGeneratorType,
    RequestLengthGeneratorType,
)

logger = init_logger(__name__)


@dataclass
class BaseRequestIntervalGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42, metadata={"help": "Random seed for the request interval generator."}
    )


@dataclass
class BaseRequestLengthGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42, metadata={"help": "Random seed for the request length generator."}
    )


@dataclass
class TraceRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/AzureFunctionsInvocationTraceForTwoWeeksJan2021Processed.csv",
        metadata={"help": "Path to the trace file for request intervals."},
    )
    start_time: str = field(
        default="1970-01-04 12:00:00", metadata={"help": "Start time for the trace."}
    )
    end_time: str = field(
        default="1970-01-04 15:00:00", metadata={"help": "End time for the trace."}
    )
    time_scale_factor: float = field(
        default=0.3,
        metadata={"help": "Factor to scale the time intervals in the trace."},
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.TRACE


@dataclass
class PoissonRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=1.0,
        metadata={"help": "Queries per second for the Poisson distribution."},
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.POISSON


@dataclass
class GammaRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=1.0, metadata={"help": "Queries per second for the Gamma distribution."}
    )
    cv: float = field(
        default=0.5,
        metadata={"help": "Coefficient of variation for the Gamma distribution."},
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.GAMMA


@dataclass
class StaticRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.STATIC


@dataclass
class TraceRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv",
        metadata={"help": "Path to the trace file for request lengths."},
    )
    prefill_scale_factor: float = field(
        default=1, metadata={"help": "Scale factor for prefill tokens."}
    )
    decode_scale_factor: float = field(
        default=1, metadata={"help": "Scale factor for decode tokens."}
    )
    num_decode_override: int = field(
        default=0, metadata={"help": "Number of decode tokoen override"}
    )
    max_tokens: int = field(
        default=8192, metadata={"help": "Maximum number of tokens allowed."}
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.TRACE


@dataclass
class ZipfRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    theta: float = field(
        default=0.6, metadata={"help": "Theta parameter for the Zipf distribution."}
    )
    scramble: bool = field(
        default=False, metadata={"help": "Whether to scramble the Zipf distribution."}
    )
    min_tokens: int = field(
        default=1024, metadata={"help": "Minimum number of tokens."}
    )
    max_tokens: int = field(
        default=4096, metadata={"help": "Maximum number of tokens."}
    )
    prefill_to_decode_ratio: float = field(
        default=20.0, metadata={"help": "Ratio of prefill tokens to decode tokens."}
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.ZIPF


@dataclass
class UniformRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    min_tokens: int = field(
        default=1024, metadata={"help": "Minimum number of tokens."}
    )
    max_tokens: int = field(
        default=4096, metadata={"help": "Maximum number of tokens."}
    )
    prefill_to_decode_ratio: float = field(
        default=20.0, metadata={"help": "Ratio of prefill tokens to decode tokens."}
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.UNIFORM


@dataclass
class FixedRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    prefill_tokens: int = field(
        default=4096, metadata={"help": "Number of prefill tokens."}
    )
    decode_tokens: int = field(
        default=512, metadata={"help": "Number of decode tokens."}
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.FIXED


@dataclass
class BaseRequestGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42, metadata={"help": "Random seed for the request generator."}
    )


@dataclass
class SyntheticRequestGeneratorConfig(BaseRequestGeneratorConfig):
    length_generator_config: BaseRequestLengthGeneratorConfig = field(
        default_factory=FixedRequestLengthGeneratorConfig
    )
    interval_generator_config: BaseRequestIntervalGeneratorConfig = field(
        default_factory=PoissonRequestIntervalGeneratorConfig
    )
    num_requests: int = field(
        default=64, metadata={"help": "Number of requests to generate."}
    )
    duration: float = field(
        default=None, metadata={"help": "Duration of the synthetic request generation."}
    )

    @staticmethod
    def get_type():
        return RequestGeneratorType.SYNTHETIC


@dataclass
class TraceRequestGeneratorConfig(BaseRequestGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/sydney_enterprise.csv",
        metadata={"help": "Path to the trace file for request generation."},
    )
    date: str = field(
        default="2023-08-21", metadata={"help": "Date for the trace data."}
    )
    prefill_scale_factor: float = field(
        default=0.3, metadata={"help": "Scale factor for prefill tokens."}
    )
    decode_scale_factor: float = field(
        default=1, metadata={"help": "Scale factor for decode tokens."}
    )
    time_scale_factor: float = field(
        default=0.04, metadata={"help": "Scale factor for time intervals."}
    )
    max_tokens: int = field(
        default=4096, metadata={"help": "Maximum number of tokens allowed."}
    )

    @staticmethod
    def get_type():
        return RequestGeneratorType.TRACE

@dataclass
class DatasetRequestGeneratorConfig(BaseRequestGeneratorConfig):
    dataset_name: str = field(
        default="rajpurkar/squad", 
        metadata={"help": "Name of the Hugging Face dataset to load (i.e. wikitext, rajpurkar/squad)"}
    )
    split: str = field(
        default="validation",
        metadata={"help": "Dataset split to use (e.g., train, validation, test)."}
    )
    max_requests: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of requests to generate. None means all."}
    )
    config_name: str = field(
        default="plain_text", 
        metadata={"help": "config of the dataset (wikitext: wikitext-103-raw-v1; rajpurkar/squad:plain_text)."}
    )
    model_name: str = field(
        default="meta-llama/Meta-Llama-3.1-8B-Instruct", 
        metadata={"help": "tokenizer model name"}
    )
    individual_request: int = field(
        default=0,
        metadata={"help": "whether use tokenized request chunk for perplexity testing, 0 means yes"}
    )
    perplexity: int = field(
        default=1,
        metadata={"help": "testing perplexity of the model"}
    )
    

    @staticmethod
    def get_type():
        return RequestGeneratorType.DATASET


@dataclass
class PromptRequestGeneratorConfig(BaseRequestGeneratorConfig):
    prompts: list[str] = field(default_factory=list)

    @staticmethod
    def get_type():
        return RequestGeneratorType.PROMPTS


@dataclass
class BenchmarkConfig(BaseEndpointConfig):
    seed: int = field(default=42, metadata={"help": "Random seed for the benchmark."})
    output_dir: str = field(
        default="exp_result/benchmark_output",
        metadata={"help": "Directory to store benchmark output."},
    )
    write_json_trace: bool = field(
        default=True, metadata={"help": "Whether to write JSON trace output."}
    )
    enable_profiling: bool = field(
        default=False, metadata={"help": "Whether to enable profiling."}
    )
    # We define two types of durations for running experiments:
    # warmup_time and time_limit.
    # Let t0 denote the timestamp when the system is fully initialized and ready.
    # 1. Warm-up Phase:
    #   - From t0 to t0 + warmup_time (denoted as w).
    #   - Requests generated during warm-up are logged and processed, but
    #     they should NOT be included in the final SLO analysis. The purpose is to
    #     bring the system to steady state.
    #   - At t0 + w, assume system is stable.
    # 2. Measurement Window:
    #   - From t0 + w onward, we begin the measurement phase.
    #   - The measurement window lasts for time_limit duration (denoted as d).
    #   - Requests arriving during [t0 + w, t0 + w + d) are considered "in-window"
    #     and shoule be evaluated for SLO adherence.
    # 3. Cool-down and Shutdown:
    #   - The system continues generating new requests and processing even after the
    #     measurement window closes (at t0 + w + d).
    #   - The system continues running until *all* in-window requests have either:
    #       (a) successfully finished, or
    #       (b) explicitly dropped.
    #   - After all in-window requests have been finished, the system shuts down immediately.
    # 4. Hard Cap / Safety Timeout:
    #   - To avoid experiments hanging indefinitely, we enforce a hard upper bound of 2 * time_limit.
    #   - At (t0 + w + d) * 5, the system will forcefully terminate, regardless of request state.
    #   - Any requests still in-flight at this forced cutoff are marked as executing (no `request_execution_time`).
    # Summary:
    #   warmup_time = w
    #   measurement window = [t0 + w, t0 + w + d)
    #   hard cutoff = (t0 + w + d) * 5
    # Calculate the throughput or goodput
    #   Let's consider throughput.
    #   *To get the total number of requests (sequences)*, sort all the
    #   requests with their arrival time and select the ones within the measurement
    #   window (i.e., all the requests with arrival time between [t0 + w, t0 + w + d]).
    #   This set of requests are the referred to as valid requests and these should be
    #   considered for any further analysis.
    #   The duration of the experiment should be calculated from t0 + w to the time
    #   when the *last valid request finished*. Note, the total duration will *NOT* be
    #   time_limit. The last valid request (sorted by arrival time) will finish
    #   after the measurement window has closed.
    #   Similarly, *to get the batch level data*, only consider the batches that
    #   process (either prefill, decode or aborted) a request from the measurement
    #   window (valid request). Next, sort these batches based on their batch id
    #   to mape it to a consistent timeline. As the batch ids are strictly
    #   increasing, this gives us the actual timeline. Additionally, it is possible
    #   that two instances happens to have a similar or close enough batch id.
    #   Therefore, before sorting the batches throught their batch id, group them
    #   based on their instance id.
    time_limit: Optional[float] = field(
        default=float("inf"), metadata={"help": "Time limit for running the experiments (in seconds)."}
    )
    warmup_time: Optional[int] = field(
        default=10, metadata={"help": "The warmup time (in seconds)."}
    )
    static_profile: Optional[int] = field(
        default=0, metadata={"help": "when enable static profile, request are directly added to time window without check time."}
    )
    num_replicas: int = field(
        default=1, metadata={"help": "Number of replicas to use."}
    )
    replica_resource_mapping: Optional[ReplicaResourceMapping] = field(
        default=None, metadata={"help": "Mapping of replicas to resources."}
    )
    request_generator_config: BaseRequestGeneratorConfig = field(
        default_factory=SyntheticRequestGeneratorConfig
    )
