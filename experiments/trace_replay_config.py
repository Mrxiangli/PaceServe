# This file generates the configuration for the trace-replay experiments.
# Unlike main_config.py (which uses Poisson arrival), these experiments use
# real request timestamps from captured wild traces to replay arrivals exactly.
#
# The request_generator_config_type is set to TRACE, which reads:
#   - arrival times from the "Time" column (seconds, absolute)
#   - token counts from "PromptTokenCount" / "CompletionTokenCount"
# All of this comes from a single CSV file — no separate length/interval generators.
#
# Two traces are configured:
#   - CODE: wild/code_distributions_processed.csv  (coding workload)
#   - CONV: wild/conv_distributions_processed.csv  (conversational workload)
#
# Schedulers: VLLM, SARATHI, PACE  (LLaMA model only)


import os
from pathlib import Path

BASE_DATA_DIR = Path("exp_result/trace_replay_data")
TIMEOUT_SECONDS = 90 * 60   # 90 minutes per individual run

# ==========================================
# 1. SEMANTIC CONSTANTS
# ==========================================
class Models:
    LLAMA = "llama"

class Traces:
    CODE   = "code"
    CONV   = "conv"
    SAMPLE = "sample"   # small subset for quick smoke-tests

class Schedulers:
    VLLM     = "VLLM"
    SARATHI  = "SARATHI"
    PACE = "PACE"
    ALL = [VLLM, SARATHI, PACE]

# ==========================================
# 2. PATHS
# ==========================================
MODEL_PATHS = {
    Models.LLAMA: "$(pwd)/downloaded_model/Llama-3.1-8B-Instruct",
}

# Each trace entry: (csv path relative to $(pwd), date string to filter)
TRACE_PATHS = {
    Traces.CODE:   ("wild/code_distributions_processed.csv", "2023-11-16"),
    Traces.CONV:   ("wild/conv_distributions_processed.csv", "2023-11-16"),
    Traces.SAMPLE: ("wild/sample.csv",                       "2023-11-16"),
}

# Per (model, trace) SLO targets
WORKLOAD_PARAMS = {
    (Models.LLAMA, Traces.CODE):   {"ttft_slo": 1.5, "tbt_slo": 0.3},
    (Models.LLAMA, Traces.CONV):   {"ttft_slo": 0.5, "tbt_slo": 0.1},
    (Models.LLAMA, Traces.SAMPLE): {"ttft_slo": 1.0, "tbt_slo": 0.5},
}

# ==========================================
# 3. EXPERIMENT SUITE BUILDER
# ==========================================
EXPERIMENT_SUITE = {}

for model_key, model_path in MODEL_PATHS.items():
    for trace_key, (trace_path, trace_date) in TRACE_PATHS.items():
        params = WORKLOAD_PARAMS.get((model_key, trace_key))
        if not params:
            continue

        target_ttft = params["ttft_slo"]
        target_tbt  = params["tbt_slo"]

        # -------------------------------------------------------
        # Global engine arguments shared across all schedulers.
        # Key difference from main_config:
        #   --request_generator_config_type TRACE
        #     → uses TraceRequestGenerator, which reads arrival
        #       timestamps directly from "Time" column (seconds).
        #   --trace_request_generator_config_time_scale_factor 1.0
        #     → replay at real speed (no compression / dilation).
        #   No --interval_generator_config_type or --poisson_* args
        #     needed; timing is fully driven by the trace file.
        # -------------------------------------------------------
        static_args = {
            "--model_config_model": model_path,
            "--model_config_download_dir": "$(pwd)/downloaded_model/",
            "--request_generator_config_type": "TRACE",
            "--trace_request_generator_config_trace_file": f"$(pwd)/{trace_path}",
            "--trace_request_generator_config_date": trace_date,
            "--trace_request_generator_config_time_scale_factor": 1.0,
            "--trace_request_generator_config_prefill_scale_factor": 1,
            "--trace_request_generator_config_decode_scale_factor": 1,
            "--trace_request_generator_config_max_tokens": 8192,
            "--model_config_max_model_len": 8192,
            "--metrics_config_keep_individual_batch_metrics": True,
            "--num_replicas": 1,
            "--instance_config_num_instance": 3,
            "--instance_config_instance_parallel": '"[(1,1),(1,1),(1,1)]"',
            "--batchregression_config_regression_model_path": f"$(pwd)/paceserve/ai_utils/regression_weights/{model_key}",
            "--worker_config_gpu_memory_utilization": 0.9,
        }

        # Shared scheduler args (prefix auto-prepended by run.py)
        scheduler_static_args = {
            "max_num_seqs": 128,
            "ttft_slo": target_ttft,
            "tbt_slo": target_tbt,
        }

        # Per-scheduler flags injected verbatim
        scheduler_specific_args = {
            Schedulers.PACE: {
                "--pace_scheduler_config_allow_drop_requests": 0,
                "--pace_scheduler_config_request_drop_threshold": 0,
                "--pace_scheduler_config_val_func": "EarliestDeadlineFirstValue",
                "--controller_config_controller_type": "pace",
                "--instance_config_instances": "low_priority low_priority high_priority",
            },
            Schedulers.VLLM: {
                "--vllm_scheduler_config_allow_drop_requests": 0,
                "--vllm_scheduler_config_request_drop_threshold": 0,
                "--instance_config_instances": "mix mix mix",
                "--controller_config_controller_type": "roundrobin",
            },
            Schedulers.SARATHI: {
                "--sarathi_scheduler_config_allow_drop_requests": 0,
                "--sarathi_scheduler_config_request_drop_threshold": 0,
                "--instance_config_instances": "mix mix mix",
                "--controller_config_controller_type": "roundrobin",
            },
        }

        exp_id = f"{model_key}_{trace_key}_trace"
        EXPERIMENT_SUITE[exp_id] = {
            "static_args": static_args,
            "scheduler_static_args": scheduler_static_args,
            "scheduler_specific_args": scheduler_specific_args,
            "grid": {
                "scheduler": Schedulers.ALL,
            },
        }
