# This files generate the configuration for all the main experiments.
# This includes running all 3 schedulers (VLLM, SARATHI, PACE) across
# 3 models (LLaMA, Qwen, Mistral) and 2 datasets (ShareGPT, LongBench)
# and sweeping across both QPS and SLOs scale.
# These are our main experiments.


import os
from pathlib import Path

import numpy as np

BASE_DATA_DIR = Path("exp_result/main_data")
TIMEOUT_SECONDS = 30 * 60   # 30 minutes per individual run

# ==========================================
# 1. SEMANTIC CONSTANTS
# ==========================================
class Models:
    LLAMA = "llama"
    QWEN = "qwen"
    MISTRAL = "mistral"

class Datasets:
    SHAREGPT = "sharegpt"
    LONGBENCH = "longbench"

class ExpTypes:
    QPS = "qps"
    SLO = "slo"

class Schedulers:
    VLLM = "VLLM"
    SARATHI = "SARATHI"
    PACE = "PACE"
    PACE_NEW = "PACE_NEW"
    ALL = [VLLM, SARATHI, PACE]

# ==========================================
# 2. PATHS & PHYSICAL LIMITS
# ==========================================
MODEL_PATHS = {
    Models.LLAMA: "$(pwd)/downloaded_model/Llama-3.1-8B-Instruct",
    Models.MISTRAL: "$(pwd)/downloaded_model/Mistral-7B-Instruct-v0.2"
}

HF_REPO_IDS = {
    Models.LLAMA: "meta-llama/Meta-Llama-3.1-8B-Instruct",
    Models.MISTRAL: "mistralai/Mistral-7B-Instruct-v0.2"
}

DATASET_PATHS = {
    Datasets.SHAREGPT: "sharegpt_distserve.csv",
    Datasets.LONGBENCH: "long_bench_filtered.csv"
}

# Maps (Models.XXX, Datasets.YYY) -> configs
WORKLOAD_PARAMS = {
    (Models.LLAMA, Datasets.SHAREGPT):    {"ttft_slo": 1.0,
                                           "tbt_slo": 0.15,
                                           "qps_sweep": range(18, 26),
                                           "constant_qps": 23,
                                           "slo_sweep": [0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.5]},
    (Models.LLAMA, Datasets.LONGBENCH):   {"ttft_slo": 2.5,
                                           "tbt_slo": 0.15,
                                           "qps_sweep": range(4, 10),
                                           "constant_qps": 7,
                                           "slo_sweep": [0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0]},
    (Models.QWEN, Datasets.SHAREGPT):     {"ttft_slo": 1.5,
                                           "tbt_slo": 0.15,
                                           "qps_sweep": [5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9],
                                           "constant_qps": 7.2,
                                           "slo_sweep": [0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0]},
    (Models.QWEN, Datasets.LONGBENCH):    {"ttft_slo": 3.0,
                                           "tbt_slo": 0.15,
                                           "qps_sweep": np.round(np.arange(1.5, 2.01, 0.1), 2).tolist(),
                                           "constant_qps": 1.7,
                                           "slo_sweep": [0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5]},
    (Models.MISTRAL, Datasets.SHAREGPT):  {"ttft_slo": 1.0,
                                           "tbt_slo": 0.15,
                                           "qps_sweep": range(18, 26),
                                           "constant_qps": 24,
                                           "slo_sweep": [0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]},
    (Models.MISTRAL, Datasets.LONGBENCH): {"ttft_slo": 2.5,
                                           "tbt_slo": 0.15,
                                           "qps_sweep": range(4, 10),
                                           "constant_qps": 7,
                                           "slo_sweep": [0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0]},
}

# ==========================================
# 3. BASE CONFIG CONSTANTS (importable for variant configs)
# ==========================================
BASE_STATIC_ARGS = {
    "--model_config_download_dir": "$(pwd)/downloaded_model/",
    "--request_generator_config_type": "SYNTHETIC",
    "--length_generator_config_type": "TRACE",
    "--interval_generator_config_type": "POISSON",
    "--poisson_request_interval_generator_config_seed": 1,
    "--model_config_max_model_len": 8192,
    "--synthetic_request_generator_config_num_requests": 2000,
    "--metrics_config_keep_individual_batch_metrics": True,
    "--trace_request_length_generator_config_prefill_scale_factor": 1,
    "--trace_request_length_generator_config_decode_scale_factor": 1,
    "--num_replicas": 1,
    "--instance_config_num_instance": 3,
    "--instance_config_instance_parallel": '"[(1,1),(1,1),(1,1)]"',
    "--worker_config_gpu_memory_utilization": 0.9,
    "--time_limit": 600,
}

BASE_SCHEDULER_STATIC_ARGS = {
    "max_num_seqs": 128,
}

SCHEDULER_SPECIFIC_ARGS = {
    Schedulers.PACE: {
        "--pace_scheduler_config_allow_drop_requests": 0,
        "--pace_scheduler_config_request_drop_threshold": 0,
        "--pace_scheduler_config_val_func": "EarliestDeadlineFirstValue",
        "--controller_config_controller_type": "pace",
        "--instance_config_instances": "low_priority low_priority high_priority",
    },
    Schedulers.PACE_NEW: {
        "--pace_scheduler_config_allow_drop_requests": 0,
        "--pace_scheduler_config_request_drop_threshold": 0,
        "--pace_scheduler_config_val_func": "EarliestDeadlineFirstValue",
        "--controller_config_controller_type": "pace",
        "--instance_config_instances": "low_priority low_priority high_priority",
    },
    Schedulers.VLLM: {
        # Add any flags here that only vLLM uses
        "--instance_config_instances": "mix mix mix",
        "--controller_config_controller_type": "roundrobin",
    },
    Schedulers.SARATHI: {
        # Add any flags here that only Sarathi uses
        "--instance_config_instances": "mix mix mix",
        "--controller_config_controller_type": "roundrobin",
    }
}

# ==========================================
# 4. EXPERIMENT SUITE BUILDER
# ==========================================
def build_experiment_suite(
    static_args_overrides: dict = None,
    scheduler_specific_args_overrides: dict = None,
    workload_params_overrides: dict = None,
) -> dict:
    """Build the full experiment suite, optionally overriding base config values.

    Args:
        static_args_overrides:
            Keys to add/replace in BASE_STATIC_ARGS for every experiment.
            Available keys and their defaults:

            "--model_config_download_dir"                               : "$(pwd)/downloaded_model/"
            "--request_generator_config_type"                           : "SYNTHETIC"
            "--length_generator_config_type"                            : "TRACE"
            "--interval_generator_config_type"                          : "POISSON"
            "--poisson_request_interval_generator_config_seed"          : 1
            "--model_config_max_model_len"                              : 8192
            "--synthetic_request_generator_config_num_requests"         : 2000
            "--metrics_config_keep_individual_batch_metrics"            : True
            "--trace_request_length_generator_config_prefill_scale_factor" : 1
            "--trace_request_length_generator_config_decode_scale_factor"  : 1
            "--num_replicas"                                            : 1
            "--instance_config_num_instance"                            : 3
            "--instance_config_instance_parallel"                       : "[(1,1),(1,1),(1,1)]"
            "--worker_config_gpu_memory_utilization"                    : 0.9
            "--time_limit"                                              : 600

            The following are set per model/dataset and can also be overridden:
            "--model_config_model"                                      : (from MODEL_PATHS)
            "--trace_request_length_generator_config_trace_file"        : (from DATASET_PATHS)
            "--batchregression_config_regression_model_path"            : (from model key)

        scheduler_specific_args_overrides:
            Per-scheduler keys to add/replace in SCHEDULER_SPECIFIC_ARGS.
            Shape: {Schedulers.XXX: {"--flag": value, ...}, ...}

            PACE / PACE_NEW available keys and their defaults:
            "--pace_scheduler_config_allow_drop_requests"    : 0
            "--pace_scheduler_config_request_drop_threshold" : 0
            "--pace_scheduler_config_val_func"               : "EarliestDeadlineFirstValue"
            "--controller_config_controller_type"            : "pace"
            "--instance_config_instances"                    : "low_priority low_priority high_priority"

            VLLM / SARATHI available keys and their defaults:
            "--instance_config_instances"                    : "mix mix mix"
            "--controller_config_controller_type"            : "roundrobin"

        workload_params_overrides:
            Per-(model, dataset) overrides merged on top of WORKLOAD_PARAMS.
            Shape: {(Models.XXX, Datasets.YYY): {"key": value, ...}, ...}
            Only the specified keys are replaced; unspecified keys keep their
            original values from WORKLOAD_PARAMS.

            Available keys per entry:
            "ttft_slo"      : float   base TTFT SLO threshold (seconds)
            "tbt_slo"       : float   base TBT SLO threshold (seconds)
            "qps_sweep"     : iterable  QPS values for the QPS-sweep experiment
            "constant_qps"  : float   fixed QPS used in the SLO-sweep experiment
            "slo_sweep"     : list    SLO scale factors for the SLO-sweep experiment

            Example:
            workload_params_overrides={
                (Models.LLAMA, Datasets.SHAREGPT): {
                    "qps_sweep": range(10, 18),
                    "constant_qps": 14,
                },
            }
    """
    static_overrides = static_args_overrides or {}
    sched_overrides = scheduler_specific_args_overrides or {}
    workload_overrides = workload_params_overrides or {}

    effective_scheduler_specific_args = {
        sched: {**args, **sched_overrides.get(sched, {})}
        for sched, args in SCHEDULER_SPECIFIC_ARGS.items()
    }

    suite = {}
    for model_key, model_path in MODEL_PATHS.items():
        for dataset_key, dataset_path in DATASET_PATHS.items():
            params = WORKLOAD_PARAMS.get((model_key, dataset_key))
            if not params:
                continue
            params = {**params, **workload_overrides.get((model_key, dataset_key), {})}

            target_ttft = params["ttft_slo"]
            target_tbt = params["tbt_slo"]
            qps_sweep_range = params["qps_sweep"]
            constant_qps = params["constant_qps"]
            slo_sweep = params["slo_sweep"]

            static_args = {
                **BASE_STATIC_ARGS,
                **static_overrides,
                "--model_config_model": model_path,
                "--trace_request_length_generator_config_trace_file": f"$(pwd)/distserve_dataset/{dataset_path}",
                "--batchregression_config_regression_model_path": f"$(pwd)/paceserve/ai_utils/regression_weights/{model_key}",
            }

            # --- 1. Generate QPS Sweep Config ---
            qps_exp_id = f"{model_key}_{dataset_key}_{ExpTypes.QPS}"
            suite[qps_exp_id] = {
                "static_args": static_args,
                "scheduler_static_args": {
                    **BASE_SCHEDULER_STATIC_ARGS,
                    "ttft_slo": target_ttft,
                    "tbt_slo": target_tbt,
                },
                "scheduler_specific_args": effective_scheduler_specific_args,
                "grid": {
                    "qps": qps_sweep_range,
                    "scheduler": Schedulers.ALL,
                }
            }

            # --- 2. Generate SLO Sweep Config ---
            slo_exp_id = f"{model_key}_{dataset_key}_{ExpTypes.SLO}"
            suite[slo_exp_id] = {
                "static_args": {
                    **static_args,
                    "--poisson_request_interval_generator_config_qps": constant_qps,
                },
                "scheduler_static_args": BASE_SCHEDULER_STATIC_ARGS,
                "scheduler_specific_args": effective_scheduler_specific_args,
                "base_slos": {
                    "ttft_slo": target_ttft,
                    "tbt_slo": target_tbt,
                },
                "grid": {
                    "slo_scale": slo_sweep,
                    "scheduler": Schedulers.ALL,
                }
            }

    return suite


EXPERIMENT_SUITE = build_experiment_suite()
