import os
from pathlib import Path
from main_config import build_experiment_suite, Models, Datasets, Schedulers, ExpTypes

# Base constants from main config, but we override instances and parallel configurations for 1 GPU
static_overrides = {
    "--instance_config_num_instance": 1,
    "--instance_config_instance_parallel": '"[(1,1)]"',
    "--synthetic_request_generator_config_num_requests": 250,
    "--time_limit": 3600,
}

scheduler_overrides = {
    Schedulers.VLLM: {
        "--instance_config_instances": "mix",
    },
    Schedulers.SARATHI: {
        "--instance_config_instances": "mix",
        "--sarathi_scheduler_config_chunk_size": 4096,
    }
}

# The single GPU capacity is about 1/3 of the 3 GPU setting, so scale down QPS accordingly
workload_overrides = {
    (Models.LLAMA, Datasets.SHAREGPT): {
        "qps_sweep": [2, 4, 6, 8, 10],
        "constant_qps": 8,
    }
}

full_suite = build_experiment_suite(
    static_args_overrides=static_overrides,
    scheduler_specific_args_overrides=scheduler_overrides,
    workload_params_overrides=workload_overrides,
)

EXPERIMENT_SUITE = {}
BASE_DATA_DIR = Path("analytical_model_veirification")
TIMEOUT_SECONDS = 10 * 60

qps_id = f"{Models.LLAMA}_{Datasets.SHAREGPT}_{ExpTypes.QPS}"
slo_id = f"{Models.LLAMA}_{Datasets.SHAREGPT}_{ExpTypes.SLO}"

for exp_id in [qps_id, slo_id]:
    if exp_id in full_suite:
        # We only want to run VLLM and SARATHI as per user request
        full_suite[exp_id]["grid"]["scheduler"] = [Schedulers.VLLM, Schedulers.SARATHI]
        EXPERIMENT_SUITE[exp_id] = full_suite[exp_id]
