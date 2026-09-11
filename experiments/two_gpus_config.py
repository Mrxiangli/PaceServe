# Two-GPU variant of main_config.py.
# Results stored under exp_result/micro_data/two_gpus instead of exp_result/main_data.
# Only the GPU-count-specific fields are overridden; everything else comes from main_config.

from pathlib import Path

from main_config import Models, Datasets, Schedulers, build_experiment_suite

BASE_DATA_DIR = Path("exp_result/micro_data/two_gpus")
TIMEOUT_SECONDS = 30 * 60   # 30 minutes per individual run

EXPERIMENT_SUITE = build_experiment_suite(
    static_args_overrides={
        "--instance_config_num_instance": 2,
        "--instance_config_instance_parallel": '"[(1,1),(1,1)]"',
        # Azam (5/21/26): I was running this experiment on i nodes (i000, i003, i004) and
        # it kept getting stuck in the startup phase. Reducing the number of CPUs per instance
        # seems to have fixed the issue, but I don't know why. I guess we give to many CPUs to
        # replica, controller, and each instance. Need to figure out a sustainable way to set
        # this for future experiments.
        "--instance_config_num_cpus_per_instance": 8,
    },
    scheduler_specific_args_overrides={
        Schedulers.PACE:     {"--instance_config_instances": "low_priority high_priority"},
        Schedulers.PACE_NEW: {"--instance_config_instances": "low_priority high_priority"},
        Schedulers.VLLM:     {"--instance_config_instances": "mix mix"},
        Schedulers.SARATHI:  {"--instance_config_instances": "mix mix"},
    },
    workload_params_overrides={
        (Models.LLAMA, Datasets.SHAREGPT): {
            "qps_sweep": range(11, 19),
        },
    },
)
