# Two-H-one-L variant of main_config.py.
# Results stored under exp_result/micro_data/2H1L instead of exp_result/main_data.
# Only the GPU-count-specific fields are overridden; everything else comes from main_config.

from pathlib import Path

from main_config import Schedulers, build_experiment_suite

BASE_DATA_DIR = Path("exp_result/micro_data/2H1L")
TIMEOUT_SECONDS = 30 * 60   # 30 minutes per individual run

EXPERIMENT_SUITE = build_experiment_suite(
    static_args_overrides={
        "--instance_config_num_cpus_per_instance": 8,
    },
    scheduler_specific_args_overrides={
        Schedulers.PACE:     {"--instance_config_instances": "low_priority high_priority high_priority"},
    },
)
