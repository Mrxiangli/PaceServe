"""Admission-control comparison based on goodput_benchmark_sweep.sh."""

from pathlib import Path

from experiments.main_config import Datasets, Models, Schedulers, build_experiment_suite

BASE_DATA_DIR = Path("exp_result/admission_control_data")
TIMEOUT_SECONDS = 20 * 60
QPS = 24.5
TTFT_SLO = 1.0
TBT_SLO = 0.15
DURATION_SECONDS = 600
EXP_ID = "llama_sharegpt_admission"
VLLM_MAX_NUM_SEQS = 256
# HP has its own limit; LP keeps Pace's configured max_num_seqs (128).
HP_MAX_NUM_SEQS = 256
HP_ENFORCE_MAX_NUM_SEQS = 1
# All three schedulers are read from the latest session for comparison.
SCHEDULERS_TO_RUN = [Schedulers.PACE, Schedulers.VLLM, Schedulers.SARATHI]
if (not isinstance(SCHEDULERS_TO_RUN, (list, tuple)) or not SCHEDULERS_TO_RUN
        or any(scheduler not in Schedulers.ALL for scheduler in SCHEDULERS_TO_RUN)
        or len(set(SCHEDULERS_TO_RUN)) != len(SCHEDULERS_TO_RUN)):
    raise ValueError("SCHEDULERS_TO_RUN must contain unique VLLM, SARATHI, or PACE schedulers")
PLOT_COMPARISON = set(SCHEDULERS_TO_RUN) == set(Schedulers.ALL)

_suite = build_experiment_suite(
    static_args_overrides={
        "--time_limit": DURATION_SECONDS,
        # Keep arrivals flowing for the full observation period; the normal
        # 2,000-request suite would exhaust its workload much too early.
        "--synthetic_request_generator_config_duration": DURATION_SECONDS,
        "--instance_config_num_cpus_per_instance": 8,
    },
    scheduler_specific_args_overrides={
        scheduler: {
            f"--{scheduler.lower()}_scheduler_config_allow_drop_requests": 1,
            f"--{scheduler.lower()}_scheduler_config_request_drop_threshold": TTFT_SLO,
            **({"--vllm_scheduler_config_max_num_seqs": VLLM_MAX_NUM_SEQS}
               if scheduler == Schedulers.VLLM else {}),
            **({"--pace_scheduler_config_hp_enforce_max_num_seqs": HP_ENFORCE_MAX_NUM_SEQS,
                "--pace_scheduler_config_hp_max_num_seqs": HP_MAX_NUM_SEQS}
               if scheduler == Schedulers.PACE else {}),
        }
        for scheduler in Schedulers.ALL
    },
    workload_params_overrides={
        (Models.LLAMA, Datasets.SHAREGPT): {
            "qps_sweep": [QPS], "ttft_slo": TTFT_SLO, "tbt_slo": TBT_SLO,
        },
    },
)
EXPERIMENT_SUITE = {EXP_ID: _suite["llama_sharegpt_qps"]}
EXPERIMENT_SUITE[EXP_ID]["grid"]["scheduler"] = list(SCHEDULERS_TO_RUN)
