#!/usr/bin/env bash
# Run the two-GPU and 2H1L QPS sweeps used by plot_micro.py.
# Usage:
#   bash experiments/micro_experiments.sh
#   bash experiments/micro_experiments.sh --dry-run
# Results: exp_result/micro_data/{two_gpus,2H1L}/
# Logs: logs/{two_gpus_config,two_h_one_l_config}/

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh" "$@"

echo "Running two-GPU LLaMA ShareGPT sweep (VLLM, SARATHI, PACE)"
run_exp "two_gpus_config" "llama_sharegpt_qps"

echo "Running 2H1L LLaMA LongBench sweep (PACE)"
run_exp "two_h_one_l_config" "llama_longbench_qps" --filter scheduler=PACE

echo ""
echo "Micro experiments complete. Logs saved under:"
echo "  ${LOG_ROOT}/two_gpus_config"
echo "  ${LOG_ROOT}/two_h_one_l_config"
