#!/usr/bin/env bash
# =============================================================================
# experiments/goodput_experiments.sh
#
# Runs all goodput (Poisson QPS / SLO sweep) experiments defined in
# main_config.py across LLaMA, Mistral, and Qwen models.
#
# Usage:
#   bash experiments/goodput_experiments.sh           # full run
#   bash experiments/goodput_experiments.sh --dry-run # print commands only
#
# Logs are written to:
#   logs/main_config/<exp_id>_<timestamp>.log
# =============================================================================

set -euo pipefail

# Load shared env, paths, and run_exp helper (common.sh lives in the same dir)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh" "$@"
RESUME_FLAG="--resume"

# =============================================================================
# LLaMA — ShareGPT + LongBench (QPS & SLO sweeps)
# =============================================================================
echo ""
echo "################################################################"
echo "#         LLaMA experiments                                    #"
echo "################################################################"

run_exp "main_config" "llama_longbench_qps"
run_exp "main_config" "llama_longbench_slo"
run_exp "main_config" "llama_sharegpt_qps"
run_exp "main_config" "llama_sharegpt_slo"

# =============================================================================
# Mistral — ShareGPT + LongBench (QPS & SLO sweeps)
# =============================================================================
echo ""
echo "################################################################"
echo "#         Mistral experiments                                  #"
echo "################################################################"

run_exp "main_config" "mistral_longbench_qps"
run_exp "main_config" "mistral_longbench_slo"
run_exp "main_config" "mistral_sharegpt_qps"
run_exp "main_config" "mistral_sharegpt_slo"

# =============================================================================
# Qwen — ShareGPT + LongBench (QPS & SLO sweeps)
# =============================================================================
echo ""
echo "################################################################"
echo "#         Qwen experiments                                     #"
echo "################################################################"

run_exp "main_config" "qwen_longbench_qps"
run_exp "main_config" "qwen_longbench_slo"
run_exp "main_config" "qwen_sharegpt_qps"
run_exp "main_config" "qwen_sharegpt_slo"

echo ""
echo "################################################################"
echo "#   ✅ Goodput experiments complete  $(date)"
echo "################################################################"
echo "   Logs saved under: ${LOG_ROOT}/main_config"
