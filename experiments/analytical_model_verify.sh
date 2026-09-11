#!/usr/bin/env bash
# =============================================================================
# experiments/single_gpu_experiments.sh
#
# Runs LLaMA ShareGPT experiments on a single GPU using VLLM and SARATHI schedulers.
# =============================================================================

set -euo pipefail

# Load shared env, paths, and run_exp helper (common.sh lives in the same dir)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh" "$@"

LOG_ROOT="${REPO_ROOT}/analytical_model_veirification/logs"

echo ""
echo "################################################################"
echo "#         Analytical model verification experiments            #"
echo "################################################################"

run_exp "single_gpu_config" "llama_sharegpt_qps"

echo ""
echo "################################################################"
echo "#   ✅ Single GPU experiments complete  $(date)"
echo "################################################################"
echo "   Logs saved under: ${LOG_ROOT}/single_gpu_config"

echo ""
echo "################################################################"
echo "#   Running analytical model plotting script"
echo "################################################################"
${PYTHON} ${REPO_ROOT}/plotting_tools/plot_analytical_model.py
