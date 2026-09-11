#!/usr/bin/env bash
# =============================================================================
# experiments/trace_experiments.sh
#
# Runs all trace-replay experiments defined in trace_replay_config.py
# (LLaMA model against CODE and CONV wild traces).
#
# Usage:
#   bash experiments/trace_experiments.sh           # full run
#   bash experiments/trace_experiments.sh --dry-run # print commands only
#
# Logs are written to:
#   logs/trace_replay_config/<exp_id>_<timestamp>.log
# =============================================================================

set -euo pipefail

# Load shared env, paths, and run_exp helper (common.sh lives in the same dir)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh" "$@"
RESUME_FLAG="--resume"

# =============================================================================
# LLaMA — CODE and CONV wild traces
# =============================================================================
echo ""
echo "################################################################"
echo "#         Trace-replay experiments                             #"
echo "################################################################"

run_exp "trace_replay_config" "llama_code_trace"
#run_exp "trace_replay_config" "llama_conv_trace"
# run_exp "trace_replay_config" "llama_sample_trace"   # smoke-test only

echo ""
echo "################################################################"
echo "#   ✅ Trace experiments complete  $(date)"
echo "################################################################"
echo "   Logs saved under: ${LOG_ROOT}/trace_replay_config"
