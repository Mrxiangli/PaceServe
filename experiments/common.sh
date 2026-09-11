#!/usr/bin/env bash
# =============================================================================
# experiments/common.sh
#
# Shared setup sourced by goodput_experiments.sh, trace_experiments.sh, and
# run_all_experiments.sh.  Do NOT execute this file directly.
#
# Provides:
#   $PYTHON        — absolute path to the Python interpreter
#   $REPO_ROOT     — absolute path to the repository root
#   $RUN_SCRIPT    — absolute path to experiments/run.py
#   $LOG_ROOT      — root directory for per-experiment log files
#   $TIMESTAMP     — session timestamp (YYYYmmdd_HHMMSS)
#   $DRY_RUN_FLAG  — "--dry-run" if the script was called with --dry-run, else ""
#   run_exp()      — helper that executes one experiment and reports elapsed time
# =============================================================================

# Keep incompatible packages in ~/.local out of the experiment environment.
export PYTHONNOUSERSITE=1

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python3"
elif command -v python3 &> /dev/null; then
    PYTHON="$(command -v python3)"
else
    PYTHON="/scratch/gilbreth/li2068/artifact/middleware-env/bin/python3"
fi

# common.sh always lives in experiments/, so go one level up to get the repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT="${REPO_ROOT}/experiments/run.py"
LOG_ROOT="${REPO_ROOT}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

DRY_RUN_FLAG=""
for arg in "$@"; do
    if [[ "${arg}" == "--dry-run" ]]; then
        DRY_RUN_FLAG="--dry-run"
        echo "🔍 DRY-RUN mode: commands will be printed but NOT executed."
        break
    fi
done



# Reuse the exported Hugging Face directory, falling back to the repo default.
export HF_HOME="${HF_HOME:-${REPO_ROOT}/downloaded_model}"
if [[ -L "${HF_HOME}" && ! -d "${HF_HOME}" ]]; then
    echo "ERROR: HF_HOME points to a symlink with a missing or inaccessible directory: ${HF_HOME}" >&2
    return 1
fi
mkdir -p "${HF_HOME}" || return 1

# Change to repo root so relative paths in configs resolve correctly
cd "${REPO_ROOT}"

# --------------------------------------------------------------------------
# run_exp <config> <exp_id>
#   Runs a single experiment via run.py, tee-ing stdout+stderr to a log file,
#   and prints a timing summary afterwards.
# --------------------------------------------------------------------------
run_exp() {
    local config="$1"
    local exp_id="$2"
    local log_dir="${LOG_ROOT}/${config}"
    
    # Ensure log directory exists for this configuration
    mkdir -p "${log_dir}"
    
    local log_file="${log_dir}/${exp_id}_${TIMESTAMP}.log"

    echo ""
    echo "============================================================"
    echo "  CONFIG  : ${config}"
    echo "  EXP ID  : ${exp_id}"
    echo "  LOG     : ${log_file}"
    echo "  STARTED : $(date)"
    echo "============================================================"

    local cmd="${PYTHON} -u ${RUN_SCRIPT} --config ${config} --exp_id ${exp_id} ${DRY_RUN_FLAG} ${RESUME_FLAG:-}"
    echo "  CMD: ${cmd}"
    echo ""

    local t_start
    t_start=$(date +%s)

    # tee so output is visible in the terminal AND persisted to the log file
    ${cmd} 2>&1 | tee "${log_file}"

    local exit_code="${PIPESTATUS[0]}"
    local t_end
    t_end=$(date +%s)
    local elapsed=$(( t_end - t_start ))
    local h=$(( elapsed / 3600 ))
    local m=$(( (elapsed % 3600) / 60 ))
    local s=$(( elapsed % 60 ))
    local elapsed_str
    if   (( h > 0 )); then elapsed_str="${h}h $(printf '%02d' ${m})m $(printf '%02d' ${s})s"
    elif (( m > 0 )); then elapsed_str="${m}m $(printf '%02d' ${s})s"
    else                   elapsed_str="${s}s"
    fi

    echo ""
    echo "  FINISHED  : $(date)"
    echo "  ELAPSED   : ${elapsed_str}"
    echo "  EXIT CODE : ${exit_code}"

    if [[ "${exit_code}" -ne 0 ]]; then
        echo "  ⚠️  WARNING: ${exp_id} exited with non-zero code ${exit_code}. Continuing to next experiment."
    fi
}
