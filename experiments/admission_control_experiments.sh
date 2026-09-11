#!/usr/bin/env bash
# Run configured schedulers, then compare all three from the latest session.
# Usage: bash experiments/admission_control_experiments.sh [--dry-run]
set -euo pipefail

for arg in "$@"; do
    case "${arg}" in
        --dry-run) ;;
        *) echo "Usage: bash experiments/admission_control_experiments.sh [--dry-run]" >&2; exit 2 ;;
    esac
done

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh" "$@"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
run_exp "experiments.admission_control_config" "llama_sharegpt_admission"

PLOT_COMPARISON=$("${PYTHON}" -c 'from experiments.admission_control_config import PLOT_COMPARISON; print(int(PLOT_COMPARISON))')
if [[ "${PLOT_COMPARISON}" != "1" ]]; then
    echo "Skipping comparison plot: all three schedulers must be selected."
elif [[ -z "${DRY_RUN_FLAG}" ]]; then
    "${PYTHON}" "${REPO_ROOT}/experiments/plot_admission_control.py"
else
    echo "After a real run, plot with: python experiments/plot_admission_control.py"
fi
