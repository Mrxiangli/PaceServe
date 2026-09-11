#!/usr/bin/env bash
# Usage: bash experiments/plot_micro.sh [--data-dir PATH] [--dry-run]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE=1
exec "${PYTHON:-python3}" "${SCRIPT_DIR}/plot_micro.py" "$@"
