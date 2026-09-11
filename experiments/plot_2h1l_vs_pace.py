"""
Compare 2H1L vs PACE goodput for a QPS sweep.

Usage:
    python plot_2h1l_vs_pace.py path_to_pace_dir path_to_2h1l_dir --title "My Plot Title" [--output figures]
"""

import sys
import argparse
import json
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

from goodput_plot import (
    get_attainment,
    interpolate_90_percent,
)
from scheduler_style import (
    SCHEDULER_COLORS,
    SCHEDULER_MARKERS,
    SCHEDULER_MAP,
    FONT_LABEL,
    FONT_TICK,
    FONT_LEGEND,
    FONT_TITLE,
    apply_plot_style,
)


def load_scheduler_data(scheduler_dir, x_axis_key="qps", default_ttft=1.0, default_tbt=0.15):
    """Load goodput data from a scheduler directory (contains runs)."""
    data = {}

    for run_dir in sorted(scheduler_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        if (run_dir / "FAILED_CRASH.txt").exists() or (run_dir / "FAILED_TIMEOUT.txt").exists():
            continue

        meta_file = run_dir / "meta.json"
        if not meta_file.exists():
            continue

        with open(meta_file) as f:
            meta = json.load(f)

        params = meta.get("params", {})
        if x_axis_key not in params:
            continue

        x_val = float(params[x_axis_key])
        run_ttft = float(params.get("ttft_slo", default_ttft))
        run_tbt = float(params.get("tbt_slo", default_tbt))

        seq_metrics = run_dir / "replica_0" / "sequence_metrics.csv"
        if not seq_metrics.exists():
            for sub in run_dir.iterdir():
                if sub.is_dir():
                    alt = sub / "replica_0" / "sequence_metrics.csv"
                    if alt.exists():
                        seq_metrics = alt
                        break

        y_val = get_attainment(seq_metrics, run_ttft, run_tbt)
        if y_val is not None:
            data[x_val] = y_val

    return data


def extract_slos_from_config(scheduler_dir):
    """Extract TTFT and TBT SLOs from the first config.yaml found."""
    for run_dir in scheduler_dir.iterdir():
        if not run_dir.is_dir():
            continue
        config_file = run_dir / "config.yaml"
        if config_file.exists():
            with open(config_file) as f:
                config = yaml.safe_load(f)
            ttft = config.get("pace_scheduler_config_ttft_slo")
            tbt = config.get("pace_scheduler_config_tbt_slo")
            if ttft and tbt:
                return float(ttft), float(tbt)
    return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare 2H1L vs PACE goodput (QPS sweep)")
    parser.add_argument("pace_dir", type=Path, help="Path to PACE scheduler directory")
    parser.add_argument("h2l_dir", type=Path, help="Path to 2H1L scheduler directory")
    parser.add_argument("--title", type=str, required=True, help="Plot title")
    parser.add_argument("--output", type=Path, default=Path("exp_result", "figures"), help="Output directory")
    args = parser.parse_args()

    ttft, tbt = extract_slos_from_config(args.pace_dir)
    if ttft is None or tbt is None:
        print(f"Error: Could not extract SLO values from config.yaml in {args.pace_dir}")
        sys.exit(1)

    pace_data = load_scheduler_data(args.pace_dir, default_ttft=ttft, default_tbt=tbt)
    h2l_data = load_scheduler_data(args.h2l_dir, default_ttft=ttft, default_tbt=tbt)

    if not pace_data or not h2l_data:
        print("Error: No data found in one or both directories")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    apply_plot_style()
    fig = plt.figure(figsize=(6, 4))

    for label, data, color_key in [("Pace-2L1H", pace_data, "PaceServe"), ("Pace-2H1L", h2l_data, "2H1L")]:
        sorted_x = sorted(data.keys())
        sorted_y = [data[x] for x in sorted_x]
        color = SCHEDULER_COLORS.get(color_key, "#888888")
        marker = SCHEDULER_MARKERS.get(color_key, "o")

        plt.plot(sorted_x, sorted_y, marker=marker, label=label, color=color,
                 linewidth=2, markersize=8)

        x_90 = interpolate_90_percent(sorted_x, sorted_y)
        if x_90:
            plt.plot([x_90, x_90], [0, 90], linestyle="--", color=color, linewidth=1)

    plt.axhline(y=90, linestyle="--", color="black", linewidth=1)
    plt.ylim(0, 105)
    plt.xlabel("QPS", fontsize=FONT_LABEL)
    plt.ylabel("SLO attainment (%)", fontsize=FONT_LABEL)
    plt.title(args.title, fontsize=FONT_TITLE, fontweight="bold")
    plt.xticks(fontsize=FONT_TICK)
    plt.yticks(fontsize=FONT_TICK)
    plt.legend(fontsize=FONT_LEGEND, loc="lower left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    output_file = args.output / "2h1l.pdf"
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    print(f"✅ Plot saved to {output_file}")
