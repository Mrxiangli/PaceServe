"""
Combined 6x2 Goodput Figure
============================
Generates a single figure with 12 subplots (6 columns × 2 rows).

Row 1 (QPS sweep):   llama_sharegpt, llama_longbench, mistral_sharegpt, mistral_longbench, qwen_sharegpt, qwen_longbench
Row 2 (SLO sweep):   llama_sharegpt, llama_longbench, mistral_sharegpt, mistral_longbench, qwen_sharegpt, qwen_longbench

Session paths are stored in goodput_plot_config.json (next to this script).
Generate it from your results directory, then run the plot:

    python experiments/generate_goodput_plot_config.py exp_result/main_data
    python experiments/goodput_plot_combined.py

Config format (goodput_plot_config.json):
    {
        "llama_sharegpt_qps": {"main": "20260507_183345", "overrides": []},
        "qwen_sharegpt_qps":  {"main": "20260509_120000", "overrides": ["20260509_140000"]},
        ...
    }

Missing experiments are displayed as empty panels.
"""

import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Reuse helpers from the single-experiment plotting script
from goodput_plot import (
    get_slo_thresholds,
    crawl_session_data,
    interpolate_90_percent,
)
from scheduler_style import (
    SCHEDULER_COLORS,
    SCHEDULER_MARKERS,
    SCHEDULER_ORDER,
    FONT_TICK,
    FONT_LABEL,
    FONT_TITLE,
    FONT_LEGEND,
    apply_plot_style,
)

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "goodput_plot_config.json"

# ── Layout definition ────────────────────────────────────────────────
COLUMNS = [
    ("llama_longbench", "Llama - LongBench"),
    ("llama_sharegpt",  "Llama - ShareGPT"),
    ("mistral_longbench", "Mistral - LongBench"),
    ("mistral_sharegpt", "Mistral - ShareGPT"),
    # ("qwen_longbench",  "Qwen - LongBench"),
    # ("qwen_sharegpt",   "Qwen - ShareGPT"),
]

ROWS = [
    ("qps", "Request Rate (QPS)"),
    ("slo", "SLO Scale Factor"),
]

ALL_EXP_IDS = [
    f"{col_prefix}_{row_suffix}"
    for row_suffix, _ in ROWS
    for col_prefix, _ in COLUMNS
]


def load_config() -> dict:
    """Load the JSON config, stripping comment keys."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH) as f:
        raw = json.load(f)

    return {k: v for k, v in raw.items() if not k.startswith("_") or k == "_data_dir"}


def resolve_session(exp_id: str, config: dict) -> tuple[str, list[str]]:
    """Return (main_session, override_sessions) for a given experiment."""
    if exp_id not in config:
        raise KeyError(f"Experiment '{exp_id}' not found in config: {CONFIG_PATH}")
    entry = config[exp_id]
    main_raw = entry.get("main", "") if isinstance(entry, dict) else entry
    overrides = entry.get("overrides", []) if isinstance(entry, dict) else []
    if not main_raw:
        raise ValueError(f"No main session specified for '{exp_id}' in config: {CONFIG_PATH}")
    return main_raw, overrides


def load_plot_data(exp_id: str, main_session: str, override_sessions: list[str],
                   data_dir: Path, static_data: dict = None) -> dict:
    """Load data for a single experiment, applying overrides and static data."""
    exp_dir = data_dir / exp_id
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    is_qps = "qps" in exp_id
    x_axis_key = "qps" if is_qps else "slo_scale"
    default_ttft, default_tbt = get_slo_thresholds(exp_id)

    # Main data
    main_dir = exp_dir / main_session
    plot_data = crawl_session_data(main_dir, x_axis_key, default_ttft, default_tbt)

    # Overrides
    for override_session in override_sessions:
        override_dir = exp_dir / override_session
        if not override_dir.exists():
            raise FileNotFoundError(f"Override directory not found: {override_dir}")
        override_data = crawl_session_data(override_dir, x_axis_key, default_ttft, default_tbt)
        for scheduler, points in override_data.items():
            if scheduler not in plot_data:
                plot_data[scheduler] = {}
            for x_val, y_val in points.items():
                plot_data[scheduler][x_val] = y_val

    # Merge static data (hardcoded x/y points from config)
    for scheduler, points in (static_data or {}).items():
        if scheduler not in plot_data:
            plot_data[scheduler] = {}
        for x_val, y_val in points.items():
            plot_data[scheduler][float(x_val)] = y_val

    return plot_data


def draw_subplot(ax, plot_data: dict, exp_id: str,
                 show_legend: bool, invert_x: bool = False, baselines: dict = None):
    """Draw a single subplot on the given Axes."""
    is_qps = "qps" in exp_id

    all_x_values = set()
    for points in plot_data.values():
        all_x_values.update(points.keys())
    sorted_x_values = sorted(list(all_x_values))

    # SLO x-values are unevenly spaced (e.g. 0.7, 0.8, 0.9, 1, 1.5, 2, 3), which
    # causes the smaller values to overlap on a linear axis. Map them to evenly-spaced
    # integer positions and label with the actual values instead.
    if not is_qps:
        x_to_pos = {x: i for i, x in enumerate(sorted_x_values)}
        to_plot_x = lambda x: x_to_pos[x]  # noqa: E731
    else:
        to_plot_x = lambda x: x  # noqa: E731

    for scheduler in SCHEDULER_ORDER:
        if scheduler not in plot_data or not plot_data[scheduler]:
            continue

        points = plot_data[scheduler]
        sorted_x = sorted(points.keys())
        sorted_y = [points[x] for x in sorted_x]
        plot_x   = [to_plot_x(x) for x in sorted_x]

        color  = SCHEDULER_COLORS.get(scheduler, '#888888')
        marker = SCHEDULER_MARKERS.get(scheduler, 'x')
        ax.plot(plot_x, sorted_y,
                marker=marker,
                label=scheduler,
                color=color,
                linewidth=2,
                markersize=6)

        if is_qps:
            x_at_90 = interpolate_90_percent(sorted_x, sorted_y)
            if x_at_90 is not None:
                ax.plot([x_at_90, x_at_90], [0, 90],
                        linestyle='--', color=color, linewidth=1)

    if is_qps and baselines:
        for baseline_name, goodput_val in baselines.items():
            color = SCHEDULER_COLORS.get(baseline_name, "#D62728")
            ax.plot([goodput_val, goodput_val], [0, 90], linestyle='--', color=color, linewidth=2, label=baseline_name)
            ax.plot([goodput_val], [90], marker='*', color=color, markersize=10)

    ax.axhline(y=90, linestyle='--', color='black', linewidth=1)
    ax.set_ylim(0, 105)

    if sorted_x_values:
        if is_qps:
            ax.set_xticks(sorted_x_values)
        else:
            ax.set_xticks(range(len(sorted_x_values)))
            ax.set_xticklabels([str(v) for v in sorted_x_values])
        ax.tick_params(axis='x', labelsize=FONT_TICK)
        plt.setp(ax.get_xticklabels(), ha='right')
    ax.tick_params(axis='y', labelsize=FONT_TICK)
    ax.grid(True, linestyle=':', alpha=0.6)

    # Invert x-axis for SLO row so scale factors decrease left-to-right
    if invert_x and sorted_x_values:
        ax.invert_xaxis()

    if show_legend:
        ax.legend(fontsize=FONT_LEGEND, loc='lower left')


def main():
    parser = argparse.ArgumentParser(description="Generate combined 6x2 goodput figure.")
    parser.add_argument(
        '--data_dir', type=Path, default=None,
        help="Root results directory (default: generated config's _data_dir, or exp_result/main_data)."
    )
    args = parser.parse_args()
    config = load_config()
    data_dir = args.data_dir or Path(config.pop("_data_dir", "exp_result/main_data"))
    baseline_config = config.pop("baselines", {})
    static_data_config = config.pop("static_data", {})

    # Resolve sessions
    print("=" * 60)
    print("Resolving session mapping:")
    resolved: dict[str, tuple[str, list[str]]] = {}
    for exp_id in ALL_EXP_IDS:
        if exp_id not in config:
            print(f"[SKIP] No session configured for {exp_id}")
            continue
        main_session, overrides = resolve_session(exp_id, config)
        resolved[exp_id] = (main_session, overrides)
    print("=" * 60 + "\n")

    # Build the figure
    apply_plot_style()
    n_rows = len(ROWS)
    n_cols = len(COLUMNS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             sharey=True, squeeze=False)

    for row_idx, (row_suffix, x_label) in enumerate(ROWS):
        for col_idx, (col_prefix, col_title) in enumerate(COLUMNS):
            exp_id = f"{col_prefix}_{row_suffix}"
            ax = axes[row_idx, col_idx]

            if exp_id not in resolved:
                ax.text(0.5, 0.5, "No session configured", ha="center", va="center",
                        transform=ax.transAxes)
                if row_idx == 0:
                    ax.set_title(col_title, fontsize=FONT_TITLE, fontweight='bold')
                continue

            main_session, overrides = resolved[exp_id]

            static_data = static_data_config.get(exp_id, {})

            print(f"📊 Loading: {exp_id} (session: {main_session})")
            plot_data = load_plot_data(exp_id, main_session, overrides, data_dir, static_data)

            if row_idx == 0:
                ax.set_title(col_title, fontsize=FONT_TITLE, fontweight='bold')

            baselines = baseline_config.get(exp_id, {})

            draw_subplot(
                ax,
                plot_data,
                exp_id,
                show_legend=(col_idx == 0 and row_idx == 0),
                invert_x=(row_suffix == "slo"),
                baselines=baselines,
            )

    fig.supylabel("SLO attainment (%)", fontsize=FONT_LABEL)
    if len(ROWS) > 1:
        # For 2x4 main figure without Qwen
        fig.tight_layout(rect=[0.001, 0, 1, 0.97], h_pad=2.5)
    else:
        # For 1x2 figure with just Qwen
        # fig.tight_layout()
        fig.tight_layout(rect=[-0.02, 0, 1, 1])

    # One centered x-label per row, placed after tight_layout so axis positions are final
    for row_idx, (_, x_label) in enumerate(ROWS):
        row_axes = axes[row_idx]
        x_center = (row_axes[0].get_position().x0 + row_axes[-1].get_position().x1) / 2
        y_bottom = min(ax.get_position().y0 for ax in row_axes)
        if len(ROWS) > 1:
            fig.text(x_center, y_bottom - 0.04, x_label,
                    ha='center', va='top', fontsize=FONT_LABEL)
        else:
            fig.text(x_center, y_bottom - 0.08, x_label,
                    ha='center', va='top', fontsize=FONT_LABEL)

    FIGURES_DIR = data_dir.parent / "figures" / data_dir.name
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "goodput_combined.pdf"
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"\n✅ Combined figure saved to: {output_path}")


if __name__ == "__main__":
    main()
