import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scheduler_style import SCHEDULER_MAP, SCHEDULER_COLORS as COLORS, SCHEDULER_LINESTYLES as LINESTYLES, SCHEDULER_ORDER, apply_plot_style

# --- Configuration & Constants ---

def get_slo_thresholds(exp_id: str) -> tuple[float, float]:
    tbt_slo = 0.3
    ttft_slo = 1.5
    return ttft_slo, tbt_slo

def load_data(session_dir: Path) -> dict:
    session_data = {}
    if not session_dir.exists():
        print(f"  [WARNING] Directory not found: {session_dir}")
        return session_data

    for scheduler_dir in session_dir.iterdir():
        if not scheduler_dir.is_dir():
            continue

        raw_scheduler = scheduler_dir.name
        if raw_scheduler not in SCHEDULER_MAP:
            continue

        display_name = SCHEDULER_MAP[raw_scheduler]

        direct_seq_metrics = scheduler_dir / "replica_0" / "sequence_metrics.csv"

        valid_runs = []
        if direct_seq_metrics.exists():
            valid_runs.append((os.path.getmtime(direct_seq_metrics), direct_seq_metrics))
        else:
            for run_dir in scheduler_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                seq_metrics_path = run_dir / "replica_0" / "sequence_metrics.csv"
                if seq_metrics_path.exists():
                    valid_runs.append((os.path.getmtime(seq_metrics_path), seq_metrics_path))

        if not valid_runs:
            print(f"  [WARNING] No valid sequence_metrics.csv found for {raw_scheduler}")
            continue

        valid_runs.sort(reverse=True)
        latest_seq_metrics = valid_runs[0][1]

        df = pd.read_csv(latest_seq_metrics)
        if df.empty:
            continue

        if 'request_in_window' in df.columns:
            df = df[df['request_in_window'] == 1]

        session_data[display_name] = df

    return session_data


def generate_timeline_plot(
    exp_id: str,
    session: str,
    data_dir: Path = None,
    start_time: float = None,
    end_time: float = None,
):
    if data_dir is None:
        data_dir = Path("exp_result/trace_replay_data")
    exp_dir = data_dir / exp_id
    if not exp_dir.exists():
        print(f"Error: Experiment directory {exp_dir} not found.")
        return

    ttft_slo, tbt_slo = get_slo_thresholds(exp_id)
    main_dir = exp_dir / session
    print(f"📊 Loading data from: {main_dir}")

    data_dict = load_data(main_dir)
    if not data_dict:
        print("No valid data found to plot.")
        return

    max_duration = 0.0
    for scheduler, df in data_dict.items():
        if not df.empty and 'request_arrived_at' in df.columns:
            local_start_time = df['request_arrived_at'].min()
            df['relative_time'] = df['request_arrived_at'] - local_start_time

            if start_time is not None:
                df = df[df['relative_time'] >= start_time]
            if end_time is not None:
                df = df[df['relative_time'] <= end_time]

            data_dict[scheduler] = df

            if not df.empty:
                duration = df['relative_time'].max()
                if duration > max_duration:
                    max_duration = duration

    actual_start_time = start_time if start_time is not None else 0.0
    actual_end_time = end_time if end_time is not None else max_duration

    plot_duration = actual_end_time - actual_start_time
    if plot_duration <= 0:
        print("Invalid time range.")
        return

    apply_plot_style()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # --------------------------------------------------
    # Subplot 1: Load (Requests per Second)
    # --------------------------------------------------
    bin_width = max(1.0, plot_duration / 200.0)
    bin_edges = np.arange(actual_start_time, actual_end_time + bin_width, bin_width)

    arrivals = pd.Series(dtype=float)
    for scheduler, df in data_dict.items():
        if not df.empty and 'relative_time' in df.columns:
            arrivals = df['relative_time']
            break

    counts, _ = np.histogram(arrivals.dropna(), bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    qps = counts / bin_width
    ax1.plot(bin_centers, qps, color='gray', linewidth=1.5, alpha=0.9)
    ax1.fill_between(bin_centers, 0, qps, color='gray', alpha=0.3)
    ax1.set_ylabel('QPS (s)')
    ax1.grid(axis='y', linestyle=':', alpha=0.6)

    # --------------------------------------------------
    # Subplot 2: Cumulative SLO Violations
    # --------------------------------------------------
    for scheduler in SCHEDULER_ORDER:
        df = data_dict.get(scheduler)
        if df is None:
            continue
        if 'prefill_e2e_time' not in df.columns or 'decode_mean_time' not in df.columns:
            continue
        if 'relative_time' not in df.columns:
            continue

        df_sorted = df.sort_values(by='relative_time')
        violations = (
            (df_sorted['prefill_e2e_time'] > ttft_slo) |
            (df_sorted['decode_mean_time'] > tbt_slo)
        )
        cumulative_violations = violations.cumsum()
        ax2.plot(
            df_sorted['relative_time'],
            cumulative_violations / 1000,
            label=scheduler,
            color=COLORS.get(scheduler, 'black'),
            linestyle=LINESTYLES.get(scheduler, '-'),
            linewidth=2.5,
            alpha=0.9,
        )

    ax2.set_ylabel('Cumulative SLO\nViolations (x1000)')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # --------------------------------------------------
    # Shared x-axis limits (set once, applies to all panels)
    # --------------------------------------------------
    ax1.set_xlim(actual_start_time, actual_end_time)

    # --------------------------------------------------
    # Shared legend at the top
    # --------------------------------------------------
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(), by_label.keys(),
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        fancybox=True,
        shadow=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    figures_dir = data_dir.parent / "figures" / data_dir.name
    figures_dir.mkdir(parents=True, exist_ok=True)

    suffix = ""
    if start_time is not None or end_time is not None:
        suffix = f"_{actual_start_time:.0f}s_to_{actual_end_time:.0f}s"

    out_name = f"timeline_trace_{exp_id}_{session}{suffix}.pdf"
    output_filename = figures_dir / out_name
    plt.savefig(output_filename, format="pdf", bbox_inches="tight")
    print(f"✅ Timeline plot successfully saved to: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Timeline Plotting Engine for Trace Replay Experiments")
    parser.add_argument("--exp_id", type=str, required=True, help="e.g., llama_sample_trace")
    parser.add_argument("--session", type=str, required=True, help="The session timestamp folder (e.g., '20260505_151914')")
    parser.add_argument("--data_dir", type=Path, default=Path("exp_result/trace_replay_data"),
                        help="Root data directory to load results from (default: exp_result/trace_replay_data).")
    parser.add_argument("--start_time", type=float, default=None, help="Start time in relative seconds")
    parser.add_argument("--end_time", type=float, default=None, help="End time in relative seconds")
    args = parser.parse_args()
    generate_timeline_plot(
        args.exp_id,
        args.session,
        data_dir=args.data_dir,
        start_time=args.start_time,
        end_time=args.end_time,
    )