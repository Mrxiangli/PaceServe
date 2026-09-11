import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration & Constants ---
# These are set at runtime from CLI args; see __main__ below.
BASE_DATA_DIR: Path = None
FIGURES_DIR: Path = None

from scheduler_style import (
    DATASET_MAP,
    SCHEDULER_MAP,
    SCHEDULER_COLORS,
    SCHEDULER_MARKERS,
    SCHEDULER_ORDER,
    FONT_LABEL,
    FONT_TICK,
    FONT_LEGEND,
    FONT_TITLE,
    apply_plot_style,
)

def get_slo_thresholds(exp_id: str) -> tuple[float, float]:
    tbt_slo = 0.15
    if "mistral" in exp_id or "llama" in exp_id:
        if "longbench" in exp_id: return 2.5, tbt_slo
        return 1.0, tbt_slo
    else: 
        if "longbench" in exp_id: return 3.0, tbt_slo
        return 1.5, tbt_slo

def get_attainment(seq_metrics: Path, ttft_slo: float, tbt_slo: float) -> float:
    if not seq_metrics.exists():
        return None

    df = pd.read_csv(seq_metrics)
    if df.empty:
        return 0.0

    request_in_sys = df['Request Id'].nunique()
    if request_in_sys == 0:
        return 0.0

    df_processed = df.dropna(subset=['prefill_e2e_time', 'decode_mean_time', 'request_num_prefill_tokens'])
    
    slo_met_mask = (df_processed['prefill_e2e_time'] <= ttft_slo) & (df_processed['decode_mean_time'] <= tbt_slo)
    slo_attainment_pct = (slo_met_mask.sum() / request_in_sys) * 100

    return float(slo_attainment_pct)

def interpolate_90_percent(x_list: list[float], y_list: list[float]) -> float | None:
    for idx in range(1, len(y_list)):
        prev_x, curr_x = x_list[idx-1], x_list[idx]
        prev_y, curr_y = y_list[idx-1], y_list[idx]
        
        if (prev_y > 90 > curr_y) or (prev_y < 90 < curr_y):
            slope = (curr_y - prev_y) / (curr_x - prev_x)
            if slope != 0: 
                return prev_x + (90 - prev_y) / slope
    return None

def crawl_session_data(session_dir: Path, x_axis_key: str, default_ttft: float, default_tbt: float) -> dict:
    """Crawls a SINGLE session directory and extracts its data."""
    session_data = {}

    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    for scheduler_dir in session_dir.iterdir():
        if not scheduler_dir.is_dir():
            continue
            
        raw_scheduler = scheduler_dir.name
        display_name = SCHEDULER_MAP.get(raw_scheduler, raw_scheduler)
        
        if display_name not in session_data:
            session_data[display_name] = {}
        
        for run_dir in scheduler_dir.iterdir():
            if not run_dir.is_dir():
                continue
                
            if (run_dir / "FAILED_CRASH.txt").exists() or (run_dir / "FAILED_TIMEOUT.txt").exists():
                print(f"  [SKIP] Ignoring failed run: {run_dir.name}")
                continue
                
            meta_file = run_dir / "meta.json"
            if not meta_file.exists():
                continue
                
            with open(meta_file, "r") as f:
                meta = json.load(f)
                
            params = meta.get("params", {})
            if x_axis_key not in params:
                continue
                
            x_val = float(params[x_axis_key])

            # For SLO-scale experiments, compute the actual SLO thresholds
            # from the scale factor and the base SLOs.
            if x_axis_key == "slo_scale":
                run_ttft = default_ttft * x_val
                run_tbt = default_tbt * x_val
            else:
                run_ttft = float(params.get("ttft_slo", default_ttft))
                run_tbt = float(params.get("tbt_slo", default_tbt))
            seq_metrics_path = run_dir / "replica_0" / "sequence_metrics.csv"
            if not seq_metrics_path.exists():
                for sub_dir in run_dir.iterdir():
                    if sub_dir.is_dir():
                        potential_path = sub_dir / "replica_0" / "sequence_metrics.csv"
                        if potential_path.exists():
                            seq_metrics_path = potential_path
                            break

            y_val = get_attainment(seq_metrics_path, run_ttft, run_tbt)
            
            if y_val is not None:
                session_data[display_name][x_val] = y_val

    return session_data

def generate_plot(exp_id: str, main_session: str, override_sessions: list, data_dir: Path):
    exp_dir = data_dir / exp_id
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    is_qps_sweep = "qps" in exp_id
    x_axis_key = "qps" if is_qps_sweep else "slo_scale"
    x_label = "QPS" if is_qps_sweep else "SLO Scale Factor"
    default_ttft, default_tbt = get_slo_thresholds(exp_id)

    # 1. Load Main Data
    main_dir = exp_dir / main_session
    print(f"📊 Loading MAIN data from: {main_dir}")
    plot_data = crawl_session_data(main_dir, x_axis_key, default_ttft, default_tbt)

    # --- 2. APPLY MULTIPLE OVERRIDES IN ORDER ---
    for override_session in override_sessions:
        override_dir = exp_dir / override_session
        if not override_dir.exists():
            raise FileNotFoundError(f"Override directory not found: {override_dir}")
            
        print(f"🔄 Applying OVERRIDE data from: {override_dir}")
        override_data = crawl_session_data(override_dir, x_axis_key, default_ttft, default_tbt)
        
        for scheduler, points in override_data.items():
            if scheduler not in plot_data:
                plot_data[scheduler] = {}
            for x_val, y_val in points.items():
                plot_data[scheduler][x_val] = y_val

    if not plot_data:
        print("No valid data found to plot.")
        return

    # 3. Generate the Figure
    figures_dir = data_dir.parent / "figures" / data_dir.name
    figures_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_style()
    plt.figure(figsize=(6, 4))
    
    all_x_values = set()
    for points in plot_data.values():
        all_x_values.update(points.keys())
    sorted_x_values = sorted(list(all_x_values))

    # SLO x-values are unevenly spaced (e.g. 0.7, 0.8, 0.9, 1, 1.5, 2, 3), which
    # causes the smaller values to overlap on a linear axis. Map them to evenly-spaced
    # integer positions and label with the actual values instead.
    if not is_qps_sweep:
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

        plt.plot(plot_x, sorted_y, marker=marker,
                 label=scheduler, color=color, linewidth=2, markersize=8)

        if is_qps_sweep:
            x_at_90 = interpolate_90_percent(sorted_x, sorted_y)
            if x_at_90 is not None:
                 plt.plot([x_at_90, x_at_90], [0, 90], linestyle='--', color=color, linewidth=1)

    plt.axhline(y=90, linestyle='--', color='black', linewidth=1)
    plt.ylim(0, 105)
    plt.legend(fontsize=FONT_LEGEND, loc='lower left')
    plt.xlabel(x_label, fontsize=FONT_LABEL)
    plt.ylabel("SLO attainment (%)", fontsize=FONT_LABEL)

    parts = exp_id.split('_')
    title_str = f"{parts[0].capitalize()}"
    if len(parts) > 1:
        title_str += f" - {DATASET_MAP[parts[1]]}"
    plt.title(title_str, fontsize=FONT_TITLE, fontweight='bold')

    if is_qps_sweep:
        plt.xticks(sorted_x_values, fontsize=FONT_TICK, ha='right')
    else:
        plt.xticks(range(len(sorted_x_values)), [str(v) for v in sorted_x_values],
                   fontsize=FONT_TICK, ha='right')
    plt.yticks(fontsize=FONT_TICK)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    # 4. Save the output
    out_name = f"goodput_{exp_id}_{main_session}"
    if override_sessions:
        out_name += f"_OVERRIDDEN"
    
    output_filename = figures_dir / f"{out_name}.pdf"
    plt.savefig(output_filename, format="pdf", bbox_inches="tight")
    print(f"\n✅ Plot successfully saved to: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting Engine with Override Support")
    parser.add_argument("--exp_id", type=str, required=True, help="e.g., llama_sharegpt_qps")
    parser.add_argument("--data_dir", type=Path, default=Path("exp_result/main_data"),
                        help="Root data directory to load results from "
                             "(default: exp_result/main_data). "
                             "e.g. exp_result/micro_data/two_gpus")
    parser.add_argument("--session", type=str, required=True,
                        help="The name of the session timestamp folder (e.g., '20260322_100000')")
    parser.add_argument("--override", nargs="*", default=[],
                        help="One or more override timestamp folders")

    args = parser.parse_args()
    generate_plot(args.exp_id, args.session, args.override, data_dir=args.data_dir)
