import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

# --- Configuration & Constants ---
BASE_DATA_DIR = Path("exp_result/trace_replay_data")
FIGURES_DIR = Path("exp_result/figures")

COLORS = ['#70A5D9', '#ECA680', '#F0868C', '#88B04B', '#9B59B6', '#F39C12']

SCHEDULER_MAP = {
    'VLLM': 'vLLM',
    'SARATHI': 'Sarathi',
    'PACE': 'PaceServe',
    'PACE_NEW': 'PaceServe (New)'
}

def get_slo_thresholds(exp_id: str) -> tuple[float, float]:
    # Hardcoded values per your trace_replay_config.py
    tbt_slo = 0.3
    ttft_slo = 1.5
    return ttft_slo, tbt_slo

def get_attainment(seq_metrics: Path, ttft_slo: float, tbt_slo: float) -> tuple[float, int, int, int, int, int, float, float]:
    if not seq_metrics.exists():
        return None, 0, 0, 0, 0, 0, 0.0, 0.0

    df = pd.read_csv(seq_metrics)
    if df.empty:
        return 0.0, 0, 0, 0, 0, 0, 0.0, 0.0

    if 'request_in_window' in df.columns:
        df = df[df['request_in_window'] == 1]

    if df.empty:
        return 0.0, 0, 0, 0, 0, 0, 0.0, 0.0

    # Identify offload-event rows: request_offloaded_time == -1.0
    # These rows have no prefill metrics but are NOT drops — they are offload events.
    # An offloaded request appears twice: once as the offload event (no metrics),
    # and once again with actual completion metrics.
    offload_col = 'request_offloaded_time'
    if offload_col in df.columns:
        offloaded_mask = df[offload_col] == -1.0
    else:
        offloaded_mask = pd.Series(False, index=df.index)

    # Total unique requests = unique Request Ids excluding offload-event rows
    df_non_offload_events = df[~offloaded_mask]
    request_in_sys = df_non_offload_events['Request Id'].nunique()
    if request_in_sys == 0:
        return 0.0, 0, 0, 0, 0, 0, 0.0, 0.0

    # Offloaded count: number of offload-event rows (each represents one offload)
    offloaded_count = int(offloaded_mask.sum())

    # Dropped: missing prefill_e2e_time AND not an offload-event row
    if 'prefill_e2e_time' in df.columns:
        missing_prefill = df['prefill_e2e_time'].isna()
        dropped_mask = missing_prefill & ~offloaded_mask
        dropped_count = df[dropped_mask]['Request Id'].nunique()
    else:
        dropped_count = 0

    df_processed = df_non_offload_events.dropna(subset=['prefill_e2e_time', 'decode_mean_time', 'request_num_prefill_tokens'])

    slo_met_mask = (df_processed['prefill_e2e_time'] <= ttft_slo) & (df_processed['decode_mean_time'] <= tbt_slo)
    slo_attainment_pct = (slo_met_mask.sum() / request_in_sys) * 100

    ttft_violations = int((df_processed['prefill_e2e_time'] > ttft_slo).sum())
    tbt_violations  = int((df_processed['decode_mean_time']  > tbt_slo).sum())

    avg_ttft = df_processed['prefill_e2e_time'].mean()
    avg_tbt = df_processed['decode_mean_time'].mean()

    return float(slo_attainment_pct), int(request_in_sys), int(offloaded_count), int(dropped_count), ttft_violations, tbt_violations, float(avg_ttft), float(avg_tbt)


def crawl_session_data(session_dir: Path, ttft_slo: float, tbt_slo: float) -> dict:
    """Crawls a SINGLE trace replay session directory and extracts its data."""
    session_data = {}

    if not session_dir.exists():
        print(f"  [WARNING] Directory not found: {session_dir}")
        return session_data

    for scheduler_dir in session_dir.iterdir():
        if not scheduler_dir.is_dir():
            continue
            
        raw_scheduler = scheduler_dir.name
        display_name = SCHEDULER_MAP.get(raw_scheduler, raw_scheduler)
        
        # Check if the directory was flattened (successful run)
        direct_seq_metrics = scheduler_dir / "replica_0" / "sequence_metrics.csv"
        
        valid_runs = []
        if direct_seq_metrics.exists():
            valid_runs.append((os.path.getmtime(direct_seq_metrics), direct_seq_metrics))
        else:
            # If not flattened, look inside timestamp directories
            for run_dir in scheduler_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                    
                seq_metrics_path = run_dir / "replica_0" / "sequence_metrics.csv"
                if seq_metrics_path.exists():
                    valid_runs.append((os.path.getmtime(seq_metrics_path), seq_metrics_path))
                    
        if not valid_runs:
            print(f"  [WARNING] No valid sequence_metrics.csv found for {raw_scheduler}")
            continue
            
        # Get the most recent run
        valid_runs.sort(reverse=True)
        latest_seq_metrics = valid_runs[0][1]
        
        attainment, req_count, offloaded_count, dropped_count, ttft_violations, tbt_violations, avg_ttft, avg_tbt = get_attainment(latest_seq_metrics, ttft_slo, tbt_slo)
        if attainment is not None:
            session_data[display_name] = {
                'attainment': attainment,
                'req_count': req_count,
                'offloaded_count': offloaded_count,
                'dropped_count': dropped_count,
                'ttft_violations': ttft_violations,
                'tbt_violations': tbt_violations,
                'avg_ttft': avg_ttft,
                'avg_tbt': avg_tbt
            }
            print(f"  Found {display_name}: {attainment:.1f}% attainment across {req_count} requests ({offloaded_count} offloaded, {dropped_count} dropped, {ttft_violations} TTFT viol, {tbt_violations} TBT viol).")

    return session_data

def generate_plot(exp_id: str, main_session: str):
    exp_dir = BASE_DATA_DIR / exp_id
    if not exp_dir.exists():
        print(f"Error: Experiment directory {exp_dir} not found.")
        return

    ttft_slo, tbt_slo = get_slo_thresholds(exp_id)

    # 1. Load Data
    main_dir = exp_dir / main_session
    print(f"📊 Loading data from: {main_dir}")
    plot_data = crawl_session_data(main_dir, ttft_slo, tbt_slo)

    if not plot_data:
        print("No valid data found to plot.")
        return

    # Print a summary table
    col_width = 130
    print("\n" + "="*col_width)
    print(f"Results for {exp_id} ({main_session})")
    print(f"Target SLOs: TTFT <= {ttft_slo}s, TBT <= {tbt_slo}s")
    print("="*col_width)
    print(f"{'Scheduler':<20} | {'Attainment':<10} | {'Requests':<10} | {'Offloaded':<11} | {'Dropped':<10} | {'Drop%':<8} | {'TTFT Viol':<11} | {'TBT Viol':<10} | {'Avg TTFT':<10} | {'Avg TBT':<10}")
    print("-"*col_width)
    for scheduler, metrics in sorted(plot_data.items()):
        req_count = metrics['req_count']
        offloaded = metrics['offloaded_count']
        dropped = metrics['dropped_count']
        drop_pct = (dropped / req_count * 100) if req_count > 0 else 0.0
        ttft_viol = metrics['ttft_violations']
        tbt_viol  = metrics['tbt_violations']
        print(f"{scheduler:<20} | {metrics['attainment']:>8.1f}% | {req_count:<10} | {offloaded:<11} | {dropped:<10} | {drop_pct:<7.1f}% | {ttft_viol:<11} | {tbt_viol:<10} | {metrics['avg_ttft']:<9.3f}s | {metrics['avg_tbt']:<9.3f}s")
    print("="*col_width + "\n")


    # 2. Generate the Figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    
    schedulers = list(plot_data.keys())
    attainments = [plot_data[s]['attainment'] for s in schedulers]
    
    # Assign colors matching plot.py
    bar_colors = [COLORS[list(SCHEDULER_MAP.values()).index(s) % len(COLORS)] if s in SCHEDULER_MAP.values() else COLORS[0] for s in schedulers]

    bars = plt.bar(schedulers, attainments, color=bar_colors, edgecolor='black', linewidth=1.2, alpha=0.9)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.axhline(y=90, linestyle='--', color='black', linewidth=1.5, label='90% Target')
    plt.ylim(0, 105)
    plt.ylabel("SLO Attainment (%)", fontsize=16)
    
    parts = exp_id.split('_')
    title_str = " ".join([p.capitalize() for p in parts])
    plt.title(f"{title_str}\n(TTFT≤{ttft_slo}s, TBT≤{tbt_slo}s)", fontsize=16)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()

    # 3. Save the output
    out_name = f"trace_replay_{exp_id}_{main_session}"
    output_filename = FIGURES_DIR / f"{out_name}.pdf"
    plt.savefig(output_filename, format="pdf", bbox_inches="tight")
    print(f"✅ Plot successfully saved to: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting Engine for Trace Replay Experiments")
    parser.add_argument("--exp_id", type=str, required=True, help="e.g., llama_sample_trace")
    parser.add_argument("--session", type=str, required=True, help="The session timestamp folder (e.g., '20260505_151914')")
    
    args = parser.parse_args()
    generate_plot(args.exp_id, args.session)
