import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

def plot_cdf(series_dict, title, xlabel, filename):
    plt.figure(figsize=(6, 4))
    for label, series in series_dict.items():
        sorted_data = np.sort(series.dropna())
        if len(sorted_data) == 0:
            continue
        yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(sorted_data, yvals, label=label, linewidth=2)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('CDF', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def parse_metrics(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Filter aborted (it is NaN or 0.0 for successful requests)
    if 'request_aborted' in df.columns:
        df = df[df['request_aborted'] != 1.0]
        
    required_cols = ['prefill_e2e_time', 'decode_mean_time']
    cols_to_drop = [c for c in required_cols if c in df.columns]
    if cols_to_drop:
        df = df.dropna(subset=cols_to_drop)
        
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, required=True, help='Path to PACE root (e.g. root/PACE/qps_25)')
    parser.add_argument('--stealing', type=str, required=True, help='Path to PACE_NEW root (e.g. root/PACE_NEW/qps_25)')
    args = parser.parse_args()
    
    base_path = Path(args.baseline) / 'replica_0' / 'sequence_metrics.csv'
    steal_path = Path(args.stealing) / 'replica_0' / 'sequence_metrics.csv'
    
    base_df = parse_metrics(base_path)
    steal_df = parse_metrics(steal_path)
    
    if base_df is None or steal_df is None:
        print("Missing metric files.")
        sys.exit(1)
        
    out_dir = Path('exp_result/images/debug_stealing')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. TTFT CDF
    plot_cdf({
        'PACE (Baseline)': base_df['prefill_e2e_time'],
        'PACE_NEW (Stealing)': steal_df['prefill_e2e_time']
    }, 'TTFT Comparison', 'TTFT (s)', out_dir / 'ttft_cdf_compare.pdf')
    
    # 2. TBT CDF
    plot_cdf({
        'PACE (Baseline)': base_df['decode_mean_time'],
        'PACE_NEW (Stealing)': steal_df['decode_mean_time']
    }, 'TBT Comparison', 'TBT (s)', out_dir / 'tbt_cdf_compare.pdf')
    
    # 3. Load Balance
    plt.figure(figsize=(8, 5))
    
    base_counts = base_df['Instance Id'].value_counts().sort_index()
    steal_counts = steal_df['Instance Id'].value_counts().sort_index()
    
    # Combine
    all_instances = sorted(list(set(base_counts.keys()) | set(steal_counts.keys())))
    
    x = np.arange(len(all_instances))
    width = 0.35
    
    base_vals = [base_counts.get(i, 0) for i in all_instances]
    steal_vals = [steal_counts.get(i, 0) for i in all_instances]
    
    plt.bar(x - width/2, base_vals, width, label='PACE (Baseline)', color='#70A5D9')
    plt.bar(x + width/2, steal_vals, width, label='PACE_NEW (Stealing)', color='#ECA680')
    
    plt.xlabel('Instance ID', fontsize=14)
    plt.ylabel('Requests Processed', fontsize=14)
    plt.title('Load Balancing: Processed Requests per Instance', fontsize=16)
    plt.xticks(x, [f'Inst {i}' for i in all_instances])
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'load_balance_compare.pdf')
    plt.close()
    
    # 4. Per-Instance TTFT Bar Chart
    plt.figure(figsize=(8, 5))
    
    base_ttft_means = base_df.groupby('Instance Id')['prefill_e2e_time'].mean().sort_index()
    steal_ttft_means = steal_df.groupby('Instance Id')['prefill_e2e_time'].mean().sort_index()
    
    base_ttft_vals = [base_ttft_means.get(i, 0) for i in all_instances]
    steal_ttft_vals = [steal_ttft_means.get(i, 0) for i in all_instances]
    
    plt.bar(x - width/2, base_ttft_vals, width, label='PACE (Baseline)', color='#70A5D9')
    plt.bar(x + width/2, steal_ttft_vals, width, label='PACE_NEW (Stealing)', color='#ECA680')
    
    for idx, val in enumerate(base_ttft_vals):
        if val > 0:
            plt.text(x[idx] - width/2, val, f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    for idx, val in enumerate(steal_ttft_vals):
        if val > 0:
            plt.text(x[idx] + width/2, val, f'{val:.3f}', ha='center', va='bottom', fontsize=10)
            
    plt.xlabel('Instance ID', fontsize=14)
    plt.ylabel('Mean TTFT (s)', fontsize=14)
    plt.title('Mean TTFT per Instance', fontsize=16)
    plt.xticks(x, [f'Inst {i}' for i in all_instances])
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'instance_ttft_compare.pdf')
    plt.close()
    
    # Print summary statistics
    print("=== SUMMARY STATISTICS ===")
    print(f"PACE - Total Processed: {len(base_df)}")
    print(f"PACE - Mean TTFT: {base_df['prefill_e2e_time'].mean():.3f}s, Mean TBT: {base_df['decode_mean_time'].mean():.3f}s")
    if 'request_num_prefill_tokens' in base_df.columns:
        print(f"PACE - Avg Prefill Tokens: {base_df['request_num_prefill_tokens'].mean():.1f}")
    
    print(f"PACE_NEW - Total Processed: {len(steal_df)}")
    print(f"PACE_NEW - Mean TTFT: {steal_df['prefill_e2e_time'].mean():.3f}s, Mean TBT: {steal_df['decode_mean_time'].mean():.3f}s")
    if 'request_num_prefill_tokens' in steal_df.columns:
        print(f"PACE_NEW - Avg Prefill Tokens: {steal_df['request_num_prefill_tokens'].mean():.1f}")
    
    print("\nPlots saved to:", out_dir)

if __name__ == '__main__':
    main()
