import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    csv_path = 'wild/conv_distributions_processed.csv'
    df = pd.read_csv(csv_path)
    
    # Calculate QPS (queries per second)
    # The 'Time' column is in seconds.
    max_time = np.ceil(df['Time'].max())
    bins = np.arange(0, max_time + 1.0, 1.0)
    qps_counts, _ = np.histogram(df['Time'], bins=bins)
    
    # Create a figure with two subplots to cover both interpretations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Time on X-axis, QPS on Y-axis (Standard request rate plot)
    time_bins = bins[:-1]
    ax1.plot(time_bins, qps_counts, color='#4A90D9', linewidth=1.0)
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('QPS (Requests per Second)', fontsize=12)
    ax1.set_title('Request Arrival Rate Over Time', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Subplot 2: QPS on X-axis, Frequency on Y-axis (Histogram of QPS)
    ax2.hist(qps_counts, bins=30, color='#E8624A', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('QPS (Requests per Second)', fontsize=12)
    ax2.set_ylabel('Frequency (Seconds)', fontsize=12)
    ax2.set_title('Distribution of QPS (QPS on X-axis)', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    out_path = '/scratch/gilbreth/li2068/azam/PaceServe/wild/sample_qps_plot.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {out_path}")
    print(f"Stats: Mean QPS: {qps_counts.mean():.2f}, Peak QPS: {qps_counts.max()}, Total Requests: {len(df)}")

if __name__ == "__main__":
    main()
