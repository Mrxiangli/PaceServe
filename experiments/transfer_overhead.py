import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
from scheduler_style import apply_plot_style


def load_and_filter(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['request_num_tokens', 'request_reschedule_interval'])
    if 'Instance Id' in df.columns:
        df = df[(df['Instance Id'] == 2) & (df['request_reschedule_interval'] != 0)]
    else:
        df = df[df['request_reschedule_interval'] != 0]
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Plot transfer overhead scatter and CDF from sequence_metrics CSVs."
    )
    parser.add_argument(
        'paths', nargs='+',
        help="One or more paths to sequence_metrics.csv files."
    )
    parser.add_argument(
        '--legends', nargs='+', metavar='LEGEND',
        help="Display label for each path (must match the number of paths)."
    )
    parser.add_argument(
        '--scatter_idx', type=int, default=0,
        help="Index of the path to use for the scatter plot (default: 0)."
    )
    parser.add_argument(
        '--output_dir', type=str, default=os.path.join('exp_result', 'figures'),
        help="Directory to save output figures (default: exp_result/figures)."
    )
    args = parser.parse_args()

    # Validate legends
    legends = args.legends if args.legends else args.paths
    if len(legends) != len(args.paths):
        parser.error(f"--legends count ({len(legends)}) must match number of paths ({len(args.paths)}).")

    if not (0 <= args.scatter_idx < len(args.paths)):
        parser.error(f"--scatter_idx {args.scatter_idx} is out of range for {len(args.paths)} path(s).")

    apply_plot_style()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Scatter plot (single dataset) ---
    scatter_path = args.paths[args.scatter_idx]
    df_filtered = load_and_filter(scatter_path)

    print(f"Scatter source: {scatter_path}")
    print(f"  Rows after filtering: {len(df_filtered)}")
    if 'Instance Id' in df_filtered.columns:
        print(f"  Unique Instance Ids: {df_filtered['Instance Id'].unique()}")

    if len(df_filtered) > 0:
        fig, ax = plt.subplots(figsize=(5, 4))
        x = df_filtered['request_num_tokens'].values
        y = df_filtered['request_reschedule_interval'].values * 1000  # convert to ms
        n_bins = 40
        x_bins = np.linspace(x.min(), x.max(), n_bins + 1)
        y_bins = np.logspace(np.log10(y.min()), np.log10(y.max()), n_bins + 1)
        counts, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = np.sqrt(y_edges[:-1] * y_edges[1:])  # geometric mean for log-spaced bins
        xi, yi = np.where(counts > 0)
        sizes = counts[xi, yi]
        sc = ax.scatter(x_centers[xi], y_centers[yi],
                        s=20 + 300 * sizes / sizes.max(),
                        c=sizes, cmap='viridis', alpha=0.7, edgecolors='none')
        fig.colorbar(sc, ax=ax, label='Count')
        ax.set_yscale('log')
        ax.set_xlabel('Prompt Length (tokens)')
        ax.set_ylabel('Transfer Overhead (ms)')
        ax.grid(True, linestyle='--', alpha=0.7)

        scatter_output = os.path.join(args.output_dir, 'reschedule_interval_scatter.pdf')
        plt.savefig(scatter_output, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to {scatter_output}")
        plt.close()
    else:
        print("No data remaining after filtering, skipping scatter plot.")

    # --- CDF plot (all datasets) ---
    plt.figure(figsize=(3.5, 2.5))
    for path, label in zip(args.paths, legends):
        df = load_and_filter(path)
        if len(df) == 0:
            print(f"No data for {label}, skipping.")
            continue
        data = (df['request_reschedule_interval'] * 1000).sort_values()
        mean_val = data.mean()
        std_val  = data.std()
        p50_val  = data.quantile(0.50)
        p90_val  = data.quantile(0.90)
        p99_val  = data.quantile(0.99)
        print(f"{label} — Mean: {mean_val:.4f} ms, Std: {std_val:.4f} ms, "
              f"P50: {p50_val:.4f} ms, P90: {p90_val:.4f} ms, P99: {p99_val:.4f} ms")
        cdf = np.arange(1, len(data) + 1) / len(data)
        plt.plot(data, cdf, label=label)

    plt.xscale('log')
    plt.xlabel('Transfer Overhead (ms)')
    plt.ylabel('CDF')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    cdf_output = os.path.join(args.output_dir, 'reschedule_interval_cdf.pdf')
    plt.savefig(cdf_output, dpi=300, bbox_inches='tight')
    print(f"CDF plot saved to {cdf_output}")
    plt.close()


if __name__ == '__main__':
    main()
