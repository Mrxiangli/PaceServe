import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import yaml

from scheduler_style import (
    apply_plot_style,
    FONT_LABEL,
    FONT_TICK,
    FONT_LEGEND,
    FONT_TITLE,
    SCHEDULER_COLORS,
)

TBT_SLO = 0.15  # seconds

# Visual style for the two series on each panel
BAR_COLOR    = SCHEDULER_COLORS['vLLM']
TTFT_COLOR   = SCHEDULER_COLORS['Sarathi']
TBT_COLOR    = SCHEDULER_COLORS['DistServe']
TTFT_MARKER  = 'o'
TBT_MARKER   = 's'


def load_ttft_slo(qps_dir):
    config_path = os.path.join(qps_dir, 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg['pace_scheduler_config_ttft_slo']
    return None


def compute_offload_stats(qps_dir, ttft_slo):
    """Returns (num_offloaded, ttft_attainment_pct, tbt_attainment_pct) or None."""
    csv_path = os.path.join(qps_dir, 'replica_0', 'sequence_metrics.csv')
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['request_reschedule_interval', 'prefill_e2e_time'])
    offloaded = df[df['request_reschedule_interval'] != 0]

    if len(offloaded) == 0:
        return None

    num_offloaded = len(offloaded)
    ttft_pct = (offloaded['prefill_e2e_time'] <= ttft_slo).sum() / num_offloaded * 100

    tbt_valid = offloaded['decode_mean_time'].dropna()
    tbt_pct = (tbt_valid <= TBT_SLO).sum() / len(tbt_valid) * 100 if len(tbt_valid) > 0 else float('nan')

    return num_offloaded, ttft_pct, tbt_pct


def load_pace_dir(pace_dir, ttft_slo_override):
    """Walk all qps_* subdirs and return lists of (qps, num_off, ttft_pct, tbt_pct)."""
    qps_dirs = sorted(
        [d for d in os.listdir(pace_dir) if d.startswith('qps_')],
        key=lambda d: int(d.split('_')[1])
    )

    rows = []
    for qps_dir_name in qps_dirs:
        qps_val  = int(qps_dir_name.split('_')[1])
        qps_path = os.path.join(pace_dir, qps_dir_name)

        ttft_slo = ttft_slo_override if ttft_slo_override is not None else load_ttft_slo(qps_path)
        stats    = compute_offload_stats(qps_path, ttft_slo)

        if stats is not None:
            num_off, ttft_pct, tbt_pct = stats
            rows.append((qps_val, num_off, ttft_pct, tbt_pct))
            print(f'  QPS {qps_val}: {num_off} offloaded, '
                  f'TTFT {ttft_pct:.1f}%, TBT {tbt_pct:.1f}%')
        else:
            print(f'  QPS {qps_val}: no offloaded requests, skipping')

    return rows


def draw_panel(ax_bar, rows, title):
    """Draw one panel: bar chart on ax_bar (left y) + two lines on twinx (right y)."""
    ax_line = ax_bar.twinx()

    qps_vals        = [r[0] for r in rows]
    num_offloaded   = [r[1] for r in rows]
    ttft_pct_list   = [r[2] for r in rows]
    tbt_pct_list    = [r[3] for r in rows]
    x = np.arange(len(qps_vals))

    bars = ax_bar.bar(x, [n / 100 for n in num_offloaded], width=0.5,
                      color=BAR_COLOR, alpha=0.45, label='# Offloaded Requests')

    line_ttft, = ax_line.plot(x, ttft_pct_list, marker=TTFT_MARKER,
                               color=TTFT_COLOR, linewidth=2, markersize=6,
                               label='TTFT Attainment')
    line_tbt,  = ax_line.plot(x, tbt_pct_list,  marker=TBT_MARKER,
                               color=TBT_COLOR,  linewidth=2, markersize=6,
                               label='TBT Attainment')

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(qps_vals, fontsize=FONT_TICK, ha='right')
    ax_bar.set_xlabel('QPS', fontsize=FONT_LABEL)
    ax_bar.set_ylabel('# Offloaded Requests (×100)', fontsize=FONT_LABEL)
    ax_bar.tick_params(axis='y', labelsize=FONT_TICK)
    ax_bar.grid(True, linestyle=':', alpha=0.5, axis='y')

    ax_line.set_ylabel('SLO Attainment (%)', fontsize=FONT_LABEL)
    ax_line.set_ylim(0, 105)
    ax_line.tick_params(axis='y', labelsize=FONT_TICK)

    ax_bar.set_title(title, fontsize=FONT_TITLE, fontweight='bold')

    # Return artist handles so the caller can build a shared legend
    return bars, line_ttft, line_tbt


def main():
    parser = argparse.ArgumentParser(
        description='Side-by-side panels: HP-offloaded count (bar) + TTFT/TBT attainment (lines) vs QPS'
    )
    parser.add_argument('pace_dir1', type=str,
                        help='First PACE directory  (e.g. .../llama_sharegpt_qps/.../PACE)')
    parser.add_argument('pace_dir2', type=str,
                        help='Second PACE directory (e.g. .../llama_longbench_qps/.../PACE)')
    parser.add_argument('--title1', type=str, default=None,
                        help='Panel title for first dataset  (default: dirname)')
    parser.add_argument('--title2', type=str, default=None,
                        help='Panel title for second dataset (default: dirname)')
    parser.add_argument('--ttft-slo', type=float, default=None,
                        help='TTFT SLO in seconds, applied to both datasets '
                             '(default: read per-QPS from config.yaml)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PDF path '
                             '(default: exp_result/figures/offload_slo_fraction.pdf)')
    args = parser.parse_args()

    title1 = args.title1 or os.path.basename(os.path.normpath(args.pace_dir1))
    title2 = args.title2 or os.path.basename(os.path.normpath(args.pace_dir2))

    print(f'Loading dataset 1: {args.pace_dir1}')
    rows1 = load_pace_dir(args.pace_dir1, args.ttft_slo)
    print(f'Loading dataset 2: {args.pace_dir2}')
    rows2 = load_pace_dir(args.pace_dir2, args.ttft_slo)

    if not rows1 and not rows2:
        print('No data found in either dataset, exiting.')
        return

    apply_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    h_bar = h_ttft = h_tbt = None
    if rows1:
        h_bar, h_ttft, h_tbt = draw_panel(ax1, rows1, title1)
    if rows2:
        b, t, tb = draw_panel(ax2, rows2, title2)
        if h_bar is None:
            h_bar, h_ttft, h_tbt = b, t, tb

    # Shared legend placed below the two panels
    if h_bar is not None:
        fig.legend(
            handles=[h_bar, h_ttft, h_tbt],
            labels=['# Offloaded Requests', 'TTFT Attainment', 'TBT Attainment'],
            loc='upper center',
            ncol=3,
            fontsize=FONT_LEGEND,
            bbox_to_anchor=(0.5, 1.08),
            frameon=True,
        )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5)

    output_dir = os.path.join('exp_result', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    output_path = args.output or os.path.join(output_dir, 'offload_slo_fraction.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nPlot saved to {output_path}')
    plt.close()


if __name__ == '__main__':
    main()
