import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import yaml
from scipy.stats import ks_2samp

from scheduler_style import (
    apply_plot_style,
    FONT_LABEL,
    FONT_TICK,
    FONT_LEGEND,
    FONT_TITLE,
    SCHEDULER_COLORS,
)


ALL_COLOR   = SCHEDULER_COLORS['vLLM']
TTFT_COLOR  = SCHEDULER_COLORS['Sarathi']
TBT_COLOR   = SCHEDULER_COLORS['DistServe']


def load_config(exp_path):
    with open(os.path.join(exp_path, 'config.yaml')) as f:
        return yaml.safe_load(f)


def load_seq_df(exp_path):
    df = pd.read_csv(os.path.join(exp_path, 'replica_0', 'sequence_metrics.csv'))
    return df.drop_duplicates(subset='Request Id', keep='last')


def plot_cdf(ax, data, label, color, linestyle='-'):
    sorted_data = np.sort(data.dropna())
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return ax.plot(sorted_data, cdf, label=label, color=color, linestyle=linestyle, linewidth=2)[0]


def draw_panel(ax, df, ttft_slo, tbt_slo, token_col, xlabel, title):
    ttft_viol = df[df['prefill_e2e_time'] > ttft_slo]
    tbt_viol  = df[df['decode_mean_time']  > tbt_slo]

    h1 = plot_cdf(ax, df[token_col],           'All requests',                   color=ALL_COLOR,  linestyle='-')
    h2 = plot_cdf(ax, ttft_viol[token_col],    f'TTFT violators', color=TTFT_COLOR, linestyle='--')
    h3 = plot_cdf(ax, tbt_viol[token_col],     f'TBT violators',   color=TBT_COLOR,  linestyle='-.')

    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel('CDF', fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    ax.grid(True, linestyle=':', alpha=0.5)

    all_tokens  = df[token_col].dropna()
    ttft_tokens = ttft_viol[token_col].dropna()
    tbt_tokens  = tbt_viol[token_col].dropna()

    ks_ttft = ks_2samp(all_tokens, ttft_tokens) if len(ttft_tokens) > 0 else None
    ks_tbt  = ks_2samp(all_tokens, tbt_tokens)  if len(tbt_tokens)  > 0 else None

    print(f'\n{title}:')
    print(f'  Total requests: {len(df)}')
    print(f'  TTFT violators: {len(ttft_viol)} ({100*len(ttft_viol)/len(df):.1f}%)', end='')
    if ks_ttft:
        print(f'  |  KS stat={ks_ttft.statistic:.3f}  p={ks_ttft.pvalue:.3e}', end='')
    print()
    print(f'  TBT  violators: {len(tbt_viol)}  ({100*len(tbt_viol)/len(df):.1f}%)', end='')
    if ks_tbt:
        print(f'  |  KS stat={ks_tbt.statistic:.3f}  p={ks_tbt.pvalue:.3e}', end='')
    print()

    return h1, h2, h3


def main():
    parser = argparse.ArgumentParser(
        description='CDF overlay of all requests vs SLO-violating requests for prefill/decode tokens.'
    )
    parser.add_argument('prefill_path', type=str,
                        help='Experiment directory for the prefill (left) panel')
    parser.add_argument('decode_path', type=str,
                        help='Experiment directory for the decode (right) panel')
    parser.add_argument('--title1', type=str, default=None,
                        help='Title for the prefill panel (default: dirname)')
    parser.add_argument('--title2', type=str, default=None,
                        help='Title for the decode panel (default: dirname)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output PDF path (default: exp_result/figures/slo_cdf_overlay.pdf)')
    args = parser.parse_args()

    title1 = args.title1 or os.path.basename(os.path.normpath(args.prefill_path))
    title2 = args.title2 or os.path.basename(os.path.normpath(args.decode_path))

    prefill_cfg = load_config(args.prefill_path)
    decode_cfg  = load_config(args.decode_path)

    prefill_df = load_seq_df(args.prefill_path)
    decode_df  = load_seq_df(args.decode_path)

    apply_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    h1, h2, h3 = draw_panel(
        ax1, prefill_df,
        ttft_slo=float(prefill_cfg['pace_scheduler_config_ttft_slo']),
        tbt_slo=float(prefill_cfg['pace_scheduler_config_tbt_slo']),
        token_col='request_num_prefill_tokens',
        xlabel='Prefill tokens',
        title=title1,
    )
    _, _, _ = draw_panel(
        ax2, decode_df,
        ttft_slo=float(decode_cfg['pace_scheduler_config_ttft_slo']),
        tbt_slo=float(decode_cfg['pace_scheduler_config_tbt_slo']),
        token_col='request_num_decode_tokens',
        xlabel='Decode tokens',
        title=title2,
    )
    ax2.set_ylabel('')

    fig.legend(
        handles=[h1, h2, h3],
        loc='upper center',
        ncol=3,
        fontsize=FONT_LEGEND,
        bbox_to_anchor=(0.5, 1.08),
        frameon=True,
    )

    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.20)

    output_dir = os.path.join('exp_result', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    output_path = args.output or os.path.join(output_dir, 'fairness_plot.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'\nPlot saved to {output_path}')
    plt.close()


if __name__ == '__main__':
    main()
