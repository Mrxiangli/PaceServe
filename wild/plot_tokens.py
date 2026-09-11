import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import gaussian_kde

# ── load ───────────────────────────────────────────────────────────────────────

code_df = pd.read_csv('/scratch/gilbreth/li2068/azam/PaceServe/code_distributions_original.csv')
conv_df = pd.read_csv('/scratch/gilbreth/li2068/azam/PaceServe/conv_distributions_original.csv')

# Normalise column names to a common schema
code_df = code_df.rename(columns={
    'num_prefill_tokens': 'context_tokens',
    'num_decode_tokens':  'generated_tokens',
})
conv_df = conv_df.rename(columns={
    'ContextTokens':   'context_tokens',
    'GeneratedTokens': 'generated_tokens',
})

datasets = [
    ('Code',  code_df, '#4A90D9', '///'),
    ('Conv',  conv_df, '#E8624A', ''),
]

# ── plotting ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Token Distribution: Code vs Conv', fontsize=15, fontweight='bold', y=1.01)

col_info = [
    ('context_tokens',   'Context Tokens (prefill)',  axes[0]),
    ('generated_tokens', 'Generated Tokens (decode)', axes[1]),
]

for col, xlabel, ax in col_info:
    all_vals = np.concatenate([df[col].dropna().values for _, df, _, _ in datasets])
    lo, hi = np.percentile(all_vals, 0.5), np.percentile(all_vals, 99.5)
    bins = np.linspace(lo, hi, 60)

    for label, df, color, hatch in datasets:
        vals = df[col].dropna().values
        vals_clipped = vals[(vals >= lo) & (vals <= hi)]

        ax.hist(vals_clipped, bins=bins, density=True,
                alpha=0.35, color=color, hatch=hatch, edgecolor=color,
                linewidth=0.6, label=label)

        if len(vals_clipped) > 1:
            kde = gaussian_kde(vals_clipped, bw_method='scott')
            x_kde = np.linspace(lo, hi, 400)
            ax.plot(x_kde, kde(x_kde), color=color, linewidth=2.0)

    # Median dashed vertical lines
    for label, df, color, _ in datasets:
        med = np.median(df[col].dropna())
        ax.axvline(med, color=color, linewidth=1.3, linestyle='--', alpha=0.8)

    ax.set_xlabel(xlabel, fontsize=12, labelpad=8)
    ax.set_ylabel('Density', fontsize=12, labelpad=8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.legend(fontsize=10, framealpha=0.9)

    stats_lines = []
    for label, df, _, _ in datasets:
        v = df[col].dropna()
        stats_lines.append(
            f"{label}: mean={v.mean():.0f}  median={v.median():.0f}"
            f"  p95={v.quantile(0.95):.0f}  n={len(v):,}"
        )
    ax.text(0.99, 0.97, '\n'.join(stats_lines), transform=ax.transAxes,
            fontsize=8, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.85, edgecolor='lightgray'))

plt.tight_layout()
out_path = '/scratch/gilbreth/li2068/azam/PaceServe/token_distributions.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {out_path}")
