import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── helpers ────────────────────────────────────────────────────────────────────

def load_and_bin(csv_path: str, bin_size: float = 1.0):
    """Return (bin_times, counts) for a CSV that has a TIMESTAMP column."""
    df = pd.read_csv(csv_path)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    df = df.sort_values('TIMESTAMP').reset_index(drop=True)
    t0 = df['TIMESTAMP'].min()
    time_sec = (df['TIMESTAMP'] - t0).dt.total_seconds()
    bins = np.arange(0, time_sec.max() + bin_size, bin_size)
    counts, edges = np.histogram(time_sec.values, bins=bins)
    bin_centers_sec = (edges[:-1] + edges[1:]) / 2
    bin_minutes = bin_centers_sec / 60.0   # elapsed minutes from 0
    return bin_minutes, counts, df, t0

# ── load both datasets ─────────────────────────────────────────────────────────

CODE_CSV = '/scratch/gilbreth/li2068/azam/PaceServe/code_distributions_original.csv'
CONV_CSV = '/scratch/gilbreth/li2068/azam/PaceServe/conv_distributions_original.csv'

code_times, code_counts, code_df, code_t0 = load_and_bin(CODE_CSV)
conv_times, conv_counts, conv_df, conv_t0 = load_and_bin(CONV_CSV)

for label, df, counts in [('code', code_df, code_counts), ('conv', conv_df, conv_counts)]:
    duration_s = (df['TIMESTAMP'].max() - df['TIMESTAMP'].min()).total_seconds()
    print(f"[{label}] rows={len(df)}  duration={duration_s:.1f}s  "
          f"mean={counts.mean():.1f} req/s  peak={counts.max()} req/s")

# ── plotting ───────────────────────────────────────────────────────────────────

CODE_COLOR = '#4A90D9'   # blue
CONV_COLOR = '#E8624A'   # orange-red

fig, ax = plt.subplots(figsize=(14, 5))

# Code distribution line
ax.fill_between(code_times, code_counts, alpha=0.15, color=CODE_COLOR)
ax.plot(code_times, code_counts, color=CODE_COLOR, linewidth=1.4,
        label='Code (req/s)')

# Conv distribution line
ax.fill_between(conv_times, conv_counts, alpha=0.15, color=CONV_COLOR)
ax.plot(conv_times, conv_counts, color=CONV_COLOR, linewidth=1.4,
        label='Conv (req/s)')

# Mean reference lines
for counts, color, times in [
    (code_counts, CODE_COLOR, code_times),
    (conv_counts, CONV_COLOR, conv_times),
]:
    mean_rate = counts.mean()
    ax.axhline(mean_rate, color=color, linewidth=1.0, linestyle=':', alpha=0.7)
    ax.text(times[-1], mean_rate + 0.3,
            f'  mean={mean_rate:.1f}', color=color, fontsize=8,
            va='bottom', ha='right', clip_on=True)

# Stats annotation (both datasets)
stats_lines = []
for label, df, counts in [('Code', code_df, code_counts), ('Conv', conv_df, conv_counts)]:
    dur = (df['TIMESTAMP'].max() - df['TIMESTAMP'].min()).total_seconds()
    stats_lines.append(
        f"{label}: mean={counts.mean():.1f} req/s  peak={counts.max():.0f}  "
        f"total={len(df):,}  dur={dur:.0f}s"
    )
ax.text(0.01, 0.97, '\n'.join(stats_lines), transform=ax.transAxes,
        fontsize=8.5, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  alpha=0.85, edgecolor='lightgray'))

# Axes styling
ax.set_xlabel('Time (minutes)', fontsize=13, labelpad=8)
ax.set_ylabel('Requests per Second', fontsize=13, labelpad=8)
ax.set_title('Request Rate over Time', fontsize=15, fontweight='bold', pad=14)

ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
plt.xticks(rotation=30, ha='right')

ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='gray')
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(fontsize=11, framealpha=0.9)

plt.tight_layout()
out_path = '/scratch/gilbreth/li2068/azam/PaceServe/request_rate.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {out_path}")
