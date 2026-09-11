import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker

def plot_two_histograms_with_bin_width(df: pd.DataFrame, column1: str, column2: str, bin_width: float, 
                                       xlabel: str, ylabel: str, legend: tuple, name:str):
    plt.figure(figsize=(4, 4))

    # Calculate bin edges based on the specified bin width
    data_min = min(df[column1].min(), df[column2].min())
    data_max = max(df[column1].max(), df[column2].max())
    bins = np.arange(data_min, data_max + bin_width, bin_width)

    print(f"Prefill mean: {np.mean(df[column1])}, Decode mean: {np.mean(df[column2])}, "
        f"Prefill std: {np.std(df[column1])}, Decode std: {np.std(df[column2])}")

    return df[column1].dropna(), df[column2].dropna(), bins, round(np.mean(df[column1]), 1), round(np.mean(df[column2]), 1)


df_gpt = pd.read_csv('/scratch/gilbreth/li2068/asplos/sarathi-serve/distserve_dataset/sharegpt_distserve.csv')
gpt_prefill, gpt_decode, gpt_bins, gpt_pref_mean, gpt_decode_mean = plot_two_histograms_with_bin_width(df_gpt, 'num_prefill_tokens', 'num_decode_tokens', 100, '# Tokens', 'Density', (r"Prompt ($\mu$=1905.0)", r"Output ($\mu$=438.7)"),'sharegpt')
df_bwb = pd.read_csv('/scratch/gilbreth/li2068/asplos/sarathi-serve/distserve_dataset/long_bench_filtered.csv')
bwb_prefill, bwb_decode, bwb_bins, bwb_pref_mean, bwb_decode_mean = plot_two_histograms_with_bin_width(df_bwb, 'num_prefill_tokens', 'num_decode_tokens', 100, '# Tokens', 'Density', (r"Prompt ($\mu$=768.2)", r"Output ($\mu$=195.9)"),'longbench')
# df_ax = pd.read_csv('/scratch/gilbreth/li2068/asplos/sarathi-serve/tracedata/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv')
# ax_prefill, ax_decode, ax_bins, ax_pref_mean, ax_decode_mean = plot_two_histograms_with_bin_width(df_ax, 'num_prefill_tokens', 'num_decode_tokens', 100, '# Tokens', 'Density', (r"Prompt ($\mu$=2340.0)", r"Output ($\mu$=438.7)"),'sharegpt')

fig, axes = plt.subplots(1, 2, figsize=(8, 3))

# Function to format axis with scientific notation
def set_scientific(ax):
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  # Only show sci notation for numbers <10^-3 or >10^3
    # ax.xaxis.set_major_formatter(formatter)
    # ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# Plot ShareGPT
axes[0].hist(gpt_prefill, bins=gpt_bins, alpha=0.6, color='blue', label=rf"Prompt ($\mu={gpt_pref_mean}$)")
axes[0].hist(gpt_decode, bins=gpt_bins, alpha=0.6, color='orange', label=rf"Output ($\mu={gpt_decode_mean}$)")
axes[0].set_xlabel("# Tokens", fontsize=18)
axes[0].set_ylabel("Density", fontsize=18)
axes[0].legend(fontsize=12)

axes[0].tick_params(axis='both', which='major', labelsize=14)
axes[0].xaxis.offsetText.set_fontsize(13)  # Increase size of scientific notation
axes[0].yaxis.offsetText.set_fontsize(13)
set_scientific(axes[0])  # Apply scientific notation

# Plot BWB
axes[1].hist(bwb_prefill, bins=bwb_bins, alpha=0.6, color='blue', label=rf"Prompt ($\mu={bwb_pref_mean}$)")
axes[1].hist(bwb_decode, bins=bwb_bins, alpha=0.6, color='orange', label=rf"Output ($\mu={bwb_decode_mean}$)")
axes[1].set_xlabel("# Tokens", fontsize=18)
axes[1].legend(fontsize=12)

axes[1].tick_params(axis='both', which='major', labelsize=14)
axes[1].xaxis.offsetText.set_fontsize(13)  # Increase size of scientific notation
axes[1].yaxis.offsetText.set_fontsize(13)
set_scientific(axes[1])  # Apply scientific notation

# Plot ArXiv
# axes[2].hist(ax_prefill, bins=ax_bins, alpha=0.6, color='blue', label=rf"Prompt ($\mu={ax_pref_mean}$)")
# axes[2].hist(ax_decode, bins=ax_bins, alpha=0.6, color='orange', label=rf"Output ($\mu={ax_decode_mean}$)")
# axes[2].set_xlabel("# Tokens", fontsize=18)
# axes[2].legend(fontsize=12)
# set_scientific(axes[2]) 
# axes[2].tick_params(axis='both', which='major', labelsize=14)
# axes[2].xaxis.offsetText.set_fontsize(13)  # Increase size of scientific notation
# axes[2].yaxis.offsetText.set_fontsize(13)
# plt.grid()
plt.savefig(f'data_analysis.pdf',bbox_inches='tight')