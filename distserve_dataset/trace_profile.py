import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_two_histograms_with_bin_width(df: pd.DataFrame, column1: str, column2: str, bin_width: float, 
                                       xlabel: str, ylabel: str, legend: tuple, name:str):
    plt.figure(figsize=(6, 4))
    
    # Calculate bin edges based on the specified bin width
    data_min = min(df[column1].min(), df[column2].min())
    data_max = max(df[column1].max(), df[column2].max())
    bins = np.arange(data_min, data_max + bin_width, bin_width)
    
    print(f"prefill mean: {np.mean(df[column1])}, decode mean: {np.mean(df[column2])}, prefill std: {np.std(df[column1])}, decode std: {np.std(df[column2])}")
    
    # Plotting the histograms with the specified bins
    plt.hist(df[column1].dropna(), bins=bins, alpha=0.5, color='blue', label=legend[0])
    plt.hist(df[column2].dropna(), bins=bins, alpha=0.5, color='orange',label=legend[1])
    
    plt.xlabel(xlabel,fontsize=18)
    plt.ylabel(ylabel,fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig(f'{name}.pdf',bbox_inches='tight')

df_gpt = pd.read_csv('/scratch/gilbreth/li2068/asplos/sarathi-serve/distserve_dataset/sharegpt_distserve.csv')
plot_two_histograms_with_bin_width(df_gpt, 'num_prefill_tokens', 'num_decode_tokens', 50, '# tokens', 'Density', ('input', 'output'),'long_bench')

