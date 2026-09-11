import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import ast

import numpy as np

def get_latest_exp_dir(repo_root):
    base_dir = repo_root / "analytical_model_veirification" / "llama_sharegpt_qps"
    # Find all timestamp directories
    subdirs = glob.glob(os.path.join(base_dir, "20*"))
    if not subdirs:
        raise FileNotFoundError(f"No experiment directories found in {base_dir}")
    # Sort by name (timestamp) and pick the latest
    latest_dir = sorted(subdirs)[-1]
    return latest_dir

def remove_outliers(df, x_col, y_col):
    if len(df) == 0:
        return df
    Q1_x = df[x_col].quantile(0.25)
    Q3_x = df[x_col].quantile(0.75)
    IQR_x = Q3_x - Q1_x
    
    Q1_y = df[y_col].quantile(0.25)
    Q3_y = df[y_col].quantile(0.75)
    IQR_y = Q3_y - Q1_y
    
    mask = (
        (df[x_col] >= Q1_x - 1.5 * IQR_x) & (df[x_col] <= Q3_x + 1.5 * IQR_x) &
        (df[y_col] >= Q1_y - 1.5 * IQR_y) & (df[y_col] <= Q3_y + 1.5 * IQR_y)
    )
    return df[mask]

def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return 0.0, 0.0
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return rmse, r2

def load_all_metrics(base_path):
    all_files = list(Path(base_path).glob("qps_*/replica_0/batch_metrics.csv"))
    dfs = []
    for f in all_files:
        df = pd.read_csv(f)
        dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

def plot_estimation():
    repo_root = Path(__file__).resolve().parent.parent
    try:
        latest_dir = get_latest_exp_dir(repo_root)
    except FileNotFoundError as e:
        print(e)
        return
        
    print(f"Using experiment directory: {latest_dir}")
    
    vllm_path = Path(latest_dir) / "VLLM"
    sarathi_path = Path(latest_dir) / "SARATHI"
    
    df_vllm = load_all_metrics(vllm_path)
    df_sarathi = load_all_metrics(sarathi_path)
    
    if len(df_vllm) == 0:
        print(f"Warning: no VLLM data found in {vllm_path}.")
        return
    if len(df_sarathi) == 0:
        print(f"Warning: no SARATHI data found in {sarathi_path}.")
        return
    
    # Pre-process columns
    for df in [df_vllm, df_sarathi]:
        if "batch_prefill_token_list" in df.columns:
            # Safely evaluate strings to lists if they are strings
            df["batch_prefill_token_list"] = df["batch_prefill_token_list"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
    
    # VLLM filtering: ONLY prefill requests (no decode tokens, >0 prefill tokens)
    df_vllm_filtered = df_vllm[(df_vllm["batch_num_decode_tokens"] == 0) & (df_vllm["batch_num_prefill_tokens"] > 0)]
    
    # SARATHI filtering: MIXED requests (>0 decode tokens, >0 prefill tokens)
    df_sarathi_filtered = df_sarathi[(df_sarathi["batch_num_decode_tokens"] > 0) & (df_sarathi["batch_num_prefill_tokens"] > 0)]
    
    x_column = "batch_execution_time"
    y_column = "batch_estimated_runtime"
    
    # Remove outliers
    df_vllm_filtered = remove_outliers(df_vllm_filtered, x_column, y_column)
    df_sarathi_filtered = remove_outliers(df_sarathi_filtered, x_column, y_column)
    
    # Create 1x2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left subplot: VLLM (Pure Prefill)
    axes[0].scatter(df_vllm_filtered[x_column], df_vllm_filtered[y_column], alpha=0.6, edgecolors="k", color="blue")
    axes[0].set_xlabel("Actual Execution Time (s)")
    axes[0].set_ylabel("Estimated Runtime (s)")
    axes[0].set_title(f"VLLM: Pure Prefill Estimation\n(n={len(df_vllm_filtered)})")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Add metrics
    if len(df_vllm_filtered) > 0:
        rmse, r2 = calculate_metrics(df_vllm_filtered[x_column], df_vllm_filtered[y_column])
        textstr = f"RMSE: {rmse:.4f}\n$R^2$: {r2:.4f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        axes[0].text(0.05, 0.95, textstr, transform=axes[0].transAxes, fontsize=11,
                     verticalalignment='top', bbox=props)
        
        max_val_0 = max(df_vllm_filtered[x_column].max(), df_vllm_filtered[y_column].max())
        axes[0].plot([0, max_val_0], [0, max_val_0], 'r--', alpha=0.8, label="y=x (Ideal)")
        axes[0].legend(loc='lower right')
    
    # Right subplot: SARATHI (Mixed)
    axes[1].scatter(df_sarathi_filtered[x_column], df_sarathi_filtered[y_column], alpha=0.6, edgecolors="k", color="green")
    axes[1].set_xlabel("Actual Execution Time (s)")
    axes[1].set_ylabel("Estimated Runtime (s)")
    axes[1].set_title(f"SARATHI: Mixed Prefill+Decode Estimation\n(n={len(df_sarathi_filtered)})")
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Add metrics
    if len(df_sarathi_filtered) > 0:
        rmse, r2 = calculate_metrics(df_sarathi_filtered[x_column], df_sarathi_filtered[y_column])
        textstr = f"RMSE: {rmse:.4f}\n$R^2$: {r2:.4f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        axes[1].text(0.05, 0.95, textstr, transform=axes[1].transAxes, fontsize=11,
                     verticalalignment='top', bbox=props)
                     
        max_val_1 = max(df_sarathi_filtered[x_column].max(), df_sarathi_filtered[y_column].max())
        axes[1].plot([0, max_val_1], [0, max_val_1], 'r--', alpha=0.8, label="y=x (Ideal)")
        axes[1].legend(loc='lower right')
    
    plt.tight_layout()
    
    out_dir = repo_root / "analytical_model_veirification" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "estimation_scatter_all_qps.png"
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    plot_estimation()