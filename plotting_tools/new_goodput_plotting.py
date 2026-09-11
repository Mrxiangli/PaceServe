import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List
from pprint import pprint

from experiments import MODEL, DATASET, ROOTS

# --- Configuration & Constants ---
COLORS = ['#70A5D9', '#ECA680', '#F0868C', '#88B04B', '#9B59B6', '#F39C12']
MARKERS = ['o', '>', '*', 's', 'D', '^']

SCHEDULER_MAP = {
    'VLLM': 'vLLM',
    'SARATHI': 'Sarathi',
    'COOR': 'Ascendra',
    'COOR_2H1L': 'Ascendra-2H1L',
    'COOR_NB': 'Ascendra',
    'COOR_NP': 'PACE_NP',
    'COOR_TOP4': 'PACE_TOP4',
    'COOR_TOP3': 'PACE_TOP3',
    'COOR_TOP2': 'PACE_TOP2',
    'COOR_SJF': 'Ascendra-SJF',
    'COOR_BASIC': 'PACE_BASIC'
}

# --- Data Structures ---
@dataclass
class AttainmentMetrics:
    slo_attainment_pct: float
    num_gpus: int
    total_processed: int
    aborted_pct: float
    avg_req_len: float
    std_req_len: float

# --- Core Functions ---
def get_attainment(seq_metrics: Path, ttft_slo: float, tbt_slo: float) -> AttainmentMetrics:
    df = pd.read_csv(seq_metrics)
    
    if df.empty:
        return AttainmentMetrics(0.0, 0, 0, 0.0, 0.0, 0.0)

    num_aborted = (df['request_aborted'] == 1).sum()
    request_in_sys = df['Request Id'].nunique()
    
    df_processed = df.dropna(subset=['prefill_e2e_time', 'decode_mean_time', 'request_num_prefill_tokens'])
    total_processed = len(df_processed)
    num_gpus = df_processed['Instance Id'].nunique() if total_processed > 0 else 0

    slo_attainment_pct = 0.0
    if request_in_sys > 0:
        slo_met_mask = (df_processed['prefill_e2e_time'] <= ttft_slo) & (df_processed['decode_mean_time'] <= tbt_slo)
        slo_attainment_pct = (slo_met_mask.sum() / request_in_sys) * 100

    aborted_pct = 0.0
    total_attempts = total_processed + num_aborted
    if total_attempts > 0:
        aborted_pct = (num_aborted / total_attempts) * 100

    avg_req_len = df_processed['request_num_prefill_tokens'].mean()
    std_req_len = df_processed['request_num_prefill_tokens'].std()

    return AttainmentMetrics(
        slo_attainment_pct=float(slo_attainment_pct),
        num_gpus=int(num_gpus),
        total_processed=int(total_processed),
        aborted_pct=float(aborted_pct),
        avg_req_len=float(0.0 if pd.isna(avg_req_len) else avg_req_len),
        std_req_len=float(0.0 if pd.isna(std_req_len) else std_req_len)
    )

def get_qps(subdir: Path) -> Tuple[float, Path, str]:
    config_yaml = subdir / 'config.yaml'
    
    with open(config_yaml, "r") as file:
        config = yaml.safe_load(file)
    
    qps = config.get("poisson_request_interval_generator_config_qps", 0.0)
    scheduler_type = config.get("scheduler_config_type", "unknown")
    seq_path = subdir / 'replica_0' / 'sequence_metrics.csv'
    
    return qps, seq_path, scheduler_type

def get_seq_paths(root_dir: Path) -> Tuple[np.ndarray, List[Path], str]:
    qps_list = []
    seq_paths = []
    final_scheduler_type = "unknown"
    
    for subdir in root_dir.iterdir():
        if not subdir.is_dir():
            continue
            
        qps, seq_path, scheduler_type = get_qps(subdir)
        
        if seq_path.exists():
            qps_list.append(qps)
            seq_paths.append(seq_path)
            final_scheduler_type = scheduler_type
            
    if qps_list:
        qps_list, seq_paths = zip(*sorted(zip(qps_list, seq_paths)))
        
    return np.array(qps_list), list(seq_paths), final_scheduler_type

def get_slo_thresholds(model: str, dataset: str) -> tuple[float, float]:
    tbt_slo = 0.15
    if model in ["mistral", "llama"]:
        if dataset == "arxiv": return 5.0, tbt_slo
        if dataset == "longbench": return 2.5, tbt_slo
        return 1.0, tbt_slo
    else:
        if dataset == "arxiv": return 5.0, tbt_slo
        if dataset == "longbench": return 3.0, tbt_slo
        return 1.5, tbt_slo

def interpolate_90_percent(qps_list: list[float], slo_list: list[float]) -> float | None:
    for idx in range(1, len(slo_list)):
        prev_qps, curr_qps = qps_list[idx-1], qps_list[idx]
        prev_slo, curr_slo = slo_list[idx-1], slo_list[idx]
        
        if (prev_slo > 90 > curr_slo) or (prev_slo < 90 < curr_slo):
            slope = (curr_slo - prev_slo) / (curr_qps - prev_qps)
            if slope != 0: 
                return prev_qps + (90 - prev_slo) / slope
    return None

# --- Plotting Functions ---
def plot_goodput(summarize_dict: dict, qps_list: list[float], model: str, dataset: str, save_dir: Path):
    plt.figure(figsize=(6, 4))
    
    for i, (scheduler, metrics) in enumerate(summarize_dict.items()):
        slos = [metrics.get(q, {}).get('slo_attainment', 0) for q in qps_list]
        plt.plot(qps_list, slos, marker=MARKERS[i % len(MARKERS)], 
                 label=scheduler, color=COLORS[i % len(COLORS)])
        
        qps_at_90 = interpolate_90_percent(qps_list, slos)
        if qps_at_90 is not None:
             plt.plot([qps_at_90, qps_at_90], [0, 90], linestyle='--', 
                      color=COLORS[i % len(COLORS)], linewidth=1)

    plt.axhline(y=90, linestyle='--', color='black', linewidth=1)
    plt.ylim(0, 105)
    plt.legend(fontsize=16, loc='lower left')
    plt.xlabel("QPS", fontsize=20)
    plt.ylabel("SLO attainment (%)", fontsize=20)
    plt.title(f"{model.capitalize()}", fontsize=20)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
    plt.savefig(save_dir / f"goodput_{model}_{dataset}.pdf", bbox_inches="tight")
    plt.close()

def plot_bar_chart(summarize_dict: dict, qps_list: list[float], metric_key: str, 
                   ylabel: str, filename: str, save_dir: Path):
    labels = list(summarize_dict.keys())
    num_groups = len(qps_list)
    num_bars_per_group = len(labels)
    bar_width = 0.22
    x = np.arange(num_groups)

    plt.figure(figsize=(4, 4))
    
    for i, scheduler in enumerate(labels):
        values = [summarize_dict[scheduler].get(q, {}).get(metric_key, 0) for q in qps_list]
        plt.bar(x + i * bar_width, values, width=bar_width, edgecolor='black', 
                label=scheduler, color=COLORS[i % len(COLORS)])

    plt.xlabel("QPS", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    tick_offset = (num_bars_per_group - 1) * bar_width / 2
    plt.xticks(x + tick_offset, qps_list, fontsize=12) 
    plt.yticks(fontsize=12)
    plt.legend(loc='lower left', fontsize=13)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.gca().yaxis.get_offset_text().set_size(12)
    
    plt.savefig(save_dir / filename, bbox_inches="tight")
    plt.close()


# --- Main Execution ---
if __name__ == "__main__":
    # Settings imported from experiments.py
    model = MODEL
    dataset = DATASET
    
    save_dir = Path(f"exp_result/images/")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    ttft_slo, tbt_slo = get_slo_thresholds(model, dataset)
    
    summarize_dict = {}
    master_qps_set = set()

    if not ROOTS:
        print("No paths provided in experiments.py. Exiting.")
        exit(0)

    # 1. Data Collection Phase
    for root_str in ROOTS:
        root_path = Path(root_str)
        
        # Safety check: skip if the directory was deleted or typed incorrectly
        if not root_path.exists():
            print(f"Warning: Skipping {root_str} (Directory not found)")
            continue

        raw_qps_list, seq_paths, raw_scheduler = get_seq_paths(root_path)
        
        if not seq_paths:
            continue
            
        scheduler = SCHEDULER_MAP.get(raw_scheduler, raw_scheduler)
        
        if scheduler not in summarize_dict:
            summarize_dict[scheduler] = {}
            
        for idx, path in enumerate(seq_paths):
            metrics = get_attainment(path, ttft_slo, tbt_slo)
            
            if metrics.num_gpus > 0:
                norm_qps = round(raw_qps_list[idx] / metrics.num_gpus, 2)
                master_qps_set.add(norm_qps)
                
                summarize_dict[scheduler][norm_qps] = {
                    'request_processed': metrics.total_processed,
                    'request_aborted': metrics.aborted_pct,
                    'slo_attainment': metrics.slo_attainment_pct,
                    'average_req_len': metrics.avg_req_len,
                    'req_len_std': metrics.std_req_len
                }

    print("\n--- Summary Dictionary ---")
    pprint(summarize_dict)
    print("--------------------------\n")

    # 2. Plotting Phase
    sorted_qps_list = sorted(list(master_qps_set))
    
    if sorted_qps_list and summarize_dict:
        # Generate Goodput Line Plot
        plot_goodput(summarize_dict, sorted_qps_list, model, dataset, save_dir)
        
        # Generate Aborted Line Plot
        plt.figure(figsize=(6, 4))
        for i, scheduler in enumerate(summarize_dict.keys()):
            aborted_rates = [summarize_dict[scheduler].get(q, {}).get('request_aborted', 0) for q in sorted_qps_list]
            plt.plot(sorted_qps_list, aborted_rates, marker=MARKERS[i % len(MARKERS)], 
                     label=scheduler, color=COLORS[i % len(COLORS)])
        plt.axhline(y=5, linestyle='--', color='black', linewidth=1)
        plt.legend(fontsize=16)
        plt.xlabel("QPS", fontsize=20)
        plt.ylabel("Request aborted (%)", fontsize=20)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.savefig(save_dir / f"request_aborted_{model}_{dataset}.pdf", bbox_inches="tight")
        plt.close()

        # Generate Bar Charts
        plot_bar_chart(summarize_dict, sorted_qps_list, 'average_req_len', 
                       "Mean prompt length", f"mean_prompt_len_{model}_{dataset}.pdf", save_dir)
                       
        plot_bar_chart(summarize_dict, sorted_qps_list, 'request_processed', 
                       "Number of processed requests", f"num_request_processed_{model}_{dataset}.pdf", save_dir)
                       
        print(f"Success! All plots saved to: {save_dir}")
    else:
        print("Warning: No valid data found to plot. Check your 'ROOTS' list and directory structure.")
