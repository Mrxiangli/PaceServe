import pandas as pd
import os
from matplotlib import pyplot as plt
import yaml
from pprint import pprint 
import numpy as np
from matplotlib.ticker import ScalarFormatter

def get_drop(seq_metrics, ttft_slo, tbt_slo, instance):
    df = pd.read_csv(seq_metrics)
    if instance != 'all':
        df = df[df['Instance Id']==instance]
    request_dropped = df[df['request_aborted']==1]
    request_not_dropped = df[df['request_aborted']!=1]
    request_offloaded = request_not_dropped[request_not_dropped['request_offloaded_time'].notna()]
    request_not_offloaded = request_not_dropped[request_not_dropped['request_offloaded_time'].isna()]
    request_offloaded_executing = request_offloaded[request_offloaded['request_execution_time'].isna()]
    request_offloaded_finished = request_offloaded[request_offloaded['request_execution_time'].notna()]
    request_offloaded_slo_attained = request_offloaded_finished[(request_offloaded_finished['prefill_e2e_time'] <= ttft_slo) & (request_offloaded_finished['decode_mean_time'] <= tbt_slo)]
    request_offloaded_slo_violated = request_offloaded_finished[(request_offloaded_finished['prefill_e2e_time'] > ttft_slo) | (request_offloaded_finished['decode_mean_time'] > tbt_slo)]
    
    request_not_offloaded_filtered = request_not_offloaded[~request_not_offloaded['Request Id'].isin(request_offloaded['Request Id'])]
    request_not_offloaded_filtered_executing = request_not_offloaded_filtered[request_not_offloaded_filtered['request_execution_time'].isna()]
    request_not_offloaded_filtered_finished = request_not_offloaded_filtered[request_not_offloaded_filtered['request_execution_time'].notna()]
    request_not_offloaded_filtered_slo_attained = request_not_offloaded_filtered_finished[(request_not_offloaded_filtered_finished['prefill_e2e_time'] <= ttft_slo) & (request_not_offloaded_filtered_finished['decode_mean_time'] <= tbt_slo)]
    request_not_offloaded_filtered_slo_violated = request_not_offloaded_filtered_finished[(request_not_offloaded_filtered_finished['prefill_e2e_time'] > ttft_slo) | (request_not_offloaded_filtered_finished['decode_mean_time'] > tbt_slo)]
    print(f"request_dropped: {len(request_dropped)}")
    print(f"request_offloaded: {len(request_offloaded)}")
    print(f"request_not_offloaded: {len(request_not_offloaded_filtered)}")
    print(f"request_offloaded_executing: {len(request_offloaded_executing)}")
    print(f"request_offloaded_attained: {len(request_offloaded_slo_attained)}")
    print(f"request_offloaded_violated: {len(request_offloaded_slo_violated)}")
    print(f"request_not_offloaded_executing: {len(request_not_offloaded_filtered_executing)}")
    print(f"request_not_offloaded_attained: {len(request_not_offloaded_filtered_slo_attained)}")
    print(f"request_not_offloaded_violated: {len(request_not_offloaded_filtered_slo_violated )}")
    
    return len(request_not_offloaded_filtered_slo_attained)+len(request_offloaded_slo_attained),len(request_dropped), len(request_not_offloaded_filtered_slo_violated)+len(request_offloaded_slo_violated)

def get_seq_path(root_dir):
    qps_list = []
    seq_paths = []
    for directory in os.listdir(root_dir):
        subdir = os.path.join(root_dir, directory)
        config_yaml = os.path.join(subdir,'config.yaml')
        with open(config_yaml, "r") as file:
            config = yaml.safe_load(file)
        qps = config.get("poisson_request_interval_generator_config_qps", [])
        scheduler_type = config.get("scheduler_config_type", [])
        seq_path = os.path.join(subdir,'replica_0/sequence_metrics.csv')
        if os.path.exists(seq_path):
            qps_list.append(qps)
            seq_paths.append(seq_path)
    qps_list, seq_paths = zip(*sorted(zip(qps_list, seq_paths)))
    return qps_list, seq_paths, scheduler_type

if __name__ == "__main__":
    ttft_slo = 1.0
    tbt_slo = 0.15
    model = 'llama'
    dataset = 'gpt'
    instance = 2
    roots = [
        '/scratch/gilbreth/li2068/asplos/PaceServe/exp_result/drop_test/Pace_2L1H/',
        #'/scratch/gilbreth/li2068/asplos/PaceServe/exp_result/drop_test/Pace_3L/',
    ]   
    
    colors = ['#70A5D9', '#ECA680', '#F0868C', '#d3d3d3']
    markers = ['o', '>', '*']
    summarize_dict = {}
    fig, ax = plt.subplots(1, 1, figsize=(11, 3))

    bar_width = 0.23
    x_base = np.arange(len(roots))  # One x position per scheduler
    print(x_base)

    labels = []
    tmp1 = []
    tmp2 = []
    tmp3 = []
    for i, root in enumerate(roots):
        qps_list, seq_paths, scheduler = get_seq_path(root)

        scheduler_map = {'SARATHI': 'Sarathi', 'VLLM': 'vLLM', 'Pace-2L1H': 'Pace-2L1H', 'Pace-3L': 'Pace-3L'}
        scheduler = scheduler_map.get(scheduler, scheduler)
        labels.append(scheduler)

        slo_attainments = []
        slo_relax_attainments = []
        request_drop = []
        slo_violates  =[]
        for path in seq_paths:
            slo_attained, request_dropped, slo_violated = get_drop(path, ttft_slo, tbt_slo, instance)
            slo_attainments.append(slo_attained)
            request_drop.append(request_dropped)
            slo_violates.append(slo_violated)

        slo = np.mean(slo_attainments)
        drop = np.mean(request_drop)
        violate = np.mean(slo_violates)

        ax.bar(x_base[i] - 1*bar_width, slo, width=bar_width, color=colors[0], edgecolor='black', label='SLO Attained' if i == 0 else "")
        ax.bar(x_base[i], drop, width=bar_width, color=colors[1], edgecolor='black' , label='Requests Dropped' if i == 0 else "")
        ax.bar(x_base[i] + 1*bar_width, violate, width=bar_width, color=colors[2], edgecolor='black',  label='Requests Violate SLO' if i == 0 else "")
        tmp1.append(slo)
        tmp2.append(drop)
        tmp3.append(violate)
    
    # for x, slo_val, drop_val, violate_val in zip(x_base, tmp1, tmp2, tmp3):
    #     ax.text(x - 1*bar_width, slo_val,     f'{round(slo_val)}', ha='center', va='bottom', fontsize=18,)
    #     ax.text(x,               drop_val,    f'{round(drop_val)}', ha='center', va='bottom', fontsize=18,)
    #     ax.text(x + 1*bar_width, violate_val, f'{round(violate_val)}', ha='center', va='bottom', fontsize=18)
        
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.set_xticks(x_base)
    ax.set_xticklabels(labels, fontsize=20)
    ax.tick_params(axis='y', labelsize=15)
    ax.yaxis.offsetText.set_fontsize(15) 
    ax.set_ylabel('# Requests', fontsize=20)
    ax.set_ylim(0, 11000)
    fig.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=4, frameon=True, fontsize=18)
    ax.grid(True, axis='y', linestyle='--')
    plt.savefig(f"/scratch/gilbreth/li2068/asplos/PaceServe/exp_result/images/drop_{model}_{dataset}.pdf", bbox_inches="tight")
    