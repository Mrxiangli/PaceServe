import pandas as pd
import os
from matplotlib import pyplot as plt
import yaml
from pprint import pprint 
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter

def get_ttft(seq_metrics, scheduler, metric, target):
    df = pd.read_csv(seq_metrics)
    # requested being dropped for non responsiveness
    df_aborted = df[df['request_aborted']==1]
    df = df.drop(columns = ['request_aborted','request_inter_arrival_delay'])
    df = df.dropna()
    num_gpus = df['Instance Id'].nunique()
    # we plot the three instance case, for roundrobin scheduler, all instance are the same
    lp1 = df[df['Instance Id']==0]
    lp2 = df[df['Instance Id']==1]
    hp1 = df[df['Instance Id']==2]
    load_lp1 = lp1['Request Id'].nunique()
    load_lp2 = lp2['Request Id'].nunique()
    load_hp1 = hp1['Request Id'].nunique()
    if metric == "load": metric = 'prefill_e2e_time'
    if target == 'mean':
        lp1_target = lp1[metric].mean()
        lp2_target = lp2[metric].mean()
        if metric == 'request_scheduling_delay' and (scheduler == 'Pace' or scheduler == 'Ascendra'):
            hp1_target = hp1['high_priority_scheduling_delay'].mean()
        else:
            hp1_target = hp1[metric].mean()
    else:
        lp1_target = lp1[metric].quantile(float(target))
        lp2_target = lp2[metric].quantile(float(target))
        if metric == 'request_scheduling_delay' and scheduler == 'Pace':
            hp1_target = hp1['high_priority_scheduling_delay'].quantile(float(target))
        else:
            hp1_target = hp1[metric].quantile(float(target))
    
    return [lp1_target, lp2_target, hp1_target], [load_lp1, load_lp2, load_hp1], num_gpus

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
    return np.array(qps_list), seq_paths, scheduler_type

if __name__ == "__main__":
    model = 'mistral'
    dataset = 'gpt'
    metric = "request_scheduling_delay"
    
    if model == "mistral" or model == "llama":
        if dataset == "arxiv":     
            ttft_slo = 5
        elif dataset == "longbench":
            ttft_slo = 2.5
        else:
            ttft_slo = 1
            
    tbt_slo = 0.15
    roots = [
        #/path/to/result/dir
    ]
    colors = ['#70A5D9', '#ECA680', '#F0868C']
    markers = ['o', '>', '*']
    summarize_dict = {}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.set_constrained_layout(True)
    for i,root in enumerate(roots):
        qps_list, seq_paths, scheduler = get_seq_path(root)
        if scheduler == 'PACE':
            scheduler = "Pace"
        if scheduler == "VLLM":
            scheduler = 'vLLM'
        if scheduler == 'SARATHI':
            scheduler = 'Sarathi'
        summarize_dict[scheduler]={}
        ttft_lists = []
        load_lists = []
        
        for idx, path in enumerate(seq_paths):
            ttft, load, num_gpus = get_ttft(path, scheduler, metric, 'mean')
            ttft_lists.append(ttft)
            load_lists.append(load)
            qps = round(qps_list[idx]/num_gpus,2)
            summarize_dict[scheduler][qps]={}
            summarize_dict[scheduler][qps]['prefill_e2e'] = ttft
            
        qps_list = [round(each/num_gpus,2) for each in qps_list]
        if 'Ascendra' in scheduler:
            labels = ['LP1', 'LP2', 'HP1']
        else:
            labels = ['INS 1', 'INS 2', 'INS 3']
        
        axs[i].plot(qps_list, [each[0] for each in ttft_lists], marker=markers[0], label=labels[0], color = colors[0])
        axs[i].plot(qps_list, [each[1] for each in ttft_lists], marker=markers[1], label=labels[1], color = colors[1])
        axs[i].plot(qps_list, [each[2] for each in ttft_lists], marker=markers[2], label=labels[2], color = colors[2])
        # if metric == "decode_mean_time":
        #     axs[i].axhline(y=tbt_slo, linestyle='--', color='black', linewidth=1)  # Added dashed line
        # else:
        #     axs[i].axhline(y=ttft_slo, linestyle='--', color='black', linewidth=1)  # Added dashed line
        axs[i].set_title(scheduler, fontsize = 24)   
        axs[i].legend(fontsize=20, loc='upper left')
        axs[i].set_xlabel("QPS", fontsize = 24)
        axs[i].tick_params(axis='both', labelsize=20)
        axs[i].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
    if metric == "decode_mean_time":
        axs[0].set_ylabel("Mean TBT (s)", fontsize = 24)
    elif metric == "prefill_e2e_time":
        axs[0].set_ylabel("P99 TTFT (s)", fontsize = 24)
    else:
        axs[0].set_ylabel("Scheduling Delay (s)", fontsize = 24)
    plt.savefig(f"/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_result/images/{metric}_{model}_{dataset}.pdf", bbox_inches="tight")
    
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.set_constrained_layout(True)
    for i,root in enumerate(roots):
        qps_list, seq_paths, scheduler = get_seq_path(root)
        if scheduler == 'PACE':
            scheduler = "Pace"
         if scheduler == "VLLM":
            scheduler = 'vLLM'
        if scheduler == 'SARATHI':
            scheduler = 'Sarathi'

        summarize_dict[scheduler]={}
        ttft_lists = []
        load_lists = []
        width=bar_width=0.1
        for idx, path in enumerate(seq_paths):
            ttft, load, num_gpus = get_ttft(path, scheduler, metric, 'mean')
            ttft_lists.append(ttft)
            load_lists.append(load)
            qps = round(qps_list[idx]/num_gpus,2)
            summarize_dict[scheduler][qps]={}
            summarize_dict[scheduler][qps]['prefill_e2e'] = ttft
        load_arrays = np.array(load_lists).T
        qps_list = [round(each/num_gpus,2) for each in qps_list]
        x = np.arange(len(qps_list))  # Evenly spaced positions for bars
        bar_width = 0.15
        if 'PACE' in scheduler:
            labels = ['LP1', 'LP2', 'HP1']
        else:
            labels = ['INS 1', 'INS 2', 'INS 3']
        
        axs[i].bar(qps_list, load_arrays[0], label=labels[0], color=colors[0], width=bar_width)
        axs[i].bar(qps_list, load_arrays[1], bottom=load_arrays[0], label=labels[1], color=colors[1], width=bar_width)
        axs[i].bar(qps_list, load_arrays[2], bottom=load_arrays[0] + load_arrays[1], label=labels[2], color=colors[2], width=bar_width)
        
        axs[i].set_title(scheduler, fontsize = 24)
        axs[i].legend(fontsize=20, loc='upper left')
        axs[i].set_xlabel("QPS", fontsize = 24)
        axs[i].tick_params(axis='both', labelsize=20)
        axs[i].set_ylim(0, 2500)
        
        axs[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axs[i].yaxis.offsetText.set_fontsize(16)

    axs[0].set_ylabel("Load (#)", fontsize = 24)

    plt.savefig(f"/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_result/images/load_{model}_{dataset}.pdf", bbox_inches="tight")

    