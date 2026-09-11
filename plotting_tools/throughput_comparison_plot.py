import pandas as pd
import os
from matplotlib import pyplot as plt
import yaml
from pprint import pprint 
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter

def get_throughput(seq_metrics, scheduler, target):
    df = pd.read_csv(seq_metrics)
    # requested being dropped for non responsiveness
    df_aborted = df[df['request_aborted']==1]
    df = df.drop(columns = ['request_aborted','request_inter_arrival_delay'])
    df = df.dropna()
    df['throughput'] = df['request_e2e_time']/df['request_num_decode_tokens']
    num_gpus = df['Instance Id'].nunique()
    # we plot the three instance case, for roundrobin scheduler, all instance are the same
    lp1 = df[df['Instance Id']==0]
    lp2 = df[df['Instance Id']==1]
    hp1 = df[df['Instance Id']==2]
    
    if target == 'mean':
        lp1_target = lp1['throughput'].mean()
        lp2_target = lp2['throughput'].mean()
        hp1_target = hp1['throughput'].mean()
    else:
        lp1_target = lp1['throughput'].quantile(float(target))
        lp2_target = lp2['throughput'].quantile(float(target))
        hp1_target = hp1['throughput'].quantile(float(target))
    
    return [lp1_target, lp2_target, hp1_target], num_gpus

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
    
    if model == "mistral" or model == "llama":
        if dataset == "arxiv":     
            ttft_slo = 5
        elif dataset == "longbench":
            ttft_slo = 2.5
        else:
            ttft_slo = 1
            
    tbt_slo = 0.15
    roots = [
        #'/path/to/result/dir/'
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
        if scheduler == 'VLLM':
            scheduler = 'vLLM'
        if scheduler == 'SARATHI':
            scheduler = 'Sarathi'
        summarize_dict[scheduler]={}
        throughput_lists = []
        
        for idx, path in enumerate(seq_paths):
            throughput, num_gpus = get_throughput(path, scheduler, 'mean')
            throughput_lists.append(throughput)
            qps = round(qps_list[idx]/num_gpus,2)
            summarize_dict[scheduler][qps]={}
            summarize_dict[scheduler][qps]['throughput'] = throughput
            
        qps_list = [round(each/num_gpus,2) for each in qps_list]
        if 'PACE' in scheduler:
            labels = ['LP1', 'LP2', 'HP1']
        else:
            labels = ['INS 1', 'INS 2', 'INS 3']
        
        axs[i].plot(qps_list, [each[0] for each in throughput_lists], marker=markers[0], label=labels[0], color = colors[0])
        axs[i].plot(qps_list, [each[1] for each in throughput_lists], marker=markers[1], label=labels[1], color = colors[1])
        axs[i].plot(qps_list, [each[2] for each in throughput_lists], marker=markers[2], label=labels[2], color = colors[2])
        axs[i].set_title(scheduler, fontsize = 24)
        axs[i].legend(fontsize=20, loc='upper left')
        axs[i].set_ylim(0, 0.25)
        axs[i].set_xlabel("QPS", fontsize = 24)
        axs[i].tick_params(axis='both', labelsize=20)
        axs[i].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))

    axs[0].set_ylabel("Normalized latency\n (s/token)", fontsize = 22)
    
    plt.savefig(f"/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_result/images/throughput_{model}_{dataset}.pdf", bbox_inches="tight")
    
    