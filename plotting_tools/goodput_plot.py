import pandas as pd
import os
from matplotlib import pyplot as plt
import yaml
from pprint import pprint 
import numpy as np

from experiments import MODEL, DATASET, ROOTS

def get_attainment(seq_metrics, ttft_slo, tbt_slo):
    df = pd.read_csv(seq_metrics)
    # requested being dropped for non responsiveness
    df_aborted = df[df['request_aborted']==1]
    df = df.drop(columns = ['request_aborted','request_inter_arrival_delay'])
    num_gpus = df['Instance Id'].nunique()
    request_in_sys = df['Request Id'].nunique()
    df = df.dropna()
    df_slo_attained = df[(df['prefill_e2e_time'] <= ttft_slo) & (df['decode_mean_time'] <= tbt_slo)]
    average_processed_request_len = df['request_num_prefill_tokens'].mean()
    slo_attainment = len(df_slo_attained)/request_in_sys
    total_request_processed = len(df)
    return slo_attainment*100, num_gpus, total_request_processed, len(df_aborted)*100/(len(df)+len(df_aborted)), average_processed_request_len, df['request_num_prefill_tokens'].std()

def get_qps(subdir):
    config_yaml = os.path.join(subdir,'config.yaml')
    with open(config_yaml, "r") as file:
        config = yaml.safe_load(file)
    qps = config.get("poisson_request_interval_generator_config_qps", [])
    scheduler_type = config.get("scheduler_config_type", [])
    seq_path = os.path.join(subdir,'replica_0/sequence_metrics.csv')
    return qps, seq_path, scheduler_type

def get_seq_path(root_dir):
    qps_list = []
    seq_paths = []
    for directory in os.listdir(root_dir):
        subdir = os.path.join(root_dir, directory)
        qps, seq_path, scheduler_type = get_qps(subdir)
        if os.path.exists(seq_path):
            qps_list.append(qps)
            seq_paths.append(seq_path)
    qps_list, seq_paths = zip(*sorted(zip(qps_list, seq_paths)))
    return np.array(qps_list), seq_paths, scheduler_type

if __name__ == "__main__":
    if MODEL == "mistral" or MODEL == "llama":
        if DATASET == "arxiv":     
            ttft_slo = 5
        elif DATASET == "longbench":
            ttft_slo = 2.5
        else:
            ttft_slo = 1
    else:
        if DATASET == "arxiv":     
            ttft_slo = 5
        elif DATASET == "longbench":
            ttft_slo = 3
        else:
            ttft_slo = 1.5
            
    tbt_slo = 0.15
    
    colors = ['#70A5D9', '#ECA680', '#F0868C']
    markers = ['o', '>', '*']
    summarize_dict = {}
    plt.figure(figsize=(6,4))
    for i,root in enumerate(ROOTS):
        qps_list, seq_paths, scheduler = get_seq_path(root)
        if scheduler == 'VLLM':
            scheduler = "vLLM"
        if scheduler == 'SARATHI':
            scheduler = "Sarathi"
        if scheduler == 'COOR':
            scheduler = "Ascendra"
        if scheduler == 'COOR_2H1L':
            scheduler = "Ascendra-2H1L"
        if scheduler == 'COOR_NB':
            scheduler = "Ascendra"
        if scheduler == 'COOR_NP':
            scheduler = "PACE_NP"
        if scheduler == 'COOR_TOP4':
            scheduler = "PACE_TOP4"
        if scheduler == 'COOR_TOP3':
            scheduler = "PACE_TOP3"
        if scheduler == 'COOR_TOP2':
            scheduler = "PACE_TOP2"
        if scheduler == 'COOR_SJF':
            scheduler = "Ascendra-SJF"
        if scheduler == 'COOR_BASIC':
            scheduler = "PACE_BASIC"
        summarize_dict[scheduler]={}
        slo_attainments = []
        for idx, path in enumerate(seq_paths):
            slo_attaintment, num_gpus, total_request_processed, request_aborted, average_req_len, std = get_attainment(path, ttft_slo, tbt_slo)
            slo_attainments.append(slo_attaintment)
            qps = round(qps_list[idx]/num_gpus,2)
            summarize_dict[scheduler][qps]={}
            summarize_dict[scheduler][qps]['request_processed'] = total_request_processed
            summarize_dict[scheduler][qps]['request_aborted'] = request_aborted
            summarize_dict[scheduler][qps]['slo_attainment'] = slo_attaintment
            summarize_dict[scheduler][qps]['average_req_len'] = average_req_len
            summarize_dict[scheduler][qps]['req_len_std'] = std
            
        qps_list = [round(each/num_gpus,2) for each in qps_list]
        plt.plot(qps_list, slo_attainments, marker=markers[i], label=scheduler, color = colors[i])
        for idx in range(1, len(slo_attainments)):
            prev_qps, curr_qps = qps_list[idx-1], qps_list[idx]
            prev_slo, curr_slo = slo_attainments[idx-1], slo_attainments[idx]
            
            # Check if 90% lies between the two points
            if (prev_slo > 90 > curr_slo) or (prev_slo < 90 < curr_slo):
                # Interpolation to find exact QPS where SLO = 90
                slope = (curr_slo - prev_slo) / (curr_qps - prev_qps)
                qps_at_90 = prev_qps + (90 - prev_slo) / slope
                plt.ylim(0,105)
                # Plotting the vertical line at the interpolated point
                plt.plot([qps_at_90, qps_at_90], [0, 90], linestyle='--', color=colors[i], linewidth=1)
                break
    plt.axhline(y=90, linestyle='--', color='black', linewidth=1)  # Added dashed line
    plt.legend(fontsize=16, loc='lower left')
    plt.xlabel("QPS", fontsize = 20)
    plt.ylabel("SLO attainment (%)", fontsize = 20)
    plt.title("Qwen-14B", fontsize = 20)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.savefig(f"/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_result/images/goodput_{MODEL}_{DATASET}.pdf", bbox_inches="tight")
    pprint(summarize_dict)
    
    plt.figure(figsize=(6,4))
    for i, scheduler in enumerate(summarize_dict.keys()):
        tmp = []
        for qps in qps_list:
            val2 = summarize_dict[scheduler][qps]['request_aborted']
            tmp.append(val2)
        plt.plot(qps_list, tmp, marker=markers[i], label=scheduler, color = colors[i])
    plt.axhline(y=5, linestyle='--', color='black', linewidth=1)  # Added dashed line
    plt.legend(fontsize=16)
    plt.xlabel("QPS", fontsize = 20)
    plt.ylabel("Request aborted (%)", fontsize = 20)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.savefig(f"/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_result/images/request_aborted_{MODEL}_{DATASET}.pdf", bbox_inches="tight")
    
    
    prompt_lens = []
    request_processed = []
    stds = []
    for qps in qps_list:
        tmp = []
        tmp_req = []
        tmp_std = []
        for scheduler in summarize_dict.keys():
            val1 = summarize_dict[scheduler][qps]['average_req_len']
            val2 = summarize_dict[scheduler][qps]['request_processed']
            std = summarize_dict[scheduler][qps]['req_len_std']
            tmp.append(val1)
            tmp_req.append(val2)
            tmp_std.append(std)
        prompt_lens.append(tmp)
        request_processed.append(tmp_req)
        stds.append(tmp_std)
        
    labels = [scheduler for scheduler in summarize_dict.keys()]
    num_groups = len(qps_list)
    num_bars_per_group = len(prompt_lens[0])  # Assuming each QPS has the same number of prompt_lens

    # Bar width and positions
    bar_width = 0.22
    x = np.arange(num_groups)  # X-axis positions for groups
    
    colors = ['#70A5D9', '#ECA680', '#F0868C']
    
    plt.figure(figsize=(4,4))
    # Plot each set of bars
    for i in range(num_bars_per_group):
        plt.bar(x + i * bar_width, [prompt_lens[j][i] for j in range(num_groups)], width=bar_width, edgecolor='black', label=f'{labels[i]}', color=colors[i])

    # Labels and title
    plt.xlabel("QPS",fontsize=14)
    plt.ylabel("Mean prompt length",  fontsize=14)
    plt.xticks(x + bar_width, qps_list, fontsize=12)  # Adjust x-ticks to center on groups
    plt.yticks(fontsize=12)
    plt.legend(loc='lower left', fontsize=13)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.gca().yaxis.get_offset_text().set_size(12)
    plt.savefig(f"/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_result/images/mean_prompt_len_{MODEL}_{DATASET}.pdf", bbox_inches="tight")
    
    plt.figure(figsize=(4,4))
    # Plot each set of bars
    for i in range(num_bars_per_group):
        plt.bar(x + i * bar_width, [request_processed[j][i] for j in range(num_groups)], width=bar_width, edgecolor='black', label=f'{labels[i]}', color=colors[i])

    # Labels and title
    plt.xlabel("QPS",fontsize=14)
    plt.ylabel("Number of processed request", fontsize=14)
    plt.xticks(x + bar_width, qps_list,fontsize=12)  # Adjust x-ticks to center on groups
    plt.yticks(fontsize=12)
    plt.legend(loc='lower left', fontsize=13)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.gca().yaxis.get_offset_text().set_size(12)
    plt.savefig(f"/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_result/images/num_request_processed_{MODEL}_{DATASET}.pdf", bbox_inches="tight")
    