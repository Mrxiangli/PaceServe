import pandas as pd
import os
from matplotlib import pyplot as plt
import yaml
from pprint import pprint 
import numpy as np

def get_attainment(seq_metrics, ttft_slo, tbt_slo, scale):
    df = pd.read_csv(seq_metrics)
    ttft_slo = ttft_slo * scale
    tbt_slo = tbt_slo * scale
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

def get_seq_path(root_dir):
    seq_paths = []
    scales_list = []
    for directory in os.listdir(root_dir):
        subdir = os.path.join(root_dir, directory)
        config_yaml = os.path.join(subdir,'config.yaml')
        with open(config_yaml, "r") as file:
            config = yaml.safe_load(file)
        scale  =  config.get("ttft_slo", [])
        scheduler_type = config.get("scheduler_config_type", [])
        seq_path = os.path.join(subdir,'replica_0/sequence_metrics.csv')
        if os.path.exists(seq_path):
            seq_paths.append(seq_path)
            scales_list.append(scale)
    scales_list, seq_paths = zip(*sorted(zip(scales_list, seq_paths)))
    scales_list = list(scales_list)
    seq_paths = list(seq_paths)
    scales_list.reverse()
    seq_paths.reverse()
    return np.array(scales_list), seq_paths, scheduler_type

if __name__ == "__main__":
    model = 'qwen'
    dataset = 'longbench'
    
    if model == "mistral" or model == "llama":
        if dataset == "arxiv":     
            ttft_slo = 5
        elif dataset == "longbench":
            ttft_slo = 2.5
        else:
            ttft_slo = 1
    else:
        if dataset == "arxiv":     
            ttft_slo = 5
        elif dataset == "longbench":
            ttft_slo = 3.0
        else:
            ttft_slo = 1.5
            
    tbt_slo = 0.15
    roots = [
        #'/path/to/result/dir/'
    ]
    colors = ['#70A5D9', '#ECA680', '#F0868C']
    markers = ['o', '>', '*']
    summarize_dict = {}
    plt.figure(figsize=(6,4))
    for i,root in enumerate(roots):
        scales_list, seq_paths, scheduler = get_seq_path(root)
        if scheduler == 'VLLM':
            scheduler = "vLLM"
        if scheduler == 'SARATHI':
            scheduler = "Sarathi"
        if scheduler == 'PACE':
            scheduler = "Pace"
        summarize_dict[scheduler]={}
        slo_attainments = []
        scales = []
        for idx, path in enumerate(seq_paths):
            
            slo_attaintment, num_gpus, total_request_processed, request_aborted, average_req_len, std = get_attainment(path, ttft_slo, tbt_slo, scales_list[idx])
            if scales_list[idx] not in summarize_dict[scheduler].keys():
                summarize_dict[scheduler][scales_list[idx]]={}
            if 'slo_attainment' not in summarize_dict[scheduler][scales_list[idx]].keys():
                summarize_dict[scheduler][scales_list[idx]]['slo_attainment'] = [slo_attaintment]
            else:
                summarize_dict[scheduler][scales_list[idx]]['slo_attainment'].append(slo_attaintment)
        
        for each in scales_list:
            slo_attainments.append(sum(summarize_dict[scheduler][each]['slo_attainment'])/len(summarize_dict[scheduler][each]['slo_attainment']))
            
        plt.plot(scales_list, slo_attainments, marker=markers[i], label=scheduler, color = colors[i])
        for idx in range(1, len(slo_attainments)):
            prev_qps, curr_qps = scales_list[idx-1], scales_list[idx]
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
    plt.gca().invert_xaxis()
    plt.legend(fontsize=16, loc='lower left')
    plt.xlabel("SLO Scale", fontsize = 20)
    plt.ylabel("SLO attainment (%)", fontsize = 20)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylim(0, 100)
    plt.savefig(f"/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_result/images/goodput_{model}_{dataset}_scale.pdf", bbox_inches="tight")
    pprint(summarize_dict)
    