import pandas as pd

# 1. Map your QPS values to their corresponding CSV files
data_map = {
    # 3: "/scratch/gilbreth/li2068/middleware/PaceServe/exp_middleware/pace_llama_goodput_benchmark_batchsize_128_3gpus_gpt_sweep_9/2026-02-19_18-10-23-326323/replica_0/sequence_metrics.csv",
    # 4: "/scratch/gilbreth/li2068/middleware/PaceServe/exp_middleware/pace_llama_goodput_benchmark_batchsize_128_3gpus_gpt_sweep_12/2026-02-19_18-18-54-384493/replica_0/sequence_metrics.csv",
    # 5: "/scratch/gilbreth/li2068/middleware/PaceServe/exp_middleware/pace_llama_goodput_benchmark_batchsize_128_3gpus_gpt_sweep_15/2026-02-19_18-30-03-798658/replica_0/sequence_metrics.csv",
    5.5: "/scratch/gilbreth/li2068/middleware/PaceServe/exp_middleware/pace_llama_goodput_benchmark_batchsize_128_3gpus_gpt_sweep_165/2026-02-19_22-02-07-756948/replica_0/sequence_metrics.csv",
    6: "/scratch/gilbreth/li2068/middleware/PaceServe/exp_middleware/pace_llama_goodput_benchmark_batchsize_128_3gpus_gpt_sweep_18/2026-02-19_18-37-31-174795/replica_0/sequence_metrics.csv",
    6.5: "/scratch/gilbreth/li2068/middleware/PaceServe/exp_middleware/pace_llama_goodput_benchmark_batchsize_128_3gpus_gpt_sweep_195/2026-02-19_21-50-54-752247/replica_0/sequence_metrics.csv",
    7: "/scratch/gilbreth/li2068/middleware/PaceServe/exp_middleware/pace_llama_goodput_benchmark_batchsize_128_3gpus_gpt_sweep_21/2026-02-19_18-45-07-543967/replica_0/sequence_metrics.csv",
    7.5: "/scratch/gilbreth/li2068/middleware/PaceServe/exp_middleware/pace_llama_goodput_benchmark_batchsize_128_3gpus_gpt_sweep_225/2026-02-19_21-45-19-785504/replica_0/sequence_metrics.csv",
    8: "/scratch/gilbreth/li2068/middleware/PaceServe/exp_middleware/pace_llama_goodput_benchmark_batchsize_128_3gpus_gpt_sweep_24/2026-02-19_21-35-46-872198/replica_0/sequence_metrics.csv",
    8.5:"/scratch/gilbreth/li2068/middleware/PaceServe/exp_middleware/pace_llama_goodput_benchmark_batchsize_128_3gpus_gpt_sweep_255/2026-02-19_22-10-06-637398/replica_0/sequence_metrics.csv"
}

def analyze_qps_metrics(mapping):
    summary_data = []

    for qps, file_path in mapping.items():
        try:
            # Load the data
            df = pd.read_csv(file_path)
            
            # Clean column names (removes leading/trailing spaces)
            df.columns = df.columns.str.strip()
            
            # Count 1: How many rows have Instance Id == 2
            instance_2_count = (df['Instance Id'] == 2).sum()
            
            # Count 2: How many rows have request_offloaded_time == -1
            offload_failed_count = (df['request_offloaded_time'] == -1).sum()
            
            # Store the counts for this QPS
            summary_data.append({
                'QPS': qps,
                'Instance_2_Total': instance_2_count-offload_failed_count,
                'Offload_Total': offload_failed_count
            })
                
        except FileNotFoundError:
            print(f"Warning: File '{file_path}' for {qps} QPS not found.")
        except KeyError as e:
            print(f"Warning: Missing column in {file_path}: {e}")

    # 2. Display the final summary table
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by QPS to ensure the table flows logically
        summary_df = summary_df.sort_values('QPS')
        
        print("\n--- Summary Metrics by QPS ---")
        print(summary_df.to_string(index=False))
        
        # Optional: Save to CSV
        # summary_df.to_csv('qps_analysis_summary.csv', index=False)
    else:
        print("\nNo data was processed.")

if __name__ == "__main__":
    analyze_qps_metrics(data_map)