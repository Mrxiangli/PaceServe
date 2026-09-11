import pandas as pd
import numpy as np

# 1. Define your dictionary mapping QPS to the respective CSV file
# Example: {QPS_Value: "filename.csv"}
data_files = {
    5.5: "/scratch/gilbreth/li2068/middleware/PaceServe/exp_middleware/pace_llama_goodput_benchmark_batchsize_128_3gpus_gpt_sweep_165/2026-02-19_22-02-07-756948/replica_0/batch_metrics.csv",
    6: "/scratch/gilbreth/li2068/middleware/PaceServe/exp_middleware/pace_llama_goodput_benchmark_batchsize_128_3gpus_gpt_sweep_18/2026-02-19_18-37-31-174795/replica_0/batch_metrics.csv",
    6.5: "/scratch/gilbreth/li2068/middleware/PaceServe/exp_middleware/pace_llama_goodput_benchmark_batchsize_128_3gpus_gpt_sweep_195/2026-02-19_21-50-54-752247/replica_0/batch_metrics.csv",
    7: "/scratch/gilbreth/li2068/middleware/PaceServe/exp_middleware/pace_llama_goodput_benchmark_batchsize_128_3gpus_gpt_sweep_21/2026-02-19_18-45-07-543967/replica_0/batch_metrics.csv",
    7.5:"/scratch/gilbreth/li2068/middleware/PaceServe/exp_middleware/pace_llama_goodput_benchmark_batchsize_128_3gpus_gpt_sweep_225/2026-02-19_21-45-19-785504/replica_0/batch_metrics.csv",
    8:"/scratch/gilbreth/li2068/middleware/PaceServe/exp_middleware/pace_llama_goodput_benchmark_batchsize_128_3gpus_gpt_sweep_24/2026-02-19_21-35-46-872198/replica_0/batch_metrics.csv",
    8.5:"/scratch/gilbreth/li2068/middleware/PaceServe/exp_middleware/pace_llama_goodput_benchmark_batchsize_128_3gpus_gpt_sweep_255/2026-02-19_22-10-06-637398/replica_0/batch_metrics.csv"
}

def process_kv_data(file_dict):
    all_qps_results = []

    for qps, file_path in file_dict.items():
        try:
            # Load the data
            df = pd.read_csv(file_path)
            
            # Ensure column names match your description
            # Formula: (max(batch_reamining_kv_blocks) - current) / current
            
            # Calculate max blocks per instance
            df['max_blocks'] = df.groupby('Instance Id')['batch_reamining_kv_blocks'].transform('max')
            
            # Calculate the specific usage ratio requested
            # Note: We use a small epsilon or handling for 0 to avoid division by zero errors
            df['usage'] = (df['max_blocks'] - df['batch_reamining_kv_blocks']) / df['batch_reamining_kv_blocks']
            
            # Average usage per instance for this specific QPS file
            instance_averages = df.groupby('Instance Id')['usage'].mean()
            
            # Convert to a DataFrame and add the QPS as a column for later merging
            res = instance_averages.to_frame().T
            res.index = [qps]
            all_qps_results.append(res)
            
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found. Skipping.")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Combine all QPS rows into one table
    if not all_qps_results:
        return None
        
    final_table = pd.concat(all_qps_results)
    final_table.index.name = 'qps'
    
    # Sort by QPS and reset index to make 'qps' the first column
    final_table = final_table.sort_index().reset_index()
    
    # Ensure columns are named consistently (Instance 0, 1, 2...)
    # This renames column headers from 0, 1, 2 to "Instance 0", etc.
    final_table.columns = [
        f"Instance {col}" if isinstance(col, (int, float)) else col 
        for col in final_table.columns
    ]
    
    return final_table

# Execute and display
result_df = process_kv_data(data_files)

if result_df is not None:
    print("Final KV Usage Table (Average per Instance):")
    print(result_df.to_string(index=False))
    
    # Optional: Save to CSV
    # result_df.to_csv("summary_kv_usage.csv", index=False)