import numpy as np
import pandas as pd
import os
import ast
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats
from sklearn.metrics import r2_score
from scipy.stats import zscore
from datetime import datetime


training_dir = '/scratch/gilbreth/li2068/asplos/sarathi-serve/sarathi/ai_utils/qwen_tp2_training'
RESULTS_DIR = '/scratch/gilbreth/li2068/asplos/sarathi-serve/sarathi/ai_utils/regression_weights/qwen-14b-tp2'

out_dir = f'{RESULTS_DIR}/{datetime.now().strftime("%Y-%m-%d-%H:%M")}'
os.makedirs(out_dir, exist_ok=True)


# LLAMA3-13B parameter
# h = 4096
# s = 128
# m = 14336
# b = 128
# n = 32
# l = 32

# Mistral-7B parameter
# h = 4096
# s = 128
# m = 14336
# b = 128
# n = 32
# l = 32

# Qwen-14B parameter
h = 5120
s = 128
m = 27392
b = 128
n = 40
l = 40


GPU_MEM_BANDWIDTH = 2 * 10**12
GPU_FLOPS = 312 * 10**12

def compute_prefill_attn_memory(row):
    prefill_attn_memory = 0
    l_list = row['batch_prefill_token_list']
    c_list = row['batch_chunk_token_list']
    for p_hat, c in zip(l_list, c_list):
        prefill_attn_memory += 2 * p_hat * s + 3 * c * s * (p_hat / b) 
        
    return prefill_attn_memory

def compute_prefill_gemm_memory(row):
    prefill_gemm_memory = 0
    l_list = row['batch_prefill_token_list']
    c_list = row['batch_chunk_token_list']
    for p_hat, c in zip(l_list, c_list):
        prefill_gemm_memory +=  8 * p_hat * h + 2 * p_hat * m
    if c_list:
        prefill_gemm_memory += 4 * h**2 + 2 * h * m
        
    return prefill_gemm_memory

def compute_prefill_attn_flops(row):
    prefill_attn_flops = 0
    l_list = row['batch_prefill_token_list']
    c_list = row['batch_chunk_token_list']
    b_d = row['batch_num_decode_tokens']
    for p_hat, c in zip(l_list, c_list):
        prefill_attn_flops += 2 * s * p_hat * c 
         
    return prefill_attn_flops

def compute_prefill_gemm_flops(row):
    c_list = row['batch_chunk_token_list']
    
    prefill_gemm = 2 * sum(c_list) * h * m + 4 * sum(c_list) * h**2 
        
    return prefill_gemm

def compute_decode_attn_memory(row):
    decode_attn_memory = 0
    l_hat_list = row['batch_processed_token_list']
    b_d = row['batch_num_decode_tokens']
    c_list = row['batch_chunk_token_list']
    decode_attn_memory += 2 * sum(l_hat_list) * s + 2 * s 
    
    return decode_attn_memory

def compute_decode_gemm_memory(row):
    decode_gemm_memory = 0
    l_hat_list = row['batch_processed_token_list']
    b_d = row['batch_num_decode_tokens']
    c_list = row['batch_chunk_token_list']
    decode_gemm_memory += 8 * b_d * h + 2 * b_d * m
    decode_gemm_memory += 4 * h**2 + 2 * h * m
    
    return decode_gemm_memory
        
def compute_decode_attn_flops(row):
    decode_attn_flops = 0
    l_hat_list = row['batch_processed_token_list']
    for p in l_hat_list:
        decode_attn_flops += 2 * p * s
    
    return decode_attn_flops

def compute_decode_gemm_flops(row):
    decode_gemm_flops = 0
    b_d = row['batch_num_decode_tokens']
    decode_gemm_flops += 4 * b_d * h**2 + 2 * b_d * h * m
    
    return decode_gemm_flops

def generate_model(model_type, out_path):
    print(f'Working for model type {model_type}...')
    csv_files = [f for f in os.listdir(training_dir) if f.endswith('.csv')]
    df_list = [pd.read_csv(os.path.join(training_dir, file)) for file in csv_files]
    df = pd.concat(df_list, ignore_index=True)

    if model_type == "mix":
        df = df[(df['batch_num_decode_tokens'] > 0) & (df['batch_num_prefill_tokens'] > 0)]
    elif model_type == "decode":
        df = df[df['batch_num_prefill_tokens'] == 0]
    else:
        df = df[df['batch_num_decode_tokens'] == 0]

    selected_columns = ['batch_num_decode_tokens', 'batch_num_prefill_tokens','batch_prefill_token_list', 'batch_chunk_token_list', 'batch_processed_token_list', 'batch_execution_time']
    df = df[selected_columns]

    for col in ['batch_prefill_token_list', 'batch_chunk_token_list', 'batch_processed_token_list']:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    df['prefill_attn_memory'] = df.apply(compute_prefill_attn_memory, axis=1)
    df['prefill_gemm_memory'] = df.apply(compute_prefill_gemm_memory, axis=1)
    df['prefill_attn_flops'] = df.apply(compute_prefill_attn_flops, axis=1)  
    df['prefill_gemm_flops'] = df.apply(compute_prefill_gemm_flops, axis=1)  
    df['decode_attn_memory'] = df.apply(compute_decode_attn_memory, axis=1) 
    df['decode_gemm_memory'] = df.apply(compute_decode_gemm_memory, axis=1)   
    df['decode_attn_flops'] = df.apply(compute_decode_attn_flops, axis=1)
    df['decode_gemm_flops'] = df.apply(compute_decode_gemm_flops, axis=1)

    df['total_memory'] = (df['prefill_attn_memory']*n + df['prefill_gemm_memory'] + df['decode_attn_memory']*n + df['decode_gemm_memory']) * l
    df['total_flops'] = (df['prefill_attn_flops']*n + df['prefill_gemm_flops'] + df['decode_attn_flops']*n + df['decode_gemm_flops']) * l
    df['memory_frac'] = df['total_memory']/GPU_MEM_BANDWIDTH
    df['flop_frac'] = df['total_flops']/GPU_FLOPS

    df['max_mem_flop'] = df[['memory_frac', 'flop_frac']].max(axis=1)
    df['sum_mem_flop'] = df['memory_frac'] + df['flop_frac']

    X = df[['max_mem_flop', 'sum_mem_flop','flop_frac', 'memory_frac']]

    y = df['batch_execution_time'] 
    z_scores = np.abs(stats.zscore(np.column_stack((X, y))))
    threshold = 3
    filtered = (z_scores < threshold).all(axis=1)
    X = X[filtered]
    y = y[filtered]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)  # 30% for validation+test
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 15% each

    model = LinearRegression()
    model.fit(X_train.values, y_train.values)

    print("Intercept:", model.intercept_)
    print("Coefficients:", model.coef_)

    y_pred_valid = model.predict(X_valid.values)
    y_pred_test = model.predict(X_test.values)

    # Compute RMSE
    rmse_valid = np.sqrt(mean_squared_error(y_valid.values, y_pred_valid))
    rmse_test = np.sqrt(mean_squared_error(y_test.values, y_pred_test))
    r2_valid = r2_score(y_valid.values, y_pred_valid)
    r2_test = r2_score(y_test.values, y_pred_test)

    print("Validation RMSE:", rmse_valid)
    print("Test RMSE:", rmse_test)
    print(f"R² valid Score: {r2_valid:.4f}")
    print(f"R² test Score: {r2_test:.4f}")

    plt.figure(figsize=(4, 4))
    sns.scatterplot(x=y_test.values, y=y_pred_test, color="blue", alpha=1, label="Predictions", s=30)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Perfect Fit")

    plt.xlabel("Actual Execution Time(s)",fontsize=17)
    plt.ylabel("Predicted Execution Time(s)",fontsize=17)
    plt.legend(fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.text(
        0.1, 0.95,  # Position (relative to axis)
        f"R² = {r2_test:.3f}\nRMSE = {rmse_test:.3f}",
        fontsize=13,
        transform=plt.gca().transAxes,  # Ensure text stays in the plot area
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')  # Add background for readability
    )
    plt.savefig(f'{out_dir}/pred_vs_actual_{model_type}.pdf', bbox_inches='tight')
    joblib.dump(model, f'{out_dir}/Qwen_14B_batch_regression_{model_type}.pkl')

for x in ['prefill','decode']:
    generate_model(x, out_dir)
