import pandas as pd
from scipy import stats

def evaluate_scheduler_fairness(results_csv, dataset_csv, ttft_slo, tbt_slo):
    # 1. Load the data
    print("Loading data...")
    results_df = pd.read_csv(results_csv)
    dataset_df = pd.read_csv(dataset_csv)
    
    # 2. Filter for SLO Violations (same logic as before)
    violators_df = results_df[
        (results_df['request_aborted'] != 1) & 
        ((results_df['prefill_e2e_time'] > ttft_slo) | (results_df['decode_mean_time'] > tbt_slo))
    ]
    
    print(f"Total requests in dataset: {len(dataset_df)}")
    print(f"Total SLO violations found: {len(violators_df)}\n")
    
    if len(violators_df) == 0:
        print("No violations found. Cannot run statistical test.")
        return

    # 3. Extract the distributions
    # Dataset distributions
    dataset_prefill = dataset_df['num_prefill_tokens'].dropna()
    dataset_decode = dataset_df['num_decode_tokens'].dropna()
    
    # Violator distributions
    violator_prefill = violators_df['request_num_prefill_tokens'].dropna()
    violator_decode = violators_df['request_num_decode_tokens'].dropna()

    # 4. Run the 2-sample Kolmogorov-Smirnov test
    print("-" * 40)
    print("KOLMOGOROV-SMIRNOV TEST RESULTS")
    print("-" * 40)
    
    # Test Prefill Fairness
    ks_stat_prefill, p_val_prefill = stats.ks_2samp(violator_prefill, dataset_prefill)
    print("Prefill Tokens Distribution:")
    print(f"  K-S Statistic: {ks_stat_prefill:.4f}")
    print(f"  P-value:       {p_val_prefill:.4f}")
    _interpret_ks_result(p_val_prefill, "Prefill")
    
    print("\n")
    
    # Test Decode Fairness
    ks_stat_decode, p_val_decode = stats.ks_2samp(violator_decode, dataset_decode)
    print("Decode Tokens Distribution:")
    print(f"  K-S Statistic: {ks_stat_decode:.4f}")
    print(f"  P-value:       {p_val_decode:.4f}")
    _interpret_ks_result(p_val_decode, "Decode")
    print("-" * 40)

def _interpret_ks_result(p_value, metric_name, alpha=0.05):
    """Helper function to explain what the p-value means for fairness."""
    if p_value > alpha:
        print(f"  -> Conclusion: FAIR. We cannot reject the null hypothesis.")
        print(f"     The {metric_name} distribution of SLO violators is statistically similar")
        print(f"     to the overall dataset. The scheduler does not significantly bias against specific lengths.")
    else:
        print(f"  -> Conclusion: POTENTIALLY UNFAIR. We reject the null hypothesis.")
        print(f"     The {metric_name} distribution of SLO violators differs significantly")
        print(f"     from the overall dataset. The scheduler may be disproportionately failing certain request sizes.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Replace with your actual file names and SLOs
    RESULTS_CSV = "/scratch/gilbreth/li2068/asplos/PaceServe/exp_middleware/coordinate_llama_goodput_benchmark_batchsize_128_3gpus_gpt_drop_enabled_1/2026-02-19_11-43-24-580545/replica_0/sequence_metrics.csv"  # The file with prefill_e2e_time, request_aborted, etc.
    DATASET_CSV = "/scratch/gilbreth/li2068/asplos/PaceServe/distserve_dataset/sharegpt_distserve.csv"            # The file with num_prefill_tokens, num_decode_tokens
    
    TTFT_SLO = 1.0   # Target TTFT
    TBT_SLO = 0.15   # Target TBT
    
    evaluate_scheduler_fairness(
        results_csv=RESULTS_CSV,
        dataset_csv=DATASET_CSV,
        ttft_slo=TTFT_SLO,
        tbt_slo=TBT_SLO
    )