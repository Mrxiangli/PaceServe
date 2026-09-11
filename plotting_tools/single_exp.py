from pathlib import Path
from new_goodput_plotting import get_attainment, get_qps

if __name__ == '__main__':
    TTFT = 1.0
    TBT = 0.15
    
    # root = Path('/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_llama8B_goodput_benchmark_batchsize_128_3gpus_gpt_ttft_1/2025-04-04_07-24-13-497076/')

    # New
    root = Path('exp_result/2026-03-17_23-29-08-228642')


    # 1. Get configuration
    qps, seq_path, scheduler_type = get_qps(root)
    
    # 2. Get metrics (This now returns the AttainmentMetrics dataclass)
    metrics = get_attainment(seq_path, TTFT, TBT)

    # 3. Calculate Normalized QPS (with a safety check for 0 GPUs)
    norm_qps = round(qps / metrics.num_gpus, 2) if metrics.num_gpus > 0 else 0.0

    # 4. Print structured, readable output
    print("\n" + "="*40)
    print(f" SUB-EXPERIMENT QUICK CHECK")
    print("="*40)
    print(f" Scheduler : {scheduler_type}")
    print(f" QPS/GPU   : {norm_qps}")
    print(f" SLO Met   : {metrics.slo_attainment_pct:.2f}%")
    print(f" Processed : {metrics.total_processed}")
    print(f" Aborted   : {metrics.aborted_pct:.2f}%")
    print(f" Avg Req   : {metrics.avg_req_len:.1f} tokens (±{metrics.std_req_len:.1f})")
    print("="*40 + "\n")
