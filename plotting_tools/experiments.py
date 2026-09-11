# --- Global Experiment Settings ---
MODEL = 'llama' # Options: 'mistral', 'llama', 'qwen'
DATASET = 'gpt' # Options: 'longbench', 'gpt', 'arxiv'

# --- Paths for the Current Figure ---
# Simply comment out the paths you want to exclude from the plot
ROOTS = [
        #NOTE 1: effect of batchsize on long bench dataset
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_longbench_ttft_2.5', 
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_longbench_ttft_2.5_nobatch',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/vllm_mistral_goodput_benchmark_batchsize_128_3gpus_longbench_ttft_2.5',
        
        #NOTE 2: effect of batchsize on sharegpt dataset
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/vllm_mistral_goodput_benchmark_batchsize_128_3gpus_gpt_ttft_1',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_gpt_ttft_1',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_gpt_ttft_1_final',
        
        #NOTE 3: effect of fraction on long bench dataset
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/vllm_mistral_goodput_benchmark_batchsize_128_3gpus_longbench_ttft_2.5',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_longbench_ttft_2.5_nobatch',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_longbench_ttft_2.5_nobatch_non_preempt'
        
        #NOTE 4: effect of fraction on gpt dataset
        #'/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/vllm_mistral_goodput_benchmark_batchsize_128_3gpus_gpt_ttft_1',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_gpt_ttft_1_final',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_gpt_ttft_1_nobatch_no_prempt',
        
        #NOTE 5: effect of top-k choice on gpt dataset
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_gpt_ttft_1_nobatch_no_prempt',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_gpt_ttft_1_nobatch_no_prempt_top4',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_gpt_ttft_1_nobatch_no_prempt_top3',

        #NOTE 6: effect of top-k choice on longbench
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_longbench_ttft_2.5_nobatch_non_preempt',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_longbench_ttft_2.5_nobatch_non_preempt_top3',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_longbench_ttft_2.5_nobatch_non_preempt_top2',
       
        #NOTE 7: effect of value function choice on longbench
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_longbench_ttft_2.5_nobatch_non_preempt',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_longbench_ttft_2.5_nobatch_non_preempt_sjf',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_longbench_ttft_2.5_nobatch_non_preempt_basic',

        #NOTE 8: effect of value function choice on gpt
        #'/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_gpt_ttft_1_final',
        #'/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_gps_ttft_1_nobatch_non_preempt_basic',
        #'/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_gps_ttft_1_nobatch_non_preempt_sjf',

        #NOTE 9: goodput comparison on GPT mistral-7B
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/vllm_mistral_goodput_benchmark_batchsize_128_3gpus_gpt_ttft_1',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/sarathi_mistral_goodput_benchmark_batchsize_128_3gpus_gps_ttft_1',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_gpt_ttft_1_final',
        
        #NOTE 10: goodput comparison on longbench mistral-7B
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/vllm_mistral_goodput_benchmark_batchsize_128_3gpus_longbench_ttft_2.5',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/sarathi_mistral_goodput_benchmark_batchsize_128_3gpus_longbench_ttft_2.5',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_mistral_goodput_benchmark_batchsize_128_3gpus_longbench_ttft_2.5_nobatch',

        #NOTE 11: goodput comparison on longbench llama-8B
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/vllm_llama8B_goodput_benchmark_batchsize_128_3gpus_longbench_ttft_2.5',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/sarathi_llama8B_goodput_benchmark_batchsize_128_3gpus_longbench_ttft_2.5',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_llama8B_goodput_benchmark_batchsize_128_3gpus_longbench_ttft_2.5',

        #NOTE 12: goodput comparison on gpt llama-8B
        '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/vllm_llama8B_goodput_benchmark_batchsize_128_3gpus_gpt_ttft_1',
        '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/sarathi_llama8B_goodput_benchmark_batchsize_128_3gpus_gpt_ttft_1',
        '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_llama8B_goodput_benchmark_batchsize_128_3gpus_gpt_ttft_1',
        
        #NOTE 13: goodput comparison on gpt Qwen-14B
        #'/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/vllm_qwen14B_goodput_benchmark_batchsize_128_3gpus_gpt',
        #'/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/sarathi_qwen14B_goodput_benchmark_batchsize_128_3gpus_gpt',
        #'/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_qwen14B_goodput_benchmark_batchsize_128_3gpus_gpt',
        #'/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_qwen14B_goodput_benchmark_batchsize_128_3gpus_gpt_3L',
        #'/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_qwen14B_goodput_benchmark_batchsize_128_3gpus_gpt_2L1H'
        
        #NOTE 13: goodput comparison on longbench Qwen-14B
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/vllm_qwen14B_goodput_benchmark_batchsize_128_3gpus_longbench',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/sarathi_qwen14B_goodput_benchmark_batchsize_128_3gpus_longbench',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_qwen14B_goodput_benchmark_batchsize_128_3gpus_longbench_2L1H_3.0',
     
        
        #NOTE 13: 3L and 2L1H comparison on longbench llama-8B
        #'/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_llama8B_goodput_benchmark_batchsize_128_3gpus_longbench_3L',
        #'/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_llama8B_benchmark_batchsize_128_3gpus_gpt_3L',
        #'/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_llama8B_goodput_benchmark_batchsize_128_3gpus_longbench_2L1H'

        #NOTE 14: 2 GPUs 1L1H on longbench llama-8B
        
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/vllm_llama_goodput_benchmark_batchsize_128_2gpus_longbench_ttft_2.5',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/sarathi_llama_goodput_benchmark_batchsize_128_2gpus_longbench_ttft_2.5',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_llama_goodput_benchmark_batchsize_128_2gpus_longbench_ttft_2.5',
        
        #NOTE : 2L1H and 2H1L comparison
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_llama_goodput_benchmark_batchsize_128_3gpus_gpt_2H1L',
        # '/scratch/gilbreth/li2068/asplos/sarathi-serve/exp_sosp/coordinate_llama8B_goodput_benchmark_batchsize_128_3gpus_gpt_ttft_1'
]
