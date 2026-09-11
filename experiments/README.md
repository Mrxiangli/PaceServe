# Middleware Replication

This document outlines the steps to set up the environment and run the experiments for middleware replication.

## 1. Environment Setup

We use Conda to manage the environment and `requirements.txt` (loaded by
`setup.py`) to install Python dependencies.

1. **Create the Conda Environment**:
   From the repository root, create the environment and install all dependencies:
   ```sh
   conda create -p "$(pwd)/middleware-env" python=3.12 pip
   conda activate "$(pwd)/middleware-env"
   conda env config vars set PYTHONNOUSERSITE=1
   export PYTHONNOUSERSITE=1
   conda install ninja
   python -m pip install -e . --extra-index-url https://flashinfer.ai/whl/cu121/torch2.3/
   python -m pip check
   python -c "import yaml, six, mdurl; import pandas; from huggingface_hub import snapshot_download; print('Dependency imports OK')"
   ```

2. **Set Up HuggingFace Token and Home**:
   Since the experiments require downloading models (like LLaMA) from HuggingFace, you must export your HuggingFace token:
   ```sh
   export HF_TOKEN="your_huggingface_token_here"
   export HF_HOME="/directory/to/downloaded/model"
   ```

## 2. Batch execution analytical model verification (Figure 5): [~20-30 minutes]

**Hardware requirement**: A single A100 GPU with at least 40GB of memory.(CUDA 13.1)

For this experiment, we use our pretrained weights (`ai_util/regression_weights`) to perform the batch execution time estimation.

1. **Run the Verification Script**:
   Execute the bash script that runs the `single_gpu_config` and then automatically generates the batch execution and estimation latency comparison figures:
   ```sh
   ./experiments/analytical_model_verify.sh
   ```
   This should create a new folder called analytical_model_verification, we can find the verification plot under analytical_model_verification/plots/.

## 3. Goodput Comparison (Figure 8) [~24 hrs]

**Hardware requirements**: 3 A100 GPUs with 80GB HBM, if 3 GPUs are on different nodes, following [ray_setup.txt](ray_setup.txt) to manually form the cluster before executing any script.

1. **Run the Goodput Scripts**
   Execute the bash script that runs the `goodput main_config` across PaceServe and Baselines across all workload
   ```sh
   ./experiments/goodput_experiments.sh
   ```
   This will save all experiments log under exp_result folder

   The goodput script skips completed runs with matching command arguments in
   earlier sessions under `exp_result/main_data`. Failed or incomplete runs are
   retried, and changed arguments trigger a new run. 

2. **Generate plotting config**
   Execute 
   ```python
   python generate_goodput_plot_config.py ../exp_result/main_data
   ```
   generate goodput_plot_config.json, then execute
   ```python
   python goodput_plot_combined.py
   ```
   generate the main result under exp_results/figures/main_data (Figure 8)

## 4. Admission Control (Figure 14) [~1 hr]

From the `PaceServe` repository root, with the benchmark environment activated
and the three-GPU Ray cluster running:

```bash
bash experiments/admission_control_experiments.sh
```

Results are saved under
`exp_result/admission_control_data/llama_sharegpt_admission/<timestamp>/`.
The `figures/` subdirectory contains PNG and PDF charts, a CSV of request
counts, and JSON metadata. Logs go to `logs/admission_control_config/`.
To plot an existing session:

```bash
python experiments/plot_admission_control.py \
  --session exp_result/admission_control_data/llama_sharegpt_admission/YYYYMMDD_HHMMSS
```

Omit `--session` to select the latest session, or use `--output-dir` to choose
where figures are written. The plot shows attained, dropped, and completed
SLO-violating request counts, with each category's ratios relative to Sarathi.


## 5. Micro Experiments (Figure 10, 11, 12, 13, 15):
1. **Real Trace Experiments (Figure 15) [~ 3hrs]** 
   Execute
   ```sh
   bash experiments/trace_experiments.sh
   ```
   Run from the repository root. The script resumes the latest trace session,
   skips matching completed runs, and restarts failed or incomplete scheduler
   runs from the beginning. It preserves replaced attempts in `.previous_attempts/`.
   Add `--dry-run` to preview the remaining runs.

2. **GPU Partition Experiments (2H1L and 1H1L) (Figure 13) [~ 4hrs]**
   Execute
   ```sh
   bash experiments/gpu_partition_experiments.sh
   ```

3. **Generate micro experiments result**
   Execute
   ```sh
   sh plot_micro.sh --data-dir ../exp_result/main_data
   ```
   generate Figure 10, 11, 12. 13 15 under exp_result/figures
