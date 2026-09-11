# PaceServe

[![arXiv paper 2504.20828](https://img.shields.io/badge/arXiv-2504.20828-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2504.20828)

PaceServe is an LLM serving system that dynamically adjusts request priority
to improve goodput under time-to-first-token and time-between-token
service-level objectives (SLOs). Its design combines a batch execution time
estimator, SLO-aware batch scheduling, and separate low- and high-priority
serving instances. By using estimated execution times and request waiting
times to identify requests at risk of missing their first-token deadline,
PaceServe promotes those requests from low-priority queues to high-priority
instances. This coordination aims to balance efficient GPU batching with
timely responses as workload demand changes. 


## Set up the environment

Create the environment once, from the repository root:

```bash
conda create -p "$(pwd)/middleware-env" python=3.12 pip
conda activate "$(pwd)/middleware-env"
conda env config vars set PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE=1
conda install ninja

python -m pip install -e . --extra-index-url https://flashinfer.ai/whl/cu121/torch2.3/
python -m pip check
python -c "import yaml, six, mdurl; import pandas; from huggingface_hub import snapshot_download; print('Dependency imports OK')"
```

The editable install uses the dependencies in `requirements.txt`.
`PYTHONNOUSERSITE=1` keeps packages in `~/.local` out of this environment.

For each new terminal session:

```bash

# From the repository root:
conda activate "$(pwd)/middleware-env"
export PYTHONNOUSERSITE=1
export HF_HOME="$(pwd)/downloaded_model"
```

Set `HF_TOKEN` to your Hugging Face access token when downloading gated models,
and ensure your account has access to the requested model. You can set
`HF_HOME` to another accessible model cache directory.

Run benchmarks on allocated GPU nodes. The main experiments use three A100
GPUs with 80 GB memory each; analytical model verification uses one A100 with
at least 40 GB. The GPU partition sweeps use two- and three-GPU configurations.
For GPUs spread across nodes, activate the same environment on each node and
follow [the Ray cluster setup instructions](experiments/ray_setup.txt),
using your head node's address.

## Run one experiment

Start by previewing a single workload. This prints the benchmark commands
without executing them:

```bash
python experiments/run.py \
  --config main_config \
  --exp_id llama_sharegpt_qps \
  --filter scheduler=PACE \
```
## Middleware replication
Follow the instructions in [experiments](experiments/README.md).

## Citation

```bibtex
@article{ikram2025ascendra,
  title={Ascendra: Dynamic request prioritization for efficient llm serving},
  author={Ikram, Azam and Li, Xiang and Elnikety, Sameh and Bagchi, Saurabh},
  journal={arXiv preprint arXiv:2504.20828},
  year={2025}
}
```
