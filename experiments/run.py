import os
import sys
import argparse
import itertools
import subprocess
import json
import shutil
import time
import re
import operator
import importlib
import shlex
from pathlib import Path


def command_signature(command):
    """Compare benchmark arguments while ignoring the session output path."""
    tokens = shlex.split(command)
    index = tokens.index("--output_dir")
    return tokens[:index] + tokens[index + 2:]


def find_completed_run(suite_dir, scheduler, folder_name, command):
    if not suite_dir.exists():
        return None
    expected = command_signature(command)
    for session in sorted(suite_dir.iterdir(), reverse=True):
        candidate = session / scheduler / folder_name
        if not candidate.is_dir() or any(candidate.glob("FAILED_*.txt")):
            continue
        try:
            with (candidate / "execution.log").open() as log:
                next(log)
                recorded = next(log).removeprefix("Command: ").strip()
                # A retry can leave multiple attempt directories, preventing
                # flatten_directory from moving successful output to the root.
                completed_dirs = [
                    candidate / Path(line.strip().removeprefix("result at: ")).name
                    for line in log if line.startswith("result at: ")
                ]
            has_results = any(
                (directory / "config.yaml").is_file()
                and (directory / "replica_0" / "sequence_metrics.csv").is_file()
                for directory in [candidate, *completed_dirs]
            )
            if (has_results or (candidate / "SUCCESS.txt").is_file()) and command_signature(recorded) == expected:
                return candidate
        except (OSError, ValueError, StopIteration):
            continue
    return None

# --- Execution Constants ---
# TIMEOUT_SECONDS is loaded from the config module at runtime.
MAX_RETRIES = 2
TIMEOUT_SECONDS = 90 * 60  # fallback default; overridden by config

def parse_filters(filter_strings):
    """Parses filters supporting =, ==, !=, >=, <=, >, <"""
    filters = []
    if not filter_strings:
        return filters

    # Regex to capture the key, the operator, and the value
    pattern = re.compile(r'^([a-zA-Z0-9_]+)(>=|<=|!=|>|<|==|=)(.*)$')
    
    for f in filter_strings:
        match = pattern.match(f)
        if match:
            k, op, v = match.groups()
            if op == '=': 
                op = '==' # Normalize single equals to standard python equality
            filters.append((k, op, v))
        else:
            print(f"Warning: Ignored invalid filter format '{f}'. Use key>=value or key=value.")
    return filters

def flatten_directory(base_dir):
    base_dir_str = str(base_dir)
    subdirs = [os.path.join(base_dir_str, d) for d in os.listdir(base_dir_str) 
               if os.path.isdir(os.path.join(base_dir_str, d))]
    
    if len(subdirs) == 1:
        inner_dir = subdirs[0]
        for item in os.listdir(inner_dir):
            shutil.move(os.path.join(inner_dir, item), base_dir_str)
        os.rmdir(inner_dir)

def _fmt_elapsed(seconds):
    """Return a human-readable elapsed time string (e.g. '1h 23m 45s')."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"

def execute_resiliently(cmd_str, output_dir, run_idx, total_runs):
    log_file_path = output_dir / "execution.log"

    for attempt in range(1, MAX_RETRIES + 1):
        run_start = time.time()
        try:
            with open(log_file_path, "w") as log_file:
                log_file.write(f"--- Attempt {attempt} ---\n")
                log_file.write(f"Command: {cmd_str}\n\n")
                log_file.flush()

                print(f"  [{run_idx}/{total_runs}] Attempt {attempt}/{MAX_RETRIES} running...")
                subprocess.run(
                    cmd_str,
                    shell=True,
                    check=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=TIMEOUT_SECONDS
                )

            elapsed = time.time() - run_start
            print(f"  -> ✅ [SUCCESS]  ⏱  {_fmt_elapsed(elapsed)}")
            flatten_directory(output_dir)
            (output_dir / "SUCCESS.txt").touch()
            break

        except subprocess.TimeoutExpired:
            elapsed = time.time() - run_start
            print(f"  -> ⚠️ [WARNING] Hung and was killed after {TIMEOUT_SECONDS/60:.0f} mins  "
                  f"(wall time: {_fmt_elapsed(elapsed)}).")
            if attempt == MAX_RETRIES:
                print(f"  -> ❌ [FAILED] Max retries reached (Timeout).")
                (output_dir / "FAILED_TIMEOUT.txt").touch()

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - run_start
            print(f"  -> ⚠️ [WARNING] Crashed with exit code {e.returncode}  "
                  f"(wall time: {_fmt_elapsed(elapsed)}).")
            if attempt == MAX_RETRIES:
                print(f"  -> ❌ [FAILED] Max retries reached (Crash).")
                with open(output_dir / "FAILED_CRASH.txt", "w") as f:
                    f.write(f"Exit code: {e.returncode}")

def archive_existing_run(output_dir, session_dir):
    """Keep replaced attempts outside the session directories used by plots."""
    if output_dir.exists():
        relative = output_dir.relative_to(session_dir)
        archive = (session_dir.parent / ".previous_attempts"
                   / session_dir.name / relative.parent
                   / f"{relative.name}_{time.time_ns()}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output_dir), str(archive))


def run_experiment(exp_id, filters=None, dry_run=False, resume=False):
    if exp_id not in EXPERIMENT_SUITE:
        print(f"Error: '{exp_id}' not found.")
        return

    exp_config = EXPERIMENT_SUITE[exp_id]

    suite_start = time.time()
    session_timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = BASE_DATA_DIR / exp_id / session_timestamp
    if resume and session_dir.parent.exists():
        sessions = sorted(
            path for path in session_dir.parent.iterdir()
            if path.is_dir() and re.fullmatch(r"\d{8}_\d{6}", path.name)
        )
        if sessions:
            session_dir = sessions[-1]
            session_timestamp = session_dir.name

    print(filters)

    if not dry_run:
        session_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Loading suite: {exp_id}")
    print(f"📂 Session Directory: {session_dir}")
    
    grid_keys = list(exp_config["grid"].keys())
    grid_values = list(itertools.product(*exp_config["grid"].values()))

    # Map string operators to actual Python math operations
    ops = {
        '==': operator.eq, '!=': operator.ne,
        '>=': operator.ge, '<=': operator.le,
        '>': operator.gt,  '<': operator.lt
    }

    # --- UPGRADED FILTER LOGIC ---
    runs_to_execute = []
    for values in grid_values:
        current_grid_params = dict(zip(grid_keys, values))
        keep_run = True
        
        for k, op_str, v in filters:
            if k not in current_grid_params:
                keep_run = False
                break 
            
            grid_val = current_grid_params[k]
            
            # Try numeric comparison first (for qps, ttft_slo, etc.)
            try:
                num_grid = float(grid_val)
                num_v = float(v)
                if not ops[op_str](num_grid, num_v):
                    keep_run = False
                    break
            except ValueError:
                # Fallback to string comparison (for scheduler names)
                if not ops[op_str](str(grid_val), v):
                    keep_run = False
                    break
                    
        if keep_run:
            runs_to_execute.append(current_grid_params)

    if not runs_to_execute:
        print("No runs matched your filters. Exiting.")
        if not dry_run and not any(session_dir.iterdir()):
            session_dir.rmdir() 
        return

    print(f"🔍 Filters applied: {[(f[0], f[1], f[2]) for f in filters] if filters else 'None'}")
    print(f"⏳ Queueing {len(runs_to_execute)} runs...\n" + "-"*50)

    # --- EXECUTION LOOP ---
    for run_idx, current_grid_params in enumerate(runs_to_execute, 1):
        scheduler = current_grid_params["scheduler"]
        
        varying_params = [f"{k}_{v}" for k, v in current_grid_params.items() if k != "scheduler"]
        folder_name = "_".join(varying_params)
        run_output_dir = session_dir / scheduler / folder_name
        
        actual_scheduler_type = "PACE" if scheduler == "PACE_NEW" else scheduler
        
        sched_prefix = f"--{actual_scheduler_type.lower()}_scheduler_config_"
        final_args = exp_config["static_args"].copy()
        
        for k, v in exp_config.get("scheduler_static_args", {}).items():
            final_args[f"{sched_prefix}{k}"] = v
            
        specific_args = exp_config.get("scheduler_specific_args", {}).get(scheduler, {})
        for k, v in specific_args.items():
            final_args[k] = v
            
        final_args["--scheduler_config_type"] = actual_scheduler_type
        
        if "qps" in current_grid_params:
            final_args["--poisson_request_interval_generator_config_qps"] = current_grid_params["qps"]
        if "ttft_slo" in current_grid_params:
            final_args[f"{sched_prefix}ttft_slo"] = current_grid_params["ttft_slo"]
        if "tbt_slo" in current_grid_params:
            final_args[f"{sched_prefix}tbt_slo"] = current_grid_params["tbt_slo"]
            
        if "slo_scale" in current_grid_params:
            base_ttft = exp_config.get("base_slos", {}).get("ttft_slo", 1.0)
            base_tbt = exp_config.get("base_slos", {}).get("tbt_slo", 0.15)
            scale = float(current_grid_params["slo_scale"])
            final_args[f"{sched_prefix}ttft_slo"] = round(base_ttft * scale, 2)
            final_args[f"{sched_prefix}tbt_slo"] = round(base_tbt * scale, 2)
            
        final_args["--output_dir"] = str(run_output_dir)

        cmd_parts = ["python", "./paceserve/benchmark/main.py"]
        for k, v in final_args.items():
            cmd_parts.extend([k, str(v)])
        cmd_str = " ".join(cmd_parts)

        if resume:
            previous = find_completed_run(
                BASE_DATA_DIR / exp_id, scheduler, folder_name, cmd_str
            )
            if previous is not None:
                if previous != run_output_dir and not dry_run:
                    archive_existing_run(run_output_dir, session_dir)
                    shutil.copytree(previous, run_output_dir)
                print(f"  [SKIP] {current_grid_params}: existing result at {previous}")
                continue

        # Ensure the model exists locally if a downloaded_model path is specified
        model_arg = final_args.get("--model_config_model", "")
        if not dry_run and "downloaded_model" in model_arg:
            actual_model_path = model_arg.replace("$(pwd)", os.getcwd())
            if not os.path.exists(os.path.join(actual_model_path, "config.json")):
                repo_id = None
                try:
                    import main_config as mc
                    model_paths = getattr(mc, "MODEL_PATHS", {})
                    hf_repo_ids = getattr(mc, "HF_REPO_IDS", {})
                except ImportError:
                    model_paths = getattr(config_module, "MODEL_PATHS", {})
                    hf_repo_ids = getattr(config_module, "HF_REPO_IDS", {})

                for m_key, m_path in model_paths.items():
                    if m_path == model_arg:
                        repo_id = hf_repo_ids.get(m_key)
                        break
                if repo_id:
                    print(f"\n📥 Model not found locally at {actual_model_path}.")
                    print(f"Downloading {repo_id} from Hugging Face Hub (this may take a while)...")
                    try:
                        from huggingface_hub import snapshot_download
                        snapshot_download(repo_id=repo_id, local_dir=actual_model_path, local_dir_use_symlinks=False)
                    except Exception as e:
                        print(f"⚠️ Failed to download model: {e}")

        print(f"\n🎯 Target: {current_grid_params}")
        
        if dry_run:
            print(f"  [COMMAND]: {cmd_str}")
        else:
            archive_existing_run(run_output_dir, session_dir)
            run_output_dir.mkdir(parents=True, exist_ok=True)
            with open(run_output_dir / "meta.json", "w") as f:
                json.dump({"exp_id": exp_id, "session": session_timestamp, "params": current_grid_params}, f, indent=4)
                
            execute_resiliently(cmd_str, run_output_dir, run_idx, len(runs_to_execute))

    if not dry_run and not any(session_dir.iterdir()):
        session_dir.rmdir()
    suite_elapsed = time.time() - suite_start
    print(f"\n{'='*50}")
    print(f"🏁 Suite '{exp_id}' complete — total time: {_fmt_elapsed(suite_elapsed)}")
    print(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execution Engine for LLM Serving Experiments",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--exp_id", type=str, required=True, help="e.g., llama_sharegpt_qps, mistral_longbench_slo, llama_code_trace")
    parser.add_argument("--config", type=str, default="main_config",
                        help="Config module to load (default: main_config). Use 'trace_replay_config' for trace-replay experiments.")
    parser.add_argument("--dry-run", action="store_true", help="Print the bash commands without running them")
    parser.add_argument("--resume", action="store_true", help="Skip configurations that have already completed successfully")
    
    filter_help = """
Filter runs using space-separated key=value pairs.
Valid keys depend on the 'grid' definition in config.py for your chosen exp_id.
Supports math operators: >=, <=, >, <, !=, =

Valid keys for QPS experiments (e.g., llama_sharegpt_qps):
  - scheduler  (e.g., scheduler=PACE)
  - qps        (e.g., qps=24)

Valid keys for SLO experiments (e.g., llama_sharegpt_slo):
  - scheduler  (e.g., scheduler=VLLM)
  - slo_scale  (e.g., slo_scale=1.5)

Examples:
  python run.py --exp_id mistral_sharegpt_qps --filter scheduler=VLLM
  python run.py --exp_id llama_sharegpt_qps --filter scheduler=PACE "qps>=21"
  python run.py --exp_id qwen_longbench_slo --filter slo_scale=3.0 "scheduler!=VLLM"
"""
    parser.add_argument("--filter", nargs="*", help=filter_help)
    
    args = parser.parse_args()

    # Dynamically load the chosen config module from the same directory as run.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    config_module = importlib.import_module(args.config)
    EXPERIMENT_SUITE = config_module.EXPERIMENT_SUITE
    BASE_DATA_DIR = config_module.BASE_DATA_DIR
    TIMEOUT_SECONDS = config_module.TIMEOUT_SECONDS  # per-run timeout from config

    parsed_filters = parse_filters(args.filter)
    run_experiment(args.exp_id, filters=parsed_filters, dry_run=args.dry_run, resume=args.resume)
