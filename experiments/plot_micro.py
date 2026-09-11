"""Run micro plots using the latest timestamped sessions in each results directory."""

import argparse
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def latest_session(root, experiment):
    directory = root / experiment
    sessions = []
    if directory.is_dir():
        for path in directory.iterdir():
            if not path.is_dir():
                continue
            try:
                stamp = datetime.strptime(path.name, "%Y%m%d_%H%M%S")
            except ValueError:
                continue
            if stamp.strftime("%Y%m%d_%H%M%S") == path.name:
                sessions.append(path)
    if not sessions:
        raise ValueError(f"No timestamped sessions in {directory}")
    selected = max(sessions)
    print(f"Selected: {selected}", flush=True)
    return selected


def require(path):
    if not path.exists():
        raise ValueError(f"Missing input: {path}")
    return path


def run_path(session, qps):
    path = session / 'PACE' / f'qps_{qps}'
    for marker in ('FAILED_CRASH.txt', 'FAILED_TIMEOUT.txt'):
        if (path / marker).exists():
            raise ValueError(f"Failed run: {path}")
    require(path / 'config.yaml')
    require(path / 'replica_0' / 'sequence_metrics.csv')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=REPO_ROOT / 'exp_result/main_data')
    parser.add_argument('--trace-data-dir', type=Path, default=REPO_ROOT / 'exp_result/trace_replay_data')
    parser.add_argument('--two-gpu-data-dir', type=Path, default=REPO_ROOT / 'exp_result/micro_data/two_gpus')
    parser.add_argument('--two-h-one-l-data-dir', type=Path, default=REPO_ROOT / 'exp_result/micro_data/2H1L')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without generating plots.')
    args = parser.parse_args()
    for key in ('data_dir', 'trace_data_dir', 'two_gpu_data_dir', 'two_h_one_l_data_dir'):
        setattr(args, key, getattr(args, key).resolve())
    cache = {}

    def session(root, experiment):
        key = (root, experiment)
        if key not in cache:
            cache[key] = latest_session(root, experiment)
        return cache[key]

    def main_session(experiment):
        return session(args.data_dir, experiment)

    plots = [
        ('timeline_plot.py', lambda: [
            '--exp_id', 'llama_code_trace', '--data_dir', args.trace_data_dir,
            '--session', session(args.trace_data_dir, 'llama_code_trace').name]),
        ('transfer_overhead.py', lambda: [
            run_path(main_session('llama_longbench_qps'), 7) / 'replica_0/sequence_metrics.csv',
            run_path(main_session('mistral_sharegpt_qps'), 22) / 'replica_0/sequence_metrics.csv',
            '--legends', 'Llama (LongBench)', 'Mistral (ShareGPT)']),
        ('offload_slo_plot.py', lambda: [
            require(main_session('llama_sharegpt_qps') / 'PACE'),
            require(main_session('llama_longbench_qps') / 'PACE'),
            '--title1', 'Llama - ShareGPT', '--title2', 'Llama - LongBench']),
        ('fairness_plot.py', lambda: [
            run_path(main_session('llama_sharegpt_qps'), 25),
            run_path(main_session('llama_longbench_qps'), 9),
            '--title1', 'LlaMA on ShareGPT (25 QPS)', '--title2', 'LlaMA on LongBench (9 QPS)']),
        ('goodput_plot.py', lambda: [
            '--exp_id', 'llama_sharegpt_qps', '--data_dir', args.two_gpu_data_dir,
            '--session', session(args.two_gpu_data_dir, 'llama_sharegpt_qps').name]),
        ('plot_2h1l_vs_pace.py', lambda: [
            require(main_session('llama_longbench_qps') / 'PACE'),
            require(session(args.two_h_one_l_data_dir, 'llama_longbench_qps') / 'PACE'),
            '--title', 'Llama - LongBench']),
    ]
    failed = False
    for script, build_args in plots:
        try:
            command = [sys.executable, str(SCRIPT_DIR / script),
                       *map(str, build_args())]
        except ValueError as exc:
            print(f'[SKIP] {script}: {exc}', flush=True)
            continue
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            result = subprocess.run(command, cwd=REPO_ROOT)
            if result.returncode:
                failed = True
                print(f'[ERROR] {script} exited with {result.returncode}', flush=True)
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
