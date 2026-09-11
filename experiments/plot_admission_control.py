"""Plot measured admission-control outcomes, normalized to Sarathi per category."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULERS = ("VLLM", "SARATHI", "PACE")
CATEGORIES = ("SLO Attained", "Requests Dropped", "Requests Violate SLO")


def count_outcomes(df, ttft, tbt):
    """Count each request once, including requests transferred between instances.

    Transfer-only records have no completed latency metrics. An eventual
    completion takes precedence over those records. Missing metrics never
    imply that a request was dropped.
    """
    required = {"Request Id", "request_aborted", "prefill_e2e_time",
                "decode_mean_time", "request_finished_at"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing metrics columns: {sorted(missing)}")
    if df.empty or df["Request Id"].isna().any():
        raise ValueError("Empty metrics or missing request IDs")
    dropped = set(df.loc[df.request_aborted.eq(1), "Request Id"])
    completed = df.dropna(subset=["prefill_e2e_time", "decode_mean_time", "request_finished_at"])
    completed = completed.sort_values("request_finished_at").drop_duplicates("Request Id", keep="last")
    if dropped & set(completed["Request Id"]):
        raise ValueError("A request has both completed and aborted records")
    attained = ((completed.prefill_e2e_time <= ttft) & (completed.decode_mean_time <= tbt)).sum()
    unresolved = set(df["Request Id"]) - dropped - set(completed["Request Id"])
    return {
        CATEGORIES[0]: int(attained),
        CATEGORIES[1]: len(dropped),
        CATEGORIES[2]: int(len(completed) - attained),
        "Unresolved recorded requests": len(unresolved),
    }


def read_session(session):
    results = {}
    reference = None
    for scheduler in SCHEDULERS:
        runs = list((session / scheduler).glob("qps_*"))
        if len(runs) != 1:
            raise ValueError(f"Expected one QPS run for {scheduler}, found {len(runs)} in {session}")
        run = runs[0]
        if list(run.glob("FAILED_*.txt")):
            raise ValueError(f"Failed run: {run}")
        with (run / "config.yaml").open() as file:
            config = yaml.safe_load(file)
        prefix = f"{scheduler.lower()}_scheduler_config_"
        if config["scheduler_config_type"] != scheduler or int(config[prefix + "allow_drop_requests"]) != 1:
            raise ValueError(f"Wrong scheduler or admission control disabled: {run}")
        keys = ["model_config_model", "trace_request_length_generator_config_trace_file",
                "poisson_request_interval_generator_config_qps",
                "poisson_request_interval_generator_config_seed", "time_limit",
                "synthetic_request_generator_config_duration", "instance_config_num_instance"]
        settings = {key: config[key] for key in keys}
        settings.update({key: config[prefix + key] for key in
                         ("ttft_slo", "tbt_slo", "request_drop_threshold")})
        if reference is not None and settings != reference:
            raise ValueError(f"Workload, duration, hardware count, or SLO settings differ: {run}")
        reference = settings
        replicas = sorted(run.glob("replica_*/sequence_metrics.csv"))
        if len(replicas) != int(config["num_replicas"]):
            raise ValueError(f"Missing replica metrics: {run}")
        counts = [count_outcomes(pd.read_csv(path), float(settings["ttft_slo"]),
                                 float(settings["tbt_slo"])) for path in replicas]
        results[scheduler] = {key: sum(count[key] for count in counts) for key in counts[0]}
    return results, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, help="Exact timestamped session directory; defaults to latest.")
    parser.add_argument("--data-dir", type=Path,
                        default=REPO_ROOT / "exp_result/admission_control_data/llama_sharegpt_admission")
    parser.add_argument("--output-dir", type=Path, help="Default: <session>/figures")
    args = parser.parse_args()
    if args.session is None:
        sessions = sorted(path for path in args.data_dir.glob("????????_??????") if path.is_dir())
        if not sessions:
            parser.error(f"No sessions found in {args.data_dir}")
        args.session = sessions[-1]
    try:
        results, settings = read_session(args.session)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    output = args.output_dir or args.session / "figures"
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame.from_dict(results, orient="index")
    summary.index.name = "Scheduler"
    summary.to_csv(output / "admission_control_counts.csv")
    (output / "admission_control_metadata.json").write_text(json.dumps({
        "session": str(args.session.resolve()), "settings": settings,
        "note": "Unresolved recorded requests are excluded from the three bars; unrecorded requests cannot be counted.",
    }, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(9, 3.3))
    x = np.arange(len(SCHEDULERS))
    width = 0.23
    for index, (category, color) in enumerate(zip(CATEGORIES, ("#70A5D9", "#ECA680", "#F0868C"))):
        values = [results[scheduler][category] for scheduler in SCHEDULERS]
        bars = ax.bar(x + (index - 1) * width, values, width, label=category,
                      color=color, edgecolor="black", linewidth=0.8)
        baseline = results["SARATHI"][category]
        labels = [f"x{value / baseline:.2f}" if baseline else "N/A" for value in values]
        ax.bar_label(bars, labels=labels, padding=3, fontsize=12)
    ax.set_xticks(x, ["vLLM", "Sarathi", "Pace"])
    ax.set_ylabel("# Requests")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_ylim(0, max(1, summary[list(CATEGORIES)].to_numpy().max()) * 1.22)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=3)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"admission_control.{suffix}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(summary.to_string())
    print(f"Figures and counts saved to {output}")


if __name__ == "__main__":
    main()
