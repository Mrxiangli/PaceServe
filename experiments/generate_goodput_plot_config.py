"""Generate the combined goodput plot config from experiment/session directories."""

import argparse
from datetime import datetime
import json
from pathlib import Path


DEFAULT_OUTPUT = Path(__file__).resolve().with_name("goodput_plot_config.json")


def generate_config(data_dir: Path, existing: dict, merge_sessions: bool = False) -> dict:
    if not data_dir.is_dir():
        raise ValueError(f"Results directory does not exist: {data_dir}")
    config = {
        "_comment": "Generated from timestamped session folders; selection does not imply a completed run.",
        "_data_dir": str(data_dir.resolve()),
    }
    for experiment in sorted(data_dir.iterdir()):
        if not experiment.is_dir() or not experiment.name.endswith(("_qps", "_slo")):
            continue
        sessions = []
        for session in sorted(experiment.iterdir()):
            if not session.is_dir():
                continue
            try:
                timestamp = datetime.strptime(session.name, "%Y%m%d_%H%M%S")
            except ValueError:
                continue
            if timestamp.strftime("%Y%m%d_%H%M%S") == session.name:
                sessions.append(session.name)
        if sessions:
            config[experiment.name] = {
                "main": sessions[0] if merge_sessions else sessions[-1],
                "overrides": sessions[1:] if merge_sessions else [],
            }
    if len(config) == 2:
        raise ValueError(f"No timestamped QPS/SLO sessions found in {data_dir}")
    for key in ("static_data", "baselines"):
        if key in existing:
            config[key] = existing[key]
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Parent results directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--merge-sessions", action="store_true",
                        help="Use all timestamped sessions, with newer points overriding older ones.")
    args = parser.parse_args()
    try:
        existing = json.loads(args.output.read_text()) if args.output.exists() else {}
        config = generate_config(args.data_dir, existing, args.merge_sessions)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.write_text(json.dumps(config, indent=4) + "\n")
    for key, entry in config.items():
        if isinstance(entry, dict) and "main" in entry:
            print(f"{key}: {entry['main']} (overrides: {entry['overrides']})")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
