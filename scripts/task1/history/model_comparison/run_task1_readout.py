#!/usr/bin/env python3
"""Independent O0/O1 M2+ target-readout control; real fits are USER RUN."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from microcture.task1_readout.protocol import load_protocol, project_path
    from microcture.task1_readout.runner import check_protocol, guard_output, run_jobs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/task1_readout.json")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Read-only metadata/source checks; no image preparation or training")
    sub.add_parser("summarize", help="Saved predictions/history only; no fitting or inference")
    run = sub.add_parser("all", help="USER RUN: 60 fixed-duration paired training jobs")
    run.add_argument("--variant", choices=("all", "O0", "O1"), default="all")
    run.add_argument("--condition", choices=("I3", "G0"))
    run.add_argument("--fold", type=int, choices=range(5))
    run.add_argument("--seed", type=int, choices=(42, 43, 44))
    args = parser.parse_args()
    config = load_protocol(project_path(ROOT, args.config))
    run_dir = guard_output(ROOT, project_path(ROOT, args.run_dir or config["run_dir"]),
                           project_path(ROOT, config["formal_run_dir"]))
    if args.command == "check":
        print(json.dumps(check_protocol(ROOT, config, run_dir, args.device), ensure_ascii=False, indent=2))
        return 0
    if args.command == "summarize":
        from microcture.task1_readout.evaluation import summarize_readout
        result = summarize_readout(run_dir)
    else:
        result = run_jobs(config, ROOT, run_dir, args.device, variant=args.variant,
                          condition=args.condition, fold=args.fold, seed=args.seed)
    print(json.dumps({key: result.get(key) for key in
                      ("status", "completed_jobs", "pending_jobs", "invalid_jobs",
                       "numerical_failure_jobs", "execution_failure_jobs")}, ensure_ascii=False, indent=2))
    return 1 if result.get("status") == "invalid" or result.get("numerical_failure_jobs", 0) or result.get("execution_failure_jobs", 0) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
