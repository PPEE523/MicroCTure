#!/usr/bin/env python3
"""Independent I3/H3 architecture CV, matched input ablations and new baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from microcture.task1_formal.protocol import load_protocol
    from microcture.task1_formal.runner import check_protocol, guard_output, project_path, run_jobs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/task1_formal.json")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Read-only signatures/IDs/config/model shapes; no image preparation or training")
    sub.add_parser("summarize", help="Rebuild derived tables from saved predictions/traces; no inference or fitting")
    for name in ("main", "ablations", "baselines", "all"):
        task = sub.add_parser(name, help="USER RUN: fixed-duration fits in the independent formal result directory")
        task.add_argument("--model", default="all", help="Neural model name or baseline name, according to stage")
        task.add_argument("--condition", choices=("I3", "I0", "G0"))
        task.add_argument("--fold", type=int, choices=range(5))
        task.add_argument("--seed", type=int, choices=(42, 43, 44))
    args = parser.parse_args()
    config = load_protocol(project_path(ROOT, args.config))
    run_dir = guard_output(ROOT, project_path(ROOT, args.run_dir or config["run_dir"]))
    if args.command == "check":
        report = check_protocol(ROOT, config, run_dir, args.device)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "summarize":
        from microcture.task1_formal.evaluation import summarize_formal
        summary = summarize_formal(run_dir)
    else:
        summary = run_jobs(config, ROOT, run_dir, args.device, stage=args.command,
                           model=args.model, condition=args.condition, fold=args.fold, seed=args.seed)
    print(json.dumps({"status": summary.get("status", summary.get("execution_status")),
                      "completed_jobs": summary.get("completed_jobs"),
                      "numerical_failure_jobs": summary.get("numerical_failure_jobs"),
                      "execution_failure_jobs": summary.get("execution_failure_jobs"),
                      "invalid_jobs": summary.get("invalid_jobs"), "pending_jobs": summary.get("pending_jobs"),
                      "run_dir": str(run_dir)}, ensure_ascii=False, indent=2))
    return 1 if (summary.get("status") == "invalid" or summary.get("numerical_failure_jobs", 0)
                 or summary.get("execution_failure_jobs", 0)) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
