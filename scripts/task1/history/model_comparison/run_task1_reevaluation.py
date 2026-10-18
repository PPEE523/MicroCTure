#!/usr/bin/env python3
"""Uniform original model families, paired final40 / inner-CE checkpoints."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from microcture.task1_reevaluation.protocol import load_protocol, project_path
    from microcture.task1_reevaluation.runner import check_protocol, guard_output, run_jobs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/task1_reevaluation.json")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Metadata/signature/model-shape check; no preprocessing or fitting")
    commands.add_parser("summarize", help="Verify saved trajectories/predictions and rebuild derived summaries")
    for name in ("main", "ablations", "baselines", "all"):
        sub = commands.add_parser(name, help="USER RUN: main=50 fits including PCA, all=155 fits")
        sub.add_argument("--model", default="all")
        sub.add_argument("--condition", choices=("I3", "I0", "G0"))
        sub.add_argument("--fold", type=int, choices=range(5))
        sub.add_argument("--seed", type=int, choices=(42, 43, 44))
    args = parser.parse_args()
    config = load_protocol(project_path(ROOT, args.config))
    run_dir = guard_output(ROOT, project_path(ROOT, args.run_dir or config["run_dir"]))
    if args.command == "check":
        print(json.dumps(check_protocol(ROOT, config, run_dir, args.device), ensure_ascii=False, indent=2))
        return 0
    if args.command == "summarize":
        from microcture.task1_reevaluation.evaluation import summarize_reevaluation
        summary = summarize_reevaluation(run_dir)
    else:
        summary = run_jobs(config, ROOT, run_dir, args.device, stage=args.command,
                           model=args.model, condition=args.condition, fold=args.fold, seed=args.seed)
    fields = ("status", "expected_fits", "completed_fits", "numerical_failure_fits", "execution_failure_fits",
              "invalid_fits", "pending_fits", "expected_endpoints", "completed_endpoints", "main_status",
              "main_completed_fits", "main_completed_endpoints", "main_complete_oof")
    print(json.dumps({**{key: summary.get(key) for key in fields}, "run_dir": str(run_dir)}, ensure_ascii=False, indent=2))
    return int(summary.get("status") == "invalid" or bool(summary.get("numerical_failure_fits"))
               or bool(summary.get("execution_failure_fits")))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
