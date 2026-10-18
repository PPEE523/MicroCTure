#!/usr/bin/env python3
"""Final task-one workflow: check, train four models, ablate, and summarize."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
_PROJECT_DIRECTORY = next(parent for parent in Path(__file__).resolve().parents
                          if (parent / "microcture/layout.py").is_file())
sys.path.insert(0, str(_PROJECT_DIRECTORY))
from microcture.layout import PROJECT_ROOT as ROOT, bootstrap, resolve
bootstrap()


def configured(args):
    from microcture.tasks.task1.protocol import load_protocol, project_path
    config = load_protocol(project_path(ROOT, args.config))
    for argument, key in (("cache", "cache_dir"), ("folds_dir", "frozen_run_dir"),
                          ("data_root", "data_root"), ("source_root", "source_root")):
        value = getattr(args, argument, None)
        if value is not None:
            config[key] = str(value)
    if getattr(args, "folds_dir", None) is None:
        prepared = project_path(ROOT, config["cache_dir"]) / "formal"
        if (prepared / "folds/fold_0.json").is_file():
            config["frozen_run_dir"] = str(prepared)
    return config


def main():
    from microcture.tasks.task1.comparison import check_protocol, guard_output, project_path, run_jobs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=resolve(ROOT, "configs/task1_local.json"))
    parser.add_argument("--cache", type=Path, help="Read-only prepared count cache, including an external original cache")
    parser.add_argument("--folds-dir", type=Path, help="Directory containing frozen folds/fold_0.json ... fold_4.json")
    parser.add_argument("--data-root", type=Path, help="Project/input root for paths in configs/shared/data.json")
    parser.add_argument("--source-root", type=Path, help="Optional relocated annotation/config provenance root for an existing cache")
    parser.add_argument("--run-dir", type=Path, help="Fresh/resumable results/task1/<experiment>/<version>; retained experiments are refused")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Read-only protocol/model metadata check; no images, fitting or output directories")
    sub.add_parser("summarize", help="Summarize saved predictions, without fitting or inference")
    for name in ("main", "ablations", "baselines", "all"):
        task = sub.add_parser(name, help="Train into a new independent result directory; completed/failed jobs are never overwritten")
        task.add_argument("--model", default="all", help="m2 / m2plus / cnn_transformer / pca_lr / registered geometry baseline")
        task.add_argument("--condition", choices=("I3", "I0", "G0"))
        task.add_argument("--fold", type=int, choices=range(5))
        task.add_argument("--seed", type=int, choices=(42, 43, 44))
    args = parser.parse_args()
    config = configured(args)
    run_dir = guard_output(ROOT, project_path(ROOT, args.run_dir or config["run_dir"]))
    if args.command == "check":
        print(json.dumps(check_protocol(ROOT, config, run_dir, args.device), ensure_ascii=False, indent=2))
        return 0
    if args.command == "summarize":
        from microcture.tasks.task1.evaluation import summarize_formal
        result = summarize_formal(run_dir)
    else:
        result = run_jobs(config, ROOT, run_dir, args.device, stage=args.command, model=args.model,
                          condition=args.condition, fold=args.fold, seed=args.seed)
    print(json.dumps({key: result.get(key) for key in ("status", "completed_jobs", "pending_jobs", "numerical_failure_jobs",
                                                     "execution_failure_jobs", "invalid_jobs")}, ensure_ascii=False, indent=2))
    return int(bool(result.get("status") == "invalid" or result.get("numerical_failure_jobs", 0)
                    or result.get("execution_failure_jobs", 0) or result.get("invalid_jobs", 0)))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
