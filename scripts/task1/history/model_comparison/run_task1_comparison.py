#!/usr/bin/env python3
"""One entry point for fresh data preparation, four-model training and reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=ROOT / "configs/task1_comparison.json")
    commands = result.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="USER RUN: stream raw .cool once and create fresh grouped folds")
    prepare.add_argument("--cache-dir", type=Path)
    train = commands.add_parser("train", help="USER RUN: fit selected models; completed folds are skipped automatically")
    train.add_argument("--model", required=True, choices=["pca_lr", "m2", "m2plus", "cnn_transformer", "all"])
    train.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    train.add_argument("--fold", type=int, nargs="+", help="Outer folds, zero based; omitted = all five")
    train.add_argument("--cache-dir", type=Path)
    train.add_argument("--run-dir", type=Path)
    summarize = commands.add_parser("summarize", help="Rebuild CSV/JSON/plots from completed outer-fold predictions")
    summarize.add_argument("--run-dir", type=Path)
    commands.add_parser("check", help="Lightweight config/model specification check; no preparation or training")
    return result


def project_path(value: Path | str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> int:
    args = parser().parse_args()
    if args.command == "summarize":
        from microcture.task1.evaluation import summarize_results
        # An explicit historical run can be summarized even when today's
        # training config has changed or is no longer present.
        if args.run_dir is not None:
            run_dir = project_path(args.run_dir)
        else:
            settings = json.loads(project_path(args.config).read_text(encoding="utf-8"))
            run_dir = project_path(settings["run_dir"])
        summary = summarize_results(run_dir)
        for row in summary["comparison"]:
            print(json.dumps(row, ensure_ascii=False))
        for error in summary["errors"]:
            print(f"INVALID EVIDENCE: {error}", file=sys.stderr)
        print(f"Results: {run_dir}")
        return 1 if summary["errors"] else 0
    from microcture.task1.training import load_config, run_training
    from microcture.task1.data import prepare_dataset
    from microcture.task1.models import model_spec

    config = load_config(project_path(args.config))
    cache_dir = project_path(getattr(args, "cache_dir", None) or config["cache_dir"])
    run_dir = project_path(getattr(args, "run_dir", None) or config["run_dir"])
    if args.command == "check":
        specs = {name: model_spec(name, config["models"]) for name in ("m2", "m2plus", "cnn_transformer")}
        print(json.dumps({"config": str(project_path(args.config)), "cache_dir": str(cache_dir),
                          "cache_ready": (cache_dir / "metadata.json").is_file(), "run_dir": str(run_dir),
                          "folds": config["data"]["folds"], "neural_seeds": config["training"]["seeds"],
                          "parameters": {name: spec["parameters"] for name, spec in specs.items()},
                          "note": "Only configuration/model definitions checked; no data preparation or training."},
                         ensure_ascii=False, indent=2))
    elif args.command == "prepare":
        dataset = prepare_dataset(config["data"], ROOT, cache_dir)
        print(json.dumps({"cache_dir": str(cache_dir), "signature": dataset["signature"],
                          "structures": len(dataset["y"]), "folds": len(dataset["folds"]),
                          "test_counts": [fold["test_counts"] for fold in dataset["folds"]]}, ensure_ascii=False, indent=2))
    else:
        summary = run_training(config, ROOT, cache_dir, run_dir, args.model, args.fold, args.device)
        for row in summary["comparison"]:
            print(json.dumps(row, ensure_ascii=False))
        print(f"Results: {run_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
