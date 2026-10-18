#!/usr/bin/env python3
"""Fixed inner-fold target-input controls, followed by H0-H3 learning checks."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from microcture.task1_controlled.runner import (load_protocol, prepare_manifest, project_path,
                                                  run_stage, summarize_controlled)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/task1_controlled.json")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="Freeze/source/target-mask preflight; no real image preprocessing or training")
    subparsers.add_parser("input", help="USER RUN: I0-I3 M2/H0 plus three geometry-only LR baselines, frozen fold2/seed42")
    heads = subparsers.add_parser("heads", help="USER RUN: H0-H3 on a historical failed cell; input controls must finish first")
    heads.add_argument("--case", required=True, help="Case name from configs/task1_controlled.json, or all")
    subparsers.add_parser("summarize", help="Rebuild only the controlled comparison and trace plots")
    args = parser.parse_args()
    config = load_protocol(project_path(ROOT, args.config))
    run_dir = project_path(ROOT, args.run_dir or config["run_dir"])
    if args.command == "check":
        manifest, _ = prepare_manifest(config, ROOT, run_dir, args.device)
        print(json.dumps({"frozen_evidence_verified": True, "target_scope": manifest["target_scope_preflight"],
                          "input_stage": config["input_stage"], "head_stage": config["head_stage"],
                          "training": config["training"], "acceptance": config["acceptance"],
                          "run_dir": str(run_dir), "training_launched": False}, ensure_ascii=False, indent=2))
        return
    if args.command == "input":
        summary = run_stage(config, ROOT, "input", None, run_dir, args.device)
    elif args.command == "heads":
        cases = list(config["head_stage"]["cases"]) if args.case == "all" else [args.case]
        for case in cases:
            summary = run_stage(config, ROOT, "heads", case, run_dir, args.device)
    else:
        summary = summarize_controlled(run_dir)
    for row in summary["comparison"]:
        print(json.dumps(row, ensure_ascii=False))
    print(f"Pending jobs: {len(summary['pending'])}; results: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
