#!/usr/bin/env python3
"""Predict supplied candidate intervals or run the packaged, raw-data-free tensor demo."""
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


def main():
    from microcture.tasks.task1.data import load_dataset
    from microcture.tasks.task1.training import resolve_device
    from microcture.tasks.task1.prediction import guard_prediction_output, load_checkpoint, load_input_npz, predict_samples, prediction_rows, prepare_candidates, read_intervals, save_predictions
    from microcture.tasks.task1.protocol import json_signature, project_path, sha256
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=resolve(ROOT, "submission/artifacts/task1/m2plus/seed_42/fold_0/final.pt"))
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input-npz", type=Path, help="Preprocessed I3 tensors, including structure_id and replicate")
    inputs.add_argument("--intervals", type=Path, help="CSV: structure_id,start,end[,center]; requires --cache")
    parser.add_argument("--cache", type=Path, help="Read-only count cache prepared by scripts/task1/prepare_task1.py")
    parser.add_argument("--fold-protocol", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, help="Optional relocated source-provenance root")
    parser.add_argument("--data-root", type=Path, help="Optional raw matrix/source root")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, help="Fresh CSV under results/task1/<experiment>/<version>/; retained evidence is refused; omitted: print JSON only")
    parser.add_argument("--verify-demo", action="store_true", help="Compare with packaged expected probabilities; not a performance evaluation")
    args = parser.parse_args()
    output = guard_prediction_output(ROOT, args.output) if args.output is not None else None
    import torch
    torch.set_num_threads(4)
    device = resolve_device(args.device)
    checkpoint = project_path(ROOT, args.checkpoint)
    model, saved = load_checkpoint(checkpoint, device)
    fold_path = project_path(ROOT, args.fold_protocol or f"submission/task1/formal/folds/fold_{saved['job']['fold']}.json")
    if args.intervals is not None:
        if args.cache is None:
            parser.error("--intervals requires --cache")
        package = json.loads(fold_path.read_text(encoding="utf-8"))
        preprocessing = package.get("preprocessing")
        frozen = package.get("frozen_fold", package)
        if preprocessing is None or frozen["fold"] != saved["job"]["fold"]:
            raise ValueError("Supply the checkpoint's matching fold protocol including saved length scaler")
        dataset = load_dataset(project_path(ROOT, args.cache), source_root=args.source_root, data_root=args.data_root)
        candidates = read_intervals(project_path(ROOT, args.intervals), int(dataset["metadata"]["chrom_length"]))
        samples = prepare_candidates(dataset, candidates, preprocessing)
    else:
        input_path = project_path(ROOT, args.input_npz or "submission/demo/task1/input.npz")
        samples = load_input_npz(input_path)
        if "fold" in samples and int(samples["fold"]) != saved["job"]["fold"]:
            raise ValueError("Input tensors and checkpoint use different folds")
        if "preprocessing_signature" in samples:
            package = json.loads(fold_path.read_text(encoding="utf-8"))
            if str(samples["preprocessing_signature"]) != package["preprocessing"]["signature"]:
                raise ValueError("Input tensors and checkpoint fold use different transformations")
    probabilities = predict_samples(model, samples, device, args.batch_size)
    rows = prediction_rows(samples, probabilities)
    result = {"status": "predicted", "checkpoint_sha256": sha256(checkpoint),
              "scope": "classification given supplied target intervals; demonstration is not performance certification",
              "structures": rows}
    if args.verify_demo:
        if args.intervals is not None:
            raise ValueError("--verify-demo is only for the packaged tensor example")
        import numpy as np
        expected = json.loads(resolve(ROOT, "submission/demo/task1/expected.json").read_text(encoding="utf-8"))
        if sha256(input_path) != expected["input_npz_sha256"] or sha256(checkpoint) != expected["checkpoint_sha256"]:
            raise ValueError("Demo input/checkpoint does not match the registered example")
        tolerance = {"rtol": 1e-5, "atol": 1e-6} if device.type == "cpu" else expected["historical_oof_cross_device_tolerance"]
        np.testing.assert_allclose(probabilities, expected["replicate_probabilities"], **tolerance)
        result["demo_probability_tolerance"] = tolerance
        if [str(value) for value in samples["structure_id"]] != expected["replicate_structure_ids"]:
            raise ValueError("Demo structure order differs")
        result["demo_verified"] = True
    if output is not None:
        save_predictions(output, rows, root=ROOT)
        result["output"] = str(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
