#!/usr/bin/env python3
"""Fixed-weight, distance-matched ROI/background permutation sensitivity.

This is an out-of-distribution engineering probe on previously inspected OOF
partitions. It does not fit, tune, relabel, establish biological causality, or
test automatic candidate discovery. ``check`` never reads weights/count bands;
``run`` uses the saved I3/H3 final endpoints and their exact frozen transforms.
"""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = _layout_root
sys.path.insert(0, str(ROOT))

import numpy as np

CLASSES = ("OPCID", "CHIN", "CHID")
MODELS = ("m2", "m2plus", "cnn_transformer")
REPEATS = (0, 1, 2)
PROTOCOL = "task1_fixed_endpoint_matched_morphology_probe_v1"
ATOL = 1e-5


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def signature(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"No rows for {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def append_owned_csv(path: Path, rows: list[dict], *, first: bool) -> None:
    """Only called for this invocation's exclusively created result directory."""
    require(bool(rows), f"No rows for {path}")
    with path.open("x" if first else "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if first:
            writer.writeheader()
        writer.writerows(rows)


def stable_rng(structure_id: str, replicate: str, scale: int, repeat: int):
    """No label, model, fold, endpoint seed or signal value enters this plan."""
    digest = hashlib.sha256(json.dumps([PROTOCOL, str(structure_id), str(replicate),
                                       int(scale), int(repeat)], separators=(",", ":")).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "little"))


def matched_shuffle_scale(channels: np.ndarray, structure_id: str, replicate: str,
                          scale: int, repeat: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Permute equal counts in each distance layer, each region separately.

    Upper and lower triangles use the same plan. Their values are permuted
    separately to preserve both multisets even if pooling has tiny rounding
    asymmetry. The diagonal, partly valid pixels and unselected pixels remain
    unchanged. Validity/mask values remain byte-for-byte unchanged, including
    fractional boundaries; a fullvalid signal pixel with fractional mask>=.5
    is eligible for the preset ROI engineering rule.
    """
    source = np.asarray(channels)
    require(source.ndim == 3 and source.shape[0] == 3 and source.shape[1] == source.shape[2],
            "Expected signal/validity/mask square channels")
    require(np.isfinite(source).all(), "Nonfinite input")
    signal, validity, mask = source
    require(np.allclose(source, source.transpose(0, 2, 1), atol=1e-6, rtol=1e-6),
            "Input channels must be symmetric before paired shuffling")
    require(np.all((validity >= 0) & (validity <= 1)) and np.all((mask >= 0) & (mask <= 1)),
            "Validity and target mask must be fractions in [0,1]")
    roi, background = source.copy(), source.copy()
    i, j = np.triu_indices(source.shape[1], k=1)
    full_valid = (validity[i, j] >= 1 - 1e-6) & (validity[j, i] >= 1 - 1e-6)
    roi_candidates = full_valid & (mask[i, j] >= .5) & (mask[j, i] >= .5)
    bg_candidates = full_valid & (mask[i, j] == 0) & (mask[j, i] == 0)
    rng = stable_rng(structure_id, replicate, scale, repeat)
    layers = []
    matched_available = matched_selected = 0
    for distance in range(1, source.shape[1]):
        a = np.flatnonzero(roi_candidates & ((j - i) == distance))
        b = np.flatnonzero(bg_candidates & ((j - i) == distance))
        k = min(len(a), len(b))
        matched_available += k
        selected = k if k >= 2 else 0
        layers.append({"distance": distance, "ROI_available_upper": len(a),
                       "background_available_upper": len(b), "matched_upper": k,
                       "permutable_upper_per_region": selected})
        if selected == 0:
            continue
        matched_selected += k
        a = rng.permutation(a)[:k]
        b = rng.permutation(b)[:k]
        permutation = rng.permutation(k)
        for destination, chosen in ((roi, a), (background, b)):
            u, v = i[chosen], j[chosen]
            destination[0, u, v] = signal[u[permutation], v[permutation]]
            destination[0, v, u] = signal[v[permutation], u[permutation]]
    roi_changed = roi[0] != signal
    bg_changed = background[0] != signal
    changed_roi = int(np.count_nonzero(roi_changed))
    changed_bg = int(np.count_nonzero(bg_changed))
    status = ("unavailable" if matched_selected == 0 else
              "matched_changed_both" if changed_roi and changed_bg else "matched_no_signal_change_in_one_or_both")
    support = {"structure_id": str(structure_id), "replicate": str(replicate), "scale": int(scale),
               "repeat": int(repeat), "status": status,
               "ROI_fullvalid_available_upper": int(roi_candidates.sum()),
               "background_fullvalid_available_upper": int(bg_candidates.sum()),
               "matched_available_upper_per_region": matched_available,
               "matched_permutable_upper_per_region": matched_selected,
               "ROI_changed_upper": int(roi_changed[i, j].sum()),
               "background_changed_upper": int(bg_changed[i, j].sum()),
               "ROI_changed_pixels": changed_roi, "background_changed_pixels": changed_bg,
               "fullvalid_upper": int(full_valid.sum()),
               "partial_valid_upper": int(((validity[i, j] > 0) & ~full_valid).sum()),
               "invalid_upper": int((validity[i, j] == 0).sum()),
               "ROI_mask_ge_half_upper": int((mask[i, j] >= .5).sum()),
               "ROI_mask_positive_upper": int((mask[i, j] > 0).sum()),
               "input_signal_symmetry_max_abs": float(np.max(np.abs(signal - signal.T))),
               "layers_json": json.dumps(layers, separators=(",", ":"))}
    return roi, background, support


def matched_shuffle_samples(samples: dict, repeat: int) -> tuple[dict, dict, list[dict]]:
    x = np.asarray(samples["x"])
    require(x.shape == (len(samples["y"]), 2, 3, 64, 64), "Expected frozen two-scale model64 inputs")
    require(repeat in REPEATS, "Use preregistered repeat seeds 0/1/2")
    roi_x, bg_x, support = x.copy(), x.copy(), []
    for row, (sid, wt) in enumerate(zip(samples["structure_id"], samples["replicate"])):
        for scale in range(2):
            roi_x[row, scale], bg_x[row, scale], item = matched_shuffle_scale(x[row, scale], sid, wt, scale, repeat)
            support.append(item)
    # Metadata and length side-channel are reused; no transform is fitted here.
    return {**samples, "x": roi_x}, {**samples, "x": bg_x}, support


def predict(model, samples: dict, device, batch_size: int = 32) -> np.ndarray:
    import torch
    require(batch_size == 32, "Fixed batch size is 32")
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(samples["x"]), batch_size):
            x = torch.from_numpy(np.ascontiguousarray(samples["x"][start:start + batch_size], dtype=np.float32)).to(device)
            length = torch.from_numpy(np.ascontiguousarray(samples["length"][start:start + batch_size], dtype=np.float32)).to(device)
            logits = model(x, length)
            require(logits.shape == (len(x), 3) and bool(torch.isfinite(logits).all()), "Invalid fixed-endpoint logits")
            probabilities.append(logits.softmax(dim=1).cpu().numpy().astype(np.float64))
    return np.concatenate(probabilities)


def verify_baseline(samples: dict, probabilities: np.ndarray, csv_path: Path) -> tuple[dict, dict]:
    """Require every unique structure's two-WT mean and argmax to reproduce."""
    from microcture.task1.evaluation import grouped_predictions
    grouped = grouped_predictions(samples, probabilities)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    ids = [row["structure_id"] for row in saved]
    require(len(ids) == len(set(ids)) and set(ids) == set(grouped["structure_id"]), "Baseline structure inventory differs")
    lookup = {row["structure_id"]: row for row in saved}
    differences, wt_differences = [], []
    for index, sid in enumerate(grouped["structure_id"]):
        row = lookup[str(sid)]
        require(int(row["y_true"]) == int(grouped["y"][index]) and int(row["group_id"]) == int(grouped["group_id"][index]),
                f"Baseline labels/group differ: {sid}")
        require(int(row["replicate_count"]) == 2 == int(grouped["replicate_count"][index]), f"Expected exactly two WT replicates: {sid}")
        actual = grouped["probabilities"][index]
        recorded = np.array([float(row[f"p_{label}"]) for label in CLASSES])
        require(np.isfinite(recorded).all() and np.allclose(actual, recorded, atol=ATOL, rtol=0),
                f"Fixed endpoint baseline probabilities differ beyond atol={ATOL}: {sid}; max_abs={np.max(np.abs(actual-recorded))}")
        require(int(row["y_pred"]) == int(recorded.argmax()) == int(actual.argmax()), f"Fixed endpoint baseline argmax differs: {sid}")
        saved_wt = json.loads(row["replicate_probabilities_json"])
        actual_wt = grouped["replicate_probabilities"][index]
        recorded_ids = [str(record["replicate"]) for record in saved_wt]
        actual_ids = [str(record["replicate"]) for record in actual_wt]
        require(len(recorded_ids) == len(set(recorded_ids)) == 2
                and set(recorded_ids) == set(actual_ids), f"WT replicate identity differs: {sid}")
        recorded_wt = {str(record["replicate"]): record for record in saved_wt}
        for record in actual_wt:
            original = recorded_wt[str(record["replicate"])]
            actual_values, original_values = np.asarray(record["probabilities"]), np.asarray(original["probabilities"])
            require(original_values.shape == (3,) and np.isfinite(original_values).all()
                    and np.allclose(actual_values, original_values, atol=ATOL, rtol=0),
                    f"Fixed endpoint WT baseline probabilities differ: {sid} WT{record['replicate']}")
            require(int(original["predicted"]) == int(original_values.argmax()) == int(actual_values.argmax()),
                    f"Fixed endpoint WT baseline argmax differs: {sid} WT{record['replicate']}")
            wt_differences.append(float(np.max(np.abs(actual_values - original_values))))
        differences.append(float(np.max(np.abs(actual - recorded))))
    return grouped, {"n_verified_structures": len(saved), "replicates_per_structure": 2,
                     "absolute_tolerance": ATOL, "relative_tolerance": 0,
                     "max_abs_probability_difference": max(differences),
                     "max_abs_WT_probability_difference": max(wt_differences),
                     "WT_identity_and_probabilities_match": True, "all_argmax_match": True}


def safe_output(run_dir: Path, output_dir: Path) -> Path:
    from microcture.task1_formal.runner import guard_output
    path = guard_output(ROOT, output_dir)
    run = run_dir.resolve()
    require(path != run and run not in path.parents and path not in run.parents,
            "Probe output must be independent from the formal scientific run")
    require(not path.exists(), f"Probe output already exists and is preserved; choose a fresh --output-dir: {path}")
    return path


def check_saved(run_dir: Path, model: str = "m2plus") -> tuple[dict, dict, list[dict], dict]:
    """Read-only metadata/source checks. Never open .pt/.npy or contact values."""
    from microcture.task1_formal.protocol import (
        _package_signature, _validate_cache_metadata, _validate_folds,
        job_directory, planned_jobs, project_path, sha256, source_signature, validate_protocol,
    )
    manifest = read_json(run_dir / "manifest.json")
    require(signature({k: v for k, v in manifest.items() if k != "experiment_signature"}) == manifest["experiment_signature"],
            "Formal manifest signature changed")
    config = validate_protocol(manifest["config"])
    require(project_path(ROOT, config["run_dir"]) == run_dir.resolve(), "Formal run path differs from saved config")
    require(source_signature(ROOT) == manifest["source_signature"], "Formal training source changed")
    for package, entrypoint in (("task1", "run_task1_comparison.py"), ("task1_controlled", "run_task1_controlled.py")):
        require(_package_signature(ROOT, package, entrypoint) == manifest["dependency_source_signature"][package],
                f"Frozen dependency source changed: {package}")
    metadata, cache_folds = _validate_cache_metadata(ROOT, project_path(ROOT, config["cache_dir"]), manifest["dataset_signature"])
    fold_signatures, fold_ids = _validate_folds(ROOT, config, metadata, cache_folds)
    require(fold_signatures == manifest["fold_signatures"] and fold_ids == manifest["fold_ids"], "Frozen fold metadata differs")
    require(len(metadata["structures"]) == manifest["n_structures"] == 344 and len(metadata["samples"]) == 2,
            "This probe requires all 344 structures and both WT replicates")
    require(model == "all" or model in MODELS, "Unknown probe backbone")
    jobs = planned_jobs(config, "main", model=model, condition="I3")
    provenance, runtimes, preprocessing_by_fold = [], [], {}
    for job in jobs:
        directory = job_directory(run_dir, job)
        index = read_json(directory / "ARTIFACTS.json")
        require(index["job"] == job and index["experiment_signature"] == manifest["experiment_signature"]
                and index["signature"] == signature(index["files"]), f"Artifact index changed: {directory}")
        names = set()
        for item in index["files"]:
            name = item["name"]
            require(Path(name).name == name and name not in names and name != "ARTIFACTS.json", "Unsafe artifact inventory")
            names.add(name)
            path = directory / name
            require(path.is_file() and not path.is_symlink() and path.stat().st_size == item["size_bytes"], f"Artifact missing/size changed: {path}")
            # A check is deliberately not full checkpoint SHA validation.
            if path.suffix in (".json", ".csv"):
                require(sha256(path) == item["sha256"], f"Scientific metadata changed: {path}")
        require({p.name for p in directory.iterdir()} == names | {"ARTIFACTS.json"}, "Artifact inventory changed")
        result, protocol, endpoint = (read_json(directory / name) for name in ("metrics.json", "protocol.json", "endpoint.json"))
        require(result["status"] == "completed" and result["job"] == protocol["job"] == endpoint["job"] == job,
                f"Incomplete or mismatched fixed job: {directory}")
        for field in ("experiment_signature", "source_signature", "dataset_signature"):
            require(result[field] == manifest[field], f"Saved provenance differs: {field}")
        preprocessing = protocol["preprocessing"]
        require(preprocessing["signature"] == result["preprocessing_signature"]
                and signature({k: v for k, v in preprocessing.items() if k != "signature"}) == preprocessing["signature"],
                "Saved preprocessing signature differs")
        previous = preprocessing_by_fold.setdefault(job["fold"], preprocessing)
        require(previous == preprocessing, "Models/seeds must reuse the same frozen fold transforms")
        checkpoint_record = next(item for item in index["files"] if item["name"] == "final.pt")
        require(endpoint["checkpoint"] == "final.pt" and endpoint["training_and_selection_complete"] is True
                and endpoint["checkpoint_sha256"] == result["checkpoint_sha256"] == checkpoint_record["sha256"],
                "Fixed checkpoint provenance differs")
        provenance.append({"job": job, "ARTIFACTS_sha256": sha256(directory / "ARTIFACTS.json"),
                           "checkpoint_sha256_recorded": checkpoint_record["sha256"],
                           "preprocessing_signature": preprocessing["signature"]})
        runtimes.append(float(result["runtime_seconds"]))
    report = {"protocol": PROTOCOL, "status": "metadata_check_passed", "source_run": str(run_dir.resolve()),
              "source_experiment_signature": manifest["experiment_signature"], "formal_source_signature": manifest["source_signature"],
              "dataset_signature": manifest["dataset_signature"], "models": sorted({job["model"] for job in jobs}),
              "jobs": len(jobs), "unique_structures": 344, "frozen_folds": config["folds"],
              "endpoint_seeds": config["seeds"], "shuffle_repeat_seeds": list(REPEATS),
              "preprocessing": "Saved inner-training-only transform; no fitting or OE refit",
              "checkpoint_content_SHA256_checked": False, "count_bands_or_weights_opened": False,
              "batch_size": 32, "forward_conditions_per_endpoint": 7,
              "total_replicate_forward_examples": 344 * 2 * 3 * 7 * len({job["model"] for job in jobs}),
              "saved_full_fit_plus_evaluation_job_seconds_median": float(np.median(runtimes)),
              "runtime_estimate": "Small GPU roughly 1–3 minutes/model, uncertain: saved durations include 40 training epochs and do not isolate forward time; includes SHA checks/input extraction overhead.",
              "scope": "Metadata-only check; run additionally hashes every indexed artifact/checkpoint and verifies baseline forward before perturbation.",
              "job_provenance": provenance}
    return manifest, metadata, jobs, report


def structure_support(support: list[dict], sid: str) -> dict:
    items = [row for row in support if row["structure_id"] == sid]
    require(len(items) == 4, f"Expected both WT replicates × two scales: {sid}")
    fields = ("matched_available_upper_per_region", "matched_permutable_upper_per_region", "ROI_changed_pixels",
              "background_changed_pixels", "ROI_fullvalid_available_upper", "background_fullvalid_available_upper")
    result = {field: sum(row[field] for row in items) for field in fields}
    available = result["matched_permutable_upper_per_region"] > 0
    effective = available and result["ROI_changed_pixels"] > 0 and result["background_changed_pixels"] > 0
    result.update({"matched_available": int(available), "matched_effective_both": int(effective),
                   "support_status": "unavailable" if not available else "matched_changed_both" if effective else "matched_no_signal_change_in_one_or_both",
                   "matched_WT_scale_count": sum(row["matched_permutable_upper_per_region"] > 0 for row in items)})
    return result


def prediction_rows(job: dict, baseline: dict, roi: dict, background: dict,
                    support: list[dict], repeat: int, metadata: dict) -> list[dict]:
    lookup = {row["structure_id"]: row for row in metadata["structures"]}
    require(np.array_equal(baseline["structure_id"], roi["structure_id"])
            and np.array_equal(baseline["structure_id"], background["structure_id"]), "Perturbed structure order differs")
    rows = []
    for index, sid in enumerate(baseline["structure_id"]):
        sid, true = str(sid), int(baseline["y"][index])
        item = lookup[sid]
        evidence = structure_support(support, sid)
        row = {"model": job["model"], "fold": job["fold"], "seed": job["seed"], "repeat": repeat,
               "structure_id": sid, "group_id": int(baseline["group_id"][index]), "y_true": true,
               "annotation_type": CLASSES[true], "length_bp": int(item["end"]) - int(item["start"]),
               "replicate_count": int(baseline["replicate_count"][index]), **evidence}
        for name, grouped in (("baseline", baseline), ("roi", roi), ("background", background)):
            row[f"{name}_prediction"] = int(grouped["predicted"][index])
            row[f"{name}_trueclass_probability"] = float(grouped["probabilities"][index, true])
            for cls, label in enumerate(CLASSES):
                row[f"{name}_p_{label}"] = float(grouped["probabilities"][index, cls])
        row["baseline_correct"] = int(row["baseline_prediction"] == true)
        # An unchanged unavailable input is not evidence of zero sensitivity.
        for name in ("roi", "background"):
            row[f"{name}_minus_baseline_trueclass_probability"] = (
                row[f"{name}_trueclass_probability"] - row["baseline_trueclass_probability"]
                if evidence["matched_effective_both"] else "")
        rows.append(row)
    return rows


def paired_summary(rows: list[dict]) -> dict:
    result = {"n_unique_structures": len(rows), "class_support": dict(Counter(row["annotation_type"] for row in rows))}
    if not rows:
        result["status"] = "unavailable"
        return result
    for condition in ("roi", "background"):
        delta = np.array([row[f"{condition}_trueclass_probability"] - row["baseline_trueclass_probability"] for row in rows])
        result[condition] = {"trueclass_probability_change_mean": float(delta.mean()),
                             "trueclass_probability_change_median": float(np.median(delta)),
                             "probability_decreased_count": int((delta < 0).sum()),
                             "prediction_changed_count": sum(row[f"{condition}_prediction"] != row["baseline_prediction"] for row in rows),
                             "correct_to_wrong_count": sum(row["baseline_correct"] and row[f"{condition}_prediction"] != row["y_true"] for row in rows),
                             "wrong_to_correct_count": sum(not row["baseline_correct"] and row[f"{condition}_prediction"] == row["y_true"] for row in rows)}
    differential = np.array([row["roi_trueclass_probability"] - row["background_trueclass_probability"] for row in rows])
    result["ROI_minus_background_paired_probability_change"] = {"mean": float(differential.mean()), "median": float(np.median(differential))}
    return result


def score_rows(rows: list[dict]) -> dict:
    from microcture.task1.evaluation import metrics
    if not rows:
        return {"n_unique_structures": 0, "class_support": {label: 0 for label in CLASSES}, "status": "unavailable"}
    scores = {name: metrics(np.array([row["y_true"] for row in rows], dtype=np.int64),
                           np.array([[row[f"{name}_p_{label}"] for label in CLASSES] for row in rows]))
              for name in ("baseline", "roi", "background")}
    degradation = {}
    for name in ("roi", "background"):
        degradation[name] = {field: scores["baseline"][field] - scores[name][field]
                             for field in ("macro_f1", "accuracy", "balanced_accuracy")}
        degradation[name]["per_class_f1"] = {label: scores["baseline"]["per_class"][label]["f1"] - scores[name]["per_class"][label]["f1"] for label in CLASSES}
    return {"n_unique_structures": len(rows), "class_support": {label: sum(row["annotation_type"] == label for row in rows) for label in CLASSES},
            "metrics": scores, "metric_degradation_baseline_minus_perturbed": degradation}


def summarize(rows: list[dict], expected_ids: set[str]) -> dict:
    comparisons = []
    for model in sorted({row["model"] for row in rows}):
        for seed in (42, 43, 44):
            for repeat in REPEATS:
                chosen = [row for row in rows if row["model"] == model and row["seed"] == seed and row["repeat"] == repeat]
                require(len(chosen) == len(expected_ids) and {row["structure_id"] for row in chosen} == expected_ids,
                        "Each endpoint seed/repeat must retain complete unique-structure OOF coverage")
                matched = [row for row in chosen if row["matched_effective_both"]]
                comparisons.append({"model": model, "seed": seed, "repeat": repeat,
                                    "full_OOF_includes_unavailable_unchanged_cases": score_rows(chosen),
                                    "matched_subset_both_conditions_actually_changed": score_rows(matched),
                                    "matched_geometry_available_count": sum(row["matched_available"] for row in chosen),
                                    "unavailable_count": sum(not row["matched_available"] for row in chosen),
                                    "matched_but_unchanged_in_one_or_both_count": sum(row["matched_available"] and not row["matched_effective_both"] for row in chosen),
                                    "paired_matched_effects": {"all": paired_summary(matched),
                                        "baseline_correct": paired_summary([row for row in matched if row["baseline_correct"]]),
                                        "baseline_wrong": paired_summary([row for row in matched if not row["baseline_correct"]])},
                                    "matched_metric_degradation_by_baseline_correctness": {
                                        "baseline_correct": score_rows([row for row in matched if row["baseline_correct"]]),
                                        "baseline_wrong": score_rows([row for row in matched if not row["baseline_correct"]])}})
    return {"comparisons": comparisons,
            "statistical_unit": "Unique structure within each original OOF seed/repeat. Seeds/repeats/WT/scales are not extra independent samples.",
            "matched_subset_definition": "At least one distance layer has k>=2, and actual changed signal pixels >0 for both ROI and background across the two WT replicates/two scales. Counts and classes disclosed per seed/repeat.",
            "unavailable_semantics": "Unmatched and unchanged cases stay in full OOF tables; they have blank paired sensitivity deltas and never enter matched effect summaries."}


def run_probe(run_dir: Path, output_dir: Path, model: str = "m2plus", requested_device: str = "auto") -> dict:
    import torch
    from threadpoolctl import threadpool_limits
    from microcture.task1.data import load_dataset
    from microcture.task1.evaluation import grouped_predictions
    from microcture.task1.training import resolve_device
    from microcture.task1_formal.data import prepare_outer
    from microcture.task1_formal.protocol import job_directory, project_path, sha256
    from microcture.task1_formal.runner import assert_resume, frozen_fold, restore_endpoint

    output = safe_output(run_dir, output_dir)
    manifest, metadata, jobs, preflight = check_saved(run_dir, model)
    config = manifest["config"]
    # All original scientific files and checkpoint contents pass before input extraction.
    for job in jobs:
        result = assert_resume(job_directory(run_dir, job), job, manifest)
        require(result is not None and result["status"] == "completed", "Fixed completed endpoint required")
    device = resolve_device(requested_device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    probe_manifest = {"protocol": PROTOCOL, "preflight": preflight, "script_sha256": sha256(Path(__file__)),
                      "device": str(device), "batch_size": 32, "shuffle_repeat_seeds": list(REPEATS),
                      "source_run_manifest_sha256": sha256(run_dir / "manifest.json"),
                      "checkpoint_content_SHA256_checked": True,
                      "region_rule": "ROI mask>=0.5 (preset engineering rule, not a biological threshold); background mask==0; both fullvalid >=1-1e-6; i<j, same abs(i-j) on the model64 pixel grid, equal k=min(ROI,bg), only k>=2; synchronized lower triangle",
                      "distance_preservation_limit": "Preserves the value multiset per model64 diagonal-offset layer/region. It does not preserve every original 10bp/100bp genomic-distance layer: coarse-window model pixels pool 4x4 native 100bp bins.",
                      "permutation_rule": "Single SHA256(structure ID,WT,scale,repeat) RNG stream selects matched sets and same permutation for both regions; independent of labels/model/endpoint seed",
                      "fixed_transforms": "Reuse prepare_outer with saved protocol preprocessing; no fit, OE refit, optimizer, gradients, training, checkpoint selection or outer tuning",
                      "interpretation": "Out-of-distribution permutation sensitivity on supplied candidate intervals; not biological causality, reliable morphology recognition, new holdout evidence, or automatic discovery"}
    probe_manifest["probe_signature"] = signature(probe_manifest)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", probe_manifest)
    started = time.perf_counter()
    all_rows, all_support, baseline_checks = [], [], []
    try:
        dataset = load_dataset(project_path(ROOT, config["cache_dir"]))
        with threadpool_limits(limits=4):
            for fold in config["folds"]:
                fold_jobs = [job for job in jobs if job["fold"] == fold]
                fold_record = frozen_fold(ROOT, config, manifest, fold)
                preprocessing = read_json(job_directory(run_dir, fold_jobs[0]) / "protocol.json")["preprocessing"]
                outer = prepare_outer(dataset, fold_record, preprocessing)
                for job in fold_jobs:
                    directory = job_directory(run_dir, job)
                    checkpoint = directory / "final.pt"
                    endpoint_hash = sha256(checkpoint)
                    require(endpoint_hash == read_json(directory / "endpoint.json")["checkpoint_sha256"], "Checkpoint changed before restoration")
                    neural = restore_endpoint(job, config, checkpoint, device)
                    try:
                        baseline, check = verify_baseline(outer, predict(neural, outer, device), directory / "outer_predictions.csv")
                        baseline_checks.append({"job": job, "checkpoint_sha256": endpoint_hash, **check})
                        print(f"Baseline reproduced: {job['model']} seed{job['seed']} fold{fold}, n={check['n_verified_structures']}, max_abs={check['max_abs_probability_difference']:.3g}", flush=True)
                        for repeat in REPEATS:
                            roi_samples, bg_samples, support = matched_shuffle_samples(outer, repeat)
                            roi = grouped_predictions(outer, predict(neural, roi_samples, device))
                            background = grouped_predictions(outer, predict(neural, bg_samples, device))
                            all_rows.extend(prediction_rows(job, baseline, roi, background, support, repeat, metadata))
                            # Plans/support do not vary with endpoint seed or model.
                            if job == fold_jobs[0]:
                                all_support.extend({"fold": fold, **row} for row in support)
                            del roi_samples, bg_samples
                        require(sha256(checkpoint) == endpoint_hash, "Fixed endpoint changed during inference")
                    finally:
                        del neural
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    # Retain partial numerical evidence without duplicate exports.
                    job_rows = [row for row in all_rows if row["model"] == job["model"] and row["seed"] == job["seed"] and row["fold"] == fold]
                    append_owned_csv(output / "predictions.csv", job_rows, first=len(baseline_checks) == 1)
                    if job == fold_jobs[0]:
                        fold_support = [row for row in all_support if row["fold"] == fold]
                        append_owned_csv(output / "matched_support.csv", fold_support, first=fold == config["folds"][0])
        expected_ids = {row["structure_id"] for row in metadata["structures"]}
        summary = {"protocol": PROTOCOL, "status": "completed", "probe_signature": probe_manifest["probe_signature"],
                   "source_run": str(run_dir.resolve()), "unique_structures": len(expected_ids), "models": preflight["models"],
                   "baseline_checks": baseline_checks, "runtime_seconds": time.perf_counter() - started,
                   "interpretation": probe_manifest["interpretation"], **summarize(all_rows, expected_ids)}
        write_json(output / "summary.json", summary)
        records = [{"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
                   for path in sorted(output.iterdir()) if path.is_file()]
        write_json(output / "ARTIFACTS.json", {"probe_signature": probe_manifest["probe_signature"],
                                               "files": records, "signature": signature(records)})
        return summary
    except Exception as error:
        write_json(output / "failure.json", {"status": "failed_preserved_no_retry_in_place", "error": str(error),
                                            "probe_signature": probe_manifest["probe_signature"],
                                            "baseline_checks_completed": baseline_checks,
                                            "runtime_seconds": time.perf_counter() - started})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/task1_formal_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/task1_morphology_probe_v1")
    parser.add_argument("--model", choices=(*MODELS, "all"), default="m2plus")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("command", choices=("check", "run"))
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    safe_output(run_dir, args.output_dir)
    if args.command == "check":
        _, _, _, report = check_saved(run_dir, args.model)
    else:
        summary = run_probe(run_dir, args.output_dir, args.model, args.device)
        report = {"status": summary["status"], "unique_structures": summary["unique_structures"],
                  "runtime_seconds": summary["runtime_seconds"], "probe_signature": summary["probe_signature"],
                  "output_dir": str(args.output_dir.resolve())}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
