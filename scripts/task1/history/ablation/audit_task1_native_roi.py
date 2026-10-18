#!/usr/bin/env python3
"""Read-only native90 scientific review; requires every fit before reading scores.

No source image, raw contact pixel, fitting or inference is read. Shared G0
checkpoints are deserialized on CPU only to test exact replay of old R1/G0.
Only the requested report directory is written. Intervals describe saved OOF
predictions conditional on this adaptively reused developmental dataset.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import importlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from microcture.task1_controlled.runner import flat_history_row
from microcture.task1_formal.evaluation import _guard_run, _identity, _fold_ids, _neural_summary, _csv
from microcture.task1_formal.protocol import json_signature, sha256
from microcture.task1_formal.runner import verify_artifact_index
from scripts.audit_task1_readout_results import scores as independent_scores, compare_scores

ARMS = (("C0", "I3"), ("C1", "I3"), ("C0", "G0"), ("C0", "V0"), ("C0", "P0"), ("C1", "P0"))
SEEDS = (42, 43, 44)
CLASSES = ("OPCID", "CHIN", "CHID")
PAIRS = (("C1/I3", "C0/I3"), ("C1/P0", "C0/P0"),
         ("C0/I3", "C0/G0"), ("C1/I3", "C0/G0"),
         ("C0/I3", "C0/V0"), ("C1/I3", "C0/V0"),
         ("C0/I3", "C0/P0"), ("C1/I3", "C1/P0"))
PROTOCOL_MODULES = {
    "task1_matched_native_roi_detail_control_v1": "microcture.task1_native_roi.experiment",
    "task1_matched_native_roi_detail_control_v2": "microcture.task1_native_roi_v2.experiment",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def safe_report(report: Path, run: Path):
    require(not any(p.is_symlink() for p in (report, *report.parents)), "Report path cannot use symlinks")
    report = report.resolve()
    require((ROOT / "reports") in report.parents and report != ROOT / "reports", "Report must be a separate project reports/ directory")
    require(not any((p / "ARTIFACTS.json").exists() or (p / "FROZEN.json").exists() for p in (report, *report.parents)),
            "Report would overlap sealed evidence")
    require(report != run and run not in report.parents, "Report cannot overwrite scientific results")
    return report


def scores_from_confusion(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support, predicted = cm.sum(-1), cm.sum(-2)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * tp, predicted + support, out=np.zeros_like(tp), where=predicted + support > 0)
    return {"macro_f1": f1.mean(-1), "CHID_f1": f1[..., 2], "CHID_precision": precision[..., 2],
            "CHID_recall": recall[..., 2], "balanced_accuracy": recall.mean(-1)}, support, predicted


def group_bootstrap(oof, identity, repeats=2000, rng_seed=20261014):
    """Same spatial-group multiplicities across every arm and endpoint seed."""
    require(repeats > 0, "Bootstrap repetition count must be positive")
    groups = sorted({r["group_id"] for r in identity.values()})
    group_index = {g: i for i, g in enumerate(groups)}
    confusion = np.zeros((len(ARMS), len(SEEDS), len(groups), 3, 3), dtype=np.int64)
    for ai, arm in enumerate(ARMS):
        for si, seed in enumerate(SEEDS):
            source = oof[(*arm, seed)]["rows"]
            require(len(source) == len(identity) and {r["structure_id"] for r in source} == set(identity), "Bootstrap requires complete matched OOF")
            for row in source:
                confusion[ai, si, group_index[int(row["group_id"])] , int(row["y_true"]), int(row["y_pred"])] += 1
    weights = np.random.default_rng(rng_seed).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=repeats)
    raw, support, predicted = scores_from_confusion(np.einsum("bg,asgij->basij", weights, confusion))
    observed, _, _ = scores_from_confusion(confusion.sum(2))
    per_arm, distributions = {}, {}
    for ai, (alignment, condition) in enumerate(ARMS):
        name = f"{alignment}/{condition}"
        per_arm[name], distributions[name] = {}, {}
        for metric in raw:
            values, seed_values = raw[metric][:, ai].mean(-1), observed[metric][ai]
            distributions[name][metric] = values
            per_arm[name][metric] = {"point_mean": float(seed_values.mean()), "seed_values": seed_values.tolist(),
                "seed_sample_sd": float(seed_values.std(ddof=1)), "ci95": np.quantile(values, [.025, .975]).tolist()}
    pairs = {}
    for first, second in PAIRS:
        pair = {}
        for metric in raw:
            seed_delta = np.array(per_arm[first][metric]["seed_values"]) - np.array(per_arm[second][metric]["seed_values"])
            values = distributions[first][metric] - distributions[second][metric]
            pair[metric] = {"point_mean": float(seed_delta.mean()), "seed_deltas": seed_delta.tolist(),
                            "ci95": np.quantile(values, [.025, .975]).tolist()}
        pairs[f"{first}-{second}"] = pair
    difference = {}
    for metric in raw:
        values = ((distributions["C1/I3"][metric] - distributions["C1/P0"][metric])
                  - (distributions["C0/I3"][metric] - distributions["C0/P0"][metric]))
        seeds = ((np.array(per_arm["C1/I3"][metric]["seed_values"]) - np.array(per_arm["C1/P0"][metric]["seed_values"]))
                 - (np.array(per_arm["C0/I3"][metric]["seed_values"]) - np.array(per_arm["C0/P0"][metric]["seed_values"])))
        difference[metric] = {"point_mean": float(seeds.mean()), "seed_deltas": seeds.tolist(),
                              "ci95": np.quantile(values, [.025, .975]).tolist()}
    return {"scope": "Conditional saved-OOF uncertainty; no refitting or correction for researcher adaptation",
        "B": repeats, "rng_seed": rng_seed, "group_count": len(groups),
        "unit": "Whole spatial groups; every structure, seed and arm uses the same sampled multiplicity",
        "seed_aggregation": "Score separately per endpoint seed, then average the three metrics",
        "interval": "Unadjusted percentile95; multiple contrasts are descriptive; zero-support draws retained using fixed three-class zero-division convention",
        "zero_CHID_support_bootstraps": int(np.sum(np.any(support[..., 2] == 0, axis=(1, 2)))),
        "zero_CHID_prediction_bootstraps": int(np.sum(np.any(predicted[..., 2] == 0, axis=(1, 2)))),
        "arms": per_arm, "pairs": pairs, "native_minus_coarse_beyond_profile": difference}


def verify_normalizer(preprocessing, frozen, metadata):
    base, shared = preprocessing["base_preprocessing"], preprocessing["native_expected"]
    binding = shared["binding"]
    require(base["signal_preprocessing"] == frozen["preprocessing"], "Frozen100bp context OE/scalers changed")
    require(base["train_ids"] == frozen["train_ids"] and base["validation_ids"] == frozen["validation_ids"], "Base preprocessing IDs changed")
    for key in ("train_ids", "validation_ids"):
        require(binding[key] == frozen[key], f"Native expected {key} differs")
    require(binding["outer_ids"] == frozen["test_ids"] and binding["frozen_fold_signature"] == json_signature(frozen), "Native expected outer/fold binding differs")
    require(shared["signature"] == preprocessing["native_expected_signature"] == json_signature({k: v for k, v in shared.items() if k != "signature"}), "Native expected signature differs")
    require(binding["min_distance_bins"] == 20 and binding["max_distance_bins"] == 2559 and
            binding["raw_resolution_bp"] == 10, "Native physical support changed")
    # Reconstruct opportunities from geometry alone: no count band/raw pixels.
    train_mask = np.zeros(metadata["n_bins"], dtype=bool)
    indices = {r["structure_id"]: i for i, r in enumerate(metadata["structures"])}
    for sid in frozen["train_ids"]:
        center = metadata["center_bin"][indices[sid]]
        train_mask[(center + np.arange(-128, 128)) % len(train_mask)] = True
    train_mask[(np.arange(len(train_mask)) + 1) * 100 > metadata["chrom_length"]] = False
    native_mask = np.repeat(train_mask, 10)[:(metadata["chrom_length"] + 9) // 10]
    spectrum = np.fft.rfft(native_mask.astype(float))
    opportunity = np.rint(np.fft.irfft(spectrum * spectrum.conj(), n=len(native_mask))[:2560]).astype(np.int64)
    require(shared["unique_training_native_bins"] == int(native_mask.sum()), "Native fit training-union size differs")
    require({r["replicate"] for r in shared["records"]} == {1, 2}, "Native expected needs both WT curves")
    for record in shared["records"]:
        amount = np.asarray(record["zero_inclusive_pair_opportunities"], dtype=float)
        counts = np.asarray(record["stored_count_sum_by_distance"], dtype=float)
        expected = np.asarray(record["expected"], dtype=float)
        require(np.array_equal(amount, opportunity) and counts.shape == expected.shape == (2560,), "Native opportunities do not match train-only geometry")
        recomputed = np.divide(counts, amount, out=np.zeros_like(counts), where=amount > 0)
        require(np.array_equal(expected, recomputed) and np.isfinite(expected).all() and (expected[20:] > 0).all(), "E10 differs from zero-inclusive count/opportunity quotient")
        require(float(counts.sum()) == record["reader"]["eligible_counts_sum"] and record["reader"]["circular_eligible_pairs"] == 0,
                "Owned count total/circular guard differs")
    scaler = preprocessing["native_first_signal_scaler"]
    require(scaler["train_ids"] == frozen["train_ids"] and scaler["fit_scope"] == "frozen inner-training replicate images only", "Native first scaler has wrong fit IDs/scope")
    return shared["signature"]


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if k != "epoch_seconds"}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def verify_g0_replay(directory, job, config):
    """G0 zeros both signal/validity scales; geometry equals historical R1/G0."""
    import torch
    old_run = ROOT / config["roi_run_dir"]
    old = old_run / "runs/R1/G0/m2plus" / f"seed_{job['seed']}" / f"fold_{job['fold']}"
    previous = read_json(old / "metrics.json")
    verify_artifact_index(old, previous["job"], read_json(old_run / "manifest.json"))
    current = read_json(directory / "metrics.json")
    first = torch.load(directory / "final.pt", map_location="cpu", weights_only=True)["state_dict"]
    second = torch.load(old / "final.pt", map_location="cpu", weights_only=True)["state_dict"]
    checks = {
        "all_weight_tensors_exact": set(first) == set(second) and all(torch.equal(first[k], second[k]) for k in first),
        "numeric_history_exact": without_timing(read_json(directory / "history.json")) == without_timing(read_json(old / "history.json")),
        "outer_CSV_bytes_exact": (directory / "outer_predictions.csv").read_bytes() == (old / "outer_predictions.csv").read_bytes(),
        "validation_CSV_bytes_exact": (directory / "validation_predictions.csv").read_bytes() == (old / "validation_predictions.csv").read_bytes(),
    }
    for field in ("initial_state_sha256", "common_initial_state_sha256", "minibatch_order_sha256", "class_weights", "acceptance"):
        checks[f"{field}_exact"] = current[field] == previous[field]
    require(all(checks.values()), f"Shared G0 failed exact R1/G0 replay; differences require investigation, not omission: {job} {checks}")
    return {"seed": job["seed"], "fold": job["fold"], "checks": checks}


def build_report(run: Path, report: Path):
    _guard_run(run)
    run = run.resolve()
    # This gate comes before loading any per-job scores or predictions.
    summary = read_json(run / "summary.json")
    require(summary["status"] == "complete" and summary["completed_jobs"] == 90 and summary["complete_oof"] == 18,
            "Review waits for all90 completed fits and18 complete OOFs; partial scores are not reviewed")
    require(not any(summary[k] for k in ("pending_jobs", "invalid_jobs", "numerical_failure_jobs", "execution_failure_jobs")), "Every registered fit must remain accounted for")
    manifest = read_json(run / "manifest.json")
    require(manifest["protocol"] in PROTOCOL_MODULES, "Unsupported native protocol version")
    implementation = importlib.import_module(PROTOCOL_MODULES[manifest["protocol"]])
    current, metadata = implementation.prepare_manifest(ROOT, manifest["config"], manifest["requested_device"])
    require(manifest == current, "Saved native source/config/dependency/environment differs")
    for field in ("experiment_signature", "source_signature", "dataset_signature"):
        require(summary[field] == manifest[field], f"Summary {field} differs")
    require(len(metadata["structures"]) == 344 and len(set(metadata["group_id"])) == 49, "Review requires registered344 structures/49 spatial groups")
    require(len(list((run / "runs").rglob("ARTIFACTS.json"))) == 90 and not (run / "failed_staging").exists()
            and not implementation.unpublished_staging(ROOT, run), "Missing/extra/unpublished scientific jobs exist")
    config, identity = manifest["config"], _identity(metadata)
    folds, jobs = _fold_ids(manifest, identity, run), implementation.planned_jobs(manifest["config"])
    records, inventory, paired, g0_replay = [], [], defaultdict(list), []
    preprocessing_by_fold, norm_by_fold, facts_by_fold, checked_norm = {}, {}, {}, set()
    learning_jobs = []
    for job in jobs:
        record = implementation._load_saved(run, job, manifest, identity, folds)
        require(record["status"] == "completed", "A registered scientific fit is missing/failed; do not silently omit it")
        records.append(record)
        directory, value, history = implementation.job_directory(run, job), record["metrics"], record["history"]
        if job["alignment"] == "C0" and job["condition"] == "G0":
            g0_replay.append(verify_g0_replay(directory, job, config))
        index = read_json(directory / "ARTIFACTS.json")
        inventory.extend({"path": str((directory / item["name"]).relative_to(run)), **{k: item[k] for k in ("size_bytes", "sha256")}} for item in index["files"])
        pp = read_json(directory / "protocol.json")["preprocessing"]
        frozen = read_json(ROOT / config["frozen_run_dir"] / "folds" / f"fold_{job['fold']}.json")
        key = (job["alignment"], job["fold"])
        if key not in checked_norm:
            signature = verify_normalizer(pp, frozen, metadata)
            checked_norm.add(key)
            if job["fold"] in norm_by_fold:
                require(norm_by_fold[job["fold"]] == signature, "C0/C1 do not share the same E10 fit")
            norm_by_fold[job["fold"]] = signature
        require(preprocessing_by_fold.setdefault(key, pp) == pp, "Condition/seed refitted preprocessing")
        init = read_json(directory / "initialization.json")
        require(init["job"] == job and init["seed"] == job["seed"], "Initialization job/seed differs")
        for field in ("initial_state_sha256", "common_initial_state_sha256"):
            require(init[field] == value[field], "Initialization fingerprint differs from sealed metric")
        paired[job["fold"], job["seed"]].append(value)
        csv_history = rows(directory / "history.csv")
        require(len(csv_history) == 41, "History CSV must contain epochs0 through40")
        counts = np.bincount([identity[sid]["y"] for sid in frozen["train_ids"]], minlength=3)
        weights = counts.sum() / (3 * counts)
        require(np.allclose(value["class_weights"], weights, atol=1e-12, rtol=0), "Class weights differ from frozen inner IDs")
        spec = value["parameter_spec"]
        require(spec["parameters"] == spec["trainable_parameters"] == 110437 and spec["readout_variant"] == "O0"
                and spec["head_version"] == "H3", "Frozen model/head/parameter budget changed")
        for original, saved in zip(history, csv_history):
            flat = flat_history_row(original)
            require(set(flat) == set(saved) and all(saved[k] == ("" if v is None else str(v)) for k, v in flat.items()), "History CSV/JSON differs")
            for view in ("train", "validation"):
                diag = original[f"{view}_diagnostics"]
                support = np.bincount([identity[sid]["y"] for sid in frozen[f"{view}_ids"]], minlength=3) * 2
                require(diag["loss_support"]["examples_count_by_class"] == support.tolist(), "Fixed-loss support differs")
                ce = np.dot(weights, diag["loss_support"]["unweighted_nll_sum_by_class"]) / np.dot(weights, support)
                require(np.isclose(ce, original[f"weighted_{view}_ce"], atol=1e-10, rtol=0), "Fixed CE arithmetic differs")
        for filename, view in (("outer_predictions.csv", "outer"), ("validation_predictions.csv", "validation")):
            score = independent_scores(rows(directory / filename))
            compare_scores(value[f"{view}_diagnostics"]["metrics"], score, f"{view} independent confusion")
        facts = read_json(directory / "roi_resampling.json")
        for part, split in (("train", "train"), ("validation", "validation"), ("outer", "test")):
            expected_ids = {(sid, rep) for sid in frozen[f"{split}_ids"] for rep in (1, 2)}
            actual_ids = [(r["structure_id"], int(r["replicate"])) for r in facts[part]]
            require(len(actual_ids) == len(set(actual_ids)) and set(actual_ids) == expected_ids, "ROI source facts do not cover exactly both WT for every frozen split member")
            for fact in facts[part]:
                require(fact["source_resolution_bp"] == 10 and fact["physical_min_pixel_distance_bp"] == 200
                        and fact["output_size"] == 64 and fact["target_physical_coverage_fraction"] >= 1-1e-8
                        and 0 < fact["target_validity_fraction"] <= 1 and fact["validity_mask_shared_byte_identical"] is True,
                        "Native input source/support/canvas/coverage facts differ")
        normalized = {part: [{k: v for k, v in r.items() if k != "alignment"} for r in data]
                      for part, data in facts.items()}
        require(facts_by_fold.setdefault(job["fold"], normalized) == normalized, "C0/C1 input geometry/coverage/source facts differ")
        tail = history[-10:]
        learning_jobs.append({**job, "accepted": value["acceptance"]["accepted"], "final_train_ce": value["final_train_ce"],
            "tail_head_gradient_min": min(r["gradient_norms"]["head"]["mean"] for r in tail),
            "effective_gradients_all_epochs": all(r["gradient_norms"][part]["finite"] and r["gradient_norms"][part]["mean"] > 1e-10
                and r["gradient_norms"][part]["nonzero_fraction"] > 0 for r in history[1:] for part in ("encoder", "head", "total")),
            "train_ce20": history[20]["weighted_train_ce"], "train_ce40": history[40]["weighted_train_ce"],
            "validation_ce20": history[20]["weighted_validation_ce"], "validation_ce40": history[40]["weighted_validation_ce"],
            "runtime_seconds": value["runtime_seconds"]})
    for key, values in paired.items():
        require(len(values) == 6, "Each fold/seed needs the six paired arms")
        for field in ("initial_state_sha256", "common_initial_state_sha256", "minibatch_order_sha256", "class_weights"):
            require(len({json.dumps(v[field], sort_keys=True) for v in values}) == 1, f"Paired {field} differs at {key}")
    oof, stats, case_rows = {}, {}, []
    for alignment, condition in ARMS:
        arm = f"{alignment}/{condition}"
        items = [r for r in records if (r["job"]["alignment"], r["job"]["condition"]) == (alignment, condition)]
        scores, seed_rows = [], []
        for seed in SEEDS:
            per = [r for r in items if r["job"]["seed"] == seed]
            merged = sorted([{**row, "fold": str(r["job"]["fold"])} for r in per for row in r["predictions"]["rows"]], key=lambda row: row["structure_id"])
            require(len(merged) == 344 and {r["structure_id"] for r in merged} == set(identity), "OOF must contain344 unique structures")
            saved = rows(run / "oof" / alignment / condition / f"seed_{seed}" / "predictions.csv")
            require(merged == saved, "Derived OOF differs from sealed per-fold CSV rows")
            score = independent_scores(merged)
            compare_scores(summary["models"][arm]["seeds"][str(seed)]["metrics"], score, "OOF summary")
            oof[alignment, condition, seed] = {"rows": merged, "metrics": score, "by_id": {r["structure_id"]: r for r in merged}}
            scores.append(score)
            cm = score["confusion_matrix"]
            seed_rows.append({"seed": seed, "predicted_counts": score["prediction_counts"], "confusion_matrix": cm,
                "CHID_TP": cm[2][2], "CHID_FP": cm[0][2] + cm[1][2], "CHID_FN": cm[2][0] + cm[2][1],
                "CHIN_to_CHID": cm[1][2], "CHID_to_CHIN": cm[2][1], "WT_disagreement": sum(int(r["replicate_disagree"]) for r in merged)})
        learning = _neural_summary(items, config)
        require(learning == summary["models"][arm]["learning"], "Saved learning summary differs")
        stable, false_negative, false_positive = [], [], []
        unanimous, all_correct = 0, 0
        for sid, ref in sorted(identity.items()):
            source = [oof[alignment, condition, seed]["by_id"][sid] for seed in SEEDS]
            predicted = [int(r["y_pred"]) for r in source]
            wrong = all(p != ref["y"] for p in predicted)
            unanimous += int(len(set(predicted)) == 1)
            all_correct += int(all(p == ref["y"] for p in predicted))
            if wrong: stable.append(sid)
            if ref["y"] == 2 and all(p != 2 for p in predicted): false_negative.append(sid)
            if ref["y"] != 2 and all(p == 2 for p in predicted): false_positive.append(sid)
            case_rows.append({"arm": arm, "structure_id": sid, "type": CLASSES[ref["y"]], "group_id": ref["group_id"],
                "length_bp": ref["length_bp"], "stable_wrong": wrong, "correct_seed_count": sum(p == ref["y"] for p in predicted),
                **{f"seed_{seed}_prediction": CLASSES[p] for seed, p in zip(SEEDS, predicted)},
                **{f"seed_{seed}_p_CHID": float(r["p_CHID"]) for seed, r in zip(SEEDS, source)}})
        require(stable == summary["models"][arm]["stable_wrong_ids"], "Stable error list differs")
        jobs_learning = [r for r in learning_jobs if r["alignment"] == alignment and r["condition"] == condition]
        comparison = next(r for r in summary["comparison"] if (r["alignment"], r["condition"]) == (alignment, condition))
        for metric in ("macro_f1", "accuracy", "balanced_accuracy"):
            values = np.array([s[metric] for s in scores])
            require(np.allclose([comparison[f"{metric}_mean"], comparison[f"{metric}_seed_sample_std"]], [values.mean(), values.std(ddof=1)], atol=1e-12, rtol=0), "Comparison mean/seed SD differs")
        for label in CLASSES:
            for metric in ("precision", "recall", "f1"):
                values = np.array([s["per_class"][label][metric] for s in scores])
                require(np.allclose([comparison[f"{label}_{metric}_mean"], comparison[f"{label}_{metric}_seed_sample_std"]],
                                    [values.mean(), values.std(ddof=1)], atol=1e-12, rtol=0), "Per-class comparison mean/seed SD differs")
        gates = config["development_gates"]
        gate_pass = learning["strict_learning_passes"] == 15 and all(s["macro_f1"] >= gates["macro_f1"] and
            all(s["per_class"][label][metric] >= gates[f"per_class_{metric}"]
                for label in CLASSES for metric in ("precision", "recall")) for s in scores)
        require(comparison["development_gate_pass"] == gate_pass and comparison["stable_wrong_count"] == len(stable)
                and comparison["strict_learning_passes"] == learning["strict_learning_passes"], "Prospective gate/learning/error counts differ")
        trajectory = {}
        for field in ("weighted_train_ce", "weighted_validation_ce", "gradient_head_mean", "gradient_encoder_mean",
                      "train_head_effective_fraction", "validation_centered_logit_variance"):
            quantiles = []
            for epoch in range(41):
                values = [flat_history_row(r["history"][epoch])[field] for r in items]
                values = [v for v in values if v is not None]
                quantiles.append(np.quantile(values, [.25, .5, .75]).tolist() if values else [None, None, None])
            trajectory[field] = quantiles
        stats[arm] = {"scores": scores, "predictions": seed_rows, "learning": learning,
            "development_gate_pass": gate_pass, "stable_wrong_ids": stable,
            "stable_FN_ids": false_negative, "stable_FP_ids": false_positive,
            "seed_unanimous_count": unanimous, "all_seed_correct_count": all_correct,
            "any_seed_wrong_count": len(identity)-all_correct,
            "train40_CE_sample_sd": float(np.std([r["final_train_ce"] for r in jobs_learning], ddof=1)),
            "validation_CE_rising_20_to40": sum(r["validation_ce40"] > r["validation_ce20"] for r in jobs_learning),
            "trajectory_quantiles_25_50_75": trajectory}
    bootstrap = group_bootstrap(oof, identity)
    old, new = set(stats["C0/I3"]["stable_wrong_ids"]), set(stats["C1/I3"]["stable_wrong_ids"])
    changes = {"removed_stable_errors": sorted(old - new), "added_stable_errors": sorted(new - old),
        "fully_repaired_all_seeds": sorted(sid for sid in old if all(int(oof["C1", "I3", s]["by_id"][sid]["y_pred"]) == identity[sid]["y"] for s in SEEDS))}
    integrity = {"status": "passed", **{k: manifest[k] for k in ("experiment_signature", "source_signature", "dataset_signature")},
        "run_dir": str(run.relative_to(ROOT)), "verified_fit_indices": 90, "verified_complete_OOF": 18,
        "indexed_scientific_files": len(inventory), "indexed_bytes": sum(r["size_bytes"] for r in inventory),
        "inventory_signature": json_signature(inventory), "scientific_inventory": inventory,
        "paired_fold_seeds": 15, "shared_native_expected_signatures": norm_by_fold, "G0_exact_R1_replay": g0_replay,
        "audit_script_sha256": sha256(Path(__file__)), "reading_scope": "Saved metadata/CSV/checkpoint hashes, CPU G0 weight-tensor equality and geometry-only FFT; no contact pixels, new inference or fitting",
        "sources": {"manifest.json": sha256(run / "manifest.json"), "summary.json": sha256(run / "summary.json")}}
    report = safe_report(report, run)
    report.mkdir(parents=True, exist_ok=True)
    for name, value in (("integrity.json", integrity), ("group_bootstrap.json", bootstrap),
                        ("error_learning.json", {"arms": stats, "changes": changes, "comparison": summary["comparison"]})):
        (report / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _csv(report / "case_review.csv", case_rows)
    _csv(report / "learning_per_job.csv", learning_jobs)
    return {"integrity": integrity, "bootstrap": bootstrap, "stats": stats, "changes": changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/task1_native_roi_v2")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/task1_native_roi_review_v1")
    args = parser.parse_args()
    result = build_report(args.run_dir, args.report_dir)
    print(json.dumps({"status": "passed", "fits": 90, "OOF": 18, "C1-I3_minus_C0-I3": result["bootstrap"]["pairs"]["C1/I3-C0/I3"], "changes": result["changes"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
