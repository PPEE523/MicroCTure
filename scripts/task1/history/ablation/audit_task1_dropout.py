#!/usr/bin/env python3
"""Audit the complete preregistered D2/D5 dropout comparison from saved evidence.

The full60/new12-OOF completion gate precedes all fit-score reads. No contact
images, model checkpoint deserialization, inference or fitting is performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from microcture.task1_dropout_v1 import experiment as D
from microcture.task1_roi import experiment as R
from microcture.task1_controlled.runner import flat_history_row
from microcture.task1_formal.evaluation import _identity, _fold_ids, _neural_summary, _csv
from microcture.task1_formal.protocol import json_signature, sha256
from scripts.audit_task1_reevaluation_results import csv_metrics, compare_score

CLASSES = ("OPCID", "CHIN", "CHID")
CONDITIONS, SEEDS = ("I3", "G0", "V0", "P0"), (42, 43, 44)
ARMS = tuple((variant, condition) for variant in ("D2", "D5") for condition in CONDITIONS)
PAIRS = tuple((f"D5/{c}", f"D2/{c}") for c in CONDITIONS) + tuple(
    (f"{v}/I3", f"{v}/{c}") for v in ("D2", "D5") for c in ("G0", "V0", "P0"))
DIFFICULT = ("CHID_8", "CHID_9", "CHID_11", "CHID_12", "CHID_15", "CHID_19")
FINGERPRINTS = ("initial_state_sha256", "common_initial_state_sha256", "minibatch_order_sha256", "class_weights")


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def safe_report(report, run):
    report = Path(report)
    require(not any(p.is_symlink() for p in (report, *report.parents)), "No report symlinks")
    report = report.resolve()
    require((ROOT / "reports") in report.parents and report != ROOT / "reports", "Use an independent reports/ directory")
    require(run != report and run not in report.parents and report not in run.parents, "Reports cannot overlap results")
    require(not any((p / "ARTIFACTS.json").exists() or (p / "FROZEN.json").exists() for p in (report, *report.parents)), "Sealed report is read-only")
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
    """Paired whole-group resampling; average metrics after scoring each seed."""
    require(repeats > 0, "Positive bootstrap repetition count required")
    groups = sorted({r["group_id"] for r in identity.values()})
    gi = {g: i for i, g in enumerate(groups)}
    cm = np.zeros((len(ARMS), len(SEEDS), len(groups), 3, 3), dtype=np.int64)
    for ai, arm in enumerate(ARMS):
        for si, seed in enumerate(SEEDS):
            source = oof[(*arm, seed)]["rows"]
            require(len(source) == len(identity) and {r["structure_id"] for r in source} == set(identity), "Matched full OOF required")
            for row in source:
                cm[ai, si, gi[int(row["group_id"])] , int(row["y_true"]), int(row["y_pred"])] += 1
    w = np.random.default_rng(rng_seed).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=repeats)
    raw, support, predicted = scores_from_confusion(np.einsum("bg,asgij->basij", w, cm))
    observed, _, _ = scores_from_confusion(cm.sum(2))
    per_arm, distributions = {}, {}
    for ai, arm in enumerate(ARMS):
        name = "/".join(arm);per_arm[name], distributions[name] = {}, {}
        for metric in raw:
            values, seeds = raw[metric][:, ai].mean(-1), observed[metric][ai]
            distributions[name][metric] = values
            per_arm[name][metric] = {"point_mean": float(seeds.mean()), "seed_values": seeds.tolist(),
                "seed_sample_sd": float(seeds.std(ddof=1)), "ci95": np.quantile(values, [.025, .975]).tolist()}
    contrasts = {}
    for first, second in PAIRS:
        contrasts[f"{first}-{second}"] = {}
        for metric in raw:
            delta = np.array(per_arm[first][metric]["seed_values"]) - np.array(per_arm[second][metric]["seed_values"])
            dist = distributions[first][metric] - distributions[second][metric]
            contrasts[f"{first}-{second}"][metric] = {"point_mean": float(delta.mean()), "seed_deltas": delta.tolist(),
                "ci95": np.quantile(dist, [.025, .975]).tolist()}
    interaction = {}
    for control in ("G0", "V0", "P0"):
        interaction[control] = {}
        for metric in raw:
            values = ((distributions["D5/I3"][metric] - distributions[f"D5/{control}"][metric])
                      - (distributions["D2/I3"][metric] - distributions[f"D2/{control}"][metric]))
            delta = ((np.array(per_arm["D5/I3"][metric]["seed_values"]) - np.array(per_arm[f"D5/{control}"][metric]["seed_values"]))
                     - (np.array(per_arm["D2/I3"][metric]["seed_values"]) - np.array(per_arm[f"D2/{control}"][metric]["seed_values"])))
            interaction[control][metric] = {"point_mean": float(delta.mean()), "seed_deltas": delta.tolist(),
                "ci95": np.quantile(values, [.025, .975]).tolist()}
    return {"scope": "Conditional saved developmental OOF uncertainty; no refit or correction for prior researcher adaptation",
        "B": repeats, "rng_seed": rng_seed, "group_count": len(groups), "unit": "Same whole-group multiplicities across every variant, condition and seed",
        "seed_aggregation": "Score each seed independently; average three seed metrics afterwards",
        "interval": "Unadjusted percentile95; multiple contrasts descriptive; zero-support draws retained using fixed three-class zero-division convention",
        "zero_CHID_support_bootstraps": int(np.any(support[..., 2] == 0, axis=(1, 2)).sum()),
        "zero_CHID_prediction_bootstraps": int(np.any(predicted[..., 2] == 0, axis=(1, 2)).sum()),
        "arms": per_arm, "pairs": contrasts, "dropout_effect_beyond_control": interaction}


def check_inventory(directory, job, manifest, variant):
    index = read_json(directory / "ARTIFACTS.json")
    require(index["job"] == job and index["experiment_signature"] == manifest["experiment_signature"] and
            index["signature"] == json_signature(index["files"]), "Artifact binding/signature differs")
    names, inventory = set(), []
    for item in index["files"]:
        relative = Path(item["name"])
        require(not relative.is_absolute() and ".." not in relative.parts and item["name"] not in names,
                "Unsafe/duplicate scientific artifact")
        names.add(item["name"]);path = directory / relative
        require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)) and
                path.stat().st_size == item["size_bytes"] and sha256(path) == item["sha256"], f"Scientific bytes differ: {path}")
        inventory.append({"variant": variant, "directory": str(directory.relative_to(ROOT)), **item})
    require({str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()} == names | {"ARTIFACTS.json"},
            "Recursive scientific inventory differs")
    return inventory


def learning_record(variant, record, identity, frozen, constant_epsilon):
    job, value, history = record["job"], record["metrics"], record["history"]
    require([r["epoch"] for r in history] == list(range(41)), "Full41 diagnostic points required")
    counts = np.bincount([identity[sid]["y"] for sid in frozen["train_ids"]], minlength=3)
    weights = counts.sum() / (3 * counts)
    require(np.allclose(weights, value["class_weights"], atol=1e-12, rtol=0), "Inner class weights differ")
    for entry in history:
        for view in ("train", "validation"):
            diag = entry[f"{view}_diagnostics"]
            support = np.bincount([identity[sid]["y"] for sid in frozen[f"{view}_ids"]], minlength=3) * 2
            require(diag["loss_support"]["examples_count_by_class"] == support.tolist(), "Fixed CE support differs")
            ce = np.dot(weights, diag["loss_support"]["unweighted_nll_sum_by_class"]) / np.dot(weights, support)
            require(np.isclose(ce, entry[f"weighted_{view}_ce"], atol=1e-10, rtol=0), "Weighted CE arithmetic differs")
    outer = record["predictions"]["rows"]
    p = np.asarray([[float(r[f"p_{c}"]) for c in CLASSES] for r in outer])
    return {"variant": variant, "condition": job["condition"], "seed": job["seed"], "fold": job["fold"],
        "accepted": value["acceptance"]["accepted"], "final_train_ce": value["final_train_ce"],
        "train_ce20": history[20]["weighted_train_ce"], "train_ce40": history[40]["weighted_train_ce"],
        "validation_ce20": history[20]["weighted_validation_ce"], "validation_ce40": history[40]["weighted_validation_ce"],
        "final_train_validation_gap": history[40]["weighted_validation_ce"] - history[40]["weighted_train_ce"],
        "tail_head_gradient_min": min(r["gradient_norms"]["head"]["mean"] for r in history[-10:]),
        "effective_gradients_all_epochs": all(r["gradient_norms"][part]["finite"] and
            r["gradient_norms"][part]["mean"] > 1e-10 and r["gradient_norms"][part]["nonzero_fraction"] > 0
            for r in history[1:] for part in ("encoder", "head", "total")),
        "outer_single_argmax_class": len({r["y_pred"] for r in outer}) == 1,
        "outer_constant_probabilities": bool(np.ptp(p, axis=0).max() <= constant_epsilon),
        "runtime_seconds": value["runtime_seconds"]}


def build_report(run, report):
    run = Path(run).resolve()
    # Do not read partial-fit outcomes to choose p, conditions, seeds or epochs.
    summary = read_json(run / "summary.json")
    require(summary["status"] == "complete" and summary["completed_jobs"] == 60 and summary["complete_oof"] == 12,
            "Review waits for all60 D5 fits and12 complete OOFs")
    require(not any(summary[k] for k in ("pending_jobs", "invalid_jobs", "execution_failure_jobs", "numerical_failure_jobs")),
            "Every registered fit must be accounted for")
    manifest = read_json(run / "manifest.json")
    current, metadata = D.prepare_manifest(ROOT, manifest["config"], manifest["requested_device"])
    require(current == manifest, "D5 source/config/dependencies/environment changed")
    require(len(list((run / "runs").rglob("ARTIFACTS.json"))) == 60 and not (run / "failed_staging").exists()
            and not D.unpublished_staging(ROOT, run), "Extra/retained scientific staging exists")
    for key in ("experiment_signature", "source_signature", "dataset_signature"):
        require(summary[key] == manifest[key], f"Summary {key} differs")
    old_run = ROOT / manifest["config"]["reference_roi_run_dir"]
    old_manifest, old_summary = read_json(old_run / "manifest.json"), read_json(old_run / "summary.json")
    require(sha256(old_run / "manifest.json") == manifest["D2_reference"]["manifest_sha256"] and
            sha256(old_run / "summary.json") == manifest["D2_reference"]["summary_sha256"], "D2 reference bytes changed")
    require(old_summary["status"] == "complete" and len(list((old_run / "runs/R1").rglob("ARTIFACTS.json"))) == 60,
            "D2 requires all60 sealed R1 reference fits")
    require(summary["D2_reference_comparison"] == [{**row, "dropout_variant": "D2", "newly_trained": False}
            for row in old_summary["comparison"] if row["alignment"] == "R1"], "Saved D2 reference comparison differs")
    identity = _identity(metadata)
    require(len(identity) == 344 and len({v["group_id"] for v in identity.values()}) == 49, "Registered344 structures/49 groups required")
    inventory, records, learning_jobs, paired, pp_shared, provenance = [], {}, [], defaultdict(list), {}, {}
    for variant, source, implementation, document, saved_summary in (
            ("D2", old_run, R, old_manifest, old_summary), ("D5", run, D, manifest, summary)):
        folds = _fold_ids(document, identity, source)
        jobs = implementation.planned_jobs(document["config"], alignment="R1")
        require(len(jobs) == 60, "Exactly60 registered fits per dropout variant required")
        provenance[variant] = {"run_dir": str(source.relative_to(ROOT)), "manifest_sha256": sha256(source / "manifest.json"),
            "summary_sha256": sha256(source / "summary.json"), "experiment_signature": document["experiment_signature"],
            "source_signature": document["source_signature"], "newly_trained_in_this_experiment": variant == "D5"}
        for job in jobs:
            directory = implementation.job_directory(source, job)
            inventory.extend(check_inventory(directory, job, document, variant))
            record = implementation._load_saved(source, job, document, identity, folds)
            require(record["status"] == "completed", "Failed/missing registered fits cannot be omitted")
            records[variant, job["condition"], job["seed"], job["fold"]] = record
            value, history = record["metrics"], record["history"]
            pp = read_json(directory / "protocol.json")["preprocessing"]
            require(pp_shared.setdefault(job["fold"], pp) == pp, "D2/D5/condition/seed preprocessing differs")
            pairing = D.assert_paired_fit(manifest["config"], job, value, pp)
            if variant == "D5":
                require(read_json(directory / "D2_pairing.json") == pairing, "Saved D5/D2 pairing proof differs")
            paired[job["fold"], job["seed"]].append(value)
            frozen = read_json(ROOT / document["config"]["frozen_run_dir"] / "folds" / f"fold_{job['fold']}.json")
            learning_jobs.append(learning_record(variant, record, identity, frozen, document["config"]["acceptance"]["constant_epsilon"]))
            saved_history = rows(directory / "history.csv")
            require(len(saved_history) == 41, "History CSV incomplete")
            for original, saved in zip(history, saved_history):
                flat = flat_history_row(original)
                require(set(flat) == set(saved) and all(saved[k] == ("" if v is None else str(v)) for k, v in flat.items()),
                        "CSV/JSON history differs")
            spec = value["parameter_spec"]
            require(spec["parameters"] == spec["trainable_parameters"] == 110437 and spec["head_version"] == "H3" and
                    spec["config"]["dropout"] == (.2 if variant == "D2" else .5), "Model/head/dropout budget differs")
            for filename, view in (("outer_predictions.csv", "outer"), ("validation_predictions.csv", "validation")):
                compare_score(csv_metrics(rows(directory / filename)), value[f"{view}_diagnostics"]["metrics"], "Independent fold CSV")
    for key, values in paired.items():
        require(len(values) == 8, "Every fold/seed requires8 paired variant/condition fits")
        for field in FINGERPRINTS:
            require(len({json.dumps(v[field], sort_keys=True) for v in values}) == 1, f"Paired {field} differs at{key}")
    oof, stats, case_rows = {}, {}, []
    for variant, condition in ARMS:
        source, saved_summary, document = (old_run, old_summary, old_manifest) if variant == "D2" else (run, summary, manifest)
        arm, key = f"{variant}/{condition}", f"R1/{condition}"
        items = [records[variant, condition, seed, fold] for seed in SEEDS for fold in range(5)]
        scores, seed_rows = [], []
        for seed in SEEDS:
            merged = sorted([{**row, "fold": str(fold)} for fold in range(5)
                for row in records[variant, condition, seed, fold]["predictions"]["rows"]], key=lambda row: row["structure_id"])
            require(len(merged) == len({r["structure_id"] for r in merged}) == 344 and
                    {r["structure_id"] for r in merged} == set(identity), "OOF requires344 unique identities")
            oof_dir = source / "oof/R1" / condition / f"seed_{seed}"
            require(rows(oof_dir / "predictions.csv") == merged, "OOF differs from exact sealed-fold merge")
            status, stored = read_json(oof_dir / "status.json"), read_json(oof_dir / "metrics.json")
            require(status["status"] == "complete" and status["oof_current"] is True and stored["oof_current"] is True and
                    status["source_signature"] == document["source_signature"] and
                    status["experiment_signature"] == stored["experiment_signature"] == document["experiment_signature"], "OOF provenance differs")
            score = csv_metrics(merged)
            compare_score(score, stored, "OOF metric JSON")
            compare_score(score, saved_summary["models"][key]["seeds"][str(seed)]["metrics"], "OOF summary")
            oof[variant, condition, seed] = {"rows": merged, "metrics": score, "by_id": {r["structure_id"]: r for r in merged}}
            scores.append(score);cm = score["confusion_matrix"]
            seed_rows.append({"seed": seed, "predicted_counts": score["prediction_counts"], "confusion_matrix": cm,
                "CHID_TP": cm[2][2], "CHID_FP": cm[0][2] + cm[1][2], "CHID_FN": cm[2][0] + cm[2][1],
                "CHIN_to_CHID": cm[1][2], "CHID_to_CHIN": cm[2][1], "WT_disagreement": sum(int(r["replicate_disagree"]) for r in merged)})
        learning = _neural_summary(items, document["config"])
        require(learning == saved_summary["models"][key]["learning"], "Saved learning summary differs")
        stable, missed, false_positive, unanimous, all_correct = [], [], [], 0, 0
        for sid, ref in sorted(identity.items()):
            rs = [oof[variant, condition, seed]["by_id"][sid] for seed in SEEDS]
            pred = [int(r["y_pred"]) for r in rs]
            unanimous += int(len(set(pred)) == 1);all_correct += int(all(p == ref["y"] for p in pred))
            if all(p != ref["y"] for p in pred): stable.append(sid)
            if ref["y"] == 2 and all(p != 2 for p in pred): missed.append(sid)
            if ref["y"] != 2 and all(p == 2 for p in pred): false_positive.append(sid)
            case_rows.append({"arm": arm, "structure_id": sid, "type": CLASSES[ref["y"]], "group_id": ref["group_id"],
                "length_bp": ref["length_bp"], "stable_wrong": all(p != ref["y"] for p in pred),
                "correct_seed_count": sum(p == ref["y"] for p in pred), "historical_difficult_CHID": sid in DIFFICULT,
                **{f"seed_{seed}_prediction": CLASSES[p] for seed, p in zip(SEEDS, pred)},
                **{f"seed_{seed}_p_CHID": float(r["p_CHID"]) for seed, r in zip(SEEDS, rs)}})
        require(stable == saved_summary["models"][key]["stable_wrong_ids"], "Stable error IDs differ")
        comparison = next(r for r in saved_summary["comparison"] if (r["alignment"], r["condition"]) == ("R1", condition))
        for field in ("macro_f1", "accuracy", "balanced_accuracy") + tuple(f"{c}_{m}" for c in CLASSES for m in ("precision", "recall", "f1")):
            values = np.array([s[field] if field in s else s["per_class"][field.split("_")[0]][field.split("_")[1]] for s in scores])
            require(np.allclose([comparison[f"{field}_mean"], comparison[f"{field}_seed_sample_std"]],
                                [values.mean(), values.std(ddof=1)], atol=1e-12, rtol=0), "Comparison mean/seed SD differs")
        gates = document["config"]["development_gates"]
        passed = learning["strict_learning_passes"] == 15 and all(s["macro_f1"] >= gates["macro_f1"] and all(
            s["per_class"][label][metric] >= gates[f"per_class_{metric}"] for label in CLASSES for metric in ("precision", "recall")) for s in scores)
        require(comparison["development_gate_pass"] == passed and comparison["stable_wrong_count"] == len(stable) and
                comparison["strict_learning_passes"] == learning["strict_learning_passes"], "Development gate/error count differs")
        trajectory = {}
        for field in ("weighted_train_ce", "weighted_validation_ce", "gradient_head_mean", "gradient_encoder_mean",
                      "train_head_effective_fraction", "validation_centered_logit_variance"):
            trajectory[field] = []
            for epoch in range(41):
                values = [flat_history_row(r["history"][epoch])[field] for r in items]
                values = [v for v in values if v is not None]
                trajectory[field].append(np.quantile(values, [.25, .5, .75]).tolist() if values else [None, None, None])
        jl = [r for r in learning_jobs if r["variant"] == variant and r["condition"] == condition]
        stats[arm] = {"scores": scores, "predictions": seed_rows, "learning": learning, "development_gate_pass": passed,
            "stable_wrong_ids": stable, "stable_FN_ids": missed, "stable_FP_ids": false_positive,
            "seed_unanimous_count": unanimous, "all_seed_correct_count": all_correct, "any_seed_wrong_count": len(identity) - all_correct,
            "effective_gradients_all_epochs_count": sum(r["effective_gradients_all_epochs"] for r in jl),
            "outer_single_argmax_class_count": sum(r["outer_single_argmax_class"] for r in jl),
            "outer_constant_probabilities_count": sum(r["outer_constant_probabilities"] for r in jl),
            "train40_CE_sample_sd": float(np.std([r["train_ce40"] for r in jl], ddof=1)),
            "validation_CE_rising_20_to40": sum(r["validation_ce40"] > r["validation_ce20"] for r in jl),
            "trajectory_quantiles_25_50_75": trajectory}
    bootstrap = group_bootstrap(oof, identity)
    require(len(summary["D5_minus_D2_paired_seed_deltas"]) == 12 and
            {(r["condition"], r["seed"]) for r in summary["D5_minus_D2_paired_seed_deltas"]} ==
            {(c, s) for c in CONDITIONS for s in SEEDS}, "All12 paired dropout deltas required")
    for row in summary["D5_minus_D2_paired_seed_deltas"]:
        for metric, field in (("macro_f1", "macro_f1_delta"), ("CHID_f1", "CHID_f1_delta")):
            pos = SEEDS.index(row["seed"])
            require(np.isclose(row[field], bootstrap["pairs"][f"D5/{row['condition']}-D2/{row['condition']}"][metric]["seed_deltas"][pos],
                               atol=1e-12, rtol=0), "Saved paired dropout delta differs")
    require(len(summary["paired_deltas"]) == 9, "All9 within-D5 ablation deltas required")
    for row in summary["paired_deltas"]:
        condition = row["comparison"].split("-")[1].split("/")[1]
        require(condition in ("G0", "V0", "P0") and row["seed"] in SEEDS, "Invalid within-D5 delta")
        for metric, field in (("macro_f1", "macro_f1_delta"), ("CHID_f1", "CHID_f1_delta")):
            require(np.isclose(row[field], bootstrap["pairs"][f"D5/I3-D5/{condition}"][metric]["seed_deltas"][SEEDS.index(row["seed"])],
                               atol=1e-12, rtol=0), "Saved within-D5 delta differs")
    old, new = set(stats["D2/I3"]["stable_wrong_ids"]), set(stats["D5/I3"]["stable_wrong_ids"])
    changes = {"removed_stable_errors": sorted(old - new), "added_stable_errors": sorted(new - old),
        "fully_repaired_all_seeds": sorted(sid for sid in old if all(int(oof["D5", "I3", s]["by_id"][sid]["y_pred"]) == identity[sid]["y"] for s in SEEDS))}
    integrity = {"status": "passed", **{k: manifest[k] for k in ("experiment_signature", "source_signature", "dataset_signature")},
        "verified_D5_fits": 60, "verified_D2_reference_fits": 60, "verified_complete_OOF": 24,
        "D2_retrained": False, "paired_fold_seeds": 15, "exact_pairing_fields": list(FINGERPRINTS) + ["preprocessing"],
        "indexed_scientific_files": len(inventory), "indexed_bytes": sum(r["size_bytes"] for r in inventory),
        "inventory_signature": json_signature(inventory), "audit_script_sha256": sha256(Path(__file__)),
        "independent_score_script_sha256": sha256(ROOT / "scripts/audit_task1_reevaluation_results.py"),
        "provenance": provenance, "reading_scope": "Saved JSON/CSV and scientific-file hashes only; no contact images, model deserialization, inference or fitting"}
    report = safe_report(report, run);report.mkdir(parents=True, exist_ok=True)
    for filename, value in (("integrity.json", integrity), ("group_bootstrap.json", bootstrap),
                            ("error_learning.json", {"arms": stats, "changes": changes})):
        (report / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _csv(report / "case_review.csv", case_rows);_csv(report / "learning_per_job.csv", learning_jobs)
    return {"integrity": integrity, "bootstrap": bootstrap, "stats": stats, "changes": changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/task1_dropout_v1")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/task1_dropout_review_v1")
    args = parser.parse_args();result = build_report(args.run_dir, args.report_dir)
    print(json.dumps({"status": "passed", "fits_new_reference": [60, 60], "OOF": 24,
        "D5_minus_D2_I3": result["bootstrap"]["pairs"]["D5/I3-D2/I3"], "changes": result["changes"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
