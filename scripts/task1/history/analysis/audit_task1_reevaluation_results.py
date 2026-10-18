#!/usr/bin/env python3
"""Read sealed paired-policy results; independently recompute CSV evidence.

No fitting, contact-array reads, checkpoint deserialization or inference.
Writes only reusable report products outside the registered experiment.
"""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = _layout_root
sys.path.insert(0, str(ROOT))
import numpy as np

from microcture.task1_formal.protocol import json_signature, sha256
from microcture.task1_formal.evaluation import _identity, _fold_ids
from microcture.task1_reevaluation.protocol import (endpoint_directory,
    endpoint_policies, job_directory, planned_jobs, prepare_manifest)
from microcture.task1_reevaluation.evaluation import _load_fit

CLASSES = ("OPCID", "CHIN", "CHID")
BACKBONES = ("m2", "m2plus", "cnn_transformer")
SCORE_NAMES = ("macro_f1", "balanced_accuracy", "accuracy", "CHID_precision", "CHID_recall", "CHID_f1")
DIFFICULT = ("CHID_8", "CHID_9", "CHID_11", "CHID_12", "CHID_15", "CHID_19")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def near(first, second, label):
    require(np.isclose(first, second, atol=1e-10, rtol=0), f"{label} differs")


def csv_metrics(rows):
    """Count unique structures after checking two-WT probability aggregation."""
    require(rows and len(rows) == len({r["structure_id"] for r in rows}), "Duplicated/empty structure IDs")
    cm = np.zeros((3, 3), dtype=np.int64)
    for row in rows:
        y, predicted = int(row["y_true"]), int(row["y_pred"])
        p = np.asarray([float(row[f"p_{name}"]) for name in CLASSES])
        require(y in range(3) and predicted in range(3), "Invalid class label")
        require(np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), "Invalid probability range")
        require(abs(float(p.sum()) - 1) < 1e-5, "Probability normalization differs")
        require(predicted == int(p.argmax()), "Argmax differs from saved prediction")
        reps = json.loads(row["replicate_probabilities_json"])
        require(len(reps) == int(row["replicate_count"]) == 2
                and {str(rep["replicate"]) for rep in reps} == {"1", "2"}, "Two distinct WT replicates required")
        rp = np.asarray([rep["probabilities"] for rep in reps], dtype=float)
        require(rp.shape == (2, 3) and np.isfinite(rp).all()
                and np.all((rp >= 0) & (rp <= 1)), "Invalid WT probabilities")
        require(np.allclose(rp.sum(axis=1), 1, atol=1e-6, rtol=0), "WT probabilities not normalized")
        require(np.allclose(p, rp.mean(axis=0), atol=1e-12, rtol=0), "WT averaging differs")
        require(all(int(rep["predicted"]) == int(np.argmax(rep["probabilities"])) for rep in reps), "WT argmax differs")
        require(int(row["replicate_disagree"]) == int(len(set(rp.argmax(axis=1))) > 1), "WT disagreement differs")
        cm[y, predicted] += 1
    n, actual = cm.sum(axis=0), cm.sum(axis=1)
    tp = cm.diagonal()
    precision = np.divide(tp, n, out=np.zeros(3), where=n > 0)
    recall = np.divide(tp, actual, out=np.zeros(3), where=actual > 0)
    f1 = np.divide(2 * tp, n + actual, out=np.zeros(3), where=n + actual > 0)
    return {"n_structures": len(rows), "confusion_matrix": cm.tolist(), "prediction_counts": dict(zip(CLASSES, n.tolist())),
            "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()), "accuracy": float(tp.sum() / len(rows)),
            "CHID_precision": float(precision[2]), "CHID_recall": float(recall[2]), "CHID_f1": float(f1[2]),
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
                                  "support": int(actual[i])} for i, name in enumerate(CLASSES)},
            "CHID_counts": {"TP": int(tp[2]), "FP": int(n[2] - tp[2]), "FN": int(actual[2] - tp[2]), "support": int(actual[2])}}


def compare_score(score, stored, label):
    require(score["confusion_matrix"] == stored["confusion_matrix"], f"{label}: confusion counts differ")
    for name in ("macro_f1", "balanced_accuracy", "accuracy"):
        near(score[name], stored[name], label + name)
    for name in CLASSES:
        for metric in ("precision", "recall", "f1", "support"):
            near(score["per_class"][name][metric], stored["per_class"][name][metric], label + name + metric)


def selected_inner_epoch(history):
    """Independent strict-less rule: epochs 1..40, first exact minimum."""
    require([row["epoch"] for row in history] == list(range(41)), "Complete 40-epoch history required")
    values = [float(row["weighted_validation_ce"]) for row in history[1:]]
    require(np.isfinite(values).all(), "Nonfinite selection trajectory")
    return 1 + int(np.argmin(values))


def describe(values):
    return {"mean": float(np.mean(values)), "seed_sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "min": float(min(values)), "max": float(max(values)), "n": len(values)}


def audit(run_dir):
    run_dir = Path(run_dir).resolve()
    manifest, saved_summary = read_json(run_dir / "manifest.json"), read_json(run_dir / "summary.json")
    require(json_signature({k: v for k, v in manifest.items() if k != "experiment_signature"}) == manifest["experiment_signature"],
            "Manifest signature differs")
    current, metadata = prepare_manifest(ROOT, manifest["config"], manifest["requested_device"])
    require(current == manifest, "Current source/environment/dependencies changed")
    require(saved_summary["status"] == "complete" and saved_summary["completed_fits"] == 155
            and saved_summary["completed_endpoints"] == 290 and saved_summary["complete_oof"] == 58, "Registered coverage incomplete")
    require(not (run_dir / "failed_staging").exists(), "Retained failure staging requires investigation")
    identity, config = _identity(metadata), manifest["config"]
    folds = _fold_ids(manifest, identity, run_dir)
    jobs = planned_jobs(config)
    require(len(jobs) == len(list((run_dir / "runs").rglob("ARTIFACTS.json"))) == 155, "Fit inventory differs")
    files, byte_count, endpoints = 0, 0, 0
    index_hashes, scored, predictions = {}, {}, {}
    expected_oof = defaultdict(list)
    histories, selected, pairings, formal_reproduction = defaultdict(list), defaultdict(list), defaultdict(list), []
    endpoint_prediction_events = defaultdict(lambda: {"endpoints": 0, "single_class_argmax_endpoints": 0,
                                                     "constant_probability_endpoints": 0})
    for job in jobs:
        directory = job_directory(run_dir, job)
        index = read_json(directory / "ARTIFACTS.json")
        require(index["job"] == job and index["experiment_signature"] == manifest["experiment_signature"]
                and json_signature(index["files"]) == index["signature"], "Fit artifact binding differs")
        indexed = set()
        for item in index["files"]:
            relative = Path(item["name"])
            require(not relative.is_absolute() and ".." not in relative.parts and item["name"] not in indexed, "Unsafe/duplicate artifact name")
            indexed.add(item["name"])
            path = directory / relative
            require(path.is_file() and not path.is_symlink() and path.stat().st_size == item["size_bytes"]
                    and sha256(path) == item["sha256"], f"Scientific artifact differs: {path}")
            files += 1
            byte_count += item["size_bytes"]
        actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
        require(actual == indexed | {"ARTIFACTS.json"}, "Recursive scientific file inventory differs")
        index_hashes[str(directory.relative_to(run_dir))] = sha256(directory / "ARTIFACTS.json")
        record = _load_fit(run_dir, job, manifest, identity, folds)
        require(record["status"] == "completed", "A failed fit cannot be omitted")
        key = f"{job['condition']}/{job['model']}"
        if job["kind"] == "neural":
            history = record["history"]
            histories[key].append(history)
            epoch = selected_inner_epoch(history)
            require(record["endpoints"]["inner_ce"]["metrics"]["selected_epoch"] == epoch, "Inner selection epoch differs")
            selected[key].append({"fold": job["fold"], "seed": job["seed"], "epoch": epoch,
                                  "selected_validation_ce": history[epoch]["weighted_validation_ce"],
                                  "final_validation_ce": history[40]["weighted_validation_ce"]})
            pairings[(job["model"], job["seed"], job["fold"])].append(record["metrics"])
            saved_history = csv_rows(directory / "history.csv")
            require([int(r["epoch"]) for r in saved_history] == list(range(41)), "History CSV incomplete")
            for a, b in zip(saved_history, history):
                for metric in ("weighted_train_ce", "weighted_validation_ce"):
                    near(float(a[metric]), b[metric], "CSV/JSON history")
        formal_directory = ROOT / config["formal_run_dir"] / directory.relative_to(run_dir)
        reproduction = {"job": job, "outer_csv_exact": False, "validation_csv_exact": False, "history_exact": None}
        for policy in endpoint_policies(job):
            endpoint = record["endpoints"][policy]
            ep_dir = endpoint_directory(run_dir, job, policy)
            outer, validation = csv_rows(ep_dir / "outer_predictions.csv"), csv_rows(ep_dir / "validation_predictions.csv")
            compare_score(csv_metrics(outer), endpoint["metrics"]["outer_metrics"], "Endpoint outer")
            compare_score(csv_metrics(validation), endpoint["metrics"].get("validation_metrics",
                          endpoint["metrics"].get("validation_diagnostics", {}).get("metrics")), "Endpoint validation")
            events = endpoint_prediction_events[f"{policy}/{job['condition']}/{job['model']}"]
            events["endpoints"] += 1
            events["single_class_argmax_endpoints"] += int(len({row["y_pred"] for row in outer}) == 1)
            p = np.asarray([[float(row[f"p_{name}"]) for name in CLASSES] for row in outer])
            events["constant_probability_endpoints"] += int(np.ptp(p, axis=0).max() <= config["acceptance"]["constant_epsilon"])
            expected_oof[(policy, job["condition"], job["model"], job["seed"])].extend([{**r, "fold": str(job["fold"])} for r in outer])
            endpoints += 1
            if policy in ("final40", "baseline"):
                for name, field in (("outer_predictions.csv", "outer_csv_exact"), ("validation_predictions.csv", "validation_csv_exact")):
                    require(csv_rows(ep_dir / name) == csv_rows(formal_directory / name), "Final40/baseline differs from original formal CSV")
                    reproduction[field] = True
        if job["kind"] == "neural":
            formal_history = read_json(formal_directory / "history.json")["epochs"]
            numeric = lambda history: [{k: v for k, v in row.items() if k != "epoch_seconds"} for row in history]
            require(numeric(record["history"]) == numeric(formal_history), "Numerical trajectory differs from formal reference")
            reproduction["history_exact"] = True
        formal_reproduction.append(reproduction)
    for pair in pairings.values():
        require(len(pair) == 3, "Three input conditions required for common initialization pairing")
        for name in ("initial_state_sha256", "common_initial_state_sha256", "minibatch_order_sha256", "class_weights"):
            require(len({json.dumps(record[name], sort_keys=True) for record in pair}) == 1, "Common initialization/stream pairing differs")
    require(endpoints == 290 and len(expected_oof) == 58, "Endpoint/OOF coverage differs")
    for key, original in expected_oof.items():
        policy, condition, model, seed = key
        directory = run_dir / "oof" / policy / condition / model / f"seed_{seed}"
        status, stored = read_json(directory / "status.json"), read_json(directory / "metrics.json")
        for document in (status, stored):
            require(document["status"] == "complete" and document["oof_current"] is True, "Stale/incomplete OOF")
            for field in ("experiment_signature", "source_signature", "dataset_signature", "selection_signature"):
                require(document[field] == manifest[field], "OOF provenance differs")
            require((document["policy"], document["condition"], document["model"], document["seed"]) == key, "OOF identity differs")
        rows = csv_rows(directory / "predictions.csv")
        by_id, expected = {r["structure_id"]: r for r in rows}, {r["structure_id"]: r for r in original}
        require(len(rows) == len(by_id) == len(expected) == 344 and set(by_id) == set(identity), "OOF unique structure coverage differs")
        for sid, row in by_id.items():
            for field, value in expected[sid].items():
                require(row[field] == value, "OOF row differs from sealed fold prediction")
        score = csv_metrics(rows)
        compare_score(score, stored, "OOF")
        compare_score(score, saved_summary["models"][f"{policy}/{condition}/{model}"]["seeds"][str(seed)]["oof"], "Summary OOF")
        name = f"{policy}/{condition}/{model}"
        scored.setdefault(name, {})[str(seed)] = score
        predictions.setdefault(name, {})[str(seed)] = by_id
    comparison = {}
    for name, per_seed in scored.items():
        policy, condition, model = name.split("/")
        rows = [r for r in saved_summary["comparison"] if (r["policy"], r["condition"], r["model"]) == (policy, condition, model)]
        require(len(rows) == 1, "Comparison key coverage differs")
        comparison[name] = {metric: describe([v[metric] for v in per_seed.values()]) for metric in SCORE_NAMES}
        for metric in SCORE_NAMES:
            near(comparison[name][metric]["mean"], rows[0][metric + "_mean"], "Summary seed mean")
            if len(per_seed) > 1:
                near(comparison[name][metric]["seed_sample_sd"], rows[0][metric + "_seed_sample_std"], "Summary sample SD")
    paired, errors, case_changes = {}, {}, []
    for model in BACKBONES:
        for condition in config["conditions"]:
            first, second = f"final40/{condition}/{model}", f"inner_ce/{condition}/{model}"
            paired[f"{condition}/{model}"] = {metric: describe([scored[second][str(seed)][metric] - scored[first][str(seed)][metric]
                                                       for seed in config["seeds"]]) for metric in SCORE_NAMES}
        for policy in ("final40", "inner_ce"):
            name = f"{policy}/I3/{model}"
            per_seed = predictions[name]
            error = {"stable_wrong": [], "stable_CHID_FN": [], "unanimous_CHID_to_CHIN": [], "stable_CHID_FP": [],
                     "seed_unanimous": 0, "correct_all_seeds": 0, "any_wrong": 0}
            for sid, ref in identity.items():
                y = ref["y"]
                pred = [int(per_seed[str(seed)][sid]["y_pred"]) for seed in config["seeds"]]
                wrong = [p != y for p in pred]
                error["seed_unanimous"] += int(len(set(pred)) == 1)
                error["correct_all_seeds"] += int(not any(wrong))
                error["any_wrong"] += int(any(wrong))
                if all(wrong): error["stable_wrong"].append(sid)
                if y == 2 and all(wrong): error["stable_CHID_FN"].append(sid)
                if y == 2 and pred == [1, 1, 1]: error["unanimous_CHID_to_CHIN"].append(sid)
                if y != 2 and pred == [2, 2, 2]: error["stable_CHID_FP"].append(sid)
            errors[name] = error
        for sid, ref in identity.items():
            row = {"model": model, "structure_id": sid, "true_class": CLASSES[ref["y"]], "length_bp": ref["length_bp"],
                   "group_id": ref["group_id"], "historical_difficult_CHID": sid in DIFFICULT}
            for policy in ("final40", "inner_ce"):
                samples = [predictions[f"{policy}/I3/{model}"][str(seed)][sid] for seed in config["seeds"]]
                row[policy + "_predictions"] = "/".join(CLASSES[int(r["y_pred"])] for r in samples)
                row[policy + "_correct_seeds"] = sum(int(r["y_true"]) == int(r["y_pred"]) for r in samples)
                row[policy + "_p_true_mean"] = float(np.mean([float(r[f"p_{CLASSES[ref['y']]}"]) for r in samples]))
                row[policy + "_fold"] = int(samples[0]["fold"])
            row["correct_seed_change"] = row["inner_ce_correct_seeds"] - row["final40_correct_seeds"]
            case_changes.append(row)
    learning = {}
    for key, trajectories in histories.items():
        fit_records = [read_json(job_directory(run_dir, job) / "metrics.json") for job in jobs
                       if job["kind"] == "neural" and f"{job['condition']}/{job['model']}" == key]
        selection_stats = describe([r["epoch"] for r in selected[key]])
        selection_stats["fit_sample_sd"] = selection_stats.pop("seed_sample_sd")
        learning[key] = {"fits": len(trajectories), "accepted": sum(r["acceptance"]["accepted"] for r in fit_records),
                         "final_train_ce_median": float(np.median([h[40]["weighted_train_ce"] for h in trajectories])),
                         "final_validation_ce_median": float(np.median([h[40]["weighted_validation_ce"] for h in trajectories])),
                         "minimum_effective_head_gradient_fraction_tail": min(r["acceptance"]["primary"]["gradients"]["head"]["nonzero_fraction_min"] for r in fit_records),
                         "minimum_effective_encoder_gradient_fraction_tail": min(r["acceptance"]["primary"]["gradients"]["encoder"]["nonzero_fraction_min"] for r in fit_records),
                         "minimum_head_gradient_norm_tail": min(r["acceptance"]["primary"]["gradients"]["head"]["mean_norm_min"] for r in fit_records),
                         "terminal_train_single_class_fits": sum(h[-1]["train_diagnostics"]["num_predicted_classes"] == 1 for h in trajectories),
                         "terminal_validation_single_class_fits": sum(h[-1]["validation_diagnostics"]["num_predicted_classes"] == 1 for h in trajectories),
                         "inner_selected_epoch": selection_stats,
                         "selected_epochs": selected[key]}
    public = {"status": "passed", "experiment_signature": manifest["experiment_signature"], "source_signature": manifest["source_signature"],
              "dataset_signature": manifest["dataset_signature"], "selection_signature": manifest["selection_signature"],
              "manifest_sha256": sha256(run_dir / "manifest.json"), "summary_sha256": sha256(run_dir / "summary.json"),
              "checks": {"fits": len(jobs), "neural_fits": len(jobs) - 20, "endpoints": endpoints, "complete_oof": len(expected_oof),
                         "structures_each_oof": 344, "sealed_scientific_files": files, "sealed_scientific_bytes": byte_count,
                         "formal_final40_baseline_exact_CSV_fits": len(formal_reproduction),
                         "formal_full40_exact_numeric_trajectories_excluding_runtime": sum(r["history_exact"] is True for r in formal_reproduction)},
              "fit_artifact_index_sha256": index_hashes, "seed_scores": scored, "comparison": comparison, "paired_policy_deltas": paired,
              "learning": learning, "I3_errors": errors, "outer_endpoint_prediction_events": dict(endpoint_prediction_events),
              "execution": "Saved evidence only: no training, preprocessing, checkpoint loading or model inference",
              "scope": "Previously inspected spatial CV; development results, not independent biological morphology validation",
              "seed_sd": "Three optimization seeds; sample SD is not a population confidence interval"}
    return {"audit": public, "case_changes": case_changes, "predictions": predictions, "histories": histories}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/task1_reevaluation_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/task1_reevaluation_review_v1")
    args = parser.parse_args()
    require(args.output.resolve().is_relative_to((ROOT / "reports").resolve()), "Write derived audit only inside reports/")
    data = audit(args.run_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(data["audit"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    changes = data["case_changes"]
    with (args.output / "case_changes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(changes[0]))
        writer.writeheader(); writer.writerows(changes)
    print(json.dumps({"status": "passed", "checks": data["audit"]["checks"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
