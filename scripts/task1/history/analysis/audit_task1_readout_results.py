"""Read-only audit of saved readout controls; no model/matrix deserialization.

Manifest/dependency checks reuse the frozen protocol validators. Prediction
arithmetic is independently recomputed from CSV confusion counts, and derived
OOF rows must match the sealed fold CSVs exactly. Only an audit JSON is written.
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
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = _layout_root
sys.path.insert(0, str(ROOT))

from microcture.task1_controlled.diagnostics import learning_acceptance
from microcture.task1_controlled.runner import flat_history_row
from microcture.task1_readout.protocol import (job_directory, json_signature,
    planned_jobs, prepare_manifest, sha256)
from microcture.task1_readout.runner import unpublished_tmp_staging

CLASSES = ("OPCID", "CHIN", "CHID")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def near(actual, expected, label: str) -> None:
    require(np.isclose(actual, expected, rtol=0, atol=1e-10), f"{label} differs: {actual!r} != {expected!r}")


def scores(predictions: list[dict]) -> dict:
    """Confusion-count arithmetic, independent of the training evaluator."""
    cm = np.zeros((3, 3), dtype=np.int64)
    for row in predictions:
        y = int(row["y_true"])
        p = np.asarray([float(row[f"p_{name}"]) for name in CLASSES])
        require(y in range(3) and np.isfinite(p).all() and np.all(p >= 0)
                and np.all(p <= 1), "Invalid prediction probabilities/label")
        require(abs(float(p.sum()) - 1) < 1e-5, "Probability sum differs from one")
        predicted = int(p.argmax())
        require(predicted == int(row["y_pred"]), "Saved argmax differs from probabilities")
        replicate_rows = json.loads(row["replicate_probabilities_json"])
        require(len(replicate_rows) == int(row["replicate_count"]) == 2
                and {str(rep["replicate"]) for rep in replicate_rows} == {"1", "2"},
                "Exactly the two WT replicates are required")
        rp = np.asarray([rep["probabilities"] for rep in replicate_rows], dtype=float)
        require(rp.shape == (2, 3) and np.isfinite(rp).all(), "Invalid replicate probabilities")
        require(np.allclose(p, rp.mean(axis=0), atol=1e-12, rtol=0),
                "Structure probability differs from replicate average")
        require(all(int(rep["predicted"]) == int(np.argmax(rep["probabilities"]))
                    for rep in replicate_rows), "Replicate argmax differs")
        require(int(row["replicate_disagree"]) == int(len(set(rp.argmax(axis=1))) > 1),
                "Replicate disagreement flag differs")
        cm[y, predicted] += 1
    support, predicted_support = cm.sum(axis=1), cm.sum(axis=0)
    precision = np.divide(cm.diagonal(), predicted_support, out=np.zeros(3), where=predicted_support > 0)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2 * cm.diagonal(), support + predicted_support,
                   out=np.zeros(3), where=support + predicted_support > 0)
    return {"accuracy": float(cm.trace() / cm.sum()), "balanced_accuracy": float(recall.mean()),
            "macro_f1": float(f1.mean()), "confusion_matrix": cm.tolist(),
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                                  "f1": float(f1[i]), "support": int(support[i])}
                          for i, name in enumerate(CLASSES)},
            "prediction_counts": dict(zip(CLASSES, predicted_support.tolist()))}


def compare_scores(actual: dict, expected: dict, label: str) -> None:
    require(actual["confusion_matrix"] == expected["confusion_matrix"], f"{label} confusion differs")
    for key in ("accuracy", "balanced_accuracy", "macro_f1"):
        near(actual[key], expected[key], f"{label}.{key}")
    for name in CLASSES:
        for key in ("precision", "recall", "f1", "support"):
            near(actual["per_class"][name][key], expected["per_class"][name][key], f"{label}.{name}.{key}")


def describe(values) -> dict:
    return {"mean": float(np.mean(values)), "seed_sample_sd": float(np.std(values, ddof=1)),
            "min": float(min(values)), "max": float(max(values))}


def audit(root: Path, run_dir: Path) -> dict:
    manifest = read_json(run_dir / "manifest.json")
    require(json_signature({k: v for k, v in manifest.items() if k != "experiment_signature"})
            == manifest["experiment_signature"], "Manifest canonical signature differs")
    current_manifest, metadata = prepare_manifest(root, manifest["config"], manifest["requested_device"])
    require(current_manifest == manifest, "Current source/dependencies/config/environment differs from trained manifest")
    config = manifest["config"]
    jobs = planned_jobs(config)
    require(len(jobs) == 60, "Expected exactly sixty registered fits")
    require(len(list((run_dir / "runs").rglob("ARTIFACTS.json"))) == 60,
            "Unexpected completed or failed artifact directories exist")
    require(not (run_dir / "failed_staging").exists()
            and not unpublished_tmp_staging(root, run_dir), "Unpublished failed staging exists")
    identity = {row["structure_id"]: {**row, "y": CLASSES.index(row["type"]),
                                     "group_id": int(group)}
                for row, group in zip(metadata["structures"], metadata["group_id"])}
    folds, transform_hashes = {}, {}
    for number in config["folds"]:
        frozen = read_json(root / config["frozen_run_dir"] / "folds" / f"fold_{number}.json")
        shared = read_json(run_dir / "folds" / f"fold_{number}.json")
        require(shared["frozen_fold"] == frozen, "Saved shared fold differs from frozen reference")
        prep = shared["preprocessing"]
        require(json_signature({k: v for k, v in prep.items() if k != "signature"}) == prep["signature"],
                "Shared preprocessing canonical signature differs")
        require(prep["signal_preprocessing"] == frozen["preprocessing"]
                and prep["signal_statistics_refitted"] is False and prep["splits_refitted"] is False
                and prep["outer_images_used"] is False, "Signal preprocessing/split fitting scope differs")
        require(prep["train_ids"] == frozen["train_ids"]
                and prep["validation_ids"] == frozen["validation_ids"], "Inner preprocessing ID list differs")
        lengths = np.asarray([np.log1p((int(identity[sid]["end"]) - int(identity[sid]["start"])) / 100)
                              for sid in frozen["train_ids"]])
        scaler = prep["length_scaler"]
        near(scaler["mean"], float(lengths.mean()), "inner-only length mean")
        near(scaler["raw_std"], float(lengths.std(ddof=0)), "inner-only length population std")
        near(scaler["std"], float(lengths.std(ddof=0)) if lengths.std(ddof=0) >= 1e-6 else 1., "length used std")
        require(scaler["fitted_unique_structures"] == len(frozen["train_ids"])
                and scaler["fit_scope"] == "frozen inner-training structures only", "Length fit scope differs")
        folds[number], transform_hashes[number] = frozen, prep["signature"]
    artifact_files, artifact_bytes = 0, 0
    record_scores, outer_rows, pair_records, trajectories = {}, defaultdict(list), defaultdict(list), defaultdict(list)
    index_hashes = {}
    for job in jobs:
        directory = job_directory(run_dir, job)
        index = read_json(directory / "ARTIFACTS.json")
        require(index["job"] == job and index["experiment_signature"] == manifest["experiment_signature"]
                and index["signature"] == json_signature(index["files"]), "Artifact index binding/signature differs")
        indexed = set()
        for item in index["files"]:
            name = item["name"]
            require(Path(name).name == name and name not in indexed and name != "ARTIFACTS.json", "Unsafe artifact name")
            indexed.add(name)
            path = directory / name
            require(path.is_file() and not path.is_symlink() and path.stat().st_size == item["size_bytes"]
                    and sha256(path) == item["sha256"], f"Artifact bytes differ: {path}")
            artifact_files += 1
            artifact_bytes += item["size_bytes"]
        require({p.name for p in directory.iterdir()} == indexed | {"ARTIFACTS.json"}, "Artifact inventory differs")
        require(indexed == {"STAGING_CONTEXT.json", "endpoint.json", "final.pt", "history.csv", "history.json",
                            "initialization.json", "metrics.json", "outer_predictions.csv", "protocol.json",
                            "validation_predictions.csv"}, "Completed scientific inventory differs")
        index_hashes[str(directory.relative_to(run_dir))] = sha256(directory / "ARTIFACTS.json")
        result, protocol = read_json(directory / "metrics.json"), read_json(directory / "protocol.json")
        require(result["status"] == "completed" and result["job"] == protocol["job"] == job, "Job completion differs")
        for key in ("experiment_signature", "source_signature", "dataset_signature"):
            require(result[key] == manifest[key], f"Job {key} binding differs")
        require(result["preprocessing_signature"] == transform_hashes[job["fold"]]
                and protocol["preprocessing"] == read_json(run_dir / "folds" / f"fold_{job['fold']}.json")["preprocessing"],
                "Job preprocessing differs from shared fitted transform")
        require(protocol["training"] == config["training"] and protocol["acceptance"] == config["acceptance"]
                and protocol["gate_beta"] == config["gate_beta"], "Job frozen training protocol differs")
        require(result["parameter_spec"]["parameters"] == result["parameter_spec"]["trainable_parameters"] == 110437,
                "Parameter matching differs")
        endpoint = read_json(directory / "endpoint.json")
        require(endpoint["job"] == job and endpoint["checkpoint"] == "final.pt"
                and endpoint["checkpoint_sha256"] == result["checkpoint_sha256"] == sha256(directory / "final.pt")
                and endpoint["outer_images_constructed"] is False and endpoint["training_and_selection_complete"] is True
                and result["outer_fit_or_selection"] is False and result["outer_constructed_after_fixed_endpoint"] is True,
                "Fixed checkpoint / pre-outer marker differs")
        initialization = read_json(directory / "initialization.json")
        require(initialization["job"] == job and initialization["seed"] == job["seed"], "Initialization binding differs")
        for key in ("initial_state_sha256", "common_initial_state_sha256"):
            require(initialization[key] == result[key], f"Initialization {key} differs")
        counts = np.bincount([identity[sid]["y"] for sid in folds[job["fold"]]["train_ids"]], minlength=3)
        expected_weights = counts.sum() / (3 * counts)
        require(np.allclose(result["class_weights"], expected_weights, atol=1e-12, rtol=0), "Class weights differ from inner IDs")
        pair_records[(job["fold"], job["seed"])].append(result)
        history = read_json(directory / "history.json")["epochs"]
        require([r["epoch"] for r in history] == list(range(41)) and result["epochs"] == 40, "History is not epochs zero through forty")
        acceptance = learning_acceptance(history, config)
        require(acceptance == result["acceptance"], "Full-trajectory learning acceptance differs from frozen criteria")
        near(result["initial_train_ce"], history[0]["weighted_train_ce"], "Initial CE")
        near(result["final_train_ce"], history[-1]["weighted_train_ce"], "Final CE")
        csv_history = rows(directory / "history.csv")
        require(len(csv_history) == 41, "History CSV row count differs")
        for row, csv_row in zip(history, csv_history):
            flat = flat_history_row(row)
            require(set(flat) == set(csv_row), "History CSV fields differ")
            for key, value in flat.items():
                require(csv_row[key] == ("" if value is None else str(value)), f"History CSV differs: {key}")
            for view in ("train", "validation"):
                diagnostic = row[f"{view}_diagnostics"]
                counts_view = np.bincount([identity[sid]["y"] for sid in folds[job["fold"]][f"{view}_ids"]], minlength=3)
                require(diagnostic["n_structures"] == counts_view.sum()
                        and diagnostic["n_replicate_examples"] == 2 * counts_view.sum(), "History diagnostic support differs")
                require(diagnostic["loss_support"]["examples_count_by_class"] == (counts_view * 2).tolist(),
                        "History fixed loss class support differs")
                nll = np.asarray(diagnostic["loss_support"]["unweighted_nll_sum_by_class"])
                fixed_ce = float((nll * expected_weights).sum() / (expected_weights * 2 * counts_view).sum())
                near(row[f"weighted_{view}_ce"], fixed_ce, "History weighted fixed CE")
        for view in ("train", "validation"):
            require(result[f"{view}_diagnostics"] == history[-1][f"{view}_diagnostics"], "Last diagnostics differ")
        outer, validation = rows(directory / "outer_predictions.csv"), rows(directory / "validation_predictions.csv")
        for view, prediction in (("test", outer), ("validation", validation)):
            require(len(prediction) == len({r["structure_id"] for r in prediction})
                    and {r["structure_id"] for r in prediction} == set(folds[job["fold"]][f"{view}_ids"]),
                    "Prediction IDs differ from frozen split")
            for row in prediction:
                ref = identity[row["structure_id"]]
                require(int(row["y_true"]) == ref["y"] and int(row["group_id"]) == ref["group_id"],
                        "Prediction identity/label/group differs")
        score = scores(outer)
        compare_scores(result["outer_metrics"], score, "Saved fold outer metrics")
        compare_scores(result["outer_diagnostics"]["metrics"], score, "Saved outer diagnostics metrics")
        compare_scores(result["validation_diagnostics"]["metrics"], scores(validation), "Saved validation metrics")
        require(result["outer_diagnostics"]["prediction_counts"] == score["prediction_counts"], "Outer predicted-class counts differ")
        record_scores[(job["condition"], job["model"], job["seed"], job["fold"])] = score
        outer_rows[(job["condition"], job["model"], job["seed"])].extend([{**row, "fold": str(job["fold"])} for row in outer])
        tail = history[-10:]
        trajectories[(job["condition"], job["model"])].append({
            "accepted": acceptance["accepted"], "final_train_ce": result["final_train_ce"],
            "effective_gradient_every_epoch": all(row["gradient_norms"][part]["finite"]
                and row["gradient_norms"][part]["mean"] > 1e-10
                and row["gradient_norms"][part]["nonzero_fraction"] > 0
                for row in history[1:] for part in ("total", "encoder", "head")),
            "tail_head_gradient_min": min(row["gradient_norms"]["head"]["mean"] for row in tail),
            "tail_head_effective_fraction_min": min(row[f"{view}_diagnostics"]["head"]["effective_fraction"]
                for row in tail for view in ("train", "validation")),
            "tail_probability_constant_epochs": sum(row[f"{view}_diagnostics"]["probabilities"]["max_coordinate_range"] <= 1e-6
                for row in tail for view in ("train", "validation")),
            "tail_argmax_constant_epochs": sum(row[f"{view}_diagnostics"]["num_predicted_classes"] == 1
                for row in tail for view in ("train", "validation")),
            "outer_argmax_constant": sum(value > 0 for value in score["prediction_counts"].values()) == 1,
            "runtime_seconds": result["runtime_seconds"]})
    for key, records in pair_records.items():
        require(len(records) == 4, "Each fold/seed requires four paired fits")
        for field in ("initial_state_sha256", "common_initial_state_sha256", "minibatch_order_sha256", "preprocessing_signature", "class_weights"):
            require(len({json.dumps(r[field], sort_keys=True) for r in records}) == 1, f"Paired {field} differs at {key}")
    oof_scores, seed_rows = {}, []
    for key, derived_rows in outer_rows.items():
        condition, variant, seed = key
        directory = run_dir / "oof" / condition / variant / f"seed_{seed}"
        stored_rows, status, saved_score = rows(directory / "predictions.csv"), read_json(directory / "status.json"), read_json(directory / "metrics.json")
        expected_rows = sorted(derived_rows, key=lambda row: row["structure_id"])
        require(stored_rows == expected_rows, f"OOF CSV differs from sealed fold CSV rows: {key}")
        require(len(stored_rows) == len({r["structure_id"] for r in stored_rows}) == 344
                and {r["structure_id"] for r in stored_rows} == set(identity), "OOF unique structure coverage differs")
        require(status["status"] == "complete" and status["oof_current"] is True
                and status["expected_folds"] == list(range(5)) and status["n_structures"] == 344, "OOF completeness status differs")
        for artifact in (status, saved_score):
            for field in ("experiment_signature", "source_signature", "dataset_signature"):
                require(artifact[field] == manifest[field], f"OOF {field} differs")
        score = scores(stored_rows)
        compare_scores(saved_score, score, "Saved OOF metrics")
        require([score["per_class"][name]["support"] for name in CLASSES] == [68, 250, 26], "OOF class support differs")
        oof_scores[key] = score
        seed_rows.append({"condition": condition, "variant": variant, "seed": seed,
                          "macro_f1": score["macro_f1"], "CHID_f1": score["per_class"]["CHID"]["f1"],
                          "balanced_accuracy": score["balanced_accuracy"], "confusion_matrix": score["confusion_matrix"],
                          "prediction_counts": score["prediction_counts"]})
    require(len(oof_scores) == 12, "Expected twelve complete OOF result sets")
    comparison, deltas = [], {}
    saved_comparison = {(r["condition"], r["model"]): r for r in rows(run_dir / "comparison.csv")}
    for condition in ("I3", "G0"):
        for variant in ("O0", "O1"):
            values = [oof_scores[(condition, variant, seed)] for seed in (42, 43, 44)]
            row = {"condition": condition, "variant": variant}
            for name in ("macro_f1", "accuracy", "balanced_accuracy"):
                stat = describe([v[name] for v in values])
                row[name] = stat
                near(float(saved_comparison[(condition, variant)][f"{name}_mean"]), stat["mean"], "Saved seed mean")
                near(float(saved_comparison[(condition, variant)][f"{name}_seed_sample_std"]), stat["seed_sample_sd"], "Saved seed sample SD")
            row["per_class"] = {}
            for name in CLASSES:
                row["per_class"][name] = {}
                for measure in ("precision", "recall", "f1"):
                    stat = describe([v["per_class"][name][measure] for v in values])
                    row["per_class"][name][measure] = stat
                    near(float(saved_comparison[(condition, variant)][f"{name}_{measure}_mean"]), stat["mean"], "Saved class seed mean")
                    near(float(saved_comparison[(condition, variant)][f"{name}_{measure}_seed_sample_std"]), stat["seed_sample_sd"], "Saved class seed SD")
            records = trajectories[(condition, variant)]
            loss_stat = describe([r["final_train_ce"] for r in records])
            loss_stat["fit_sample_sd"] = loss_stat.pop("seed_sample_sd")
            row["learning"] = {"passes": sum(r["accepted"] for r in records), "fits": len(records),
                "effective_gradient_every_epoch_fits": sum(r["effective_gradient_every_epoch"] for r in records),
                "final_train_ce": loss_stat,
                "tail_head_gradient_min": min(r["tail_head_gradient_min"] for r in records),
                "tail_head_effective_fraction_min": min(r["tail_head_effective_fraction_min"] for r in records),
                "tail_probability_constant_train_or_val_epochs": sum(r["tail_probability_constant_epochs"] for r in records),
                "tail_argmax_constant_train_or_val_epochs": sum(r["tail_argmax_constant_epochs"] for r in records),
                "outer_argmax_constant_fits": sum(r["outer_argmax_constant"] for r in records),
                "runtime_seconds_total": sum(r["runtime_seconds"] for r in records)}
            comparison.append(row)
        paired = [{"seed": seed,
                   "macro_f1": oof_scores[(condition, "O1", seed)]["macro_f1"] - oof_scores[(condition, "O0", seed)]["macro_f1"],
                   "CHID_f1": oof_scores[(condition, "O1", seed)]["per_class"]["CHID"]["f1"] - oof_scores[(condition, "O0", seed)]["per_class"]["CHID"]["f1"]}
                  for seed in (42, 43, 44)]
        deltas[f"{condition}:O1-O0"] = {"seed_deltas": paired,
            "macro_f1": describe([r["macro_f1"] for r in paired]), "CHID_f1": describe([r["CHID_f1"] for r in paired])}
    summary = read_json(run_dir / "summary.json")
    require(summary["status"] == "complete" and summary["expected_jobs"] == summary["completed_jobs"] == 60
            and all(summary[key] == 0 for key in ("pending_jobs", "invalid_jobs", "numerical_failure_jobs", "execution_failure_jobs"))
            and summary["errors"] == [], "Saved summary completion/integrity differs")
    require(len(summary["comparison"]) == 4, "Saved summary must contain four comparisons")
    for row in summary["comparison"]:
        csv_row = saved_comparison[(row["condition"], row["model"])]
        for key in csv_row:
            if key.endswith("_mean") or key.endswith("_seed_sample_std"):
                near(row[key], float(csv_row[key]), f"Summary JSON versus checked CSV {key}")
    paired_rows = rows(run_dir / "paired_deltas.csv")
    require(len(paired_rows) == 60, "Expected sixty fold/seed rows in the four paired contrasts")
    require(len({(r["comparison"], r["seed"], r["fold"]) for r in paired_rows}) == 60,
            "Repeated saved paired fold/seed rows")
    for row in paired_rows:
        label = row["comparison"]
        seed, fold = int(row["seed"]), int(row["fold"])
        if label in ("I3:O1-O0", "G0:O1-O0"):
            condition = label.split(":")[0]
            first, second = (condition, "O1", seed, fold), (condition, "O0", seed, fold)
        else:
            require(label in ("O0:I3-G0", "O1:I3-G0"), "Unknown saved paired contrast")
            variant = label.split(":")[0]
            first, second = ("I3", variant, seed, fold), ("G0", variant, seed, fold)
        a, b = record_scores[first], record_scores[second]
        near(float(row["macro_f1_delta"]), a["macro_f1"] - b["macro_f1"], "Paired fold macro-F1 delta")
        near(float(row["balanced_accuracy_delta"]), a["balanced_accuracy"] - b["balanced_accuracy"], "Paired fold balanced-accuracy delta")
        near(float(row["CHID_f1_delta"]), a["per_class"]["CHID"]["f1"] - b["per_class"]["CHID"]["f1"], "Paired fold CHID-F1 delta")
    for label, values in deltas.items():
        saved = summary["paired_comparisons"][label]["complete_seed_oof_deltas"]
        require(len(saved) == 3, "Saved paired complete-OOF rows differ")
        for expected, actual in zip(values["seed_deltas"], saved):
            require(actual["seed"] == expected["seed"], "Paired complete-OOF seed differs")
            near(actual["macro_f1_delta"], expected["macro_f1"], "Paired complete-OOF macro delta")
            near(actual["CHID_f1_delta"], expected["CHID_f1"], "Paired complete-OOF CHID delta")
    # O0 was retrained in this round. Exact replay is checked against the old
    # control CSVs, rather than substituting the earlier aggregate score.
    replay_jobs = 0
    for condition in ("I3", "G0"):
        for seed in (42, 43, 44):
            for fold in range(5):
                current = run_dir / "runs" / condition / "O0" / f"seed_{seed}" / f"fold_{fold}" / "outer_predictions.csv"
                formal = root / config["formal_run_dir"] / "runs" / condition / "m2plus" / f"seed_{seed}" / f"fold_{fold}" / "outer_predictions.csv"
                require(current.read_bytes() == formal.read_bytes(), "Retrained O0 CSV is not the exact deterministic formal replay")
                replay_jobs += 1
    return {"status": "passed", "protocol": manifest["protocol"],
        "experiment_signature": manifest["experiment_signature"], "source_signature": manifest["source_signature"],
        "dataset_signature": manifest["dataset_signature"], "audit_script_sha256": sha256(Path(__file__)),
        "manifest_sha256": sha256(run_dir / "manifest.json"),
        "checks": {"current_source_dependency_manifest_equal": True, "jobs": 60,
            "artifact_indices": 60, "artifact_files_sha256_verified": artifact_files, "artifact_bytes_sha256_verified": artifact_bytes,
            "artifact_index_set_signature": json_signature(index_hashes), "full_epoch_0_to_40_histories": 60,
            "history_csv_json_exact_matches": 60, "learning_acceptance_recomputed": 60,
            "inner_only_length_scalers_recomputed": 5, "shared_preprocessing_folds": 5,
            "four_way_initialization_minibatch_pairs": len(pair_records), "fixed_endpoint_before_outer_markers": 60,
            "saved_validation_metrics_recomputed": 60, "saved_outer_metrics_recomputed": 60,
            "oof_row_sets_bound_to_sealed_fold_csvs": 12, "unique_structures_per_oof": 344,
            "paired_fold_seed_delta_rows_recomputed": 60, "unpublished_staging": 0,
            "O0_retrained_exact_formal_outer_csv_replays": replay_jobs,
            "model_deserializations": 0, "contact_matrix_reads": 0, "training_calls": 0},
        "comparison": comparison, "paired_complete_oof_deltas": deltas, "seed_oof": seed_rows,
        "interpretation": "Developmental paired spatial CV. Seed SD measures optimizer variability, not an independent biological confidence interval.",
        "limits": ["Pre-outer ordering is supported by sealed marker fields and signature-verified runner order; no independent execution timestamp attestation exists.",
            "Checkpoint bytes were hashed but not deserialized; epoch40 is verified from recorded full trajectories and bound checkpoint markers.",
            "Original contact matrices and cached bands were metadata checked by the frozen protocol, not read or reprocessed."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/task1_readout_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/task1_readout_review_v1/integrity.json")
    args = parser.parse_args()
    result = audit(ROOT, args.run_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": result["checks"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
