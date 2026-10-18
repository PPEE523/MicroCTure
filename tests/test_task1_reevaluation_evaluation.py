"""Handwritten CSV/history fixtures; no images, model inference or fitting."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from microcture.task1.evaluation import CLASS_NAMES, metrics, save_prediction_csv, _read_prediction_csv
from microcture.task1_controlled.diagnostics import learning_acceptance
from microcture.task1_reevaluation import evaluation as ev
from microcture.task1_reevaluation.protocol import endpoint_directory, json_signature, planned_jobs, sha256
from microcture.task1_reevaluation.selection import endpoint_selection_records

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/task1_reevaluation.json").read_text())


def diagnostic(probability, labels):
    probability = np.asarray(probability, dtype=float)
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(probability.argmax(axis=1), minlength=3)
    logits = np.log(probability)
    ranges = np.ptp(logits, axis=0)
    variance = float(np.var(logits, axis=0).mean())
    probability_variance = float(np.var(probability, axis=0).mean())
    score = metrics(labels, probability)
    return {"n_structures": len(labels), "n_replicate_examples": 2 * len(labels), "metrics": score,
        "logits": {"max_coordinate_range": float(ranges.max())},
        "centered_logits": {"max_coordinate_range": float(ranges.max())},
        "probabilities": {"max_coordinate_range": float(np.ptp(probability, axis=0).max())},
        "head": {"available": True, "active_units": 32, "effective_fraction": 1., "epsilon": 1e-6},
        "logit_between_structure_var": variance, "centered_logit_between_structure_var": variance,
        "probability_between_structure_var": probability_variance,
        "logits_constant": bool(ranges.max() <= 1e-6), "centered_logits_constant": bool(ranges.max() <= 1e-6),
        "probabilities_constant": bool(np.ptp(probability, axis=0).max() <= 1e-6),
        "majority_prediction_fraction": float(counts.max() / len(labels)), "num_predicted_classes": int(np.count_nonzero(counts)),
        "prediction_counts": dict(zip(CLASS_NAMES, counts.tolist()))}


def fixture_history():
    y = np.arange(3)
    correct = np.eye(3) * .85 + .05
    wrong = np.roll(correct, 1, axis=1)
    constant = np.tile([.25, .5, .25], (3, 1))
    history = []
    for epoch in range(41):
        train_probability = np.eye(3) * (.02 * epoch + .02) + (1 - (.02 * epoch + .02)) / 3
        train = diagnostic(train_probability, y)
        validation_probability = correct if epoch == 0 else constant if epoch == 5 else wrong
        validation = diagnostic(validation_probability, y)
        history.append({"epoch": epoch, "weighted_train_ce": float(-np.log(train_probability[y, y]).mean()),
            "weighted_validation_ce": float(-np.log(validation_probability[y, y]).mean()),
            "train_diagnostics": train, "validation_diagnostics": validation,
            "gradient_norms": None if epoch == 0 else {name: {"mean": 1., "min": .5, "max": 2., "finite": True,
                "nonzero_fraction": 1.} for name in ("total", "encoder", "head")}})
    return history


def tiny_metadata():
    structures = [{"structure_id": f"{name}_{fold}", "type": name, "start": str(fold * 1000 + cls * 100),
                   "end": str(fold * 1000 + cls * 100 + 50)} for fold in range(5) for cls, name in enumerate(CLASS_NAMES)]
    groups = [fold for fold in range(5) for cls in range(3)]
    ids_by_fold = [[f"{name}_{fold}" for name in CLASS_NAMES] for fold in range(5)]
    folds = {str(fold): {"test_ids": ids_by_fold[fold], "validation_ids": ids_by_fold[(fold + 1) % 5],
             "train_ids": [sid for number, ids in enumerate(ids_by_fold) if number not in (fold, (fold + 1) % 5) for sid in ids]}
             for fold in range(5)}
    return {"signature": "fixture_dataset", "structures": structures, "group_id": groups}, folds


def prediction_fixture(directory, fold, policy="final40"):
    labels = np.repeat(np.arange(3), 2)
    probability = np.repeat(np.tile([.25, .5, .25], (3, 1)) if policy == "inner_ce" else np.eye(3) * .85 + .05, 2, axis=0)
    samples = {"y": labels, "structure_id": np.repeat([f"{name}_{fold}" for name in CLASS_NAMES], 2),
               "group_id": np.full(6, fold), "replicate": np.tile(["1", "2"], 3)}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "predictions.csv"
    save_prediction_csv(path, samples, probability)
    return _read_prediction_csv(path)


class ReevaluationEvaluationTests(unittest.TestCase):
    def test_saved_endpoint_epoch_and_weight_hash_checked_against_inner_history(self):
        with tempfile.TemporaryDirectory(prefix="reevaluation-endpoint-fixture-", dir="/tmp") as temporary:
            run_dir = Path(temporary)
            metadata, fold_ids = tiny_metadata()
            identity = ev._identity(metadata)
            job = planned_jobs(CONFIG, "main", "m2", fold=0, seed=42)[0]
            history = fixture_history()
            signature = json_signature(CONFIG["selection"])
            manifest = {"config": CONFIG, "experiment_signature": "fixture_experiment", "source_signature": "fixture_source",
                        "dataset_signature": metadata["signature"], "selection_signature": signature}
            endpoint = endpoint_directory(run_dir, job, "inner_ce")
            endpoint.mkdir(parents=True)
            (endpoint / "weights.pt").write_bytes(b"fixture weights; never deserialized")
            outer = prediction_fixture(run_dir / "outer-fixture", 0, "inner_ce")
            validation = prediction_fixture(run_dir / "validation-fixture", 1, "inner_ce")
            (endpoint / "outer_predictions.csv").write_bytes((run_dir / "outer-fixture/predictions.csv").read_bytes())
            (endpoint / "validation_predictions.csv").write_bytes((run_dir / "validation-fixture/predictions.csv").read_bytes())
            checkpoint_hash = sha256(endpoint / "weights.pt")
            marker = {"job": job, "policy": "inner_ce", "selected_epoch": 5, "selection_signature": signature,
                      "checkpoint": "weights.pt", "checkpoint_sha256": checkpoint_hash,
                      "outer_images_constructed": False, "training_and_selection_complete": True}
            selected = history[5]
            value = {**manifest, "status": "completed", "job": job, "policy": "inner_ce", "selected_epoch": 5,
                     "selection_signature": signature, "checkpoint_sha256": checkpoint_hash,
                     "preprocessing_signature": "fixture_preprocessing", "outer_fit_or_selection": False,
                     "outer_constructed_after_fixed_endpoint": True,
                     "train_diagnostics": selected["train_diagnostics"], "validation_diagnostics": selected["validation_diagnostics"],
                     "weighted_train_ce": selected["weighted_train_ce"], "weighted_validation_ce": selected["weighted_validation_ce"],
                     "outer_metrics": outer["metrics"], "outer_diagnostics": diagnostic(outer["probabilities"], np.arange(3))}
            (endpoint / "endpoint.json").write_text(json.dumps(marker))
            (endpoint / "metrics.json").write_text(json.dumps(value))
            common = {"preprocessing_signature": "fixture_preprocessing"}
            result = ev._load_endpoint(run_dir, job, "inner_ce", manifest, common, history, identity,
                                       {int(key): fold for key, fold in fold_ids.items()})
            self.assertEqual(result["metrics"]["selected_epoch"], 5)
            self.assertEqual(result["metrics"]["outer_diagnostics"]["num_predicted_classes"], 1)
            marker["selected_epoch"] = value["selected_epoch"] = 6
            (endpoint / "endpoint.json").write_text(json.dumps(marker))
            (endpoint / "metrics.json").write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "preregistered inner-only selection"):
                ev._load_endpoint(run_dir, job, "inner_ce", manifest, common, history, identity,
                                  {int(key): fold for key, fold in fold_ids.items()})
            marker["selected_epoch"] = value["selected_epoch"] = 5
            (endpoint / "endpoint.json").write_text(json.dumps(marker))
            (endpoint / "metrics.json").write_text(json.dumps(value))
            (endpoint / "weights.pt").write_bytes(b"tampered fixture weight")
            with self.assertRaisesRegex(ValueError, "checkpoint identity/bytes"):
                ev._load_endpoint(run_dir, job, "inner_ce", manifest, common, history, identity,
                                  {int(key): fold for key, fold in fold_ids.items()})

    def run_summary(self, complete=True, failed_learning=False):
        with tempfile.TemporaryDirectory(prefix="reevaluation-eval-fixture-", dir="/tmp") as temporary:
            directory = Path(temporary)
            metadata, folds = tiny_metadata()
            config = copy.deepcopy(CONFIG)
            config["run_dir"] = str(directory)
            manifest = {"config": config, "requested_device": "cpu", "protocol": CONFIG["protocol"],
                "source_signature": "fixture_source", "dataset_signature": metadata["signature"],
                "selection_signature": json_signature(CONFIG["selection"]), "fold_ids": folds}
            manifest["experiment_signature"] = json_signature(manifest)
            (directory / "manifest.json").write_text(json.dumps(manifest))
            history = fixture_history()
            acceptance = learning_acceptance(history, CONFIG)
            self.assertTrue(acceptance["accepted"])
            predictions = {(fold, policy): prediction_fixture(directory / "fixture" / policy / str(fold), fold, policy)
                           for fold in range(5) for policy in ("final40", "inner_ce", "baseline")}
            if not complete:
                historical = directory / "oof/final40/I3/m2/seed_42/metrics.json"
                historical.parent.mkdir(parents=True, exist_ok=True)
                historical.write_text('{"historical": true}\n')
            def saved_fit(run_dir, job, supplied_manifest, identity, frozen_folds):
                if not complete and (job["condition"], job["model"], job["seed"], job["fold"]) == ("I3", "m2", 42, 0):
                    return {"job": job, "status": "pending", "endpoints": {}}
                value = {"runtime_seconds": 1., "preprocessing_signature": f"fold_{job['fold']}",
                    "initial_state_sha256": f"{job['model']}_{job['seed']}", "common_initial_state_sha256": "common",
                    "minibatch_order_sha256": f"seed_{job['seed']}", "class_weights": [1., 1., 1.],
                    "final_train_ce": history[-1]["weighted_train_ce"], "acceptance": copy.deepcopy(acceptance),
                    "train_diagnostics": history[-1]["train_diagnostics"], "validation_diagnostics": history[-1]["validation_diagnostics"]}
                if failed_learning:
                    value["acceptance"]["accepted"] = False
                endpoints = {}
                policies = ("final40", "inner_ce") if job["kind"] == "neural" else ("baseline",)
                for policy in policies:
                    selected = 5 if policy == "inner_ce" else 40
                    endpoint_value = {"selected_epoch": selected if job["kind"] == "neural" else None,
                        "weighted_train_ce": history[selected]["weighted_train_ce"], "weighted_validation_ce": history[selected]["weighted_validation_ce"],
                        "train_diagnostics": history[selected]["train_diagnostics"], "validation_diagnostics": history[selected]["validation_diagnostics"],
                        "outer_diagnostics": diagnostic(predictions[(job["fold"], policy)]["probabilities"], np.arange(3))}
                    endpoints[policy] = {"status": "completed", "metrics": endpoint_value, "predictions": predictions[(job["fold"], policy)]}
                return {"job": job, "status": "completed", "metrics": value, "history": history, "endpoints": endpoints}
            with patch.object(ev, "prepare_manifest", return_value=(manifest, metadata)), patch.object(ev, "_load_fit", side_effect=saved_fit):
                summary = ev.summarize_reevaluation(directory, metadata)
            if not complete:
                self.assertEqual(historical.read_text(), '{"historical": true}\n')
                stale_status = json.loads((historical.parent / "status.json").read_text())
                self.assertFalse(stale_status["oof_current"])
                self.assertTrue(stale_status["previous_numeric_artifacts_retained"])
            return summary

    def test_two_policy_constant_early_checkpoint_is_scored_and_not_full40_terminal(self):
        result = self.run_summary()
        self.assertEqual((result["status"], result["completed_fits"], result["completed_endpoints"], result["complete_oof"]), ("complete", 155, 290, 58))
        self.assertEqual((result["main_status"], result["main_completed_fits"], result["main_completed_endpoints"], result["main_complete_oof"]), ("complete", 50, 95, 19))
        self.assertEqual(result["total_fit_runtime_seconds"], 155.)
        self.assertEqual(sum(v["strict_learning_passes"] for v in result["shared_full40_learning"].values()), 135)
        fit = result["shared_full40_learning"]["I3/m2"]
        self.assertEqual(fit["prediction_events"]["validation"]["single_class_argmax"]["final_cases"], 0)
        endpoint = result["models"]["inner_ce/I3/m2"]["selected_endpoint_events"]
        self.assertEqual(endpoint["validation"]["events"]["single_class_argmax"], 15)
        self.assertEqual(endpoint["outer"]["events"]["constant_probability"], 15)
        inner_score = next(row for row in result["comparison"] if (row["policy"], row["condition"], row["model"]) == ("inner_ce", "I3", "m2"))
        self.assertAlmostEqual(inner_score["macro_f1_mean"], 1 / 6)
        self.assertEqual(inner_score["macro_f1_seed_sample_std"], 0.)
        self.assertEqual(result["paired_policy_comparisons"]["I3/m2"]["complete_seed_pairs"], 3)
        self.assertEqual(result["paired_policy_comparisons"]["inner_ce/m2/I3-G0"]["complete_seed_pairs"], 3)

    def test_missing_fold_blocks_seed_and_aggregate_without_zero_fill(self):
        result = self.run_summary(complete=False)
        self.assertEqual((result["status"], result["pending_fits"], result["complete_oof"]), ("pending", 1, 56))
        row = next(row for row in result["comparison"] if (row["policy"], row["condition"], row["model"]) == ("final40", "I3", "m2"))
        self.assertEqual(row["complete_seeds"], 2)
        self.assertIsNone(row["macro_f1_mean"])
        self.assertIsNone(result["paired_policy_comparisons"]["I3/m2"]["macro_f1_delta_mean"])

    def test_failed_learning_criteria_do_not_filter_completed_predictions(self):
        result = self.run_summary(failed_learning=True)
        self.assertEqual((result["completed_fits"], result["completed_endpoints"], result["complete_oof"]), (155, 290, 58))
        self.assertEqual(sum(v["strict_learning_passes"] for v in result["shared_full40_learning"].values()), 0)
        baseline = next(row for row in result["comparison"] if row["model"] == "pca_lr")
        self.assertIsNone(baseline["macro_f1_seed_sample_std"])


if __name__ == "__main__":
    unittest.main()
