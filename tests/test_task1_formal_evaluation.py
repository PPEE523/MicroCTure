"""Handwritten stored-result fixtures; no training or model inference."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from microcture.task1.evaluation import metrics, save_prediction_csv
from microcture.task1_controlled.diagnostics import learning_acceptance
from microcture.task1_formal.evaluation import summarize_formal
from microcture.task1_formal.protocol import job_directory, json_signature


class FormalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="microcture-formal-eval-", dir="/tmp")
        self.directory = Path(self.temporary.name)
        self.metadata = {"signature": "data", "group_id": list(range(6)), "structures": [
            {"structure_id": f"S{index}", "type": ("OPCID", "CHIN", "CHID")[index % 3],
             "start": 1000 * index, "end": 1000 * index + 100} for index in range(6)]}
        self.manifest = {"config": {"training": {"epochs": 40}, "acceptance": {"window": 10}},
                         "source_signature": "source", "dataset_signature": "data",
                         "fold_ids": {"0": {"train_ids": ["S3", "S4"], "validation_ids": ["S5"], "test_ids": ["S0", "S1", "S2"]},
                                      "1": {"train_ids": ["S0", "S1"], "validation_ids": ["S2"], "test_ids": ["S3", "S4", "S5"]}}}
        self.manifest["experiment_signature"] = json_signature(self.manifest)
        (self.directory / "manifest.json").write_text(json.dumps(self.manifest))

    def tearDown(self):
        self.temporary.cleanup()

    def job(self, fold, seed=42, condition="I3", model="m2", neural=False):
        value = {"stage": "formal", "kind": "neural" if neural else "baseline", "model": model,
                 "condition": condition, "fold": fold, "seed": seed, "input_variant": condition,
                 "name": "H3" if neural else model}
        if neural:
            value.update(backbone=model, head="H3", learning_rate=1e-4)
        return value

    def write(self, job, *, constant=False, status="completed", wrong_group=False, loss=.1, init="init"):
        directory = job_directory(self.directory, job)
        directory.mkdir(parents=True, exist_ok=True)
        fold = job["fold"]
        indices = list(range(fold * 3, fold * 3 + 3))
        y = np.asarray([index % 3 for index in indices], dtype=np.int64)
        p = np.full((3, 3), .05)
        p[np.arange(3), y] = .9
        if constant:
            p[:] = [.05, .9, .05]
        score = metrics(y, p)
        value = {"status": status, "job": job, "experiment_signature": self.manifest["experiment_signature"], "source_signature": "source",
                 "dataset_signature": "data", "preprocessing_signature": f"fit-{fold}", "outer_metrics": score,
                 "train_structures": 2, "validation_structures": 1, "test_structures": 3, "runtime_seconds": 1.0,
                 "initial_train_ce": 1.1, "final_train_ce": loss, "class_weights": [1., 1., 1.]}
        if job["kind"] == "neural":
            diagnostic = {"probabilities_constant": constant, "centered_logits_constant": constant,
                          "logit_between_structure_var": 0. if constant else 1.,
                          "centered_logit_between_structure_var": 0. if constant else 1.,
                          "logits": {"max_coordinate_range": 0. if constant else 1.},
                          "centered_logits": {"max_coordinate_range": 0. if constant else 1.},
                          "probabilities": {"max_coordinate_range": float(np.ptp(p, axis=0).max())},
                          "probability_between_structure_var": float(np.var(p, axis=0).mean()),
                          "head": {"available": True, "active_units": 0 if constant else 32,
                                   "effective_fraction": 0. if constant else 1., "epsilon": 1e-6},
                          "num_predicted_classes": 1 if constant else 3,
                          "majority_prediction_fraction": 1. if constant else 1 / 3,
                          "prediction_counts": dict(zip(("OPCID", "CHIN", "CHID"), np.bincount(p.argmax(axis=1), minlength=3).tolist())),
                          "metrics": score}
            value.update(outer_diagnostics=diagnostic, train_diagnostics=diagnostic, validation_diagnostics=diagnostic,
                         initial_state_sha256=init,
                         common_initial_state_sha256="common", minibatch_order_sha256="order", epochs=40)
            epochs = [{"epoch": epoch, "weighted_train_ce": 1.1 if epoch == 0 else loss,
                       "train_diagnostics": diagnostic, "validation_diagnostics": diagnostic,
                       "gradient_norms": {name: {"mean": 0. if constant and name == "encoder" else 1.,
                                                 "min": 0. if constant and name == "encoder" else 1.,
                                                 "max": 0. if constant and name == "encoder" else 1.,
                                                 "finite": True, "nonzero_fraction": 0. if constant and name == "encoder" else 1.}
                                          for name in ("total", "encoder", "head")}}
                      for epoch in range(41 if status == "completed" else 4)]
            (directory / "history.json").write_text(json.dumps({"epochs": epochs}))
            value["acceptance"] = learning_acceptance(epochs, self.manifest["config"])
        (directory / "metrics.json").write_text(json.dumps(value))
        if status == "completed":
            groups = np.asarray(indices, dtype=np.int64)
            if wrong_group:
                groups[0] = 99
            save_prediction_csv(directory / "outer_predictions.csv",
                                {"structure_id": np.asarray([f"S{index}" for index in indices]), "y": y, "group_id": groups}, p)
        return directory

    def summarize(self, jobs):
        with patch("microcture.task1_formal.evaluation.planned_jobs", return_value=jobs):
            return summarize_formal(self.directory, self.metadata)

    def test_complete_oof_scores_all_unique_structures(self):
        jobs = [self.job(fold) for fold in range(2)]
        for job in jobs:
            self.write(job)
        result = self.summarize(jobs)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["comparison"][0]["macro_f1_mean"], 1.)
        self.assertEqual(result["models"]["I3/m2"]["seeds"]["42"]["oof"]["n_structures"], 6)
        self.assertTrue((self.directory / "oof/I3/m2/seed_42/predictions.csv").is_file())

    def test_missing_fold_never_produces_partial_oof(self):
        jobs = [self.job(fold) for fold in range(2)]
        self.write(jobs[0])
        result = self.summarize(jobs)
        self.assertEqual(result["status"], "pending")
        self.assertIsNone(result["comparison"][0]["macro_f1_mean"])
        self.assertFalse((self.directory / "oof/I3/m2/seed_42/predictions.csv").exists())
        self.assertEqual(json.loads((self.directory / "oof/I3/m2/seed_42/status.json").read_text())["status"], "pending")

    def test_baseline_acceptance_null_is_valid(self):
        jobs = [self.job(fold) for fold in range(2)]
        for job in jobs:
            directory = self.write(job)
            path = directory / "metrics.json"
            value = json.loads(path.read_text())
            value["acceptance"] = None
            path.write_text(json.dumps(value))
        result = self.summarize(jobs)
        self.assertEqual(result["status"], "complete")
        self.assertIsNone(result["jobs"][0]["learning_pass"])

    def test_numerical_failure_is_not_missing_or_complete(self):
        jobs = [self.job(fold, neural=True) for fold in range(2)]
        self.write(jobs[0])
        self.write(jobs[1], status="numerical_failure", constant=True)
        result = self.summarize(jobs)
        self.assertEqual(result["numerical_failure_jobs"], 1)
        self.assertEqual(result["pending_jobs"], 0)
        detail = result["models"]["I3/m2"]
        self.assertEqual(detail["seeds"]["42"]["numerical_failure_folds"], [1])
        self.assertEqual(detail["learning_diagnostics"]["prediction_events"]["train"]["constant_probability"]["events"], 3)
        self.assertEqual(detail["learning_diagnostics"]["prediction_events"]["outer"]["observed_events"], 1)

    def test_execution_failure_keeps_partial_trace_and_separate_count(self):
        jobs = [self.job(fold, neural=True) for fold in range(2)]
        self.write(jobs[0])
        self.write(jobs[1], status="execution_failure", constant=True)
        result = self.summarize(jobs)
        self.assertEqual(result["execution_failure_jobs"], 1)
        self.assertEqual(result["numerical_failure_jobs"], 0)
        self.assertEqual(result["pending_jobs"], 0)
        detail = result["models"]["I3/m2"]
        self.assertEqual(detail["seeds"]["42"]["execution_failure_folds"], [1])
        self.assertEqual(detail["learning_diagnostics"]["execution_failure_cases"], 1)
        self.assertEqual(detail["learning_diagnostics"]["prediction_events"]["train"]["constant_probability"]["events"], 3)
        self.assertFalse((self.directory / "oof/I3/m2/seed_42/predictions.csv").exists())

    def test_constant_completed_classifier_remains_in_performance(self):
        jobs = [self.job(fold, neural=True) for fold in range(2)]
        for job in jobs:
            self.write(job, constant=True, loss=1.1)
        result = self.summarize(jobs)
        self.assertEqual(result["status"], "complete")
        self.assertAlmostEqual(result["comparison"][0]["macro_f1_mean"], 1 / 6)
        diagnostic = result["models"]["I3/m2"]["learning_diagnostics"]
        self.assertEqual(diagnostic["strict_learning_passes"], 0)
        self.assertEqual(diagnostic["effective_gradients"]["encoder"]["effective_cases"], 0)
        self.assertEqual(diagnostic["effective_gradients"]["head"]["effective_cases"], 2)
        self.assertEqual(diagnostic["prediction_events"]["train"]["observed_events"], 80)
        self.assertEqual(diagnostic["prediction_events"]["train"]["constant_probability"]["events"], 80)
        self.assertEqual(diagnostic["prediction_events"]["outer"]["constant_probability"]["events"], 2)

    def test_constant_pca_baseline_keeps_performance_and_probability_only_events(self):
        jobs = [self.job(fold, model="pca_lr") for fold in range(2)]
        for job in jobs:
            self.write(job, constant=True)
        result = self.summarize(jobs)
        self.assertEqual(result["status"], "complete")
        self.assertAlmostEqual(result["comparison"][0]["macro_f1_mean"], 1 / 6)
        self.assertEqual(result["jobs"][0]["outer_prediction_counts"], {"OPCID": 0, "CHIN": 3, "CHID": 0})
        detail = result["models"]["I3/pca_lr"]
        self.assertNotIn("learning_diagnostics", detail)
        events = detail["outer_prediction_events"]
        self.assertEqual(events["observed_cases"], 2)
        for key in ("constant_probability", "single_class_argmax", "majority_concentration"):
            self.assertEqual(events[key], {"cases": 2, "rate": 1., "denominator": 2})
        self.assertNotIn("constant_centered_logits", events)

    def test_wrong_spatial_group_invalidates_fit(self):
        jobs = [self.job(fold) for fold in range(2)]
        self.write(jobs[0], wrong_group=True)
        self.write(jobs[1])
        result = self.summarize(jobs)
        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["invalid_jobs"], 1)
        self.assertFalse((self.directory / "oof/I3/m2/seed_42/predictions.csv").exists())

    def test_corrupt_stored_metric_invalidates_fit(self):
        jobs = [self.job(fold) for fold in range(2)]
        for job in jobs:
            directory = self.write(job)
        path = directory / "metrics.json"
        value = json.loads(path.read_text())
        value["outer_metrics"]["macro_f1"] = .5
        path.write_text(json.dumps(value))
        result = self.summarize(jobs)
        self.assertEqual(result["invalid_jobs"], 1)
        self.assertIn("differs from structure predictions", result["errors"][0]["reason"])

    def test_cross_seed_agreement_aligns_by_structure(self):
        jobs = [self.job(fold, seed=seed) for seed in (42, 43, 44) for fold in range(2)]
        for job in jobs:
            self.write(job, constant=job["seed"] == 44)
        result = self.summarize(jobs)
        detail = result["models"]["I3/m2"]["seed_prediction_agreement"]
        self.assertEqual(detail["n_structures"], 6)
        self.assertEqual(detail["n_seeds"], 3)
        self.assertAlmostEqual(detail["unanimous_rate"], 1 / 3)
        self.assertIsNotNone(result["comparison"][0]["macro_f1_seed_sample_std"])

    def test_incomplete_seed_blocks_main_mean_but_retains_complete_seed_oof(self):
        jobs = [self.job(fold, seed=seed) for seed in (42, 43) for fold in range(2)]
        for job in jobs[:-1]:
            self.write(job)
        result = self.summarize(jobs)
        row = result["comparison"][0]
        self.assertEqual(row["complete_seeds"], 1)
        self.assertIsNone(row["macro_f1_mean"])
        self.assertTrue((self.directory / "oof/I3/m2/seed_42/predictions.csv").is_file())
        self.assertFalse((self.directory / "oof/I3/m2/seed_43/predictions.csv").exists())

    def test_paired_initialization_detects_changed_ablation(self):
        jobs = [self.job(fold, condition=condition, neural=True) for condition in ("I3", "I0") for fold in range(2)]
        for job in jobs:
            self.write(job, init="changed" if job["condition"] == "I0" else "init")
        result = self.summarize(jobs)
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("Input ablations have different initial_state" in error["reason"] for error in result["errors"]))

    def test_paired_deltas_keep_fold_seed_pairs(self):
        jobs = [self.job(fold, condition=condition, neural=True) for condition in ("I3", "G0") for fold in range(2)]
        for job in jobs:
            self.write(job, constant=job["condition"] == "G0")
        result = self.summarize(jobs)
        comparison = result["paired_comparisons"]["m2:I3-G0"]
        self.assertEqual(comparison["completed_pairs"], 2)
        self.assertAlmostEqual(comparison["macro_f1_delta"]["mean"], 5 / 6)
        self.assertEqual(len(comparison["complete_seed_oof_deltas"]), 1)

    def test_duplicate_outer_identity_in_frozen_folds_fails_before_summarizing(self):
        self.manifest["fold_ids"]["1"]["test_ids"] = ["S0", "S4", "S5"]
        (self.directory / "manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError):
            self.summarize([self.job(0), self.job(1)])

    def test_training_history_missing_epoch_invalidates_completion(self):
        jobs = [self.job(fold, neural=True) for fold in range(2)]
        for job in jobs:
            directory = self.write(job)
        history_path = directory / "history.json"
        value = json.loads(history_path.read_text())
        value["epochs"].pop(12)
        history_path.write_text(json.dumps(value))
        result = self.summarize(jobs)
        self.assertEqual(result["invalid_jobs"], 1)
        self.assertIn("contiguous", result["errors"][0]["reason"])

    def test_saved_acceptance_cannot_override_full_trajectory(self):
        jobs = [self.job(fold, neural=True) for fold in range(2)]
        for job in jobs:
            directory = self.write(job)
        path = directory / "metrics.json"
        value = json.loads(path.read_text())
        value["acceptance"]["accepted"] = False
        path.write_text(json.dumps(value))
        result = self.summarize(jobs)
        self.assertEqual(result["invalid_jobs"], 1)
        self.assertIn("acceptance differs", result["errors"][0]["reason"])

    def test_saved_endpoint_diagnostics_must_equal_trajectory(self):
        jobs = [self.job(fold, neural=True) for fold in range(2)]
        for job in jobs:
            directory = self.write(job)
        path = directory / "metrics.json"
        value = json.loads(path.read_text())
        value["validation_diagnostics"]["num_predicted_classes"] = 1
        path.write_text(json.dumps(value))
        result = self.summarize(jobs)
        self.assertEqual(result["invalid_jobs"], 1)
        self.assertIn("Final validation diagnostics differ", result["errors"][0]["reason"])

    def test_outer_collapse_diagnostic_is_checked_against_saved_csv(self):
        jobs = [self.job(fold, neural=True) for fold in range(2)]
        for job in jobs:
            directory = self.write(job)
        path = directory / "metrics.json"
        value = json.loads(path.read_text())
        value["outer_diagnostics"]["probabilities"]["max_coordinate_range"] = 0.
        path.write_text(json.dumps(value))
        result = self.summarize(jobs)
        self.assertEqual(result["invalid_jobs"], 1)
        self.assertIn("probability range differs", result["errors"][0]["reason"])

    def test_canonical_manifest_signature_is_required(self):
        value = dict(self.manifest)
        value["source_signature"] = "changed-without-signing"
        (self.directory / "manifest.json").write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "canonical"):
            self.summarize([self.job(0), self.job(1)])
        self.assertFalse((self.directory / "summary.json").exists())

    def test_any_internal_symlink_is_rejected_before_outputs(self):
        (self.directory / "unrelated-link").symlink_to(self.directory / "manifest.json")
        with self.assertRaisesRegex(ValueError, "Symbolic links"):
            self.summarize([self.job(0), self.job(1)])
        self.assertFalse((self.directory / "summary.json").exists())

    def test_derived_oof_cannot_overwrite_an_artifact_sealed_job(self):
        destination = self.directory / "oof/I3/m2/seed_42"
        destination.mkdir(parents=True)
        (destination / "ARTIFACTS.json").write_text("{}")
        metrics_path = destination / "metrics.json"
        metrics_path.write_text('{"preserve": true}')
        with self.assertRaisesRegex(ValueError, "ARTIFACTS"):
            self.summarize([self.job(0), self.job(1)])
        self.assertEqual(metrics_path.read_text(), '{"preserve": true}')

    def test_stale_oof_numeric_files_are_retained_but_invalidated(self):
        jobs = [self.job(fold) for fold in range(2)]
        for job in jobs:
            directory = self.write(job)
        result = self.summarize(jobs)
        self.assertEqual(result["status"], "complete")
        oof = self.directory / "oof/I3/m2/seed_42"
        original_metrics = (oof / "metrics.json").read_text()
        original_predictions = (oof / "predictions.csv").read_text()
        # Preserve this fixture's numeric result while simulating absent job metadata.
        (directory / "metrics.json").rename(directory / "metrics.unavailable.json")
        result = self.summarize(jobs)
        self.assertEqual(result["status"], "pending")
        state = json.loads((oof / "status.json").read_text())
        self.assertFalse(state["oof_current"])
        self.assertTrue(state["previous_numeric_artifacts_retained"])
        self.assertEqual(state["experiment_signature"], self.manifest["experiment_signature"])
        self.assertEqual((oof / "metrics.json").read_text(), original_metrics)
        self.assertEqual((oof / "predictions.csv").read_text(), original_predictions)

    def test_complete_oof_artifacts_are_bound_to_provenance(self):
        jobs = [self.job(fold) for fold in range(2)]
        for job in jobs:
            self.write(job)
        self.summarize(jobs)
        path = self.directory / "oof/I3/m2/seed_42"
        value = json.loads((path / "metrics.json").read_text())
        status = json.loads((path / "status.json").read_text())
        self.assertEqual(status["status"], "complete")
        self.assertTrue(status["oof_current"])
        for key in ("experiment_signature", "source_signature", "dataset_signature"):
            self.assertEqual(value[key], self.manifest[key])
            self.assertEqual(status[key], self.manifest[key])

    def test_relative_metadata_path_is_resolved_against_project_not_cwd(self):
        project = self.directory / "fake-project"
        cache = project / "cache"
        cache.mkdir(parents=True)
        (cache / "metadata.json").write_text(json.dumps(self.metadata))
        value = {key: entry for key, entry in self.manifest.items() if key != "experiment_signature"}
        value["config"] = {**value["config"], "cache_dir": "cache"}
        value["experiment_signature"] = json_signature(value)
        (self.directory / "manifest.json").write_text(json.dumps(value))
        with patch("microcture.task1_formal.evaluation._PROJECT_ROOT", project), \
             patch("microcture.task1_formal.evaluation.planned_jobs", return_value=[self.job(0), self.job(1)]):
            result = summarize_formal(self.directory)
        self.assertEqual(result["n_structures"], 6)
        self.assertEqual(result["pending_jobs"], 2)


if __name__ == "__main__":
    unittest.main()
