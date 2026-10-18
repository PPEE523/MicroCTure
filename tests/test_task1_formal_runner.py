"""Ordering/resume integration with handwritten samples and fake fits only."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from microcture.task1.evaluation import save_prediction_csv
from microcture.task1_controlled.evidence import json_signature
from microcture.task1_formal import runner
from microcture.task1_formal.protocol import planned_jobs


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/task1_formal.json").read_text())


def samples(ids, labels, groups):
    # One handwritten observation per structure; no raw data extraction.
    n = len(ids)
    return {"x": np.ones((n, 2, 3, 64, 64), dtype=np.float32),
            "length": np.linspace(-1, 1, n, dtype=np.float32)[:, None],
            "y": np.asarray(labels, dtype=np.int64), "structure_id": np.asarray(ids),
            "group_id": np.asarray(groups, dtype=np.int64), "replicate": np.full(n, "fixture")}


def probabilities(sample):
    value = np.full((len(sample["y"]), 3), 0.1, dtype=np.float32)
    value[np.arange(len(value)), sample["y"]] = 0.8
    return value


class CountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, length):
        self.calls += 1
        value = length[:, 0]
        return torch.stack((value, -value, torch.zeros_like(value)), dim=1)


class FormalRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir="/tmp", prefix="microcture-formal-test-")
        self.root = Path(self.temporary.name)
        self.config = copy.deepcopy(CONFIG)
        self.config["run_dir"] = "results/new"
        self.config["frozen_run_dir"] = "results/frozen"
        self.job = planned_jobs(self.config, "baselines", "length_only", fold=0)[0]
        self.pre = {"signature": "fixture-preprocessing"}
        labels = ["OPCID", "CHIN", "CHID"]
        rows = [{"structure_id": f"{labels[i % 3]}_{i}", "type": labels[i % 3],
                 "start": i * 1000, "end": i * 1000 + 100} for i in range(15)]
        self.metadata = {"signature": "fixture-dataset", "structures": rows, "group_id": list(range(15))}
        self.fold_ids = {}
        self.fold_signatures = {}
        for number in range(5):
            test = list(range(number * 3, number * 3 + 3))
            val = list(range(((number + 1) % 5) * 3, ((number + 1) % 5) * 3 + 3))
            train = [i for i in range(15) if i not in test + val]
            frozen = {"fold": number, "train_ids": [rows[i]["structure_id"] for i in train],
                      "validation_ids": [rows[i]["structure_id"] for i in val],
                      "test_ids": [rows[i]["structure_id"] for i in test], "preprocessing": {}}
            path = self.root / "results/frozen/folds" / f"fold_{number}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(frozen))
            self.fold_ids[str(number)] = {key: frozen[key] for key in ("train_ids", "validation_ids", "test_ids")}
            self.fold_signatures[str(number)] = json_signature(frozen)
            if number == 0:
                self.fold = frozen
                self.train = samples(frozen["train_ids"], [i % 3 for i in train], train)
                self.validation = samples(frozen["validation_ids"], [i % 3 for i in val], val)
                self.outer = samples(frozen["test_ids"], [i % 3 for i in test], test)
        self.manifest = {"config": self.config, "source_signature": "fixture-source", "dataset_signature": "fixture-dataset",
                         "fold_ids": self.fold_ids,
                         "fold_signatures": self.fold_signatures}
        self.manifest["experiment_signature"] = json_signature(self.manifest)
        self.events = []

    def tearDown(self):
        self.temporary.cleanup()

    def fake_fit(self, name, train, validation, config, output):
        self.events.append("fit_inner")
        self.assertNotIn("prepare_outer", self.events)
        (output / "model.joblib").write_text("handwritten fitted-estimator fixture")
        save_prediction_csv(output / "validation_predictions.csv", validation, probabilities(validation))
        return object(), {"status": "completed", "acceptance": None}

    def fake_outer(self, dataset, frozen, preprocessing):
        self.events.append("prepare_outer")
        self.assertIn("fit_inner", self.events)
        # Endpoint creation has already happened at this point.
        return self.outer

    def fit(self, output):
        with patch.object(runner, "fit_baseline", side_effect=self.fake_fit), \
             patch.object(runner, "prepare_outer", side_effect=self.fake_outer), \
             patch.object(runner, "predict_baseline", side_effect=lambda estimator, name, sample: probabilities(sample)):
            return runner.fit_one(self.job, self.train, self.validation, self.pre, {}, self.fold,
                                  self.config, self.manifest, output, torch.device("cpu"))

    def test_inner_endpoint_precedes_outer_and_completed_case_can_resume(self):
        output = self.root / "job"
        output.mkdir()
        def outer(dataset, frozen, preprocessing):
            self.assertTrue((output / "endpoint.json").is_file())
            endpoint = json.loads((output / "endpoint.json").read_text())
            self.assertTrue(endpoint["training_and_selection_complete"])
            self.assertFalse(endpoint["outer_images_constructed"])
            self.assertEqual(endpoint["checkpoint_sha256"], runner.sha256(output / "model.joblib"))
            return self.fake_outer(dataset, frozen, preprocessing)
        with patch.object(runner, "fit_baseline", side_effect=self.fake_fit), \
             patch.object(runner, "prepare_outer", side_effect=outer), \
             patch.object(runner, "predict_baseline", return_value=probabilities(self.outer)):
            result = runner.fit_one(self.job, self.train, self.validation, self.pre, {}, self.fold,
                                    self.config, self.manifest, output, torch.device("cpu"))
        self.assertEqual(self.events, ["fit_inner", "prepare_outer"])
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["outer_constructed_after_fixed_endpoint"])
        self.assertEqual(runner.assert_resume(output, self.job, self.manifest), result)

    def test_failed_inner_fit_never_builds_outer_and_failure_is_not_retried(self):
        output = self.root / "job"
        output.mkdir()
        with patch.object(runner, "fit_baseline", side_effect=RuntimeError("handwritten numeric failure")), \
             patch.object(runner, "prepare_outer") as outer:
            result = runner.fit_one(self.job, self.train, self.validation, self.pre, {}, self.fold,
                                    self.config, self.manifest, output, torch.device("cpu"))
        outer.assert_not_called()
        self.assertEqual(result["status"], "numerical_failure")
        self.assertFalse(result["outer_images_used"])
        self.assertEqual(runner.assert_resume(output, self.job, self.manifest)["status"], "numerical_failure")

    def test_failed_outer_keeps_inner_evidence_and_execution_failure_status(self):
        output = self.root / "job"
        output.mkdir()
        with patch.object(runner, "fit_baseline", side_effect=self.fake_fit), \
             patch.object(runner, "prepare_outer", return_value=self.outer), \
             patch.object(runner, "predict_baseline", side_effect=ValueError("handwritten outer failure")):
            result = runner.fit_one(self.job, self.train, self.validation, self.pre, {}, self.fold,
                                    self.config, self.manifest, output, torch.device("cpu"))
        self.assertEqual(result["status"], "execution_failure")
        self.assertEqual(result["failure_stage"], "outer_evaluation")
        self.assertTrue((output / "model.joblib").is_file())
        self.assertTrue((output / "failure.json").is_file())
        self.assertEqual(runner.assert_resume(output, self.job, self.manifest)["status"], "execution_failure")

    def test_neural_g0_fit_and_outer_receive_the_same_registered_condition(self):
        output = self.root / "job"
        output.mkdir()
        job = planned_jobs(self.config, "ablations", "m2", "G0", fold=0, seed=42)[0]
        events = []

        def fit_neural(actual_job, train, validation, config, model_config, device, destination):
            events.append("fit_inner")
            self.assertEqual(actual_job["head"], "H3")
            self.assertEqual(actual_job["learning_rate"], 1e-4)
            self.assertTrue(np.all(train["x"][:, :, :2] == 0))
            self.assertTrue(np.all(validation["x"][:, :, :2] == 0))
            np.testing.assert_array_equal(train["length"], self.train["length"])
            (destination / "final.pt").write_text("handwritten final checkpoint")
            return {"status": "completed", "job": actual_job, "acceptance": {"accepted": False},
                    "epochs": 40, "initial_train_ce": 1.0, "final_train_ce": 0.4}

        def prepare(dataset, frozen, preprocessing):
            events.append("outer")
            self.assertTrue((output / "endpoint.json").is_file())
            return self.outer

        def diagnose(model, outer, config, device):
            events.append("predict")
            self.assertTrue(np.all(outer["x"][:, :, :2] == 0))
            self.assertTrue(np.all(outer["x"][:, :, 2] == 1))
            return probabilities(outer), {"fixture": True}

        with patch.object(runner, "fit_diagnostic", side_effect=fit_neural), \
             patch.object(runner, "prepare_outer", side_effect=prepare), \
             patch.object(runner, "restore_endpoint", return_value=object()), \
             patch.object(runner, "outer_neural_endpoint", side_effect=diagnose):
            result = runner.fit_one(job, self.train, self.validation, self.pre, {}, self.fold,
                                    self.config, self.manifest, output, torch.device("cpu"))
        self.assertEqual(events, ["fit_inner", "outer", "predict"])
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["acceptance"]["accepted"])  # A failed acceptance does not suppress outer scoring.
        self.assertTrue((output / "outer_predictions.csv").is_file())

    def test_restore_rejects_a_checkpoint_with_an_old_head(self):
        job = planned_jobs(self.config, "main", "m2", fold=0, seed=42)[0]
        saved = {"job": job, "epoch": 40, "model_config": self.config["models"], "head_version": "H0"}
        with patch.object(runner.torch, "load", return_value=saved), \
             self.assertRaisesRegex(ValueError, "preregistered final endpoint"):
            runner.restore_endpoint(job, self.config, self.root / "fake.pt", torch.device("cpu"))

    def test_changed_prediction_is_refused_before_resume(self):
        output = self.root / "job"
        output.mkdir()
        self.fit(output)
        (output / "outer_predictions.csv").write_text("changed saved probabilities")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            runner.assert_resume(output, self.job, self.manifest)

    def test_missing_validation_evidence_is_not_skipped(self):
        output = self.root / "job"
        output.mkdir()
        self.fit(output)
        (output / "validation_predictions.csv").unlink()
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            runner.assert_resume(output, self.job, self.manifest)

    def test_outer_probabilities_are_from_the_single_diagnostic_forward(self):
        config = copy.deepcopy(self.config)
        config["training"]["batch_size"] = 2
        model = CountingModel().eval()
        values, diag = runner.outer_neural_endpoint(model, self.outer, config, torch.device("cpu"))
        self.assertEqual(model.calls, 2)  # Three examples, two batches; no second pass.
        np.testing.assert_allclose(values.sum(axis=1), 1, atol=1e-6)
        self.assertEqual(diag["n_structures"], 3)
        self.assertEqual(len(model._forward_hooks), 0)

    def test_output_guard_rejects_frozen_tree_and_nested_symlink(self):
        frozen = self.root / "results/frozen"
        (frozen / "FROZEN.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "frozen evidence"):
            runner.guard_output(self.root, frozen / "new")
        output = self.root / "results/new"
        output.mkdir()
        (output / "linked").symlink_to(frozen, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symbolic-link artifacts"):
            runner.guard_output(self.root, output)

    def test_mocked_stage_publishes_and_summarizes_one_fit_then_skips_it(self):
        output = self.root / "results/new"
        with patch.object(runner, "prepare_manifest", return_value=(self.manifest, self.metadata)), \
             patch.object(runner, "load_dataset", return_value={"signature": "fixture-dataset"}) as loader, \
             patch.object(runner, "prepare_inner", return_value=(self.train, self.validation, self.pre)) as inner, \
             patch.object(runner, "fit_baseline", side_effect=self.fake_fit) as fitter, \
             patch.object(runner, "prepare_outer", side_effect=self.fake_outer), \
             patch.object(runner, "predict_baseline", return_value=probabilities(self.outer)):
            summary = runner.run_jobs(self.config, self.root, output, "cpu", stage="baselines", model="length_only", fold=0)
            self.assertEqual(summary["completed_jobs"], 1)
            self.assertEqual(summary["pending_jobs"], 154)
            fitter.assert_called_once()
            inner.assert_called_once()
            self.assertTrue((output / "summary.json").is_file())
            again = runner.run_jobs(self.config, self.root, output, "cpu", stage="baselines", model="length_only", fold=0)
            self.assertEqual(again["completed_jobs"], 1)
            fitter.assert_called_once()
            loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
