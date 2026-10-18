"""Mocked loop/publication examples only; no real training or contact images."""
from __future__ import annotations

import copy
import errno
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from microcture.task1.evaluation import save_prediction_csv
from microcture.task1_reevaluation import runner
from microcture.task1_reevaluation.protocol import planned_jobs, PROTOCOL
from microcture.task1_reevaluation.selection import endpoint_selection_records

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/task1_reevaluation.json").read_text())


def samples():
    return {"x": np.ones((3, 2, 3, 64, 64), dtype=np.float32),
            "length": np.asarray([[-1], [0], [1]], dtype=np.float32),
            "y": np.arange(3, dtype=np.int64),
            "structure_id": np.asarray(["OPCID_0", "CHIN_0", "CHID_0"]),
            "group_id": np.arange(3, dtype=np.int64), "replicate": np.full(3, "fixture")}


def mock_history():
    return [{"epoch": e, "weighted_train_ce": 1 / (e + 1),
             "weighted_validation_ce": 0.01 if e == 0 else (0.2 if e in (7, 12) else 0.5),
             "train_diagnostics": {"tag": f"train_epoch_{e}"},
             "validation_diagnostics": {"tag": f"validation_epoch_{e}"}}
            for e in range(41)]


class RunnerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="microcture-reevaluation-test-", dir="/tmp")
        self.root = Path(self.temp.name)
        self.config = copy.deepcopy(CONFIG)
        self.run_dir = self.root / self.config["run_dir"]
        self.job = planned_jobs(self.config, "all", "m2plus", "I3", 0, 42)[0]
        self.sample = samples()
        self.pre = {"signature": "fixture-pre"}
        ids = self.sample["structure_id"].tolist()
        self.fold = {"fold": 0, "train_ids": ids, "validation_ids": ids, "test_ids": ids}
        self.manifest = {"protocol": PROTOCOL, "config": self.config,
                         "source_signature": "fixture-source", "dataset_signature": "fixture-data",
                         "experiment_signature": "fixture-exp", "evaluation_scope": "fixture"}
        self.output = self.root / "fit"
        self.output.mkdir()
        self.prob = np.eye(3) * 0.7 + 0.1

    def tearDown(self):
        self.temp.cleanup()

    def fake_fit(self, job, train, validation, config, model_config, device, output):
        history = mock_history()
        selections = endpoint_selection_records(history, config)
        runner.write_json(output / "history.json", {"epochs": history})
        runner.write_json(output / "initialization.json", {
            "initial_state_sha256": "fixture-init", "common_initial_state_sha256": "fixture-common"})
        for policy in ("final40", "inner_ce"):
            directory = output / "endpoints" / policy
            directory.mkdir(parents=True)
            (directory / "weights.pt").write_bytes(f"fixture_weight_{policy}".encode())
        return {"status": "completed", "job": job, "acceptance": {"accepted": True},
                "epochs": 40, "initial_train_ce": history[0]["weighted_train_ce"],
                "final_train_ce": history[-1]["weighted_train_ce"],
                "train_diagnostics": history[-1]["train_diagnostics"],
                "validation_diagnostics": history[-1]["validation_diagnostics"],
                "selection_signature": runner.selection_signature(config),
                "endpoint_selections": selections, "fit_units": 1, "endpoint_count": 2}

    def complete_fit(self, outer_callback=None):
        with patch.object(runner, "fit_diagnostic", side_effect=self.fake_fit), \
                patch.object(runner, "restore_endpoint", return_value=object()), \
                patch.object(runner, "predict", return_value=self.prob), \
                patch.object(runner, "outer_neural_endpoint", return_value=(self.prob, {"tag": "outer"})), \
                patch.object(runner, "prepare_outer", side_effect=outer_callback or (lambda *args: self.sample)):
            return runner.fit_one(self.job, self.sample, self.sample, self.pre, {}, self.fold,
                                  self.config, self.manifest, self.output, torch.device("cpu"))

    def test_both_weights_are_fixed_before_exactly_one_outer_construction(self):
        calls = []

        def outer(*args):
            fixed = json.loads((self.output / "endpoints_fixed.json").read_text())
            self.assertEqual(set(fixed["checkpoint_sha256_by_policy"]), {"final40", "inner_ce"})
            self.assertEqual(fixed["selected_epoch_by_policy"], {"final40": 40, "inner_ce": 7})
            self.assertFalse(fixed["outer_images_constructed"])
            for policy in ("final40", "inner_ce"):
                directory = self.output / "endpoints" / policy
                marker = json.loads((directory / "endpoint.json").read_text())
                self.assertTrue(marker["both_neural_endpoints_fixed_before_outer"])
                self.assertEqual(marker["fixed_endpoint_hashes"], fixed["checkpoint_sha256_by_policy"])
                self.assertTrue((directory / "validation_predictions.csv").is_file())
            calls.append(True)
            return self.sample

        result = self.complete_fit(outer)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["fit_units"], 1)
        self.assertEqual(result["endpoint_count"], 2)
        self.assertEqual(result["train_diagnostics"], {"tag": "train_epoch_40"})
        for policy, epoch in (("final40", 40), ("inner_ce", 7)):
            value = json.loads((self.output / "endpoints" / policy / "metrics.json").read_text())
            self.assertEqual(value["train_diagnostics"], {"tag": f"train_epoch_{epoch}"})
            self.assertEqual(value["selected_epoch"], epoch)
            self.assertTrue(value["shared_fit_unit"])
            self.assertEqual(value["runtime_seconds"], result["runtime_seconds"])
        runner.verify_artifact_index(self.output, self.job, self.manifest)
        self.assertEqual(runner.assert_resume(self.output, self.job, self.manifest)["status"], "completed")

    def test_resealed_missing_or_bad_shared_fixation_marker_is_rejected(self):
        self.complete_fit()
        path = self.output / "endpoints_fixed.json"
        value = json.loads(path.read_text())
        value["checkpoint_sha256_by_policy"].pop("inner_ce")
        runner.write_json(path, value)
        runner.write_artifact_index(self.output, self.job, self.manifest)
        with self.assertRaises(ValueError):
            runner.assert_resume(self.output, self.job, self.manifest)

    def test_partial_second_endpoint_failure_keeps_first_endpoint_predictions(self):
        def endpoint(*args):
            if endpoint.calls:
                raise RuntimeError("fixture second endpoint failure")
            endpoint.calls += 1
            return self.prob, {"tag": "outer"}

        endpoint.calls = 0
        with patch.object(runner, "fit_diagnostic", side_effect=self.fake_fit), \
                patch.object(runner, "restore_endpoint", return_value=object()), \
                patch.object(runner, "predict", return_value=self.prob), \
                patch.object(runner, "outer_neural_endpoint", side_effect=endpoint), \
                patch.object(runner, "prepare_outer", return_value=self.sample):
            result = runner.fit_one(self.job, self.sample, self.sample, self.pre, {}, self.fold,
                                    self.config, self.manifest, self.output, torch.device("cpu"))
        self.assertEqual(result["status"], "numerical_failure")
        self.assertEqual(result["completed_endpoint_policies"], ["final40"])
        self.assertTrue((self.output / "endpoints/final40/outer_predictions.csv").is_file())
        self.assertTrue((self.output / "endpoints/inner_ce/weights.pt").is_file())
        runner.verify_artifact_index(self.output, self.job, self.manifest)
        self.assertEqual(runner.assert_resume(self.output, self.job, self.manifest)["status"], "numerical_failure")

    def test_classical_constant_prediction_is_one_retained_baseline_endpoint(self):
        job = planned_jobs(self.config, "baselines", "mask_only", "I3", 0, 42)[0]
        probabilities = np.tile([0.1, 0.1, 0.8], (3, 1))
        calls = []

        def fit(name, train, validation, config, output):
            output.mkdir(parents=True)
            (output / "model.joblib").write_bytes(b"fixture_baseline_model")
            save_prediction_csv(output / "validation_predictions.csv", validation, probabilities)
            return object(), {"status": "completed", "acceptance": None,
                              "validation_metrics": runner.score_samples(validation, probabilities)}

        def outer(*args):
            calls.append(True)
            marker = json.loads((self.output / "endpoints/baseline/endpoint.json").read_text())
            self.assertIsNone(marker["selected_epoch"])
            self.assertFalse(marker["outer_images_constructed"])
            return self.sample

        with patch.object(runner, "fit_baseline", side_effect=fit), \
                patch.object(runner, "predict_baseline", return_value=probabilities), \
                patch.object(runner, "prepare_outer", side_effect=outer):
            result = runner.fit_one(job, self.sample, self.sample, self.pre, {}, self.fold,
                                    self.config, self.manifest, self.output, torch.device("cpu"))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["fit_units"], 1)
        self.assertEqual(result["endpoint_count"], 1)
        self.assertEqual(result["completed_endpoint_policies"], ["baseline"])
        self.assertEqual(len(calls), 1)
        endpoint = json.loads((self.output / "endpoints/baseline/metrics.json").read_text())
        self.assertEqual(endpoint["outer_metrics"]["confusion_matrix"], [[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        self.assertEqual(runner.assert_resume(self.output, job, self.manifest)["status"], "completed")

    def test_interrupt_seals_partial_trajectory_without_constructing_outer(self):
        def interrupted(*args):
            output = args[-1]
            runner.write_json(output / "history.json", {"epochs": mock_history()[:2]})
            directory = output / "endpoints/inner_ce"
            directory.mkdir(parents=True)
            (directory / "weights.pt").write_bytes(b"fixture_partial_weight")
            raise KeyboardInterrupt()

        with patch.object(runner, "fit_diagnostic", side_effect=interrupted), \
                patch.object(runner, "prepare_outer") as outer:
            result = runner.fit_one(self.job, self.sample, self.sample, self.pre, {}, self.fold,
                                    self.config, self.manifest, self.output, torch.device("cpu"))
        self.assertEqual(result["status"], "execution_failure")
        self.assertTrue(result["run_stop_requested"])
        self.assertFalse(outer.called)
        self.assertEqual(len(json.loads((self.output / "history.json").read_text())["epochs"]), 2)
        self.assertTrue((self.output / "endpoints/inner_ce/weights.pt").is_file())
        runner.verify_artifact_index(self.output, self.job, self.manifest)

    def test_run_interrupt_publishes_once_and_stops_remaining_fits(self):
        count = []

        def interrupted_fit(job, *args):
            output = args[-2]
            runner.write_json(output / "protocol.json", {"job": job, "selection": self.config["selection"],
                                                         "preprocessing": self.pre})
            runner.write_json(output / "history.json", {"epochs": mock_history()[:2]})
            value = {"status": "execution_failure", "job": job, "run_stop_requested": True,
                     "error_type": "KeyboardInterrupt", "preprocessing_signature": self.pre["signature"],
                     **{k: self.manifest[k] for k in ("source_signature", "dataset_signature", "experiment_signature")}}
            runner.write_json(output / "metrics.json", value)
            runner.write_artifact_index(output, job, self.manifest)
            count.append(job)
            return value

        with patch.object(runner, "prepare_manifest", return_value=(self.manifest, {})), \
                patch.object(runner, "resolve_device", return_value=torch.device("cpu")), \
                patch.object(runner, "load_dataset", return_value={"signature": "fixture-data"}), \
                patch.object(runner, "frozen_fold", return_value=self.fold), \
                patch.object(runner, "prepare_inner", return_value=(self.sample, self.sample, self.pre)), \
                patch.object(runner, "fit_one", side_effect=interrupted_fit):
            with self.assertRaises(runner.RunStopped):
                runner.run_jobs(self.config, self.root, self.run_dir, stage="main", model="m2plus", seed=42)
        self.assertEqual(len(count), 1)
        published = runner.job_directory(self.run_dir, count[0])
        self.assertTrue((published / "history.json").is_file())
        self.assertEqual(runner.assert_resume(published, count[0], self.manifest)["status"], "execution_failure")
        self.assertFalse((self.run_dir / ".training.lock").exists())

    def test_empty_filter_cannot_create_manifest_or_open_dataset(self):
        with patch.object(runner, "prepare_manifest") as prepare, patch.object(runner, "load_dataset") as load:
            with self.assertRaises(ValueError):
                runner.run_jobs(self.config, self.root, self.run_dir, stage="main", model="pca_lr", seed=43)
        self.assertFalse(prepare.called)
        self.assertFalse(load.called)
        self.assertFalse(self.run_dir.exists())

    def test_recursive_artifact_inventory_rejects_changed_and_unindexed_files(self):
        self.complete_fit()
        (self.output / "endpoints/inner_ce/new.txt").write_text("fixture")
        with self.assertRaises(ValueError):
            runner.verify_artifact_index(self.output, self.job, self.manifest)
        (self.output / "endpoints/inner_ce/new.txt").unlink()
        (self.output / "endpoints/inner_ce/weights.pt").write_bytes(b"changed_fixture")
        with self.assertRaises(ValueError):
            runner.verify_artifact_index(self.output, self.job, self.manifest)

    def test_recursive_publisher_cross_filesystem_keeps_metrics_last(self):
        self.complete_fit()
        destination = self.root / "published"
        copied = []
        original_copy = shutil.copy2

        def copy_file(source, target):
            copied.append(Path(source).relative_to(self.output).as_posix())
            return original_copy(source, target)

        with patch.object(runner.os, "replace", side_effect=OSError(errno.EXDEV, "fixture filesystem")), \
                patch.object(runner.shutil, "copy2", side_effect=copy_file):
            runner.publish_fold(self.output, destination)
        self.assertEqual(copied[-1], "metrics.json")
        self.assertTrue(self.output.exists())
        runner.verify_artifact_index(destination, self.job, self.manifest)

    def test_failed_cross_filesystem_copy_preserves_source_and_partial_destination(self):
        self.complete_fit()
        destination = self.root / "partial"
        with patch.object(runner.os, "replace", side_effect=OSError(errno.EXDEV, "fixture filesystem")), \
                patch.object(runner.shutil, "copy2", side_effect=OSError("fixture disk fault")):
            with self.assertRaises(OSError):
                runner.publish_fold(self.output, destination)
        self.assertTrue((self.output / "history.json").is_file())
        self.assertTrue(destination.exists())
        self.assertFalse((destination / "metrics.json").exists())

    def test_double_storage_fault_keeps_temporary_scientific_files(self):
        (self.output / "history.json").write_text('{"fixture": true}')
        with patch.object(runner.shutil, "move", side_effect=OSError("fixture persistent disk fault")), \
                patch.object(runner, "write_staging_context", side_effect=OSError("fixture temporary marker fault")), \
                patch("builtins.print") as message:
            retained = runner.preserve_failed_staging(self.output, self.run_dir, self.job,
                                                       OSError("original fixture fault"), self.root)
        self.assertEqual(retained, self.output)
        self.assertTrue((retained / "history.json").is_file())
        self.assertTrue(any("automatic retry prevention cannot be persisted" in str(call)
                            for call in message.call_args_list))

    def test_guard_accepts_own_manifest_and_protects_previous_experiments(self):
        self.run_dir.mkdir(parents=True)
        runner.write_json(self.run_dir / "manifest.json", self.manifest)
        self.assertEqual(runner.guard_output(self.root, self.run_dir), self.run_dir)
        for directory in ("results/task1_formal_v1", "results/task1_readout_v1", "results/task1_morphology_probe_v1"):
            with self.assertRaises(ValueError):
                runner.guard_output(self.root, self.root / directory)


class MockedLoopTests(unittest.TestCase):
    """An integer counter replaces optimization; only checkpoint I/O is real."""

    def test_mocked_counter_loop_saves_actual_earliest_best_weights_and_full40(self):
        class CounterModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.counter = nn.Parameter(torch.zeros(1))
                self.encoder = nn.Identity()
                self.head = nn.Identity()

            def forward(self, x, length):
                return self.counter.expand(len(x), 3)

        model = CounterModel()
        updates = []

        class CounterOptimizer:
            def zero_grad(self, **kwargs):
                pass

            def step(self):
                with torch.no_grad():
                    model.counter.add_(1)
                updates.append(int(model.counter))

        class FakeLoss:
            def backward(self):
                pass

            def detach(self):
                return 1.0

        diagnosis_calls = []

        def diagnose(model, data, *args, **kwargs):
            epoch = int(model.counter)
            view = "train" if len(diagnosis_calls) % 2 == 0 else "validation"
            diagnosis_calls.append((epoch, view))
            ce = 0.01 if epoch == 0 else (0.2 if epoch in (3, 4) else 0.5)
            return {"ce": 1 / (epoch + 1) if view == "train" else ce,
                    "metrics": {"per_class": {"CHID": {"f1": 1.0 if epoch == 20 else 0.0}}},
                    "head": {"effective_fraction": 1.0}, "centered_logit_between_structure_var": 1.0}

        config = copy.deepcopy(CONFIG)
        config["training"]["batch_size"] = 32
        job = planned_jobs(config, "all", "m2plus", "I3", 0, 42)[0]
        with tempfile.TemporaryDirectory(prefix="microcture-reevaluation-mock-loop-", dir="/tmp") as tmp:
            output = Path(tmp)
            with patch.object(runner, "seed_everything"), \
                    patch.object(runner, "build_controlled_model", return_value=model), \
                    patch.object(runner.torch.optim, "AdamW", return_value=CounterOptimizer()), \
                    patch.object(runner.nn, "CrossEntropyLoss", return_value=lambda *args: FakeLoss()), \
                    patch.object(runner.torch, "isfinite", return_value=True), \
                    patch.object(runner, "diagnose_model", side_effect=diagnose), \
                    patch.object(runner, "weighted_ce", side_effect=lambda diagnostic, weights: diagnostic["ce"]), \
                    patch.object(runner, "_gradient_norm", return_value=1.0), \
                    patch.object(runner, "_gradient_summary", return_value={"fixture": True}), \
                    patch.object(runner, "flat_history_row", side_effect=lambda row: {"epoch": row["epoch"]}), \
                    patch.object(runner, "learning_acceptance", return_value={"accepted": True}), \
                    patch.object(runner, "model_spec", return_value={"parameters": 1}), \
                    patch("builtins.print"):
                result = runner.fit_diagnostic(job, samples(), samples(), config, config["models"],
                                               torch.device("cpu"), output)
            best = torch.load(output / "endpoints/inner_ce/weights.pt", weights_only=True)
            final = torch.load(output / "endpoints/final40/weights.pt", weights_only=True)
            self.assertEqual(updates, list(range(1, 41)))
            self.assertEqual(len(diagnosis_calls), 82)
            self.assertEqual(best["epoch"], 3)
            self.assertEqual(float(best["state_dict"]["counter"]), 3.0)
            self.assertEqual(final["epoch"], 40)
            self.assertEqual(float(final["state_dict"]["counter"]), 40.0)
            self.assertEqual(result["endpoint_selections"]["inner_ce"]["selected_epoch"], 3)
            self.assertEqual(len(json.loads((output / "history.json").read_text())["epochs"]), 41)


if __name__ == "__main__":
    unittest.main()
