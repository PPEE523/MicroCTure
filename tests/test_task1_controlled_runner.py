"""Controlled orchestration with synthetic metadata and fake fits only.

No model forward, optimizer, raw matrix, preprocessing fit or sklearn fit is
run here. Fake traces exercise persistence, gating and diagnostic field names.
"""
from contextlib import ExitStack, nullcontext, redirect_stdout
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from microcture.task1_controlled import runner
from microcture.task1_controlled.diagnostics import learning_acceptance


ROOT = Path(__file__).resolve().parents[1]


def fake_diagnostic():
    return {
        "n_structures": 3, "n_replicate_examples": 6,
        "metrics": {"macro_f1": 0.5, "per_class": {"CHID": {"f1": 0.25}}},
        "logit_between_structure_var": 0.1,
        "centered_logit_between_structure_var": 0.08,
        "probability_between_structure_var": 0.02,
        "logits": {"max_coordinate_range": 0.8},
        "centered_logits": {"max_coordinate_range": 0.7},
        "probabilities": {"max_coordinate_range": 0.2},
        "majority_prediction_fraction": 2 / 3,
        "head": {"available": True, "active_units": 4, "total_units": 4,
                 "positive_fraction": 0.0, "effective_fraction": 1.0,
                 "inactive_units": 0, "epsilon": 1e-6},
        "loss_support": {"unweighted_nll_sum_by_class": [1.0, 1.0, 1.0],
                         "examples_count_by_class": [2, 2, 2]},
    }


def fake_history(count=41):
    gradients = {name: {"mean": 0.1, "min": 0.08, "max": 0.12,
                        "finite": True, "nonzero_fraction": 1.0}
                 for name in ("total", "encoder", "head")}
    return [{"epoch": epoch, "weighted_train_ce": 1.1 - 0.01 * epoch,
             "weighted_validation_ce": 0.9, "optimized_training_loss": None if epoch == 0 else 0.8,
             "train_diagnostics": fake_diagnostic(), "validation_diagnostics": fake_diagnostic(),
             "gradient_norms": None if epoch == 0 else copy.deepcopy(gradients),
             "epoch_seconds": 0.001}
            for epoch in range(count)]


class ControlledRunnerTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/task1_controlled.json").read_text())
        self.config.update({"frozen_run_dir": "frozen", "cache_dir": "unused-cache", "evidence_dir": "unused-evidence"})
        # One historical cell is enough to test four-way pairing; preserve 40 epochs.
        name = "m2plus_f2_s42"
        self.config["head_stage"]["cases"] = {name: self.config["head_stage"]["cases"][name]}
        self.case = name
        self.fit_records = []
        self.geometry_records = []
        self.failure_job = None
        self.train = self._samples("train")
        self.validation = self._samples("val")

    def _samples(self, prefix):
        return {"x": np.zeros((6, 2, 3, 2, 2), dtype=np.float32),
                "length": np.zeros((6, 1), dtype=np.float32),
                "y": np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
                "structure_id": np.asarray([prefix + str(i // 2) for i in range(6)]),
                "group_id": np.repeat(np.arange(3), 2),
                "replicate": np.asarray(["1", "2"] * 3)}

    def _fixture(self, root):
        frozen = root / "frozen"
        (frozen / "folds").mkdir(parents=True)
        fold = {"train_ids": ["train0", "train1", "train2"],
                "validation_ids": ["val0", "val1", "val2"],
                "test_ids": ["outer-never-read"]}
        (frozen / "folds/fold_2.json").write_text(json.dumps(fold))
        (frozen / "manifest.json").write_text(json.dumps({"config": {"models": {"synthetic": True}}}))

    def _prepare_manifest(self, config, root, run_dir, requested_device):
        return {"protocol": config["protocol"], "config": config,
                "experiment_signature": "fake-controlled-signature"}, {"synthetic": True}

    def _write_history(self, output, history):
        runner.write_json(output / "history.json", {"epochs": history})
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            rows = [runner.flat_history_row(row) for row in history]
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _fit(self, job, train, validation, config, model_config, device, output):
        self.fit_records.append({"job": copy.deepcopy(job), "training": copy.deepcopy(config["training"]),
                                 "train_variant": train["input_variant"],
                                 "validation_variant": validation["input_variant"],
                                 "train_x_identity": id(train["x"]), "train_length_identity": id(train["length"])})
        initial = "fake-initial-H01" if job["head"] in ("H0", "H1") else "fake-initial-H23"
        runner.write_json(output / "initialization.json", {"initial_state_sha256": initial,
                                                          "common_initial_state_sha256": "fake-common-initial"})
        if job["name"] == self.failure_job:
            self._write_history(output, fake_history(3))
            raise RuntimeError("mock numerical failure; no actual optimization")
        history = fake_history(config["training"]["epochs"] + 1)
        self._write_history(output, history)
        (output / "final.pt").write_text("fake final checkpoint; never load as a model")
        return {"status": "completed", "job": job, "acceptance": learning_acceptance(history, config),
                "initial_state_sha256": initial, "common_initial_state_sha256": "fake-common-initial",
                "minibatch_order_sha256": "fake-paired-batches", "class_weights": [1.0, 1.0, 1.0],
                "parameter_spec": {"parameters": 12}, "initial_train_ce": history[0]["weighted_train_ce"],
                "final_train_ce": history[-1]["weighted_train_ce"],
                "train_diagnostics": history[-1]["train_diagnostics"],
                "validation_diagnostics": history[-1]["validation_diagnostics"],
                "epochs": config["training"]["epochs"], "outer_images_used": False,
                "checkpoint_selection": "final epoch only; fake fit"}

    def _geometry(self, job, train, validation, config, output):
        self.geometry_records.append(copy.deepcopy(job))
        (output / "model.joblib").write_text("fake geometry baseline; never load")
        return {"status": "completed", "job": job, "acceptance": None,
                "validation_metrics": fake_diagnostic()["metrics"], "outer_images_used": False}

    def _patches(self):
        stack = ExitStack()
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(patch.object(runner, "prepare_manifest", side_effect=self._prepare_manifest))
        self.dataset_load = stack.enter_context(patch.object(runner, "load_dataset", return_value={"synthetic": True}))
        stack.enter_context(patch.object(runner, "prepare_controlled_fold", return_value=(self.train, self.validation, {"signature": "fake-preprocess"})))
        stack.enter_context(patch.object(runner, "apply_input_variant", side_effect=lambda data, variant: {**data, "input_variant": variant}))
        stack.enter_context(patch.object(runner, "fit_diagnostic", side_effect=self._fit))
        stack.enter_context(patch.object(runner, "fit_geometry", side_effect=self._geometry))
        stack.enter_context(patch.object(runner, "resolve_device", return_value=torch.device("cpu")))
        stack.enter_context(patch.object(runner.torch, "set_num_threads"))
        stack.enter_context(patch.object(runner, "threadpool_limits", side_effect=lambda **kwargs: nullcontext()))
        self.plot = stack.enter_context(patch.object(runner, "plot_traces"))
        return stack

    def _run(self, root, stage):
        return runner.run_stage(self.config, root, stage, self.case if stage == "heads" else None,
                                root / "controlled", "cpu")

    def test_heads_require_all_seven_completed_input_controls(self):
        with tempfile.TemporaryDirectory(prefix="microcture-control-test-", dir="/tmp") as temporary, self._patches():
            root = Path(temporary)
            self._fixture(root)
            with self.assertRaisesRegex(ValueError, "Complete.*input controls"):
                self._run(root, "heads")
            self.assertEqual(self.fit_records, [])
            self.dataset_load.assert_not_called()
            summary = self._run(root, "input")
            self.assertEqual(len(summary["comparison"]), 7)
            self.assertEqual(summary["execution_status"], "partial")
            for job in runner.planned_jobs(self.config, "input"):
                path = runner.job_directory(root / "controlled", job) / "metrics.json"
                record = json.loads(path.read_text())
                record["status"] = "numerical_failure"
                path.write_text(json.dumps(record))
                with self.subTest(job=job["name"]), self.assertRaisesRegex(ValueError, "incomplete"):
                    self._run(root, "heads")
                record["status"] = "completed"
                path.write_text(json.dumps(record))
            self.assertEqual(len(self.fit_records), 4)

    def test_41_epoch_schema_resume_and_four_head_pairing(self):
        with tempfile.TemporaryDirectory(prefix="microcture-control-test-", dir="/tmp") as temporary, self._patches():
            root = Path(temporary)
            self._fixture(root)
            self._run(root, "input")
            initial_loads = self.dataset_load.call_count
            self._run(root, "input")
            self.assertEqual(len(self.fit_records), 4)
            self.assertEqual(len(self.geometry_records), 3)
            self.assertEqual(self.dataset_load.call_count, initial_loads)
            summary = self._run(root, "heads")
            self.assertEqual(summary["execution_status"], "complete")
            self.assertEqual(len(self.fit_records), 8)
            heads = self.fit_records[4:]
            self.assertEqual([r["job"]["name"] for r in heads], ["H0", "H1", "H2", "H3"])
            self.assertEqual([r["job"]["learning_rate"] for r in heads], [0.001, 0.001, 0.001, 0.0001])
            self.assertEqual(len({r["train_x_identity"] for r in heads}), 1)
            self.assertEqual(len({r["train_length_identity"] for r in heads}), 1)
            self.assertTrue(all(r["train_variant"] == r["validation_variant"] == "I3" for r in heads))
            self.assertTrue(all(r["training"] == self.config["training"] for r in heads))
            pairing = summary["pairing_checks"][self.case]
            self.assertTrue(pairing["common_initial_weights_match"])
            self.assertTrue(pairing["minibatch_order_match"])
            self.assertTrue(pairing["preprocessing_match"])
            self.assertTrue(pairing["class_weights_match"])
            self.assertTrue(pairing["comparison_ready"])
            self.assertTrue(pairing["H0_H1_full_initial_weights_match"])
            self.assertTrue(pairing["H2_H3_full_initial_weights_match"])
            self.assertEqual(pairing["completed_versions"], 4)
            for row in summary["comparison"]:
                if row["name"] not in ("H0", "H1", "H2", "H3", "I0", "I1", "I2", "I3"):
                    continue
                self.assertEqual(row["validation_CHID_F1"], 0.25)
                self.assertEqual(row["validation_centered_logit_variance"], 0.08)
                self.assertEqual(row["validation_head_effective_fraction"], 1.0)
                self.assertTrue(row["accepted"])
            directory = root / "controlled/heads" / self.case / "H3"
            records = json.loads((directory / "history.json").read_text())["epochs"]
            self.assertEqual([r["epoch"] for r in records], list(range(41)))
            self.assertIsNone(records[0]["gradient_norms"])
            self.assertIsNotNone(records[-1]["gradient_norms"])
            with (directory / "history.csv").open(newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 41)
            self.assertEqual(float(csv_rows[-1]["validation_chid_f1"]), 0.25)
            self.assertEqual(float(csv_rows[-1]["validation_centered_logit_variance"]), 0.08)
            self.assertEqual(float(csv_rows[-1]["train_head_effective_fraction"]), 1.0)
            self.assertEqual(float(csv_rows[-1]["gradient_encoder_mean"]), 0.1)
            self._run(root, "heads")
            self.assertEqual(len(self.fit_records), 8)

    def test_numerical_failure_retains_history_and_blocks_heads(self):
        self.failure_job = "I2"
        with tempfile.TemporaryDirectory(prefix="microcture-control-test-", dir="/tmp") as temporary, self._patches():
            root = Path(temporary)
            self._fixture(root)
            summary = self._run(root, "input")
            directory = root / "controlled/input/I2"
            self.assertTrue((directory / "failure.json").is_file())
            self.assertTrue((directory / "initialization.json").is_file())
            failed_record = json.loads((directory / "metrics.json").read_text())
            self.assertEqual(failed_record["common_initial_state_sha256"], "fake-common-initial")
            self.assertEqual(len(json.loads((directory / "history.json").read_text())["epochs"]), 3)
            failed = next(row for row in summary["comparison"] if row["name"] == "I2")
            self.assertEqual(failed["status"], "numerical_failure")
            self.assertFalse(failed["accepted"])
            self.assertIsNone(failed["validation_CHID_F1"])
            self._run(root, "input")
            self.assertEqual(len(self.fit_records), 4)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                self._run(root, "heads")

    def test_wrong_job_metadata_is_rejected_despite_matching_signature(self):
        with tempfile.TemporaryDirectory(prefix="microcture-control-test-", dir="/tmp") as temporary, self._patches():
            root = Path(temporary)
            self._fixture(root)
            self._run(root, "input")
            path = root / "controlled/input/I0/metrics.json"
            record = json.loads(path.read_text())
            record["job"]["name"] = "I1"
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                runner.summarize_controlled(root / "controlled")
            with self.assertRaises(ValueError):
                self._run(root, "input")
            with self.assertRaises(ValueError):
                self._run(root, "heads")

    def test_pairing_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="microcture-control-test-", dir="/tmp") as temporary, self._patches():
            root = Path(temporary)
            self._fixture(root)
            self._run(root, "input")
            path = root / "controlled/input/I1/metrics.json"
            original = json.loads(path.read_text())
            edits = {"common_initial_state_sha256": "mismatched-fake-pairing",
                     "minibatch_order_sha256": "mismatched-fake-pairing",
                     "preprocessing_signature": "mismatched-fake-preprocessing",
                     "class_weights": [1.0, 1.0, 2.0]}
            for field, value in edits.items():
                record = copy.deepcopy(original)
                record[field] = value
                path.write_text(json.dumps(record))
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Paired"):
                    runner.summarize_controlled(root / "controlled")

    def test_head_full_initial_state_pair_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="microcture-control-test-", dir="/tmp") as temporary, self._patches():
            root = Path(temporary)
            self._fixture(root)
            self._run(root, "input")
            self._run(root, "heads")
            for head in ("H1", "H3"):
                path = root / "controlled/heads" / self.case / head / "metrics.json"
                original = json.loads(path.read_text())
                record = copy.deepcopy(original)
                record["initial_state_sha256"] = "mismatched-full-initial-state"
                path.write_text(json.dumps(record))
                with self.subTest(head=head), self.assertRaisesRegex(ValueError, "Paired"):
                    runner.summarize_controlled(root / "controlled")
                path.write_text(json.dumps(original))

    def test_protocol_requires_all_geometry_baselines_and_fixed_c(self):
        with tempfile.TemporaryDirectory(prefix="microcture-control-test-", dir="/tmp") as temporary:
            path = Path(temporary) / "protocol.json"
            for edit in ({"geometry_baselines": []}, {"baseline_C": 10.0}):
                config = copy.deepcopy(self.config)
                config["input_stage"].update(edit)
                path.write_text(json.dumps(config))
                with self.subTest(edit=edit), self.assertRaises(ValueError):
                    runner.load_protocol(path)


if __name__ == "__main__":
    unittest.main()
