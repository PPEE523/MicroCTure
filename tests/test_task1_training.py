"""Exercise orchestration with fake fitting and tiny synthetic tensors only."""

from contextlib import ExitStack, nullcontext
import errno
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from microcture.task1 import training
from microcture.task1.evaluation import summarize_results


class FakePipeline:
    def __getitem__(self, index):
        return SimpleNamespace(coef_=np.zeros((3, 2)), intercept_=np.zeros(3))

    def predict_proba(self, x):
        return np.eye(3)[np.arange(len(x)) % 3]


class MockTrainingTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "protocol": "mock_nested_grouped_cv", "data": {"train_augmentation": "none"},
            "training": {"seeds": [42, 43], "cpu_threads": 1, "inner_split_seed": 7, "batch_size": 4},
            "baseline": {"seed": 42}, "models": {},
        }
        self.dataset = {
            "config": self.config["data"], "signature": "mock-dataset",
            "y": np.asarray([0, 1, 2, 0, 1, 2]),
            "structure_id": np.asarray(list("abcdef")),
            "label_order": ["OPCID", "CHIN", "CHID"],
            "metadata": {"evaluation_scope": "synthetic tests only"},
            "folds": [
                {"fold": 0, "train_ids": list("def"), "test_ids": list("abc")},
                {"fold": 1, "train_ids": list("abc"), "test_ids": list("def")},
            ],
        }
        self.events = []
        self.fit_count = 0
        self.transform_count = 0

    def _samples(self, ids):
        labels = np.asarray(["abcdef".index(sid) % 3 for sid in ids], dtype=np.int64)
        return {"x": np.zeros((len(ids), 2, 2, 2, 2), dtype=np.float32), "y": labels,
                "structure_id": np.asarray(ids), "group_id": np.asarray(["abcdef".index(sid) for sid in ids]),
                "replicate": np.asarray(["synthetic"] * len(ids))}

    def _prepare(self, dataset, train_ids, validation_ids, config):
        return self._samples(train_ids), self._samples(validation_ids), {"signature": "mock-preprocessing"}

    def _fake_fit_neural(self, name, train, validation, config, seed, device, output):
        self.fit_count += 1
        self.events.append(("fit", name, seed))
        (output / "best.pt").write_text("fake checkpoint; no model training", encoding="utf-8")
        # Deliberately different from outer scores: summary must read test.
        return object(), {"macro_f1": 0.123, "accuracy": 0.123}, {"selected_epoch": 1}

    def _fake_fit_pca(self, train, validation, config, output):
        self.fit_count += 1
        self.events.append(("fit", "pca_lr", config["baseline"]["seed"]))
        (output / "model.joblib").write_text("fake pipeline; no PCA fitting", encoding="utf-8")
        return FakePipeline(), {"macro_f1": 0.123, "accuracy": 0.123}, {"pca_components": 2}

    def _transform(self, dataset, test_ids, preprocessing):
        # Constructing outer tensors before fit would violate this invariant.
        self.assertEqual(self.fit_count, self.transform_count + 1)
        self.transform_count += 1
        self.events.append(("outer_transform", tuple(test_ids)))
        return self._samples(test_ids)

    def _patches(self):
        stack = ExitStack()
        stack.enter_context(patch.object(training, "load_dataset", return_value=self.dataset))
        stack.enter_context(patch.object(training, "source_signature", return_value="mock-source"))
        stack.enter_context(patch.object(training, "environment_record", return_value={"test": "mock"}))
        stack.enter_context(patch.object(training, "split_train_validation",
                                        side_effect=lambda dataset, ids, config, seed: (ids[:2], ids[2:])))
        stack.enter_context(patch.object(training, "prepare_fold", side_effect=self._prepare))
        stack.enter_context(patch.object(training, "transform_heldout", side_effect=self._transform))
        stack.enter_context(patch.object(training, "fit_neural", side_effect=self._fake_fit_neural))
        stack.enter_context(patch.object(training, "fit_pca", side_effect=self._fake_fit_pca))
        stack.enter_context(patch.object(training, "predict_neural",
                                        side_effect=lambda model, data, device, batch: np.eye(3)[data["y"]]))
        stack.enter_context(patch.object(training, "model_spec", return_value={"parameters": 12}))
        stack.enter_context(patch.object(training.torch, "set_num_threads"))
        stack.enter_context(patch.object(training, "threadpool_limits", side_effect=lambda **kwargs: nullcontext()))
        return stack

    def test_partial_resume_and_full_oof_schema(self):
        with tempfile.TemporaryDirectory(prefix="microcture-mock-", dir="/tmp") as temporary, self._patches():
            root = Path(temporary)
            run_dir = root / "run"
            partial = training.run_training(self.config, root, root / "unused-cache", run_dir,
                                            "m2", [0], "cpu")
            self.assertEqual(partial["status"], "pending")
            self.assertEqual(self.fit_count, 2)
            self.assertEqual(self.transform_count, 2)
            m2 = next(row for row in partial["comparison"] if row["model"] == "m2")
            self.assertEqual(m2["complete_seeds"], 0)
            self.assertEqual(m2["completed_folds"], 2)
            self.assertIsNone(m2["macro_f1_mean"])
            self.assertIsNone(partial["models"]["m2"]["seed_prediction_agreement"])

            # Completed destinations are skipped; no fake or real fitting occurs.
            training.run_training(self.config, root, root / "unused-cache", run_dir, "m2", [0], "cpu")
            self.assertEqual(self.fit_count, 2)
            self.assertEqual(self.transform_count, 2)

            complete = training.run_training(self.config, root, root / "unused-cache", run_dir, "all", None, "cpu")
            self.assertEqual(complete["status"], "complete", complete["errors"])
            self.assertEqual(self.fit_count, 14)  # 2 PCA + 3 neural x 2 seeds x 2 folds.
            self.assertEqual(self.transform_count, 14)
            self.assertTrue(all(self.events[i][0] == "fit" and self.events[i + 1][0] == "outer_transform"
                                for i in range(0, len(self.events), 2)))
            for row in complete["comparison"]:
                self.assertEqual(row["macro_f1_mean"], 1.0)
                self.assertIsNotNone(row["runtime_seconds"])
                self.assertGreaterEqual(row["runtime_seconds"], 0)
                if row["model"] != "pca_lr":
                    self.assertEqual(row["parameter_count"], 12)
                    self.assertEqual(row["unanimous_rate"], 1.0)
                    agreement = complete["models"][row["model"]]["seed_prediction_agreement"]
                    self.assertEqual(agreement["n_structures"], 6)
                    self.assertEqual(agreement["n_seeds"], 2)
                else:
                    self.assertIsNone(row["unanimous_rate"])
                    self.assertEqual(row["expected_seeds"], 1)
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            evidence = json.loads((run_dir / "runs/m2/seed_42/fold_0/metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(evidence["run_signature"], manifest["run_signature"])
            self.assertEqual(evidence["source_signature"], manifest["source_signature"])
            self.assertEqual(evidence["validation"]["macro_f1"], 0.123)
            self.assertEqual(evidence["test"]["macro_f1"], 1.0)

            # A copied fold from another run must never enter the OOF scores.
            evidence["run_signature"] = "different-run"
            (run_dir / "runs/m2/seed_42/fold_0/metrics.json").write_text(json.dumps(evidence), encoding="utf-8")
            corrupted = summarize_results(run_dir)
            self.assertEqual(corrupted["status"], "pending")
            self.assertTrue(any("run_signature" in error for error in corrupted["errors"]))

    def test_cross_filesystem_publish_keeps_completion_marker(self):
        with tempfile.TemporaryDirectory(prefix="microcture-publish-", dir="/tmp") as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "best.pt").write_bytes(b"synthetic artifact")
            (source / "metrics.json").write_text(json.dumps({"complete": True}), encoding="utf-8")
            with patch.object(training.os, "replace", side_effect=OSError(errno.EXDEV, "different filesystem")):
                training.publish_fold(source, root / "destination")
            self.assertEqual((root / "destination/best.pt").read_bytes(), b"synthetic artifact")
            self.assertEqual(json.loads((root / "destination/metrics.json").read_text()), {"complete": True})

    def test_failed_publish_removes_only_new_incomplete_destination(self):
        with tempfile.TemporaryDirectory(prefix="microcture-publish-", dir="/tmp") as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "best.pt").write_bytes(b"synthetic artifact")
            (source / "metrics.json").write_text("{}", encoding="utf-8")
            with patch.object(training.os, "replace", side_effect=OSError(errno.EXDEV, "different filesystem")), \
                    patch.object(training.shutil, "copy2", side_effect=OSError("synthetic failure")):
                with self.assertRaises(OSError):
                    training.publish_fold(source, root / "destination")
            self.assertTrue((source / "best.pt").is_file())
            self.assertFalse((root / "destination").exists())


if __name__ == "__main__":
    unittest.main()
