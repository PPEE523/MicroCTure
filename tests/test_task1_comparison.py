"""Small synthetic tests; do not read matrices or train any model."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from microcture.task1.evaluation import grouped_predictions, metrics, save_prediction_csv, summarize_results


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "structure_id": np.asarray(["a", "a", "b", "b"]),
            "y": np.asarray([0, 0, 2, 2]),
            "group_id": np.asarray([10, 10, 11, 11]),
            "replicate": np.asarray(["rep1", "rep2", "rep1", "rep2"]),
        }
        self.probabilities = np.asarray([[0.8, 0.1, 0.1], [0.4, 0.5, 0.1],
                                         [0.1, 0.1, 0.8], [0.1, 0.2, 0.7]])

    def test_replicates_are_one_scored_structure(self):
        result = grouped_predictions(self.data, self.probabilities)
        self.assertEqual(result["structure_id"].tolist(), ["a", "b"])
        self.assertEqual(result["replicate_count"].tolist(), [2, 2])
        np.testing.assert_allclose(result["probabilities"][0], [0.6, 0.3, 0.1])
        self.assertEqual(result["replicate_disagreement"]["disagreement_count"], 1)
        self.assertEqual(metrics(result["y"], result["probabilities"])["n_structures"], 2)

    def test_missing_class_does_not_shrink_macro_denominator(self):
        score = metrics(np.asarray([0, 2]), np.asarray([[1., 0., 0.], [0., 0., 1.]]))
        self.assertEqual(score["accuracy"], 1.0)
        self.assertAlmostEqual(score["macro_f1"], 2 / 3)
        self.assertAlmostEqual(score["balanced_accuracy"], 2 / 3)
        self.assertEqual(score["confusion_matrix"], [[1, 0, 0], [0, 0, 0], [0, 0, 1]])
        self.assertEqual(score["per_class"]["CHIN"]["support"], 0)

    def test_invalid_probabilities_are_rejected(self):
        for bad in (np.zeros((4, 2)), np.zeros((4, 3)),
                    np.full((4, 3), np.nan), np.tile([-0.1, 0.5, 0.6], (4, 1))):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                grouped_predictions(self.data, bad)

    def test_label_and_group_disagreement_are_rejected(self):
        for key in ("y", "group_id"):
            inconsistent = {name: value.copy() for name, value in self.data.items()}
            inconsistent[key][1] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                grouped_predictions(inconsistent, self.probabilities)

    def test_prediction_csv_contains_replicates_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="microcture-test-", dir="/tmp") as temporary:
            path = Path(temporary) / "predictions.csv"
            save_prediction_csv(path, self.data, self.probabilities)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(json.loads(rows[0]["replicate_probabilities_json"])), 2)
            with self.assertRaises(FileExistsError):
                save_prediction_csv(path, self.data, self.probabilities)

    def _write_fold(self, root, model, seed, fold, ids, labels, probabilities, groups):
        destination = root / "runs" / model / f"seed_{seed}" / f"fold_{fold}"
        destination.mkdir(parents=True)
        data = {"structure_id": np.asarray(ids), "y": np.asarray(labels), "group_id": np.asarray(groups)}
        save_prediction_csv(destination / "predictions.csv", data, np.asarray(probabilities))
        metadata = {"model": model, "seed": seed, "fold": fold, "test": metrics(np.asarray(labels), probabilities),
                    "validation": {"macro_f1": 0.123},
                    "parameter_count": 12, "runtime_seconds": 2.0, "protocol": "synthetic", "dataset_signature": "test"}
        (destination / "metrics.json").write_text(json.dumps(metadata), encoding="utf-8")

    def test_summary_scores_oof_and_marks_missing_model_pending(self):
        with tempfile.TemporaryDirectory(prefix="microcture-test-", dir="/tmp") as temporary:
            root = Path(temporary)
            manifest = {"models": ["m2", "m2plus"], "model_seeds": {"m2": [42], "m2plus": [42, 43]}, "folds": 2,
                        "n_structures": 3, "protocol": "synthetic", "dataset_signature": "test"}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self._write_fold(root, "m2", 42, 0, ["a"], [0], [[1., 0., 0.]], [0])
            self._write_fold(root, "m2", 42, 1, ["b", "c"], [1, 2], [[0., 1., 0.], [0., 0., 1.]], [1, 2])
            summary = summarize_results(root)
            self.assertEqual(summary["status"], "pending")
            row = summary["comparison"][0]
            self.assertEqual(row["status"], "complete")
            self.assertEqual(row["macro_f1_mean"], 1.0)
            self.assertAlmostEqual(row["fold_macro_f1_mean"], 0.5)
            self.assertIsNone(row["macro_f1_seed_std"])
            self.assertEqual(summary["comparison"][1]["status"], "pending")
            self.assertEqual(len(summary["pending"]), 4)
            self.assertEqual(summary["comparison"][1]["expected_seeds"], 2)
            self.assertTrue((root / "comparison.png").is_file())
            self.assertTrue((root / "oof" / "m2" / "seed_42" / "confusion_matrix.png").is_file())

    def test_summary_rejects_cross_fold_leakage_group(self):
        with tempfile.TemporaryDirectory(prefix="microcture-test-", dir="/tmp") as temporary:
            root = Path(temporary)
            manifest = {"models": ["m2"], "seeds": [42], "folds": 2,
                        "protocol": "synthetic", "dataset_signature": "test"}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self._write_fold(root, "m2", 42, 0, ["a"], [0], [[1., 0., 0.]], [10])
            self._write_fold(root, "m2", 42, 1, ["b"], [1], [[0., 1., 0.]], [10])
            summary = summarize_results(root)
            self.assertEqual(summary["status"], "pending")
            self.assertTrue(any("leakage group" in error for error in summary["errors"]))
            self.assertEqual(summary["comparison"][0]["complete_seeds"], 0)


if __name__ == "__main__":
    unittest.main()
