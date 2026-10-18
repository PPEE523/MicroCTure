"""Handwritten report fixtures; no fitting, inference, or scientific-result writes."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.task1_formal_performance_plots import complete_seed_scores, performance_payload
from scripts.task1_formal_stability_plots import (
    MODELS, SEEDS, _chid_grid, _curve_matrix, _diagnostic_matrix, plot_stability,
)


def score(*, swap: bool = False) -> dict:
    """Six cases: either all right, or both CHIN and both CHID swapped."""
    values = {"precision": 0. if swap else 1., "recall": 0. if swap else 1.,
              "f1": 0. if swap else 1.}
    return {"n_structures": 6,
            "confusion_matrix": [[2, 0, 0], [0, 0, 2], [0, 2, 0]] if swap else [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            "macro_f1": 1 / 3 if swap else 1., "accuracy": 1 / 3 if swap else 1.,
            "balanced_accuracy": 1 / 3 if swap else 1.,
            "per_class": {"OPCID": {"precision": 1., "recall": 1., "f1": 1.},
                          "CHIN": dict(values), "CHID": dict(values)}}


def fixture() -> dict:
    summary = {"class_names": ["OPCID", "CHIN", "CHID"], "n_structures": 6,
               "class_support": {"OPCID": 2, "CHIN": 2, "CHID": 2},
               "comparison": [], "models": {}}
    groups = {"I3/m2": [False, True, True], "I3/m2plus": [False, False, False],
              "I3/cnn_transformer": [False, True, True], "I3/pca_lr": [False],
              "I0/m2": [True, True, True], "G0/m2": [True, True, True]}
    for key, swaps in groups.items():
        condition, model = key.split("/")
        summary["comparison"].append({"condition": condition, "model": model,
                                      "status": "complete", "expected_seeds": len(swaps),
                                      "complete_seeds": len(swaps)})
        summary["models"][key] = {"seeds": {str(seed): {"status": "complete", "oof": score(swap=swaps[index])}
                                                  for index, seed in enumerate(SEEDS[:len(swaps)])}}
    return summary


class FormalVisualizationTests(unittest.TestCase):
    def test_full_seed_mean_sample_sd_and_single_seed_no_sd(self):
        payload = performance_payload(fixture())
        metrics = payload["groups"]["I3/m2"]["metrics"]["macro_f1"]
        self.assertAlmostEqual(metrics["mean"], 5 / 9)
        self.assertAlmostEqual(metrics["sample_sd"], 2 / (3 * math.sqrt(3)))
        self.assertIsNone(payload["groups"]["I3/pca_lr"]["metrics"]["macro_f1"]["sample_sd"])

    def test_partial_seed_set_never_yields_selected_subset(self):
        summary = fixture()
        del summary["models"]["I3/m2"]["seeds"]["44"]
        scores, reason = complete_seed_scores(summary, "I3", "m2")
        self.assertEqual(scores, [])
        self.assertIsNotNone(reason)
        self.assertNotIn("I3/m2", performance_payload(summary)["groups"])

    def test_pairing_is_same_seed_difference_without_fold_replication(self):
        pair = performance_payload(fixture())["paired"]["I3: Transformer - M2+"]
        self.assertEqual(pair["seeds"], [42, 43, 44])
        self.assertEqual(len(pair["metrics"]["macro_f1"]["values"]), 3)
        self.assertAlmostEqual(pair["metrics"]["macro_f1"]["mean"], -4 / 9)
        self.assertAlmostEqual(pair["metrics"]["macro_f1"]["sample_sd"], 2 / (3 * math.sqrt(3)))
        summary = fixture()
        seeds = summary["models"]["I3/cnn_transformer"]["seeds"]
        seeds["45"] = seeds.pop("44")
        with self.assertRaisesRegex(ValueError, "same complete seeds"):
            performance_payload(summary)

    def test_confusion_counts_retain_six_structure_sample_size(self):
        group = performance_payload(fixture())["groups"]["I3/m2"]
        self.assertAlmostEqual(float(group["confusion_mean_count"].sum()), 6.)
        np.testing.assert_allclose(group["confusion_mean_row_fraction"].sum(axis=1), [1., 1., 1.])
        self.assertAlmostEqual(group["confusion_mean_row_fraction"][2, 2], 1 / 3)
        summary = fixture()
        summary["models"]["I3/m2"]["seeds"]["42"]["oof"]["confusion_matrix"][0][0] = 3
        with self.assertRaisesRegex(ValueError, "support or confusion counts"):
            complete_seed_scores(summary, "I3", "m2")

    def test_missing_curve_measurements_remain_nan(self):
        epochs, values = _curve_matrix([
            {"rows": [{"epoch": 0, "loss": 1.}, {"epoch": 2, "loss": 0.}]},
            {"rows": [{"epoch": 1, "loss": .5}, {"epoch": 2}]},
        ], ("loss",))
        np.testing.assert_array_equal(epochs, [0, 1, 2])
        self.assertTrue(np.isnan(values[0, 1]))
        self.assertTrue(np.isnan(values[1, 0]))
        self.assertTrue(np.isnan(values[1, 2]))
        self.assertEqual(values[0, 2], 0.)

    def test_single_class_predictions_do_not_imply_constant_probabilities(self):
        summary = {"models": {"I3/m2": {"learning_diagnostics": {
            "strict_learning_passes": 15, "expected_cases": 15,
            "prediction_events": {"validation": {"final_observed_cases": 15,
                "constant_probability": {"final_cases": 0},
                "single_class_argmax": {"final_cases": 15}}}}}}}
        constant, labels = _diagnostic_matrix(summary, "constant_probability")
        single, _ = _diagnostic_matrix(summary, "single_class_argmax")
        self.assertEqual(constant[0, 2], 0.)
        self.assertEqual(labels[0][2], "0/15")
        self.assertEqual(single[0, 2], 1.)
        self.assertTrue(np.isnan(single[0, 0]))

    def test_chid_order_uses_all_nine_predictions_and_checks_fold_identity(self):
        rows = {"CHID_1": {"y_true": 2, "y_pred": 2, "group_id": "3", "fold": "4"},
                "CHID_2": {"y_true": 2, "y_pred": 1, "group_id": "3", "fold": "4"},
                "CHID_8": {"y_true": 2, "y_pred": 1, "group_id": "5", "fold": "0"}}
        oof = {(model, seed): deepcopy(rows) for model in MODELS for seed in SEEDS}
        identifiers, matrix, labels = _chid_grid(oof)
        self.assertEqual(identifiers, ["CHID_2", "CHID_8", "CHID_1"])
        self.assertEqual(matrix.shape, (3, 9))
        np.testing.assert_array_equal(matrix[0], [1] * 9)
        self.assertIn("g3, f4", labels[0])
        oof[(MODELS[-1], SEEDS[-1])]["CHID_2"]["fold"] = "3"
        with self.assertRaisesRegex(ValueError, "Inconsistent OOF group/fold"):
            _chid_grid(oof)

    def test_plotting_refuses_the_scientific_directory_and_children(self):
        with tempfile.TemporaryDirectory(prefix="microcture-visual-fixture-", dir="/tmp") as directory:
            run_dir = Path(directory)
            for destination in (run_dir, run_dir / "figures"):
                with self.assertRaisesRegex(ValueError, "outside the scientific result directory"):
                    plot_stability(run_dir, {}, destination)
            self.assertFalse((run_dir / "figures").exists())


if __name__ == "__main__":
    unittest.main()
