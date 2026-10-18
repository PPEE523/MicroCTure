"""Hand-computed saved-OOF bootstrap and incomplete-review safeguards."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import audit_task1_native_roi as A
from scripts import plot_task1_native_roi as P


def fixture_oof(native_improves=False):
    identity = {f"s{i}": {"group_id": i} for i in range(3)}
    oof = {}
    for arm in A.ARMS:
        for seed in A.SEEDS:
            values = []
            for label in range(3):
                prediction = label
                if native_improves and arm != ("C1", "I3") and label == 2:
                    prediction = 1
                values.append({"structure_id": f"s{label}", "group_id": str(label), "y_true": str(label), "y_pred": str(prediction)})
            oof[(*arm, seed)] = {"rows": values}
    return oof, identity


class NativeBootstrapTests(unittest.TestCase):
    def test_identical_arms_have_exact_zero_paired_uncertainty(self):
        oof, identity = fixture_oof()
        result = A.group_bootstrap(oof, identity, repeats=100, rng_seed=9)
        for pair in result["pairs"].values():
            for value in pair.values():
                self.assertEqual(value["point_mean"], 0)
                self.assertEqual(value["ci95"], [0, 0])
        self.assertEqual(result["group_count"], 3)
        self.assertGreater(result["zero_CHID_support_bootstraps"], 0)

    def test_hand_confusion_gain_and_difference_beyond_profile(self):
        oof, identity = fixture_oof(native_improves=True)
        result = A.group_bootstrap(oof, identity, repeats=80, rng_seed=11)
        pair = result["pairs"]["C1/I3-C0/I3"]
        self.assertAlmostEqual(pair["macro_f1"]["point_mean"], 4 / 9)
        self.assertEqual(pair["CHID_f1"]["point_mean"], 1)
        np.testing.assert_allclose(pair["macro_f1"]["seed_deltas"], [4 / 9] * 3, atol=1e-15, rtol=0)
        self.assertAlmostEqual(result["native_minus_coarse_beyond_profile"]["macro_f1"]["point_mean"], 4 / 9)

    def test_seed_metrics_averaged_after_scoring_not_prediction_pooling(self):
        oof, identity = fixture_oof(native_improves=True)
        oof["C1", "I3", 44]["rows"][-1]["y_pred"] = "1"
        result = A.group_bootstrap(oof, identity, repeats=30, rng_seed=3)
        expected = (1 + 1 + 5 / 9) / 3
        self.assertAlmostEqual(result["arms"]["C1/I3"]["macro_f1"]["point_mean"], expected)
        self.assertAlmostEqual(result["pairs"]["C1/I3-C0/I3"]["macro_f1"]["point_mean"], 8 / 27)

    def test_repeated_seed_rows_cannot_pass_as_complete_oof(self):
        oof, identity = fixture_oof()
        oof["C0", "I3", 42]["rows"][-1] = oof["C0", "I3", 42]["rows"][0]
        with self.assertRaisesRegex(ValueError, "complete matched OOF"):
            A.group_bootstrap(oof, identity, repeats=10)

    def test_confusion_arithmetic_keeps_three_classes_with_zero_support(self):
        score, support, predicted = A.scores_from_confusion(np.diag([2, 1, 0]))
        self.assertAlmostEqual(float(score["macro_f1"]), 2 / 3)
        self.assertEqual(float(score["CHID_f1"]), 0)
        self.assertEqual(int(support[2]), 0)


class ReviewCompletionGate(unittest.TestCase):
    def test_partial_run_rejected_before_importing_scientific_loader(self):
        with tempfile.TemporaryDirectory(prefix="microcture-native-review-test-", dir="/tmp") as temporary:
            run = Path(temporary)
            (run / "summary.json").write_text(json.dumps({"status": "pending", "completed_jobs": 7, "complete_oof": 0}))
            with patch.object(A.importlib, "import_module") as scientific:
                with self.assertRaisesRegex(ValueError, "partial scores are not reviewed"):
                    A.build_report(run, run / "report")
            scientific.assert_not_called()
            self.assertFalse((run / "report").exists())

    def test_g0_replay_compares_weight_tensors_and_ignores_only_time(self):
        import torch
        with tempfile.TemporaryDirectory(prefix="microcture-native-g0-replay-", dir="/tmp") as temporary:
            root = Path(temporary)
            old = root / "old/runs/R1/G0/m2plus/seed_42/fold_0"
            new = root / "new"
            old.mkdir(parents=True); new.mkdir()
            fields = {"job": {"fixture": True}, "initial_state_sha256": "initial", "common_initial_state_sha256": "common",
                      "minibatch_order_sha256": "order", "class_weights": [1, 1, 1], "acceptance": {"accepted": True}}
            (root / "old/manifest.json").write_text("{}")
            for index, directory in enumerate((old, new)):
                torch.save({"state_dict": {"weight": torch.tensor([1., 2.])}}, directory / "final.pt")
                (directory / "metrics.json").write_text(json.dumps(fields))
                (directory / "history.json").write_text(json.dumps({"epochs": [{"epoch": 0, "ce": 2., "epoch_seconds": 10 + index}]}))
                for filename in ("outer_predictions.csv", "validation_predictions.csv"):
                    (directory / filename).write_bytes(b"fixture identical CSV\n")
            with patch.object(A, "ROOT", root), patch.object(A, "verify_artifact_index") as seal:
                result = A.verify_g0_replay(new, {"seed": 42, "fold": 0}, {"roi_run_dir": "old"})
                self.assertTrue(all(result["checks"].values()))
                seal.assert_called_once()
                torch.save({"state_dict": {"weight": torch.tensor([1., 3.])}}, new / "final.pt")
                with self.assertRaisesRegex(ValueError, "failed exact R1/G0 replay"):
                    A.verify_g0_replay(new, {"seed": 42, "fold": 0}, {"roi_run_dir": "old"})


class ScientificFigureFixture(unittest.TestCase):
    def test_all_four_figures_render_all26_chid_without_real_results(self):
        identity = {"OPCID_1": {"group_id": 0}, "CHIN_1": {"group_id": 1},
                    **{f"CHID_{i}": {"group_id": i + 1} for i in range(1, 27)}}
        source = [{"structure_id": sid, "group_id": str(ref["group_id"]),
                   "y_true": "0" if sid.startswith("OPCID") else "1" if sid.startswith("CHIN_") else "2"}
                  for sid, ref in identity.items()]
        for row in source:
            row["y_pred"] = row["y_true"]
            truth = int(row["y_true"])
            probability = [.1, .1, .1]; probability[truth] = .8
            row.update({f"p_{c}": str(probability[i]) for i, c in enumerate(A.CLASSES)})
            row.update(replicate_count="2", replicate_disagree="0", replicate_probabilities_json=json.dumps([
                {"replicate": rep, "probabilities": probability, "predicted": truth} for rep in (1, 2)]))
        scores = A.independent_scores(source)
        result, cases, oof = {"arms": {}}, [], {}
        for alignment, condition in A.ARMS:
            arm = f"{alignment}/{condition}"
            for seed in A.SEEDS:
                oof[alignment, condition, seed] = {"rows": source}
            trajectory = {field: [[.8 / (epoch + 1), 1 / (epoch + 1), 1.2 / (epoch + 1)] for epoch in range(41)]
                          for field in ("weighted_train_ce", "weighted_validation_ce", "gradient_head_mean",
                                        "train_head_effective_fraction", "validation_centered_logit_variance")}
            result["arms"][arm] = {"scores": [scores] * 3, "predictions": [{"confusion_matrix": scores["confusion_matrix"],
                    "predicted_counts": scores["prediction_counts"]}] * 3, "stable_wrong_ids": [],
                    "trajectory_quantiles_25_50_75": trajectory, "learning": {"strict_learning_passes": 15}}
            cases.extend({"arm": arm, "structure_id": row["structure_id"], "type": A.CLASSES[int(row["y_true"])],
                          **{f"seed_{s}_prediction": A.CLASSES[int(row["y_pred"])] for s in A.SEEDS}} for row in source)
        bootstrap = A.group_bootstrap(oof, identity, repeats=20, rng_seed=9)
        previous = os.environ.get("MPLCONFIGDIR")
        try:
            with tempfile.TemporaryDirectory(prefix="microcture-native-plot-fixture-", dir="/tmp") as temporary:
                output = Path(temporary)
                os.environ["MPLCONFIGDIR"] = str(output / "mpl")
                import matplotlib
                matplotlib.use("Agg")
                P.performance(result, output)
                P.paired_effects(bootstrap, output)
                P.errors(result, cases, output)
                P.learning(result, output)
                for name in ("performance", "paired_effects", "predictions_and_CHID", "learning"):
                    for extension in ("png", "pdf"):
                        self.assertGreater((output / f"{name}.{extension}").stat().st_size, 1000)
        finally:
            if previous is None: os.environ.pop("MPLCONFIGDIR", None)
            else: os.environ["MPLCONFIGDIR"] = previous


if __name__ == "__main__":
    unittest.main()
