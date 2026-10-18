"""Hand-counted fixtures for saved-evidence audit, with no training."""
import json
import unittest

from scripts.audit_task1_reevaluation_results import (CLASSES, csv_metrics,
    selected_inner_epoch)


def prediction(number, actual, predicted):
    probabilities = [.1, .1, .1]
    probabilities[predicted] = .8
    replicates = [{"replicate": str(r), "predicted": predicted, "probabilities": probabilities} for r in (1, 2)]
    return {"structure_id": str(number), "y_true": str(actual), "y_pred": str(predicted),
            **{f"p_{name}": str(value) for name, value in zip(CLASSES, probabilities)},
            "replicate_count": "2", "replicate_disagree": "0",
            "replicate_probabilities_json": json.dumps(replicates)}


class AuditFixtureTests(unittest.TestCase):
    def test_hand_counted_confusion(self):
        rows = [prediction(i, y, p) for i, (y, p) in enumerate([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 1)])]
        score = csv_metrics(rows)
        self.assertEqual(score["confusion_matrix"], [[1, 1, 0], [0, 1, 1], [0, 1, 1]])
        self.assertAlmostEqual(score["macro_f1"], (2/3 + 2/5 + 1/2) / 3)
        self.assertAlmostEqual(score["CHID_precision"], .5)
        self.assertAlmostEqual(score["CHID_recall"], .5)
        self.assertEqual(score["CHID_counts"], {"TP": 1, "FP": 1, "FN": 1, "support": 2})

    def test_absent_class_still_in_macro(self):
        score = csv_metrics([prediction(0, 1, 1)])
        self.assertAlmostEqual(score["macro_f1"], 1/3)
        self.assertEqual(score["CHID_f1"], 0)

    def test_duplicate_not_extra_biological_sample(self):
        row = prediction(1, 2, 2)
        with self.assertRaisesRegex(ValueError, "Duplicated"):
            csv_metrics([row, row.copy()])

    def test_replicate_average_corruption_detected(self):
        row = prediction(1, 2, 2)
        reps = json.loads(row["replicate_probabilities_json"])
        reps[0]["probabilities"] = [.2, .1, .7]
        row["replicate_probabilities_json"] = json.dumps(reps)
        with self.assertRaisesRegex(ValueError, "WT averaging"):
            csv_metrics([row])

    def test_wrong_saved_argmax_detected(self):
        row = prediction(1, 2, 2); row["y_pred"] = "1"
        with self.assertRaisesRegex(ValueError, "Argmax"):
            csv_metrics([row])

    def test_inner_selection_ignores_epoch_zero_and_exact_tie_is_early(self):
        history = [{"epoch": i, "weighted_validation_ce": 5.} for i in range(41)]
        history[0]["weighted_validation_ce"] = 0.
        history[7]["weighted_validation_ce"] = 1.
        history[25]["weighted_validation_ce"] = 1.
        self.assertEqual(selected_inner_epoch(history), 7)
        history[25]["weighted_validation_ce"] = 1. - 1e-12
        self.assertEqual(selected_inner_epoch(history), 25)

    def test_nonfinite_selection_fails(self):
        history = [{"epoch": i, "weighted_validation_ce": 5.} for i in range(41)]
        history[8]["weighted_validation_ce"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            selected_inner_epoch(history)


if __name__ == "__main__":
    unittest.main()
