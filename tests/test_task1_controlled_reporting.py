"""Handwritten saved-metadata fixtures; no training code or model imports."""

import copy
import unittest

from scripts.task1_controlled_reporting import aggregate_heads


CASES = [f"case_{index}" for index in range(6)]
CONFIG = {"training": {"epochs": 4, "learning_rate": 0.001},
          "head_stage": {"H3_learning_rate": 0.0001},
          "acceptance": {"window": 2, "CHID_F1_threshold": None}}


def diagnostic():
    return {"logit_between_structure_var": 1.0, "centered_logit_between_structure_var": 0.5,
            "logits": {"max_coordinate_range": 2.0}, "centered_logits": {"max_coordinate_range": 1.0},
            "probabilities": {"max_coordinate_range": 0.1},
            "prediction_counts": {"OPCID": 2, "CHIN": 6, "CHID": 2},
            "majority_prediction_fraction": 0.6,
            "head": {"available": True, "total_units": 32, "inactive_units": 0, "active_units": 32,
                     "effective_fraction": 0.5, "epsilon": 1e-6}}


def record(case="case_0", head="H1", endpoint=0.4):
    losses = [1.2, 0.9, 0.7, endpoint + 0.02, endpoint]
    history = []
    for epoch, loss in enumerate(losses):
        history.append({"epoch": epoch, "weighted_train_ce": loss,
                        "train_diagnostics": diagnostic(), "validation_diagnostics": diagnostic(),
                        "gradient_norms": {group: {"mean": 1.0, "min": 0.5, "max": 2.0,
                                                  "finite": True, "nonzero_fraction": 1.0}
                                           for group in ("total", "encoder", "head")} if epoch else None})
    return {"case": case, "head": head, "status": "completed",
            "metrics": {"acceptance": {"accepted": True}}, "history": history}


class ControlledReportingTests(unittest.TestCase):
    def test_missing_cases_are_pending_and_never_count_as_failures(self):
        result = aggregate_heads([record()], CASES, CONFIG)
        head = result["head_summaries"]["H1"]
        self.assertEqual(head["observed_cases"], 1)
        self.assertEqual(head["completed_cases"], 1)
        self.assertEqual(head["missing_cases"], 5)
        self.assertEqual(head["observed_not_accepted_cases"], 0)
        self.assertEqual(head["strict_accepted"], {"numerator": 1, "denominator": 1, "rate": 1.0})
        self.assertFalse(result["decision_ready"])
        self.assertEqual(result["candidate_recommendation"]["status"], "defer")

    def test_common_class_offset_is_not_discriminative_variation(self):
        entry = record()
        for row in entry["history"][1:]:
            for split in ("train", "validation"):
                diag = row[split + "_diagnostics"]
                diag["centered_logit_between_structure_var"] = 0.0
                diag["centered_logits"]["max_coordinate_range"] = 0.0
                diag["probabilities"]["max_coordinate_range"] = 0.0
        result = aggregate_heads([entry], CASES, CONFIG)
        incidence = result["head_summaries"]["H1"]["event_incidence"]["validation"]
        self.assertEqual(incidence["raw_logits_constant"]["post_epoch_incidence"]["numerator"], 0)
        self.assertEqual(incidence["centered_logits_constant"]["post_epoch_incidence"], {"numerator": 4, "denominator": 4, "rate": 1.0})
        self.assertEqual(incidence["centered_logits_constant"]["epoch_zero_case_incidence"]["numerator"], 0)
        self.assertEqual(incidence["probabilities_constant"]["final_epoch_case_incidence"]["numerator"], 1)
        self.assertEqual(result["head_summaries"]["H1"]["strict_accepted"]["numerator"], 0)

    def test_bias_gradient_does_not_certify_encoder_learning(self):
        entry = record(head="H0")
        for row in entry["history"][1:]:
            row["gradient_norms"]["encoder"].update(mean=0.0, min=0.0, max=0.0, nonzero_fraction=0.0)
            for split in ("train", "validation"):
                row[split + "_diagnostics"]["head"].update(inactive_units=32, active_units=0, effective_fraction=0.0)
        result = aggregate_heads([entry], CASES, CONFIG)
        head = result["head_summaries"]["H0"]
        self.assertEqual(head["effective_gradient_cases"]["head"]["numerator"], 1)
        self.assertEqual(head["effective_gradient_cases"]["total"]["numerator"], 1)
        self.assertEqual(head["effective_gradient_cases"]["encoder"]["numerator"], 0)
        self.assertEqual(head["strict_accepted"]["numerator"], 0)
        self.assertEqual(head["failure_category_counts"]["dead_head"], 1)

    def test_case_population_statistics_use_final_endpoints(self):
        result = aggregate_heads([record(endpoint=0.1), record(case="case_1", endpoint=0.3)], CASES, CONFIG)
        stats = result["head_summaries"]["H1"]["final_train_ce_by_case"]
        self.assertEqual(stats["n_cases"], 2)
        self.assertAlmostEqual(stats["mean"], 0.2)
        self.assertAlmostEqual(stats["median"], 0.2)
        self.assertAlmostEqual(stats["std_population"], 0.1)
        self.assertEqual((stats["min"], stats["max"]), (0.1, 0.3))

    def test_single_class_predictions_and_constant_probabilities_are_distinct(self):
        entry = record()
        for row in entry["history"][1:]:
            for split in ("train", "validation"):
                row[split + "_diagnostics"].update(prediction_counts={"OPCID": 0, "CHIN": 10, "CHID": 0}, majority_prediction_fraction=1.0)
        head = aggregate_heads([entry], CASES, CONFIG)["head_summaries"]["H1"]
        events = head["event_incidence"]["train"]
        self.assertEqual(events["single_class_predictions"]["post_epoch_incidence"]["numerator"], 4)
        self.assertEqual(events["probabilities_constant"]["post_epoch_incidence"]["numerator"], 0)
        self.assertEqual(head["failure_category_counts"]["class_concentration"], 1)
        self.assertEqual(head["failure_category_counts"]["dead_head"], 0)

    def test_small_complete_case_list_cannot_bypass_six_case_gate(self):
        records = [record(head=head) for head in ("H0", "H1", "H2", "H3")]
        result = aggregate_heads(records, ["case_0"], CONFIG)
        self.assertTrue(result["coverage_complete_for_supplied_expected_cases"])
        self.assertFalse(result["decision_ready"])
        self.assertEqual(result["comparison_status"], "pending")

    def test_six_case_candidates_prefer_simpler_h1_without_using_f1(self):
        records = [record(case=case, head=head) for case in CASES for head in ("H0", "H1", "H2", "H3")]
        for entry in records:
            entry["metrics"]["validation_CHID_F1"] = 0.01 if entry["head"] == "H1" else 0.99
        result = aggregate_heads(records, CASES, CONFIG)
        self.assertTrue(result["decision_ready"])
        self.assertEqual(result["candidate_recommendation"]["recipe"]["head_architecture"], "H1")
        self.assertEqual(len(result["H1_H2_same_lr_pairs"]), 6)

    def test_only_h3_recipe_keeps_untested_lower_lr_h1_explicit(self):
        records = [record(case=case, head=head) for case in CASES for head in ("H0", "H1", "H2", "H3")]
        for entry in records:
            if entry["head"] in ("H1", "H2"):
                entry["metrics"]["acceptance"]["accepted"] = False
                for row, loss in zip(entry["history"], [1.2, 0.1, 0.1, 0.8, 0.9]): row["weighted_train_ce"] = loss
        result = aggregate_heads(records, CASES, CONFIG)
        recommendation = result["candidate_recommendation"]
        self.assertEqual(recommendation["recipe"]["head_architecture"], "H2")
        self.assertEqual(recommendation["recipe"]["head_version"], "H3")
        self.assertEqual(recommendation["recipe"]["learning_rate"], 0.0001)
        self.assertIn("not tested", recommendation["untested_comparison"])

    def test_explicit_numerical_failure_is_observed_but_missing_is_not(self):
        failed = {"case": "case_0", "head": "H1", "status": "numerical_failure", "metrics": {}, "history": []}
        head = aggregate_heads([failed], CASES, CONFIG)["head_summaries"]["H1"]
        self.assertEqual(head["strict_accepted"], {"numerator": 0, "denominator": 1, "rate": 0.0})
        self.assertEqual(head["numerical_failure_cases"], 1)
        self.assertEqual(head["failure_category_counts"]["ineffective_encoder_gradient"], 0)
        self.assertEqual(head["missing_cases"], 5)
        self.assertEqual(head["final_train_ce_by_case"]["n_cases"], 0)

    def test_duplicate_records_and_unsorted_epochs_are_rejected(self):
        entry = record()
        with self.assertRaises(ValueError): aggregate_heads([entry, copy.deepcopy(entry)], CASES, CONFIG)
        entry["history"][1], entry["history"][2] = entry["history"][2], entry["history"][1]
        with self.assertRaises(ValueError): aggregate_heads([entry], CASES, CONFIG)


if __name__ == "__main__":
    unittest.main()
