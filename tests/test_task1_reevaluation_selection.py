"""Handwritten selection examples; no models, data preprocessing or fitting."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from microcture.task1_reevaluation.selection import (
    InnerCESelector, choose_inner_ce_epoch, endpoint_selection_records, selection_signature,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/task1_reevaluation.json").read_text())


def trajectory():
    return [{"epoch": e, "weighted_train_ce": 1 / (e + 1),
             "weighted_validation_ce": 0.01 if e == 0 else (0.2 if e in (7, 12) else 0.5),
             "validation_CHID_F1": 1.0 if e == 33 else 0.0,
             "outer_F1": 1.0 if e == 40 else 0.0} for e in range(41)]


class SelectionTests(unittest.TestCase):
    def test_epoch_zero_is_excluded_even_when_its_ce_is_smallest(self):
        self.assertEqual(choose_inner_ce_epoch(trajectory()), 7)

    def test_exact_tie_keeps_earliest_and_not_highest_CHID_or_outer_F1(self):
        rows = trajectory()
        self.assertEqual(choose_inner_ce_epoch(rows), 7)
        rows[12]["weighted_validation_ce"] = 0.19999999999999998
        self.assertEqual(choose_inner_ce_epoch(rows), 12)

    def test_live_strict_comparison_matches_fixed_history(self):
        selector = InnerCESelector()
        changes = [row["epoch"] for row in trajectory()
                   if selector.consider(row["epoch"], row["weighted_validation_ce"])]
        self.assertEqual(changes, [1, 7])
        self.assertEqual(selector.selected_epoch, 7)

    def test_partial_reordered_and_repeated_epochs_are_rejected(self):
        for rows in (trajectory()[:-1], list(reversed(trajectory())), trajectory()[:10] + trajectory()[9:]):
            with self.assertRaises(ValueError):
                choose_inner_ce_epoch(rows)

    def test_nonfinite_and_negative_ce_are_rejected_in_any_registered_epoch(self):
        for value in (float("nan"), float("inf"), float("-inf"), -0.001):
            for epoch in (0, 4, 40):
                rows = trajectory()
                rows[epoch]["weighted_validation_ce"] = value
                with self.assertRaises(ValueError):
                    choose_inner_ce_epoch(rows)

    def test_final40_record_is_distinct_and_both_records_share_the_full_fit(self):
        rows = trajectory()
        records = endpoint_selection_records(rows, CONFIG)
        self.assertEqual(list(records), ["final40", "inner_ce"])
        self.assertEqual(records["final40"]["selected_epoch"], 40)
        self.assertEqual(records["inner_ce"]["selected_epoch"], 7)
        for policy, record in records.items():
            self.assertEqual(record["weighted_validation_ce"], rows[record["selected_epoch"]]["weighted_validation_ce"])
            self.assertEqual(record["selection_signature"], selection_signature(CONFIG))
            self.assertEqual(record["full_training_epochs"], 40)
            self.assertFalse(record["early_stopping"])
            self.assertFalse(record["refit"])

    def test_selection_signature_binds_only_the_frozen_rule(self):
        config = copy.deepcopy(CONFIG)
        config["run_dir"] = "results/fixture_another_directory"
        self.assertEqual(selection_signature(config), selection_signature(CONFIG))
        config["selection"]["tie_break"] = "latest"
        self.assertNotEqual(selection_signature(config), selection_signature(CONFIG))

    def test_live_epoch_outside_registered_budget_is_rejected(self):
        for epoch in (-1, 41, True, 1.0):
            with self.assertRaises(ValueError):
                InnerCESelector().consider(epoch, 1.0)


if __name__ == "__main__":
    unittest.main()
