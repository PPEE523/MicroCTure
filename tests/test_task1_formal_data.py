"""Synthetic and mocked checks of the formal fold-isolation contract."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from microcture.task1 import data as original
from microcture.task1_controlled import data as controlled
from microcture.task1_formal import data


class FormalDataChecks(unittest.TestCase):
    def setUp(self):
        centers = np.array([400, 900, 1400, 1900, 2400, 2900])
        ids = np.array(["T0", "T1", "V0", "V1", "O0", "O1"])
        groups, footprints = original._group_windows(centers, 4000, 256, 10)
        self.dataset = {"signature": "synthetic dataset", "structure_id": ids,
                        "config": {"resolution_bp": 100, "window_bins": [64, 256],
                                   "output_size": 64, "guard_bins": 10},
                        "center_bin": centers, "n_bins": 4000,
                        "group_id": groups, "window_footprints": footprints,
                        "metadata": {"chrom_length": 400000, "structures": [
                            {"structure_id": str(sid), "start": str(center * 100 - 1000),
                             "end": str(center * 100 + 1000)}
                            for sid, center in zip(ids, centers)]}}
        signal = {"dataset_signature": self.dataset["signature"], "train_ids": ["T0", "T1"],
                  "expected": [], "scalers": []}
        signal["signature"] = original._json_hash(signal)
        self.frozen = {"fold": 0, "train_ids": ["T0", "T1"], "validation_ids": ["V0", "V1"],
                       "test_ids": ["O0", "O1"], "preprocessing": signal}
        self.preprocessing = {
            "fold": 0, "dataset_signature": self.dataset["signature"],
            "frozen_fold_signature": controlled._signature(self.frozen),
            "frozen_signal_preprocessing_signature": signal["signature"],
            "signal_preprocessing": copy.deepcopy(signal),
            "train_ids": ["T0", "T1"], "validation_ids": ["V0", "V1"],
            "outer_images_used": False, "signal_statistics_refitted": False, "splits_refitted": False,
            "length_scaler": {"mean": 3.0, "std": 0.5, "fitted_unique_structures": 2,
                              "fit_scope": "frozen inner-training structures only"},
        }
        self.preprocessing["signature"] = controlled._signature(self.preprocessing)
        self.samples = {"x": np.arange(2 * 2 * 3 * 64 * 64, dtype=np.float32).reshape(2, 2, 3, 64, 64),
                        "length": np.array([[1.0], [2.0]], dtype=np.float32), "y": np.array([0, 1])}

    def test_inner_delegates_only_to_existing_controlled_preparation(self):
        train, validation = {"synthetic": "train"}, {"synthetic": "validation"}
        with patch.object(controlled, "prepare_controlled_fold",
                          return_value=(train, validation, self.preprocessing)) as prepare, \
                patch.object(original, "prepare_fold", side_effect=AssertionError("Unexpected signal fit")), \
                patch.object(original, "split_train_validation", side_effect=AssertionError("Unexpected split fit")):
            returned = data.prepare_inner(self.dataset, self.frozen)
        prepare.assert_called_once_with(self.dataset, self.frozen)
        self.assertIs(returned[0], train)
        self.assertIs(returned[1], validation)
        self.assertIs(returned[2], self.preprocessing)

    def test_outer_reuses_exact_fits_and_length_scaler_without_fitting(self):
        signal = {"x": np.zeros((2, 2, 2, 64, 64), dtype=np.float32),
                  "structure_id": np.array(["O0", "O1"]), "y": np.array([0, 2]),
                  "group_id": np.array([4, 5])}
        before = copy.deepcopy(self.preprocessing)
        with patch.object(original, "transform_heldout", return_value=signal) as transform, \
                patch.object(controlled, "_fit_length_scaler", side_effect=AssertionError("Unexpected length fit")), \
                patch.object(original, "_fit_expected", side_effect=AssertionError("Unexpected OE fit")), \
                patch.object(original, "prepare_fold", side_effect=AssertionError("Unexpected signal fit")):
            outer = data.prepare_outer(self.dataset, self.frozen, self.preprocessing)
        transform.assert_called_once_with(self.dataset, ["O0", "O1"], self.preprocessing["signal_preprocessing"])
        self.assertEqual(outer["x"].shape, (2, 2, 3, 64, 64))
        expected_length = (np.log1p(2000 / 100) - 3.0) / 0.5
        np.testing.assert_allclose(outer["length"], expected_length, atol=1e-7)
        self.assertEqual(before, self.preprocessing)

    def test_rejects_outer_spatial_group_leakage_before_extraction(self):
        changed = copy.deepcopy(self.dataset)
        changed["group_id"][4] = changed["group_id"][0]
        with patch.object(original, "transform_heldout", side_effect=AssertionError("Images must not be read")):
            with self.assertRaisesRegex(ValueError, "Spatial groups"):
                data.prepare_outer(changed, self.frozen, self.preprocessing)

    def test_rejects_validation_outer_window_guard_overlap(self):
        changed = copy.deepcopy(self.dataset)
        changed["window_footprints"][4] = changed["window_footprints"][2] + 3
        with self.assertRaisesRegex(ValueError, "window/guard overlap"):
            data.prepare_inner(changed, self.frozen)

    def test_rejects_preprocessing_and_frozen_fold_tampering(self):
        cases = []
        changed = copy.deepcopy(self.preprocessing)
        changed["length_scaler"]["mean"] = 99.0
        cases.append((self.frozen, changed, "signature mismatch"))
        changed_fold = copy.deepcopy(self.frozen)
        changed_fold["fold"] = 1
        cases.append((changed_fold, self.preprocessing, "different frozen fold"))
        changed = copy.deepcopy(self.preprocessing)
        changed["length_scaler"]["fit_scope"] = "train plus validation"
        changed["signature"] = controlled._signature({k: v for k, v in changed.items() if k != "signature"})
        cases.append((self.frozen, changed, "inner-training structures only"))
        changed = copy.deepcopy(self.preprocessing)
        changed["signal_preprocessing"]["scalers"] = [{"mean": 99.0}]
        changed["signature"] = controlled._signature({k: v for k, v in changed.items() if k != "signature"})
        cases.append((self.frozen, changed, "signal preprocessing differs"))
        for fold, preprocessing, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    data.prepare_outer(self.dataset, fold, preprocessing)

    def test_rejects_missing_or_duplicate_partition_ids(self):
        for ids in (["O0"], ["O0", "O0"], ["unknown", "O1"]):
            changed = copy.deepcopy(self.frozen)
            changed["test_ids"] = ids
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError):
                    data.prepare_inner(self.dataset, changed)

    def test_conditions_copy_inputs_and_remove_exact_channels(self):
        before_x, before_length = self.samples["x"].copy(), self.samples["length"].copy()
        for condition in data.CONDITIONS:
            result = data.apply_condition(self.samples, condition)
            self.assertEqual(result["x"].shape, before_x.shape)
            self.assertEqual(result["length"].shape, before_length.shape)
            np.testing.assert_array_equal(result["x"][:, :, :2], 0 if condition == "G0" else before_x[:, :, :2])
            np.testing.assert_array_equal(result["x"][:, :, 2], 0 if condition == "I0" else before_x[:, :, 2])
            np.testing.assert_array_equal(result["length"], 0 if condition == "I0" else before_length)
            result["x"][:] = -1
            result["length"][:] = -1
        np.testing.assert_array_equal(self.samples["x"], before_x)
        np.testing.assert_array_equal(self.samples["length"], before_length)

    def test_rejects_unknown_condition_or_malformed_layout(self):
        with self.assertRaisesRegex(ValueError, "Unknown formal condition"):
            data.apply_condition(self.samples, "I2")
        with self.assertRaisesRegex(ValueError, "input layout"):
            data.apply_condition({**self.samples, "length": np.zeros(2)}, "I3")


if __name__ == "__main__":
    unittest.main()
