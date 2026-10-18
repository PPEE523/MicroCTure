"""Hand-derived ROI area cases and frozen-input invariants; no model fitting."""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import numpy as np

from microcture.task1_roi import data


class AreaResamplingTests(unittest.TestCase):
    def test_exact_fractional_endpoint_weights(self):
        weights = data.roi_area_weights(25, 175, np.array([0, 100]), output_size=3)
        np.testing.assert_allclose(weights, [[1, 0], [.5, .5], [0, 1]], atol=0, rtol=0)
        np.testing.assert_allclose(weights.sum(1), 1, atol=0, rtol=0)

    def test_constant_signal_and_full_target_mask(self):
        signal = np.full((2, 2), 7.0)
        channels, facts = data.resample_roi(signal, np.ones_like(signal), np.array([0, 100]),
                                           25, 175, output_size=3)
        np.testing.assert_array_equal(channels[0], np.full((3, 3), 7, dtype=np.float32))
        np.testing.assert_array_equal(channels[1:], np.ones((2, 3, 3), dtype=np.float32))
        self.assertEqual(facts["native_bin_count"], 2)
        self.assertEqual(facts["resampling_mode"], "upsampling")
        self.assertFalse(facts["target_source_resolution_refined"])

    def test_validity_weighting_excludes_missing_values(self):
        signal = np.array([[2., 1000.], [1000., 6.]])
        validity = np.eye(2)
        channels, _ = data.resample_roi(signal, validity, np.array([0, 100]), 0, 200, output_size=1)
        np.testing.assert_array_equal(channels[:, 0, 0], [4., .5, 1.])

    def test_all_missing_and_partial_physical_coverage_differ(self):
        channels, facts = data.resample_roi(np.ones((1, 1)), np.zeros((1, 1)), np.array([0]),
                                           0, 200, output_size=2)
        np.testing.assert_array_equal(channels[0:2], np.zeros((2, 2, 2)))
        np.testing.assert_array_equal(channels[2], [[1, 0], [0, 0]])
        self.assertEqual(facts["target_physical_coverage_fraction"], .25)
        self.assertEqual(facts["target_validity_fraction"], 0)

    def test_symmetric_values_with_arbitrary_source_order(self):
        signal = np.array([[1., 2., 3.], [2., 4., 5.], [3., 5., 6.]])
        validity = np.array([[1., .5, 0], [.5, 1, .5], [0, .5, 1]])
        ordered, _ = data.resample_roi(signal, validity, np.array([0, 100, 200]), 30, 280, output_size=4)
        permutation = [2, 0, 1]
        reordered, _ = data.resample_roi(signal[np.ix_(permutation, permutation)],
                                       validity[np.ix_(permutation, permutation)], np.array([200, 0, 100]),
                                       30, 280, output_size=4)
        np.testing.assert_allclose(ordered, reordered, atol=1e-7, rtol=0)
        np.testing.assert_allclose(ordered, ordered.transpose(0, 2, 1), atol=1e-7, rtol=0)

    def test_terminal_partial_bin_clipped_to_physical_end(self):
        weights = data.roi_area_weights(100, 150, np.array([100]), output_size=2, chrom_length=150)
        np.testing.assert_array_equal(weights, [[1], [1]])
        with self.assertRaises(ValueError):
            data.roi_area_weights(100, 151, np.array([100]), chrom_length=150)


def fixture_samples(sid: str) -> dict:
    rng = np.random.default_rng(3 if sid == "A" else 4)
    return {"x": rng.normal(size=(2, 2, 3, 64, 64)).astype(np.float32),
            "length": np.array([[.3], [.3]], dtype=np.float32), "y": np.array([1, 1]),
            "structure_id": np.array([sid, sid]), "replicate": np.array([1, 2]),
            "sample_id": np.array(["rep1", "rep2"]), "group_id": np.array([4, 4])}


class FrozenInputTests(unittest.TestCase):
    def setUp(self):
        self.dataset = {"config": {"resolution_bp": 100, "output_size": 64, "window_bins": [64, 256]},
                        "structure_id": np.array(["A", "B"]),
                        "metadata": {"chrom_length": 6400, "samples": [{"replicate": 1}, {"replicate": 2}],
                                     "structures": [{"start": 1000, "end": 3000}, {"start": 1000, "end": 3000}]}}
        self.base = {"signature": "base", "signal_preprocessing": {"expected": [{}, {}]},
                     "dataset_signature": "dataset", "frozen_fold_signature": "fold", "train_ids": ["A"]}
        self.train, self.validation = fixture_samples("A"), fixture_samples("B")

    def native(self, dataset, structure_index, replicate_index, expected):
        return np.full((64, 64), 2.0 if structure_index == 0 else 9.0), np.ones((64, 64)), np.arange(64) * 100

    def test_r0_exact_original_objects_and_no_refit(self):
        with patch.object(data.formal, "prepare_inner", return_value=(self.train, self.validation, self.base)):
            train, validation, preprocessing = data.prepare_roi_inner(self.dataset, {}, "R0")
        self.assertIs(train, self.train)
        self.assertIs(validation, self.validation)
        self.assertIsNone(preprocessing["roi_first_signal_scaler"])
        with patch.object(data.formal, "prepare_outer", return_value=self.validation):
            self.assertIs(data.prepare_roi_outer(self.dataset, {}, preprocessing, "R0"), self.validation)

    def test_r1_only_first_scale_and_fit_inner_only(self):
        preserved = copy.deepcopy(self.train)
        with patch.object(data.formal, "prepare_inner", return_value=(self.train, self.validation, self.base)), \
             patch.object(data, "_native_source", side_effect=self.native):
            train, validation, preprocessing = data.prepare_roi_inner(self.dataset, {}, "R1")
        scaler = preprocessing["roi_first_signal_scaler"]
        self.assertEqual(scaler["mean"], 2.)
        self.assertEqual(scaler["train_ids"], ["A"])
        np.testing.assert_array_equal(train["x"][:, 0, 0], 0)
        self.assertTrue((validation["x"][:, 0, 0] > 0).all())
        for field in ("length", "y", "structure_id", "replicate", "sample_id", "group_id"):
            self.assertEqual(train[field].tobytes(), preserved[field].tobytes())
        self.assertEqual(train["x"][:, 1].tobytes(), preserved["x"][:, 1].tobytes())
        self.assertEqual(self.train["x"].tobytes(), preserved["x"].tobytes())
        np.testing.assert_array_equal(train["x"][:, 0, 2], 1)

    def test_outer_reuses_fit_and_rejects_tampering(self):
        with patch.object(data.formal, "prepare_inner", return_value=(self.train, self.validation, self.base)), \
             patch.object(data, "_native_source", side_effect=self.native):
            _, _, preprocessing = data.prepare_roi_inner(self.dataset, {}, "R1")
        frozen = copy.deepcopy(preprocessing)
        with patch.object(data.formal, "prepare_outer", return_value=self.validation), \
             patch.object(data, "_native_source", side_effect=self.native):
            outer = data.prepare_roi_outer(self.dataset, {}, preprocessing, "R1")
        self.assertEqual(preprocessing, frozen)
        self.assertEqual(outer["x"][:, 1].tobytes(), self.validation["x"][:, 1].tobytes())
        preprocessing["roi_first_signal_scaler"]["mean"] = 8
        with self.assertRaises(ValueError):
            data.prepare_roi_outer(self.dataset, {}, preprocessing, "R1")


if __name__ == "__main__":
    unittest.main()
