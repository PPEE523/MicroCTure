"""Synthetic tests for target masks and frozen inner-fold diagnostics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from microcture.task1 import data as original
from microcture.task1_controlled import data


class ControlledDataChecks(unittest.TestCase):
    def setUp(self):
        n_bins = 4000
        config = original._config({"split_search_trials": 1})
        centers = np.array([500, 500, 950, 950, 1750, 2300, 3000])
        ids = np.array(["CHIN_A", "CHID_A", "CHIN_B", "CHID_B", "OPCID_C", "CHID_D", "OUTER"])
        lengths = [2000, 8000, 2000, 6000, 8000, 2000, 4000]
        rows = [{"structure_id": str(sid), "start": str(int(center * 100 - length // 2)),
                 "end": str(int(center * 100 + length // 2)), "coordinate_status": "provisional"}
                for sid, center, length in zip(ids, centers, lengths)]
        groups, footprints = original._group_windows(centers, n_bins, 256, 10)
        band = np.broadcast_to(np.arange(1, 258, dtype=np.float64), (n_bins, 257))
        self.dataset = {"n_bins": n_bins, "config": config, "center_bin": centers,
                        "structure_id": ids, "y": np.array([1, 2, 1, 2, 0, 2, 0]),
                        "group_id": groups, "window_footprints": footprints,
                        "bands": [band, band], "bin_occupancy": np.ones(n_bins),
                        "replicate": np.array([1, 2]), "signature": "synthetic frozen dataset",
                        "metadata": {"chrom_length": n_bins * 100, "structures": rows,
                                     "coordinate_status": ["provisional"],
                                     "samples": [{"sample_id": "WT1"}, {"sample_id": "WT2"}]}}
        self.train_ids, self.validation_ids = ids[:4].tolist(), ids[4:6].tolist()
        self.original_train, self.original_validation, fit = original.prepare_fold(
            self.dataset, self.train_ids, self.validation_ids, config)
        self.frozen = {"fold": 0, "train_ids": self.train_ids, "validation_ids": self.validation_ids,
                       "test_ids": ["OUTER"], "preprocessing": fit}

    def test_shared_signal_windows_get_different_target_geometry(self):
        train, validation, preprocessing = data.prepare_controlled_fold(self.dataset, self.frozen)
        np.testing.assert_array_equal(train["x"][:, :, :2], self.original_train["x"])
        np.testing.assert_array_equal(validation["x"][:, :, :2], self.original_validation["x"])
        a = np.flatnonzero(train["structure_id"] == "CHIN_A")[0]
        b = np.flatnonzero(train["structure_id"] == "CHID_A")[0]
        np.testing.assert_array_equal(train["x"][a, :, :2], train["x"][b, :, :2])
        self.assertFalse(np.array_equal(train["x"][a, :, 2], train["x"][b, :, 2]))
        self.assertNotEqual(train["length"][a, 0], train["length"][b, 0])
        self.assertEqual(train["x"][a, 0, 1, 32, 32], 0)
        self.assertEqual(train["x"][a, 0, 2, 32, 32], 1)
        self.assertFalse(preprocessing["outer_images_used"])
        self.assertEqual(preprocessing["frozen_signal_preprocessing_signature"], self.frozen["preprocessing"]["signature"])

    def test_fractional_target_area_and_pooling(self):
        mask = data.target_masks_for_interval(48525, 50575, 500, n_bins=4000, chrom_length=400000)
        small_bins = np.arange(468, 532)
        fraction = np.maximum(0, np.minimum((small_bins + 1) * 100, 50575) - np.maximum(small_bins * 100, 48525)) / 100
        np.testing.assert_allclose(mask[0], np.outer(fraction, fraction))
        large_bins = np.arange(372, 628)
        large_fraction = np.maximum(0, np.minimum((large_bins + 1) * 100, 50575) - np.maximum(large_bins * 100, 48525)) / 100
        dense_target = np.outer(large_fraction, large_fraction)
        np.testing.assert_allclose(mask[1], dense_target.reshape(64, 4, 64, 4).mean(axis=(1, 3)))
        self.assertEqual(mask[0, 17, 17], 0.75 ** 2)
        self.assertAlmostEqual(float(mask[0].sum()), ((50575 - 48525) / 100) ** 2)

    def test_geometry_reverse_matches_both_axes_and_keeps_length(self):
        # Reflect a fractional off-centre interval about the exact crop centre.
        original_mask = data.target_masks_for_interval(48525, 50575, 500, n_bins=4000, chrom_length=400000)
        reflected_mask = data.target_masks_for_interval(49425, 51475, 500, n_bins=4000, chrom_length=400000)
        np.testing.assert_array_equal(original_mask[:, ::-1, ::-1], reflected_mask)
        train, _, _ = data.prepare_controlled_fold(self.dataset, self.frozen)
        before = train["length"].copy()
        reversed_x = train["x"][..., ::-1, ::-1]
        np.testing.assert_array_equal(reversed_x[:, :, 2], train["x"][:, :, 2, ::-1, ::-1])
        np.testing.assert_array_equal(train["length"], before)

    def test_circular_window_and_terminal_fraction_geometry(self):
        origin = data.target_masks_for_interval(0, 100, 0, n_bins=4000, chrom_length=400000)
        self.assertEqual(origin[0, 32, 32], 1)
        self.assertEqual(np.count_nonzero(origin[0]), 1)
        terminal = data.target_masks_for_interval(399925, 399952, 3999,
                                                  n_bins=4000, chrom_length=399952)
        self.assertAlmostEqual(float(terminal[0, 32, 32]), (27 / 100) ** 2, places=7)
        self.assertAlmostEqual(float(terminal[0].sum()), (27 / 100) ** 2, places=7)

    def test_length_scaler_uses_unique_inner_training_structures_only(self):
        train, _, preprocessing = data.prepare_controlled_fold(self.dataset, self.frozen)
        expected = np.log1p(np.array([2000, 8000, 2000, 6000]) / 100)
        self.assertAlmostEqual(preprocessing["length_scaler"]["mean"], expected.mean())
        self.assertAlmostEqual(preprocessing["length_scaler"]["std"], expected.std())
        self.assertEqual(preprocessing["length_scaler"]["fitted_unique_structures"], 4)
        changed = copy.deepcopy(self.dataset)
        changed["metadata"]["structures"][4]["start"] = "150000"
        changed["metadata"]["structures"][4]["end"] = "200000"
        train2, validation2, preprocessing2 = data.prepare_controlled_fold(changed, self.frozen)
        self.assertEqual(preprocessing["length_scaler"], preprocessing2["length_scaler"])
        np.testing.assert_array_equal(train["length"], train2["length"])
        np.testing.assert_array_equal(train["x"], train2["x"])
        self.assertTrue(np.isfinite(validation2["length"]).all())

    def test_constant_training_lengths_use_safe_unit_scale(self):
        changed = copy.deepcopy(self.dataset)
        for row, center in zip(changed["metadata"]["structures"][:4], changed["center_bin"][:4]):
            row["start"], row["end"] = str(center * 100 - 1000), str(center * 100 + 1000)
        train, validation, preprocessing = data.prepare_controlled_fold(changed, self.frozen)
        self.assertTrue(preprocessing["length_scaler"]["constant_training_lengths"])
        self.assertEqual(preprocessing["length_scaler"]["std"], 1.0)
        np.testing.assert_array_equal(train["length"], np.zeros_like(train["length"]))
        self.assertTrue(np.isfinite(validation["length"]).all())
        self.assertLess(float(np.abs(validation["length"]).max()), 5)

    def test_variants_preserve_signal_and_original_inputs(self):
        train, _, _ = data.prepare_controlled_fold(self.dataset, self.frozen)
        original_x, original_length = train["x"].copy(), train["length"].copy()
        for variant in data.INPUT_VARIANTS:
            modified = data.apply_input_variant(train, variant)
            self.assertEqual(modified["x"].shape, train["x"].shape)
            self.assertEqual(modified["length"].shape, train["length"].shape)
            np.testing.assert_array_equal(modified["x"][:, :, :2], train["x"][:, :, :2])
            expected_mask = np.zeros_like(train["x"][:, :, 2]) if variant in ("I0", "I2") else train["x"][:, :, 2]
            expected_length = np.zeros_like(train["length"]) if variant in ("I0", "I1") else train["length"]
            np.testing.assert_array_equal(modified["x"][:, :, 2], expected_mask)
            np.testing.assert_array_equal(modified["length"], expected_length)
            modified["x"][:] = -123
            modified["length"][:] = -123
        np.testing.assert_array_equal(train["x"], original_x)
        np.testing.assert_array_equal(train["length"], original_length)

    def test_no_signal_refit_or_outer_image_extraction(self):
        extract = original._extract_inputs
        accessed = []

        def guarded_extract(dataset, indices, expected):
            identifiers = dataset["structure_id"][indices].tolist()
            self.assertNotIn("OUTER", identifiers)
            accessed.extend(identifiers)
            return extract(dataset, indices, expected)

        with patch.object(original, "_extract_inputs", side_effect=guarded_extract), \
                patch.object(original, "_fit_expected", side_effect=AssertionError("Unexpected OE refit")), \
                patch.object(original, "prepare_fold", side_effect=AssertionError("Unexpected signal refit")), \
                patch.object(original, "split_train_validation", side_effect=AssertionError("Unexpected split refit")):
            data.prepare_controlled_fold(self.dataset, self.frozen)
        self.assertEqual(set(accessed), set(self.train_ids + self.validation_ids))

    def test_target_geometry_does_not_encode_class_or_source_metadata(self):
        train, _, _ = data.prepare_controlled_fold(self.dataset, self.frozen)
        changed = copy.deepcopy(self.dataset)
        changed["y"] = (changed["y"] + 1) % 3
        for row in changed["metadata"]["structures"]:
            row["type"] = "arbitrary class label"
            row["source_sheet"] = "arbitrary origin"
        train2, _, _ = data.prepare_controlled_fold(changed, self.frozen)
        np.testing.assert_array_equal(train["x"], train2["x"])
        np.testing.assert_array_equal(train["length"], train2["length"])

    def test_runner_augmentation_reverses_all_channels_and_duplicates_length(self):
        from microcture.task1_controlled.runner import augmented_arrays
        train, _, _ = data.prepare_controlled_fold(self.dataset, self.frozen)
        x, length, y = augmented_arrays(train, "reverse")
        n = len(train["y"])
        np.testing.assert_array_equal(x[:n], train["x"])
        np.testing.assert_array_equal(x[n:], train["x"][..., ::-1, ::-1])
        np.testing.assert_array_equal(length[:n], length[n:])
        np.testing.assert_array_equal(y[:n], y[n:])

    def test_geometry_baselines_ignore_signal_and_validity(self):
        from microcture.task1_controlled.runner import geometry_features
        train, _, _ = data.prepare_controlled_fold(self.dataset, self.frozen)
        changed = {**train, "x": train["x"].copy()}
        changed["x"][:, :, :2] = np.nan
        for name in ("length_only", "mask_only", "mask_length"):
            expected = geometry_features(train, name)
            actual = geometry_features(changed, name)
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_array_equal(actual, expected)

    def test_runner_stages_only_inner_views_with_mocked_fits(self):
        from microcture.task1_controlled import runner
        config = json.loads(Path("configs/task1_controlled.json").read_text(encoding="utf-8"))
        manifest = {"experiment_signature": "synthetic controlled manifest", "config": config}
        accessed = []

        def fake_fit(job, training, validation, *arguments):
            self.assertNotIn("OUTER", training["structure_id"])
            self.assertNotIn("OUTER", validation["structure_id"])
            self.assertEqual(training["x"].shape[1:], (2, 3, 64, 64))
            self.assertEqual(training["length"].shape, (len(training["y"]), 1))
            self.assertEqual(training["input_variant"], job["input_variant"])
            if job["input_variant"] in ("I0", "I2"):
                self.assertTrue(np.all(training["x"][:, :, 2] == 0))
            if job["input_variant"] in ("I0", "I1"):
                self.assertTrue(np.all(training["length"] == 0))
            accessed.append(job["name"])
            return {"status": "completed", "outer_images_used": False}

        with tempfile.TemporaryDirectory(prefix="microcture-runner-check-", dir="/tmp") as temporary:
            root = Path(temporary)
            frozen_run = root / config["frozen_run_dir"]
            (frozen_run / "folds").mkdir(parents=True)
            fold = {**self.frozen, "fold": config["input_stage"]["fold"]}
            (frozen_run / "folds" / f"fold_{fold['fold']}.json").write_text(json.dumps(fold))
            (frozen_run / "manifest.json").write_text(json.dumps({"config": {"models": {}}}))
            with patch.object(runner, "prepare_manifest", return_value=(manifest, {})), \
                    patch.object(runner, "load_dataset", return_value=self.dataset), \
                    patch.object(runner, "fit_diagnostic", side_effect=fake_fit), \
                    patch.object(runner, "fit_geometry", side_effect=fake_fit), \
                    patch.object(runner, "summarize_controlled", return_value={"status": "mocked"}), \
                    patch.object(original, "_fit_expected", side_effect=AssertionError("Unexpected OE fit")), \
                    patch.object(original, "prepare_fold", side_effect=AssertionError("Unexpected signal fit")):
                result = runner.run_stage(config, root, "input", None, root / "controlled", "cpu")
            self.assertEqual(result, {"status": "mocked"})
            self.assertEqual(accessed, config["input_stage"]["variants"] + config["input_stage"]["geometry_baselines"])


if __name__ == "__main__":
    unittest.main()
