"""Synthetic numerical fidelity and blinded evidence/export protection checks."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "morphology_evidence", ROOT / "scripts/build_task1_morphology_evidence.py")
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


class EvidenceChecks(unittest.TestCase):
    def setUp(self):
        n_bins, length = 1001, 100050
        centers = np.array([500, 10])
        rng = np.random.default_rng(92)
        bands = [rng.poisson(2 + replicate, size=(n_bins, 257)).astype(float)
                 for replicate in range(2)]
        self.dataset = {
            "config": {"resolution_bp": 100, "window_bins": [64, 256], "output_size": 64},
            "n_bins": n_bins, "bands": bands,
            "bin_occupancy": ((np.arange(n_bins) + 1) * 100 <= length).astype(float),
            "center_bin": centers,
            "window_footprints": (centers[:, None] + np.arange(-128, 128)) % n_bins,
            "structure_id": np.array(["CHID_1", "OPCID_2"]), "y": np.array([2, 0]),
            "group_id": np.array([0, 1]), "replicate": np.array([1, 2]),
            "metadata": {"chrom_length": length,
                "label_order": ["OPCID", "CHIN", "CHID"],
                "samples": [{"sample_id": "WT1"}, {"sample_id": "WT2"}],
                "structures": [{"structure_id": "CHID_1", "type": "CHID",
                    "start": "48550", "end": "52250", "coordinate_status": "provisional"},
                    {"structure_id": "OPCID_2", "type": "OPCID",
                     "start": "120", "end": "2500", "coordinate_status": "provisional"}]} }
        self.preprocessing = {"signal_preprocessing": {
            "expected": [{"expected": [1.0] * 257}, {"expected": [2.0] * 257}],
            "scalers": [{"mean": 1.2, "std": .6}, {"mean": 1.1, "std": .5}]},
            "length_scaler": {"mean": 3.0, "std": .4}}

    def test_native_and_compressed_views_match_existing_model_inputs(self):
        before = copy.deepcopy(self.preprocessing)
        with patch.object(evidence.original, "_fit_expected", side_effect=AssertionError("Must not fit OE")), \
                patch.object(evidence.original, "prepare_fold", side_effect=AssertionError("Must not fit scalers")), \
                patch.object(evidence.controlled, "_fit_length_scaler", side_effect=AssertionError("Must not fit length")):
            records, facts = evidence.extract_case(self.dataset, 0, self.preprocessing)
        self.assertEqual(len(records), 4)
        self.assertTrue(facts["exact_model_signal_and_mask_verified"])
        self.assertEqual(before, self.preprocessing)
        for row in records:
            self.assertEqual(row["raw_counts"].shape, (row["width"], row["width"]))
            self.assertEqual(row["model_signal"].shape, (64, 64))
            self.assertEqual(row["model_mask"].shape, (64, 64))
            self.assertLessEqual(row["interval_coverage_fraction"], 1)
            self.assertTrue(np.isfinite(row["model_signal"]).all())
        small = records[0]
        np.testing.assert_array_equal(small["model_valid"], small["native_valid"])

    def test_origin_wrapping_terminal_partial_bin_and_near_diagonal_exclusions(self):
        records, _ = evidence.extract_case(self.dataset, 1, self.preprocessing)
        for row in records:
            self.assertTrue(row["window_wraps_origin"])
            self.assertEqual(row["terminal_partial_bins"], 1)
            self.assertTrue((np.diag(row["native_valid"]) == 0).all())
            self.assertTrue((np.diag(row["native_valid"], 1) == 0).all())
            terminal = np.flatnonzero((10 + np.arange(-row["width"] // 2, row["width"] // 2)) % 1001 == 1000)[0]
            self.assertTrue((row["native_valid"][terminal] == 0).all())
            self.assertAlmostEqual(row["interval_coverage_fraction"], 1)

    def test_exact_half_open_partial_bin_geometry_and_target_clipping(self):
        geometry = evidence.interval_geometry(50, 250, np.arange(4), 450, 2)
        np.testing.assert_allclose(geometry["axis"], [.5, 1, .5, 0])
        self.assertEqual(geometry["visible_segments"], [[.25, 1.25]])
        self.assertEqual(geometry["interval_coverage_fraction"], 1)
        clipped = evidence.interval_geometry(0, 10000, np.arange(32, 96), 12000)
        self.assertAlmostEqual(clipped["interval_coverage_fraction"], .64)
        self.assertEqual(clipped["cropped_target_bp"], 3600)
        origin = evidence.interval_geometry(50, 250, np.array([9, 0, 1, 2]), 950)
        self.assertEqual(origin["visible_segments"], [[1.5, 3.5]])

    def test_registered_blind_key_and_case_selection_accept_no_silent_subset(self):
        with tempfile.TemporaryDirectory(prefix="microcture-test-morphology-", dir="/tmp") as temporary:
            path = Path(temporary)
            (path / "reveal_key.csv").write_text("case_id,structure_id\nB001,CHID_1\nB002,OPCID_2\n")
            registered = evidence.read_reveal_key(path, self.dataset["metadata"])
            (path / "cases.csv").write_text("structure_id\nCHID_1\nOPCID_2\n")
            self.assertEqual(evidence.selected_ids(path / "cases.csv", registered), {"CHID_1", "OPCID_2"})
            (path / "priority.txt").write_text("B002\n")
            self.assertEqual(evidence.selected_ids(path / "priority.txt", registered), {"OPCID_2"})
            (path / "priority.txt").write_text("unknown\n")
            with self.assertRaisesRegex(ValueError, "unregistered"):
                evidence.selected_ids(path / "priority.txt", registered)
            (path / "reveal_key.csv").write_text("case_id,structure_id\nCHID_B001,CHID_1\nB002,OPCID_2\n")
            with self.assertRaisesRegex(ValueError, "class name"):
                evidence.read_reveal_key(path, self.dataset["metadata"])

    def test_protected_and_symlink_outputs_are_refused(self):
        for value in ("data/visuals", "results/task1_formal_v1", "reports/task1_controlled_review_v1/new",
                      "reports", "reports/task1_morphology_review_v1/new"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evidence.safe_output(value)
        self.assertEqual(evidence.safe_output("reports/task1_morphology_evidence_v1"),
                         ROOT / "reports/task1_morphology_evidence_v1")

    def test_fixed_display_scales_and_saturation_disclosure_do_not_use_case_quantiles(self):
        caps = evidence.display_limits()
        self.assertEqual(caps["raw_count_cap"], 100)
        self.assertAlmostEqual(caps["log1p_oe_vmax"], np.log1p(20))
        self.assertEqual(caps["standardized_abs_vmax"], 3)
        row = {"raw_counts": np.array([[0., 100.]]),
               "native_log1p_oe": np.array([[0., 10.]]), "native_valid": np.array([[1, 0]]),
               "pooled_log1p_oe": np.array([[0., 10.]]), "model_valid": np.array([[1, 1]]),
               "model_signal": np.array([[-3., 0.]])}
        fractions = evidence.saturation_fractions(row, caps)
        self.assertEqual(fractions["raw_log1p_count"], .5)
        self.assertEqual(fractions["native_log1p_oe"], 0)
        self.assertEqual(fractions["pooled_log1p_oe"], .5)
        self.assertEqual(fractions["absolute_model_z"], .5)
        for kwargs in ({"raw_count_cap": 0}, {"oe_cap": float("nan")}, {"z_cap": -1}):
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                evidence.display_limits(**kwargs)

    def test_one_synthetic_render_is_blinded_and_integrity_verified(self):
        records, facts = evidence.extract_case(self.dataset, 0, self.preprocessing)
        with tempfile.TemporaryDirectory(prefix="microcture-test-morphology-plot-", dir="/tmp") as temporary:
            destination = Path(temporary) / "B001"
            destination.mkdir()
            with patch.dict(os.environ, {"MPLBACKEND": "Agg", "MPLCONFIGDIR": str(Path(temporary) / "mpl")}):
                payload = evidence.render_case("B001", records, facts, destination, dpi=80)
            text = json.dumps(payload)
            for forbidden in ("CHID_1", "CHID", "OPCID_2", "y_true", "y_pred", "source_sheet"):
                self.assertNotIn(forbidden, text)
            self.assertEqual(evidence.verify_case(destination, "B001"), payload)
            self.assertTrue((destination / "contact_evidence.png").is_file())
            self.assertTrue((destination / "geometry_and_validity.pdf").is_file())
            (destination / "evidence.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                evidence.verify_case(destination, "B001")


if __name__ == "__main__":
    unittest.main()
