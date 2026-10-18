import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from microcture.layout import bootstrap
bootstrap()
import review_saved_task56
from review_saved_task56 import (compare_metrics, contact_metrics, correlation,
                                 image_metrics, main, output_guard,
                                 structure_metrics, within_distance_correlation,
                                 write_once)


class RetainedReviewTest(unittest.TestCase):
    def test_image_metrics_constant_hand_calculation(self):
        target = np.full((12, 12), .5)
        actual = image_metrics(np.zeros_like(target), target, np.ones_like(target, bool), 3)
        self.assertEqual(actual["mse"], .25)
        self.assertAlmostEqual(actual["psnr"], 10 * np.log10(4))
        self.assertAlmostEqual(actual["ssim"], .0001 / .2501)
        self.assertIsNone(actual["shape_correlation"])

    def test_structure_boundary_shift_hand_calculation(self):
        target, prediction = np.ones((8, 8)), np.ones((8, 8))
        target[2, 3], prediction[3, 4] = 0, 0
        cfg = dict(profile_flank_bins=1, target_resolution_bp=200, boundary_radius_bins=2)
        row = dict(start=600, end=600, start_bin=0)
        actual = structure_metrics(prediction, target, np.ones((8, 8), bool), row, cfg)
        self.assertEqual(actual["evaluated_boundaries"], 2)
        self.assertEqual(actual["boundary_shift_bp"], 200)

    def test_grouped_correlation_excludes_small_and_constant_groups(self):
        # First distance has corr=+1 (3 observations); second has corr=-1
        # (4 observations). Small/constant groups contribute no weight.
        separation = np.array([2] * 3 + [3] * 4 + [4] * 2 + [5] * 3)
        x = np.array([0, 1, 2, 0, 1, 2, 3, 1, 2, 1, 1, 1.])
        y = np.array([0, 1, 2, 3, 2, 1, 0, 1, 2, 1, 2, 3.])
        self.assertAlmostEqual(within_distance_correlation(x, y, separation), -1 / 7)
        self.assertIsNone(correlation(np.ones(4), np.arange(4)))

    def test_contact_inverse_transform_exact_hand_case(self):
        xyz = np.array([[0., 0, 0], [1, 0, 0], [3, 0, 0], [6, 0, 0]])
        a, b = np.array([0, 0, 1]), np.array([1, 2, 3])
        counts = np.zeros((4, 4))
        counts[a, b] = 1 / np.array([1., 3., 5.]) - .1
        cfg = dict(contact_exponent=1., pseudocount=.1)
        fitted = dict(distance_scale=1., log_contact_background=[0., 0., 0.])
        actual = contact_metrics(xyz, a, b, np.ones(3, bool), counts, cfg, fitted)
        self.assertAlmostEqual(actual["log_distance_rmse"], 0)
        self.assertAlmostEqual(actual["contact_log_pearson"], 1)
        self.assertAlmostEqual(actual["contact_spearman"], 1)

    def test_changed_metric_fails_instead_of_reported_pass(self):
        with self.assertRaisesRegex(ValueError, "differs"):
            compare_metrics(dict(psnr="23"), dict(psnr=24.), {})

    def test_report_cannot_write_to_input_workspace_or_symlink(self):
        with tempfile.TemporaryDirectory(prefix="microcture-retained-test-") as temp:
            directory = Path(temp)
            data = directory / "data"
            data.mkdir()
            with self.assertRaisesRegex(ValueError, "outside"):
                output_guard(data, data / "new_report")
            link = directory / "link"
            link.symlink_to(data, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "outside"):
                output_guard(data, link / "report")

    def test_scientific_output_identical_rerun_and_conflict(self):
        with tempfile.TemporaryDirectory(prefix="microcture-retained-test-") as temp:
            path = Path(temp) / "result.csv"
            write_once(path, "a,b\n1,2\n")
            write_once(path, "a,b\n1,2\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                write_once(path, "a,b\n1,3\n")
            self.assertEqual(path.read_text(), "a,b\n1,2\n")

    def test_check_command_inventories_missing_files_without_training(self):
        with tempfile.TemporaryDirectory(prefix="microcture-retained-test-") as temp:
            directory = Path(temp)
            data, report = directory / "input", directory / "reports/task5/review/v1"
            data.mkdir()
            with mock.patch.object(review_saved_task56, "ROOT", directory), mock.patch("optional_common.ROOT", directory), contextlib.redirect_stdout(io.StringIO()):
                code = main(["check", "--data-root", str(data), "--report-dir", str(report)])
            self.assertEqual(code, 2)
            self.assertEqual(list(data.iterdir()), [])
            result = json.loads((report / "inputs.json").read_text())
            self.assertEqual(len(result["missing_checkpoints"]), 12)
            self.assertFalse(result["checkpoint_restoration_performed"])
            self.assertFalse(result["training_performed"])
            self.assertEqual({p.name for p in report.iterdir()}, {"inputs.json"})


if __name__ == "__main__":
    unittest.main()
