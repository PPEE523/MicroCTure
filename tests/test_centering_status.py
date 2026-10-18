"""Missing calibration is not a negative; no model fitting or real data."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from centering_status import (assess, decision_from_row, normalize_summary,
                              summarize_candidate, aggregate, metric_count)
from review_saved_centering import output_path


class CenteringStatusTests(unittest.TestCase):
    def test_finite_assessed_positive_and_negative(self):
        self.assertIs(assess([2, 3], [1, 2], True, 10, 10)['outside_both'], True)
        self.assertIs(assess([1, 3], [1, 2], True, 10, 10)['outside_both'], False)

    def test_insufficient_nonfinite_quality_and_missing_pairs_are_unknown(self):
        cases = [([2, 3], [1, 2], True, 8), ([2, 3], [float('nan'), 2], True, 10),
                 ([2, 3], [1, 2], False, 10), ([float('nan'), 3], [1, 2], True, 10),
                 ([], [], True, 10), ([2], [1, 2], True, 10)]
        for distances, thresholds, quality, count in cases:
            result = assess(distances, thresholds, quality, count, 10)
            self.assertEqual(result['assessment_status'], 'not_evaluable')
            self.assertIsNone(result['outside_both'])

    def test_legacy_false_with_nan_does_not_become_negative(self):
        row = dict(outside_both='False', quality_pass='True', rep1_cutoff='nan', rep2_cutoff='nan',
                   rep1_distance='1', rep2_distance='2')
        self.assertIsNone(decision_from_row(row))
        summary = dict(calibration_sufficient='False', all_variants_quality_pass='True',
                       original_outside_variants='0', boundary_outside_variants='0',
                       boundary_w64_outside_fraction='0', robust_recentered_anomaly='False')
        result = normalize_summary(summary, 8, 10)
        self.assertIsNone(result['original_outside_variants'])
        self.assertIsNone(result['boundary_w64_outside_fraction'])
        self.assertIsNone(result['robust_recentered_anomaly'])
        self.assertEqual(summary['robust_recentered_anomaly'], 'False')

    def test_complete_grid_unknown_and_real_zero_remain_distinct(self):
        settings = [dict(anchor=a, width_bins=w) for a in ['original', 'boundary'] for w in [64, 128, 256]]
        for flag, sufficient, expected in [(None, False, None), (False, True, False), (True, True, True)]:
            flags = {(i, m): flag for i in range(6) for m in ['pca', 'shape']}
            summary = summarize_candidate(settings, flags, True, True, sufficient, .8)
            self.assertIs(summary['robust_recentered_anomaly'], expected)
            total = aggregate([summary])
            self.assertEqual(total['robust_recentered_anomalies'], None if expected is None else int(expected))
        self.assertNotEqual(metric_count(None, False), 0)

    def test_output_cannot_escape_clone_or_replace_old_report(self):
        with tempfile.TemporaryDirectory(prefix='microcture-centering-fixture-', dir='/tmp') as tmp:
            clone = Path(tmp) / 'clone'
            clone.mkdir()
            data = Path(tmp) / 'old'
            data.mkdir()
            with self.assertRaises(ValueError):
                output_path(data / 'reports/new', data, root=clone)
            good = output_path('reports/new', data, root=clone)
            good.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                output_path(good, data, root=clone)


if __name__ == '__main__':
    unittest.main()
