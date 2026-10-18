"""Hand-made numerical contracts; no real data, weights or bulk inference."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts.probe_task1_morphology_dependence import (
    CLASSES, matched_shuffle_scale, matched_shuffle_samples, safe_output,
    score_rows, structure_support, summarize, verify_baseline,
)


def fixture(size=8):
    signal = np.arange(size * size, dtype=np.float32).reshape(size, size)
    signal = signal + signal.T
    validity = np.ones((size, size), dtype=np.float32)
    mask = np.zeros((size, size), dtype=np.float32)
    mask[:4, :4] = 1
    return np.stack((signal, validity, mask))


class MorphologyProbeTests(unittest.TestCase):
    def test_symmetry_geometry_multisets_and_matched_distance_counts(self):
        source = fixture()
        roi, bg, support = matched_shuffle_scale(source, 'unlabelled_123', 'WT1', 0, 0)
        self.assertGreater(support['matched_permutable_upper_per_region'], 0)
        i, j = np.triu_indices(8, 1)
        self.assertEqual(support['ROI_changed_upper'], support['background_changed_upper'])
        for changed, region in ((roi, source[2] >= .5), (bg, source[2] == 0)):
            np.testing.assert_array_equal(changed[1:], source[1:])
            np.testing.assert_array_equal(changed, changed.transpose(0, 2, 1))
            np.testing.assert_array_equal(np.diag(changed[0]), np.diag(source[0]))
            np.testing.assert_array_equal(changed[0][~region], source[0][~region])
            for distance in range(1, 8):
                chosen = region[i, j] & ((j - i) == distance)
                np.testing.assert_array_equal(np.sort(changed[0, i[chosen], j[chosen]]),
                                              np.sort(source[0, i[chosen], j[chosen]]))
        for layer in __import__('json').loads(support['layers_json']):
            self.assertEqual(layer['matched_upper'], min(layer['ROI_available_upper'], layer['background_available_upper']))
            self.assertEqual(layer['permutable_upper_per_region'], layer['matched_upper'] if layer['matched_upper'] >= 2 else 0)

    def test_partial_valid_pixels_and_fractional_boundary_not_changed(self):
        source = fixture()
        source[1, 0, 1] = source[1, 1, 0] = .5
        source[2, 2, 3] = source[2, 3, 2] = .25
        roi, bg, support = matched_shuffle_scale(source, 'case', '1', 1, 1)
        for changed in (roi, bg):
            self.assertEqual(changed[0, 0, 1], source[0, 0, 1])
            self.assertEqual(changed[0, 2, 3], source[0, 2, 3])
            np.testing.assert_array_equal(changed[1:], source[1:])
        self.assertEqual(support['partial_valid_upper'], 1)

    def test_empty_region_all_invalid_and_singleton_are_unavailable(self):
        for mode in ('no_ROI', 'no_background', 'all_invalid', 'singleton'):
            source = fixture()
            if mode == 'no_ROI':
                source[2] = 0
            elif mode == 'no_background':
                source[2] = 1
            elif mode == 'all_invalid':
                source[1] = 0
            else:
                source[2] = 0
                source[2, 0, 1] = source[2, 1, 0] = 1
            roi, bg, support = matched_shuffle_scale(source, 'case', '1', 0, 0)
            self.assertEqual(support['status'], 'unavailable')
            self.assertEqual(support['matched_permutable_upper_per_region'], 0)
            np.testing.assert_array_equal(roi, source)
            np.testing.assert_array_equal(bg, source)

    def test_constant_signal_keeps_available_distinct_from_changed(self):
        source = fixture()
        source[0] = 3
        roi, bg, support = matched_shuffle_scale(source, 'case', '1', 0, 0)
        self.assertGreater(support['matched_permutable_upper_per_region'], 0)
        self.assertEqual(support['ROI_changed_pixels'], 0)
        self.assertEqual(support['status'], 'matched_no_signal_change_in_one_or_both')
        aggregate = structure_support([dict(support, scale=scale, replicate=wt)
                                       for scale in (0, 1) for wt in ('1', '2')], 'case')
        self.assertEqual(aggregate['matched_available'], 1)
        self.assertEqual(aggregate['matched_effective_both'], 0)

    def test_signal_values_do_not_change_random_plan(self):
        source = fixture()
        roi, bg, _ = matched_shuffle_scale(source, 'case', 'WT1', 0, 2)
        transformed = source.copy()
        transformed[0] = source[0] * 11 + 7
        roi2, bg2, _ = matched_shuffle_scale(transformed, 'case', 'WT1', 0, 2)
        np.testing.assert_array_equal(roi2[0], roi[0] * 11 + 7)
        np.testing.assert_array_equal(bg2[0], bg[0] * 11 + 7)
        again, _, _ = matched_shuffle_scale(source, 'case', 'WT1', 0, 2)
        np.testing.assert_array_equal(again, roi)

    def test_whole_sample_shares_length_and_does_not_mutate_inputs(self):
        base = fixture(64)
        samples = {'x': np.stack([np.stack([base, base]), np.stack([base, base])]),
                   'length': np.array([[1.], [1.]], dtype=np.float32),
                   'structure_id': np.array(['case', 'case']), 'replicate': np.array(['1', '2']),
                   'y': np.array([2, 2]), 'group_id': np.array([1, 1])}
        original = samples['x'].copy()
        roi, bg, support = matched_shuffle_samples(samples, 0)
        self.assertIs(roi['length'], samples['length'])
        self.assertIs(bg['length'], samples['length'])
        np.testing.assert_array_equal(samples['x'], original)
        self.assertEqual(len(support), 4)
        # Changing labels leaves every selected pixel and permutation identical.
        roi2, bg2, _ = matched_shuffle_samples({**samples, 'y': np.array([0, 0])}, 0)
        np.testing.assert_array_equal(roi2['x'], roi['x'])
        np.testing.assert_array_equal(bg2['x'], bg['x'])

    def test_baseline_two_WT_mean_and_argmax_are_required(self):
        samples = {'structure_id': np.array(['case', 'case']), 'y': np.array([2, 2]),
                   'group_id': np.array([1, 1]), 'replicate': np.array(['1', '2'])}
        probabilities = np.array([[.1, .2, .7], [.2, .2, .6]])
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            path = Path(temporary) / 'pred.csv'
            row = {'structure_id': 'case', 'y_true': 2, 'y_pred': 2, 'group_id': 1,
                   'p_OPCID': .15, 'p_CHIN': .2, 'p_CHID': .65, 'replicate_count': 2,
                   'replicate_probabilities_json': json.dumps([
                       {'replicate': str(i+1), 'probabilities': p.tolist(), 'predicted': 2}
                       for i, p in enumerate(probabilities)])}
            def save():
                with path.open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
            save()
            _, check = verify_baseline(samples, probabilities, path)
            self.assertEqual(check['n_verified_structures'], 1)
            row['p_CHID'] = .64
            save()
            with self.assertRaisesRegex(ValueError, 'probabilities differ'):
                verify_baseline(samples, probabilities, path)
            row['p_CHID'] = .65
            row['y_pred'] = 1
            save()
            with self.assertRaisesRegex(ValueError, 'argmax differs'):
                verify_baseline(samples, probabilities, path)
            row['y_pred'] = 2
            # WT deviations can cancel exactly in the mean and still must fail.
            wrong_wt = probabilities.copy()
            wrong_wt[0, 0] += .01
            wrong_wt[0, 2] -= .01
            wrong_wt[1, 0] -= .01
            wrong_wt[1, 2] += .01
            row['replicate_probabilities_json'] = json.dumps([
                {'replicate': str(i+1), 'probabilities': p.tolist(), 'predicted': 2}
                for i, p in enumerate(wrong_wt)])
            save()
            with self.assertRaisesRegex(ValueError, 'WT baseline probabilities differ'):
                verify_baseline(samples, probabilities, path)

    def test_unavailable_summary_has_no_zero_effect_claim(self):
        rows = []
        for seed in (42, 43, 44):
            for repeat in (0, 1, 2):
                row = {'model': 'm2plus', 'seed': seed, 'repeat': repeat, 'structure_id': 'case',
                       'y_true': 0, 'annotation_type': 'OPCID', 'baseline_correct': 1,
                       'matched_available': 0, 'matched_effective_both': 0}
                for condition in ('baseline', 'roi', 'background'):
                    row[f'{condition}_prediction'] = 0
                    row[f'{condition}_trueclass_probability'] = .8
                    row.update({f'{condition}_p_{label}': p for label, p in zip(CLASSES, (.8, .1, .1))})
                rows.append(row)
        summary = summarize(rows, {'case'})
        comparison = summary['comparisons'][0]
        self.assertEqual(comparison['unavailable_count'], 1)
        self.assertEqual(comparison['full_OOF_includes_unavailable_unchanged_cases']['n_unique_structures'], 1)
        self.assertEqual(comparison['paired_matched_effects']['all']['status'], 'unavailable')
        self.assertNotIn('roi', comparison['paired_matched_effects']['all'])
        self.assertEqual(score_rows([])['status'], 'unavailable')

    def test_output_rejects_formal_and_existing_results(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            root = Path(temporary)
            (root / 'results/formal').mkdir(parents=True)
            with mock.patch('scripts.probe_task1_morphology_dependence.ROOT', root):
                with self.assertRaisesRegex(ValueError, 'independent'):
                    safe_output(root / 'results/formal', root / 'results/formal/probe')
                (root / 'results/probe').mkdir()
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    safe_output(root / 'results/formal', root / 'results/probe')


if __name__ == '__main__':
    unittest.main()
