"""Small audit contracts: geometry, seed/error semantics and blinding."""
from __future__ import annotations

import unittest

from scripts.audit_task1_morphology_errors import (
    MODELS, REVIEW_FIELDS, _blind_cases, assert_pending_register, case_flags, interval_geometry, priority_cases,
)


class MorphologyErrorAuditTests(unittest.TestCase):
    def test_coarse_area_pooling_keeps_fractional_small_interval(self):
        # 450 bp covers six native bins but just two 400 bp output bins.
        geometry = interval_geometry(9990, 10440, 102, chrom_length=100000, n_bins=1000)
        self.assertEqual(geometry['fine_ROI_covered_bp'], 450)
        self.assertEqual(geometry['coarse_ROI_coverage_fraction'], 1.)
        self.assertEqual(geometry['coarse_ROI_nonzero_output_axis_bins'], 2)
        self.assertAlmostEqual(geometry['coarse_ROI_effective_output_axis_bins'], 450 / 400)
        self.assertAlmostEqual(geometry['coarse_ROI_effective_output_area_bins'], (450 / 400) ** 2)

    def test_target_longer_than_fine_window_has_partial_geometry(self):
        geometry = interval_geometry(5000, 15000, 100, chrom_length=100000, n_bins=1000)
        self.assertEqual(geometry['fine_ROI_covered_bp'], 6400)
        self.assertEqual(geometry['fine_ROI_coverage_fraction'], .64)
        self.assertEqual(geometry['fine_ROI_clipped'], 1)
        self.assertEqual(geometry['coarse_ROI_clipped'], 0)

    def test_origin_wrap_and_partial_last_chromosome_bin(self):
        geometry = interval_geometry(5, 135, 0, chrom_length=100050, n_bins=1001)
        self.assertEqual(geometry['fine_window_wraps_origin'], 1)
        self.assertEqual(geometry['fine_ROI_covered_bp'], 130)
        self.assertAlmostEqual(geometry['fine_ROI_effective_output_axis_bins'], 1.3)

    def test_stability_uses_all_seeds_and_chid_fp_excludes_true_chid(self):
        mixed = {model: [2, 1, 2] for model in MODELS}
        flags = case_flags(2, mixed)
        self.assertEqual(flags['m2plus_CHID_stable_FN'], 0)
        self.assertEqual(flags['m2plus_CHID_any_FN'], 1)
        self.assertEqual(flags['m2plus_CHID_any_FP'], 0)
        stable = {model: [2, 2, 2] for model in MODELS}
        flags = case_flags(1, stable)
        self.assertEqual(flags['m2plus_CHID_stable_FP'], 1)
        self.assertEqual(flags['m2plus_CHIN_to_CHID_stable'], 1)

    def test_stable_wrong_can_have_different_wrong_classes(self):
        values = {model: [0, 1, 0] for model in MODELS}
        flags = case_flags(2, values)
        self.assertEqual(flags['m2plus_stable_wrong'], 1)
        self.assertEqual(flags['m2plus_seed_unanimous'], 0)
        self.assertEqual(flags['m2plus_CHID_stable_FN'], 1)
        self.assertEqual(flags['m2plus_CHID_to_CHIN_stable'], 0)

    def test_blinding_drops_original_id_class_prediction_and_error_priority(self):
        source = {'case_id': 'case_0001', 'chrom': 'chr', 'start_bp': 0, 'end_bp': 100,
                  'center_bin': 0, 'length_bp': 100, 'group_id': 1, 'fold': 0,
                  'coordinate_system': '0-based-half-open', 'coordinate_status': 'provisional',
                  'structure_id': 'CHID_1', 'annotation_type': 'CHID', 'm2plus_seed_42_prediction': 'CHIN',
                  'm2plus_stable_wrong': 1, 'source_sheet': 'CHID annotations'}
        blind = _blind_cases([source])[0]
        for field in ('structure_id', 'annotation_type', 'm2plus_seed_42_prediction', 'm2plus_stable_wrong', 'source_sheet'):
            self.assertNotIn(field, blind)
        for field, default in REVIEW_FIELDS.items():
            self.assertEqual(blind[field], default)

    def test_reference_matching_prefers_group_then_length_not_confidence(self):
        def item(sid, label, group, fold, length, correct, fp=0):
            return {'case_id': sid, 'structure_id': sid, 'annotation_type': label,
                    'group_id': group, 'fold': fold, 'length_bp': length,
                    'm2plus_stable_correct': correct, 'm2plus_CHID_stable_FP': fp}
        cases = [item('a', 'CHID', 1, 0, 1000, 0),
                 item('b', 'CHIN', 1, 0, 2000, 1),
                 item('c', 'CHIN', 2, 0, 1001, 1),
                 item('d', 'OPCID', 3, 1, 1000, 1),
                 item('e', 'CHID', 2, 0, 3000, 1)]
        selected, result = priority_cases(cases)
        match = [row for row in result['matches'] if row['anchor'] == 'a' and row['reference_class'] == 'CHIN'][0]
        self.assertEqual(match['reference'], 'b')
        self.assertEqual(match['match_scope'], 'same_group')
        self.assertTrue({'a', 'e'}.issubset({row['structure_id'] for row in selected}))

    def test_human_observations_and_coordinates_cannot_be_silently_reset(self):
        source = {'case_id': 'case_0001', 'chrom': 'chr', 'start_bp': 0, 'end_bp': 100,
                  'center_bin': 0, 'length_bp': 100, 'group_id': 1, 'fold': 0,
                  'coordinate_system': '0-based-half-open', 'coordinate_status': 'provisional'}
        untouched = [{key: str(value) for key, value in _blind_cases([source])[0].items()}]
        assert_pending_register(untouched, [source])
        edited = [dict(untouched[0], observation_notes='Boundary looks unclear')]
        with self.assertRaisesRegex(ValueError, 'must be preserved'):
            assert_pending_register(edited, [source])
        edited = [dict(untouched[0], start_bp='10')]
        with self.assertRaisesRegex(ValueError, 'must be preserved'):
            assert_pending_register(edited, [source])


if __name__ == '__main__':
    unittest.main()
