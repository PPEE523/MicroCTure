import sys
import unittest
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from discovery_v2_core import bin_roles, window_role, train_background, symmetric_mask, quota_select, training_percentiles


class DiscoveryTrainingTest(unittest.TestCase):
    def test_heldout_counts_cannot_change_training_expected_or_qc(self):
        n, width = 60, 6
        roles = np.repeat([0, 1, 2], 20)
        band = dict(band=np.ones((n, width)), coverage=np.ones(n), valid=np.ones(n, bool))
        fitted = train_background(band, roles)
        changed = dict(band)
        changed['band'] = band['band'].copy()
        for d in range(2, width):
            heldout_endpoint = (roles != 0) | (np.roll(roles, -d) != 0)
            changed['band'][heldout_endpoint, d] *= 10000
        altered = train_background(changed, roles)
        np.testing.assert_equal(fitted['expected'], altered['expected'])
        np.testing.assert_equal(fitted['valid'][roles == 0], altered['valid'][roles == 0])
        self.assertEqual(fitted['threshold'], altered['threshold'])

    def test_window_cannot_cross_spatial_block(self):
        cfg = dict(spatial_block_bins=10, split_block_modulus=3, train_block_remainders=[0],
                   validation_block_remainders=[1], test_block_remainders=[2])
        roles = bin_roles(60, cfg)
        self.assertEqual(window_role(10, 10, roles, 10), 1)
        self.assertEqual(window_role(9, 2, roles, 10), -1)
        self.assertEqual(window_role(-1, 3, roles, 10), -1)

    def test_corruption_is_symmetric_and_target_is_untouched(self):
        x = torch.ones(2, 2, 8, 8)
        corrupt = symmetric_mask(x, .4, torch.Generator().manual_seed(42))
        torch.testing.assert_close(corrupt, corrupt.transpose(2, 3))
        self.assertTrue(torch.equal(x, torch.ones_like(x)))
        self.assertTrue((corrupt[:, 1] == 0).any())
        torch.testing.assert_close(symmetric_mask(x, 0, torch.Generator().manual_seed(42)), x)

    def test_ranks_fit_training_only_per_scale(self):
        widths = np.array([4, 4, 4, 8, 8, 8])
        values = np.array([1., 3, 2, 10, 30, 20])
        train = np.array([True, True, False, True, True, False])
        ranks = training_percentiles(values, widths, train, np.ones(6, bool))
        np.testing.assert_equal(ranks, [.5, 1, .5, .5, 1, .5])
        values[~train] = 100000
        changed = training_percentiles(values, widths, train, np.ones(6, bool))
        np.testing.assert_equal(ranks[train], changed[train])

    def test_scale_quotas_and_nonoverlap(self):
        rows = [dict(start_bin=0, width_bins=4), dict(start_bin=2, width_bins=4),
                dict(start_bin=8, width_bins=2), dict(start_bin=12, width_bins=2)]
        selected = quota_select(np.array([4, 3, 2, 1]), rows, np.ones(4, bool), {'4': 1, '2': 2}, 20)
        self.assertEqual(selected, [0, 2, 3])
        with self.assertRaises(ValueError):
            quota_select(np.array([4, 3, 2, 1]), rows, np.ones(4, bool), {'4': 2}, 20)


if __name__ == '__main__':
    unittest.main()
