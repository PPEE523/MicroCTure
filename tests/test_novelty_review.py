import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from review_novelty import boundary_profile, circular_gap


class NoveltyReviewTest(unittest.TestCase):
    def test_uniform_oe_has_no_boundary(self):
        oe = np.ones((32, 32))
        d = np.abs(np.arange(32)[:, None] - np.arange(32)[None, :])
        profile = boundary_profile(oe, d >= 2, 8, 20)
        np.testing.assert_allclose(profile[np.isfinite(profile)], 0)

    def test_weak_cross_contacts_localize_known_cut(self):
        oe = np.ones((32, 32))
        oe[:16, 16:] = .1
        oe[16:, :16] = .1
        d = np.abs(np.arange(32)[:, None] - np.arange(32)[None, :])
        profile = boundary_profile(oe, d >= 2, 8, 20)
        self.assertEqual(int(np.nanargmin(profile)), 16)
        self.assertLess(profile[16], -3)

    def test_circular_annotation_gap(self):
        self.assertEqual(circular_gap(5, 10, dict(start=95, end=98), 100), 7)
        self.assertEqual(circular_gap(5, 10, dict(start=7, end=8), 100), 0)


if __name__ == '__main__':
    unittest.main()
