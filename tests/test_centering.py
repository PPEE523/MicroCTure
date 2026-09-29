import sys
import unittest
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from centering_sensitivity import occlusion_mask, variants


class CenteringTest(unittest.TestCase):
    def test_full_grid_has_equal_interventions_for_both_anchors(self):
        root=Path(__file__).resolve().parents[1]
        cfg=json.loads((root/'configs/centering_sensitivity.json').read_text())
        grid=variants(cfg)
        self.assertEqual(len(grid),54)
        self.assertEqual(len({r['variant_id'] for r in grid}),54)
        groups=[{(r['width_bins'],r['shift_bins'],r['occlusion']) for r in grid if r['anchor']==a} for a in cfg['anchors']]
        self.assertEqual(groups[0],groups[1])

    def test_masks_are_symmetric_deterministic_and_do_not_mask_diagonal(self):
        a=occlusion_mask(64,.1,17)
        np.testing.assert_array_equal(a,a.T)
        np.testing.assert_array_equal(a,occlusion_mask(64,.1,17))
        self.assertTrue(np.diag(a).all())
        self.assertTrue(np.diag(a,1).all())
        self.assertTrue(occlusion_mask(64,0,17).all())
        self.assertFalse(np.array_equal(a,occlusion_mask(64,.1,29)))


if __name__=='__main__':unittest.main()
