import importlib.util
from pathlib import Path
import tempfile
import unittest

import cooler
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('plot_examples', Path(__file__).resolve().parents[1]/'scripts/plot_examples.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LocalMatricesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'example.cool'
        starts = np.arange(0, 1052, 10)
        bins = pd.DataFrame({'chrom': 'chr', 'start': starts, 'end': np.minimum(starts+10,1052)})
        pixels = [(i,i,1) for i in range(0,106,10)] + [(0,20,4),(0,90,6)]
        pixels = pd.DataFrame(sorted(pixels), columns=['bin1_id','bin2_id','count'])
        cooler.create_cooler(str(self.path), bins, pixels)

    def tearDown(self):
        self.temp.cleanup()

    def test_expected_includes_zeros_and_wrap_pairs(self):
        background = module.expected_background(self.path, 100, 3, 4)
        self.assertEqual(background['opportunities'][2], 9)
        self.assertAlmostEqual(background['expected'][2], 10/9)
        self.assertFalse(background['valid'][-1])
        self.assertEqual(background['total'], 21)

    def test_wrapped_fetch_keeps_cross_origin_contacts(self):
        matrix = module.aggregate_local(cooler.Cooler(str(self.path)), np.array([9,10,0]), 10)
        self.assertEqual(matrix[0,2], 6)
        np.testing.assert_equal(matrix, matrix.T)

    def test_aggregation_and_mask(self):
        clr = cooler.Cooler(str(self.path))
        ids = np.array([0,1,2])
        matrix = module.aggregate_local(clr, ids, 10)
        self.assertEqual(matrix[0,2], 4)
        bg = module.expected_background(self.path,100,3,4)
        raw, depth, oe = module.normalize(matrix, ids, bg)
        self.assertTrue(np.isnan(raw[0,1]))
        self.assertAlmostEqual(depth[0,2],4e8/21)
        self.assertAlmostEqual(oe[0,2],3.6)


if __name__ == '__main__':
    unittest.main()
