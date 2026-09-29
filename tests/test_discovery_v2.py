import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_discovery_v2 import adjusted_rand


class ClusteringAgreementTest(unittest.TestCase):
    def test_label_names_do_not_affect_agreement(self):
        self.assertEqual(adjusted_rand([0, 0, 1, 1], [4, 4, 2, 2]), 1.0)

    def test_crossed_partitions_have_negative_adjusted_agreement(self):
        self.assertAlmostEqual(adjusted_rand([0, 0, 1, 1], [0, 1, 0, 1]), -0.5)

    def test_identical_single_cluster(self):
        self.assertEqual(adjusted_rand([0, 0, 0], [2, 2, 2]), 1.0)


if __name__ == '__main__':
    unittest.main()
