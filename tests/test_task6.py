import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_task6 import aggregate_upper
from train_task6 import pair_partition, fit_geometry_inputs, classical_mds, GraphDecoder, distance_loss


class ReconstructionTest(unittest.TestCase):
    def test_aggregation_counts_diagonal_once(self):
        result = aggregate_upper(np.array([0, 0, 1]), np.array([0, 1, 1]), np.array([3, 7, 2]), 2)
        np.testing.assert_array_equal(result, [[3, 7], [7, 2]])
        self.assertEqual(np.triu(result).sum(), 12)

    def test_mds_recovers_euclidean_distances(self):
        points = np.random.default_rng(42).normal(size=(12, 3))
        reconstructed = classical_mds(squareform(pdist(points)))
        np.testing.assert_allclose(pdist(reconstructed), pdist(points), atol=1e-7)

    def test_heldout_edges_do_not_change_graph_or_mds_input(self):
        n = 50
        matrix = np.random.default_rng(42).uniform(1, 100, (n, n))
        matrix = (matrix+matrix.T)/2
        a, b, roles = pair_partition(n, 42)
        cfg = dict(pseudocount=.5, contact_exponent=1/3)
        before = fit_geometry_inputs(matrix, a, b, roles, cfg)
        matrix[a[roles != 0], b[roles != 0]] = 1e9
        matrix[b[roles != 0], a[roles != 0]] = 1e9
        after = fit_geometry_inputs(matrix, a, b, roles, cfg)
        for i in [1, 2]: np.testing.assert_array_equal(before[i], after[i])
        self.assertEqual(before[3], after[3])
        self.assertTrue(np.all(before[2][a[roles != 0], b[roles != 0]] == 0))

    def test_decoder_and_loss_are_finite(self):
        initial = torch.randn(20, 3)
        model = GraphDecoder(initial, 8)
        xyz = model(torch.eye(20))
        a, b = torch.arange(10), torch.arange(10, 20)
        loss = distance_loss(xyz, a, b, torch.ones(10))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(xyz.mean(0), torch.zeros(3), atol=1e-6, rtol=0)
