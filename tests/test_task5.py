import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from train_task5 import pool, image_metrics, SuperResolutionCNN, masked_loss


class SuperResolutionTest(unittest.TestCase):
    def test_pool_ignores_invalid_values(self):
        matrix = np.arange(64, dtype=float).reshape(8, 8)
        mask = np.ones((8, 8), bool)
        mask[0, 0] = False
        expected, fraction = pool(matrix, mask, 4)
        matrix[0, 0] = 1e9
        actual, _ = pool(matrix, mask, 4)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(fraction[0, 0], 15/16)

    def test_metrics_perfect_and_degraded(self):
        rng = np.random.default_rng(3)
        target = rng.random((32, 32))
        mask = np.ones_like(target, bool)
        good = image_metrics(target, target, mask)
        bad = image_metrics(np.zeros_like(target), target, mask)
        self.assertAlmostEqual(good['ssim'], 1)
        self.assertGreater(good['psnr'], bad['psnr'])
        self.assertGreater(good['ssim'], bad['ssim'])

    def test_symmetric_output_and_masked_gradient(self):
        for residual in [True, False]:
            model = SuperResolutionCNN(4, residual)
            output = model(torch.rand(2, 2, 8, 8))
            torch.testing.assert_close(output, output.transpose(-1, -2))
            mask = torch.ones_like(output)
            mask[:, :, :2] = 0
            target = torch.zeros_like(output)
            loss = masked_loss(output, target, mask)
            target[:, :, :2] = 1e8
            torch.testing.assert_close(loss, masked_loss(output, target, mask))
            loss.backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters()))
