"""Forward-only checks for the paired controlled models; no training."""

import unittest

import torch
from torch import nn

from microcture.task1.models import build_model as build_original_model
from microcture.task1_controlled.models import BACKBONES, HEAD_VERSIONS, build_controlled_model, model_spec


class ControlledModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def _paired(self, backbone, head="H0", seed=2026):
        torch.manual_seed(seed)
        original = build_original_model(backbone).eval()
        original_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(seed)
        controlled = build_controlled_model(backbone, head_version=head).eval()
        self.assertTrue(torch.equal(original_rng, torch.random.get_rng_state()))
        return original, controlled

    def test_h0_matches_original_with_zero_added_fields(self):
        inputs = torch.randn(2, 2, 2, 64, 64)
        expanded = torch.cat((inputs, torch.zeros(2, 2, 1, 64, 64)), dim=2)
        for backbone in BACKBONES:
            with self.subTest(backbone=backbone):
                original, controlled = self._paired(backbone)
                with torch.inference_mode():
                    torch.testing.assert_close(
                        controlled(expanded, torch.zeros(2, 1)), original(inputs),
                        atol=2e-6, rtol=1e-5,
                    )
                self.assertTrue(torch.equal(controlled.encoder[0].weight[:, :2], original.encoder[0].weight))
                self.assertTrue(torch.equal(controlled.encoder[0].bias, original.encoder[0].bias))
                self.assertTrue(torch.equal(controlled.head.input_linear.weight[:, :-1], original.head[0].weight))
                self.assertTrue(torch.equal(controlled.head.input_linear.bias, original.head[0].bias))
                self.assertTrue(torch.equal(controlled.head.output_linear.weight, original.head[3].weight))
                self.assertTrue(torch.equal(controlled.head.output_linear.bias, original.head[3].bias))
                self.assertTrue((controlled.encoder[0].weight[:, 2] == 0).all().item())
                self.assertTrue((controlled.head.input_linear.weight[:, -1] == 0).all().item())
                self.assertTrue(controlled.encoder[0].weight.requires_grad)
                self.assertTrue(controlled.head.input_linear.weight.requires_grad)

    def test_head_versions_share_backbone_linear_initialization_and_rng(self):
        for backbone in BACKBONES:
            snapshots = {}
            random_states = []
            for head in HEAD_VERSIONS:
                torch.manual_seed(42)
                model = build_controlled_model(backbone, head_version=head)
                random_states.append(torch.random.get_rng_state().clone())
                snapshots[head] = model.state_dict()
            for head in HEAD_VERSIONS[1:]:
                with self.subTest(backbone=backbone, head=head):
                    for name, value in snapshots["H0"].items():
                        self.assertTrue(torch.equal(value, snapshots[head][name]), name)
            for state in random_states[1:]:
                self.assertTrue(torch.equal(random_states[0], state))
            self.assertEqual(set(snapshots["H2"]), set(snapshots["H3"]))
            for name, value in snapshots["H2"].items():
                self.assertTrue(torch.equal(value, snapshots["H3"][name]), name)

    def test_h1_only_changes_activation_and_h2_adds_normalization(self):
        h0 = build_controlled_model("m2plus", head_version="H0")
        h1 = build_controlled_model("m2plus", head_version="H1")
        h2 = build_controlled_model("m2plus", head_version="H2")
        self.assertIsInstance(h0.head.activation, nn.ReLU)
        self.assertIsInstance(h1.head.activation, nn.GELU)
        self.assertIsInstance(h0.head.normalization, nn.Identity)
        self.assertIsInstance(h1.head.normalization, nn.Identity)
        self.assertIsInstance(h2.head.normalization, nn.LayerNorm)
        self.assertEqual(h2.head.normalization.normalized_shape, (2049,))
        self.assertTrue((h2.head.normalization.weight == 1).all().item())
        self.assertTrue((h2.head.normalization.bias == 0).all().item())
        self.assertIs(h2.head.last, h2.head.output_linear)
        self.assertNotIn("head.last.weight", h2.state_dict())
        self.assertEqual(h0.head.dropout.p, 0.2)

    def test_added_fields_start_neutral_and_remain_usable_parameters(self):
        model = build_controlled_model("m2plus").eval()
        inputs = torch.randn(1, 2, 3, 64, 64)
        absent = inputs.clone()
        absent[:, :, 2] = 0
        with torch.inference_mode():
            before = model(inputs, torch.ones(1, 1))
            torch.testing.assert_close(before, model(absent, torch.zeros(1, 1)))
        # Forward-only sensitivity checks: manually assign the added columns,
        # without gradients/optimizer, to verify they are actually connected.
        with torch.no_grad():
            model.head.input_linear.weight[:, -1].fill_(1)
            model.head.input_linear.bias.fill_(2)
            model.head.output_linear.weight.fill_(0.1)
            with_length = model(absent, torch.ones(1, 1))
            without_length = model(absent, torch.zeros(1, 1))
            self.assertGreater((with_length - without_length).abs().max().item(), 0.1)
            readout_before = model.spatial_readout(inputs)
            model.encoder[0].weight[:, 2].fill_(0.1)
            readout_after = model.spatial_readout(inputs)
            self.assertGreater((readout_before - readout_after).abs().max().item(), 1e-4)

    def test_all_heads_accept_shared_input_and_return_finite_three_class_logits(self):
        image = torch.randn(2, 2, 3, 64, 64)
        length = torch.randn(2, 1)
        for backbone in BACKBONES:
            for head in HEAD_VERSIONS:
                with self.subTest(backbone=backbone, head=head), torch.inference_mode():
                    logits = build_controlled_model(backbone, head_version=head).eval()(image, length)
                    self.assertEqual(tuple(logits.shape), (2, 3))
                    self.assertTrue(torch.isfinite(logits).all().item())

    def test_spec_preserves_rng_and_describes_canonical_h2_h3_architecture(self):
        for head in HEAD_VERSIONS:
            before = torch.random.get_rng_state().clone()
            spec = model_spec("m2plus", head_version=head)
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
            self.assertEqual(spec["parameters"], sum(p.numel() for p in build_controlled_model("m2plus", head_version=head).parameters()))
            self.assertEqual(sum(spec["parameter_groups"].values()), spec["parameters"])
            self.assertEqual(spec["input_shape"], [2, 3, 64, 64])
            self.assertEqual(spec["paired_initialization"]["additional_cpu_rng_draws"], 0)
        h2 = model_spec("cnn_transformer", head_version="H2")
        h3 = model_spec("cnn_transformer", head_version="H3")
        self.assertEqual(h2["parameters"], h3["parameters"])
        self.assertEqual(h2["canonical_architecture"], h3["canonical_architecture"])

    def test_invalid_backbone_head_image_and_length_are_rejected(self):
        with self.assertRaises(ValueError):
            build_controlled_model("unknown")
        with self.assertRaises(ValueError):
            build_controlled_model("m2plus", head_version="H4")
        model = build_controlled_model("m2plus")
        for shape in ((2, 2, 2, 64, 64), (2, 2, 3, 32, 32), (2, 3, 64, 64)):
            with self.subTest(image_shape=shape), self.assertRaises(ValueError):
                model(torch.zeros(shape), torch.zeros(2, 1))
        image = torch.zeros(2, 2, 3, 64, 64)
        for shape in ((2,), (1, 1), (2, 2)):
            with self.subTest(length_shape=shape), self.assertRaises(ValueError):
                model(image, torch.zeros(shape))


if __name__ == "__main__":
    unittest.main()
