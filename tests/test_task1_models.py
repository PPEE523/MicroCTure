"""Small forward-only model checks: no training, optimizer or matrix reads."""

import unittest

import torch
from torch import nn

from microcture.task1.models import MODEL_NAMES, build_model, model_spec


class ArchivedM2(nn.Module):
    """Archived architecture retained here to check compatibility independently."""

    def __init__(self):
        super().__init__()
        layers = []
        incoming = 2
        for outgoing in (16, 32, 64):
            layers.extend((nn.Conv2d(incoming, outgoing, 3, padding=1),
                           nn.GroupNorm(4, outgoing), nn.ReLU(), nn.MaxPool2d(2)))
            incoming = outgoing
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 3))

    def forward(self, inputs):
        batch, scales, channels, height, width = inputs.shape
        features = self.encoder(inputs.reshape(batch * scales, channels, height, width))
        return self.head(features.reshape(batch, scales * 64))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_primary_models_and_ablation_accept_the_shared_input(self):
        inputs = torch.randn(2, 2, 2, 64, 64)
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                model = build_model(name).eval()
                with torch.inference_mode():
                    logits = model(inputs)
                self.assertEqual(tuple(logits.shape), (2, 3))
                self.assertTrue(torch.isfinite(logits).all().item())

    def test_default_m2_is_architecture_and_weight_compatible_with_archive(self):
        archived = ArchivedM2().eval()
        restarted = build_model("m2").eval()
        restarted.load_state_dict(archived.state_dict(), strict=True)
        inputs = torch.randn(2, 2, 2, 64, 64)
        with torch.inference_mode():
            self.assertTrue(torch.equal(archived(inputs), restarted(inputs)))

    def test_spatial_readout_receives_actual_grid_tokens(self):
        # A ramp in the encoder output must remain spatially varying when it
        # becomes tokens. A GAP-then-token implementation would erase it.
        model = build_model("m2plus").eval()
        captured = []
        grid = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
        features = grid.expand(2, 64, 8, 8)
        encoder_hook = model.encoder.register_forward_hook(lambda _m, _i, _o: features)
        token_hook = model.blocks.register_forward_pre_hook(lambda _m, args: captured.append(args[0]))
        try:
            with torch.inference_mode():
                model(torch.zeros(1, 2, 2, 64, 64))
        finally:
            encoder_hook.remove()
            token_hook.remove()
        tokens = captured[0].reshape(1, 2, 16, 64)
        self.assertEqual(tuple(captured[0].shape), (1, 32, 64))
        spatial = (tokens - model.position_encoding[None, None]
                   - model.scale_embedding.weight[None, :, None])
        self.assertGreater(spatial[0, 0, :, 0].std().item(), 1.0)

    def test_deterministic_spatial_pooling_matches_adaptive_bin_means(self):
        features = torch.randn(2, 64, 8, 8)
        for grid in (1, 2, 4, 8):
            with self.subTest(grid=grid):
                model = build_model("m2plus", {"spatial_grid": grid})
                pooled = model.spatial_pool(features)
                expected = nn.AdaptiveAvgPool2d((grid, grid))(features)
                torch.testing.assert_close(pooled, expected)
        with self.assertRaises(ValueError):
            build_model("cnn_transformer", {"spatial_grid": 3})

    def test_no_attention_ablation_matches_m2plus_with_identical_weights(self):
        m2plus = build_model("m2plus").eval()
        ablation = build_model("cnn_transformer_no_attention").eval()
        ablation.load_state_dict(m2plus.state_dict(), strict=True)
        inputs = torch.randn(1, 2, 2, 64, 64)
        with torch.inference_mode():
            self.assertTrue(torch.equal(m2plus(inputs), ablation(inputs)))

    def test_specification_preserves_rng_and_counts_parameters(self):
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                previous_rng = torch.random.get_rng_state().clone()
                spec = model_spec(name)
                self.assertTrue(torch.equal(previous_rng, torch.random.get_rng_state()))
                count = sum(parameter.numel() for parameter in build_model(name).parameters())
                self.assertEqual(spec["parameters"], count)
                self.assertEqual(sum(spec["parameter_groups"].values()), count)

    def test_invalid_input_and_config_fail_before_training(self):
        for name in ("m2", "m2plus", "cnn_transformer"):
            with self.subTest(name=name):
                model = build_model(name)
                for shape in ((2, 2, 64, 64), (2, 1, 2, 64, 64), (2, 2, 2, 32, 32)):
                    with self.subTest(shape=shape), self.assertRaises(ValueError):
                        model(torch.zeros(shape))
        with self.assertRaises(ValueError):
            build_model("unknown")
        for config in ({"channels": [16, 32, 65]}, {"token_dim": 63},
                       {"token_dim": 64, "attention_heads": 3}, {"dropout": 1},
                       {"transformer_layers": 0}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                build_model("cnn_transformer", config)


if __name__ == "__main__":
    unittest.main()
