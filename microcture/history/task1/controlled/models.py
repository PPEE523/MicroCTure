"""Paired model initialization for the controlled input/head experiments.

The original Task 1 source stays frozen. Every controlled input variant uses
three image channels (signal, validity, target scope) and a length scalar, even
when its scope/length fields are zero. Added feature weights start at zero and
all pre-existing weights are copied exactly from the original initialization.

H0 retains ReLU. H1 changes only that activation to GELU. H2 adds LayerNorm on
the entire readout-plus-length vector and otherwise matches H1. H3 is the same
architecture and initialization as H2; its lower learning rate belongs to the
runner, not to this module. H2/H3 normalization includes the length scalar, so
length can influence normalized features even when its direct weight is zero.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from microcture.task1.models import build_model as build_original_model
from microcture.task1.models import model_spec as original_model_spec


BACKBONES = ("m2", "m2plus", "cnn_transformer")
HEAD_VERSIONS = ("H0", "H1", "H2", "H3")


def _expanded_input_convolution(original: nn.Conv2d) -> nn.Conv2d:
    """Add a trainable zero-weight target channel without advancing the RNG."""
    if original.in_channels != 2 or original.groups != 1:
        raise ValueError("Controlled initialization requires the original two-channel stem")
    with torch.random.fork_rng(devices=[]):
        expanded = nn.Conv2d(
            3, original.out_channels, kernel_size=original.kernel_size,
            stride=original.stride, padding=original.padding, dilation=original.dilation,
            groups=original.groups, bias=original.bias is not None,
            padding_mode=original.padding_mode, device=original.weight.device,
            dtype=original.weight.dtype,
        )
    with torch.no_grad():
        expanded.weight.zero_()
        expanded.weight[:, :2].copy_(original.weight)
        if original.bias is not None:
            expanded.bias.copy_(original.bias)
    return expanded


class ControlledHead(nn.Module):
    """Named head components support forward-only activation instrumentation."""

    def __init__(self, original: nn.Sequential, version: str) -> None:
        super().__init__()
        incoming: nn.Linear = original[0]
        self.version = version
        # Reusing every original coefficient also preserves fan-in scaling;
        # changing initialization scale would confound the input comparison.
        with torch.random.fork_rng(devices=[]):
            self.input_linear = nn.Linear(
                incoming.in_features + 1, incoming.out_features,
                bias=incoming.bias is not None, device=incoming.weight.device,
                dtype=incoming.weight.dtype,
            )
        with torch.no_grad():
            self.input_linear.weight.zero_()
            self.input_linear.weight[:, :-1].copy_(incoming.weight)
            if incoming.bias is not None:
                self.input_linear.bias.copy_(incoming.bias)
        self.normalization: nn.Module = (
            nn.LayerNorm(incoming.in_features + 1, device=incoming.weight.device,
                         dtype=incoming.weight.dtype)
            if version in ("H2", "H3") else nn.Identity()
        )
        self.activation: nn.Module = nn.ReLU() if version == "H0" else nn.GELU()
        self.dropout = original[2]
        self.output_linear = original[3]

    @property
    def last(self) -> nn.Linear:
        """Alias without duplicate module registration/state-dictionary keys."""
        return self.output_linear

    def forward(self, readout_with_length: Tensor) -> Tensor:
        hidden = self.input_linear(self.normalization(readout_with_length))
        return self.output_linear(self.dropout(self.activation(hidden)))


class ControlledModel(nn.Module):
    """Original backbone with three channels and a controlled classification head."""

    def __init__(self, backbone_name: str, original: nn.Module, head_version: str) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.head_version = head_version
        self.config = original.config
        self.head = ControlledHead(original.head, head_version)
        original.encoder[0] = _expanded_input_convolution(original.encoder[0])
        original.head = nn.Identity()
        self.backbone = original

    @property
    def encoder(self) -> nn.Sequential:
        return self.backbone.encoder

    def spatial_readout(self, x: Tensor) -> Tensor:
        """Retain the original pooling/token operations before adding length."""
        if x.ndim != 5 or tuple(x.shape[1:]) != (2, 3, 64, 64):
            raise ValueError(f"Expected image shape [N, 2, 3, 64, 64], got {tuple(x.shape)}")
        if x.shape[0] < 1:
            raise ValueError("Controlled models require a nonempty batch")
        batch = x.shape[0]
        features = self.encoder(x.reshape(batch * 2, 3, 64, 64))
        if self.backbone_name == "m2":
            return features.reshape(batch, 2 * self.config["channels"][-1])
        features = self.backbone.spatial_pool(features)
        tokens = self.backbone.token_projection(features.flatten(2).transpose(1, 2))
        tokens = tokens.reshape(batch, 2, -1, self.config["token_dim"])
        tokens = (
            tokens + self.backbone.position_encoding[None, None]
            + self.backbone.scale_embedding.weight[None, :, None]
        )
        tokens = self.backbone.blocks(tokens.reshape(batch, -1, self.config["token_dim"]))
        return tokens.flatten(1)

    def forward(self, x: Tensor, length: Tensor) -> Tensor:
        if length.ndim != 2 or tuple(length.shape) != (x.shape[0], 1):
            raise ValueError(f"Expected length shape [N, 1], got {tuple(length.shape)}")
        readout = self.spatial_readout(x)
        length = length.to(device=readout.device, dtype=readout.dtype)
        return self.head(torch.cat((readout, length), dim=1))


def build_controlled_model(
    backbone_name: str,
    model_config: Mapping[str, Any] | None = None,
    head_version: str = "H0",
) -> ControlledModel:
    """Initialize paired models from the frozen original builder.

    Seeding immediately before this function gives matching original weights
    for every head/input variant. Expansion does not consume additional CPU
    RNG draws, so dropout starts from the same RNG state across head versions.
    """
    if backbone_name not in BACKBONES:
        raise ValueError(f"Unknown backbone {backbone_name!r}; choose from {BACKBONES}")
    if head_version not in HEAD_VERSIONS:
        raise ValueError(f"Unknown head version {head_version!r}; choose from {HEAD_VERSIONS}")
    original = build_original_model(backbone_name, model_config)
    return ControlledModel(backbone_name, original, head_version)


def model_spec(
    backbone_name: str,
    model_config: Mapping[str, Any] | None = None,
    head_version: str = "H0",
) -> dict[str, Any]:
    """JSON-ready specifications; input variants do not change parameter counts."""
    original = original_model_spec(backbone_name, model_config)
    with torch.random.fork_rng(devices=[]):
        model = build_controlled_model(backbone_name, model_config, head_version)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    canonical_head = "H2" if head_version == "H3" else head_version
    return {
        **original,
        "name": backbone_name,
        "experiment_family": "controlled_input_and_head_ablation",
        "input_shape": [2, 3, 64, 64],
        "image_channel_order": ["signal", "valid_fraction", "target_scope"],
        "length_shape": [1],
        "head_version": head_version,
        "canonical_head_architecture": canonical_head,
        "canonical_architecture": f"{backbone_name}/{canonical_head}/three_channels_plus_length",
        "parameters": parameter_count,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "original_parameters": original["parameters"],
        "additional_parameters": parameter_count - original["parameters"],
        "parameter_groups": {
            group: sum(parameter.numel() for parameter in module.parameters())
            for group, module in model.named_children()
        },
        "head": {
            "readout_features_without_length": model.head.input_linear.in_features - 1,
            "readout_features_with_length": model.head.input_linear.in_features,
            "normalization": "LayerNorm" if head_version in ("H2", "H3") else "Identity",
            "normalization_scope": "entire_readout_plus_length" if head_version in ("H2", "H3") else None,
            "activation": "ReLU" if head_version == "H0" else "GELU",
            "dropout": model.head.dropout.p,
            "hidden_features": model.head.input_linear.out_features,
            "output_classes": model.head.output_linear.out_features,
        },
        "paired_initialization": {
            "original_coefficients_copied_exactly": True,
            "target_channel_initial_weights": "zero_trainable",
            "length_column_initial_weights": "zero_trainable",
            "LayerNorm_initial_gamma_beta": [1, 0] if head_version in ("H2", "H3") else None,
            "additional_cpu_rng_draws": 0,
        },
        "H3_difference": "runner_learning_rate_only; architecture_and_initialization_equal_H2",
        "input_variant_parameter_budget": "identical for every input variant under the same backbone/head",
    }
