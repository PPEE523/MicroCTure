"""Task 1 shared backbones and paired target-aware classification heads.

The registered module attribute names and state-dictionary keys are unchanged,
so historical state-dictionary checkpoints load strictly without conversion."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


# Backbone architectures.
MODEL_NAMES = ("m2", "m2plus", "cnn_transformer", "cnn_transformer_no_attention")

DEFAULT_CONFIG: dict[str, Any] = {
    "channels": [16, 32, 64],
    "dropout": 0.2,
    "spatial_grid": 4,
    "token_dim": 64,
    "transformer_layers": 1,
    "attention_heads": 4,
    "ffn_dim": 128,
    "head_dim": 32,
}


def _settings(config: Mapping[str, Any] | None) -> dict[str, Any]:
    settings = {**DEFAULT_CONFIG, **(dict(config) if config is not None else {})}
    channels = settings["channels"]
    if len(channels) != 3 or any(
        isinstance(channel, bool) or int(channel) != channel
        or int(channel) <= 0 or int(channel) % 4 for channel in channels
    ):
        raise ValueError("channels must contain three positive multiples of 4")
    settings["channels"] = [int(channel) for channel in channels]
    settings["dropout"] = float(settings["dropout"])
    if not 0.0 <= settings["dropout"] < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    for key in (
        "spatial_grid", "token_dim", "transformer_layers", "attention_heads",
        "ffn_dim", "head_dim",
    ):
        value = settings[key]
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError(f"{key} must be a positive integer")
        settings[key] = int(value)
    if settings["spatial_grid"] not in (1, 2, 4, 8):
        raise ValueError("spatial_grid must divide the 8 x 8 CNN feature map: 1, 2, 4 or 8")
    if settings["token_dim"] % 4:
        raise ValueError("token_dim must be divisible by 4 for 2D position encoding")
    if settings["token_dim"] % settings["attention_heads"]:
        raise ValueError("token_dim must be divisible by attention_heads")
    return settings


def _encoder(channels: list[int]) -> nn.Sequential:
    """The same convolutional stem for every neural model."""
    layers: list[nn.Module] = []
    in_channels = 2
    for out_channels in channels:
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ])
        in_channels = out_channels
    return nn.Sequential(*layers)


def _classification_head(in_features: int, config: dict[str, Any]) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, config["head_dim"]),
        nn.ReLU(),
        nn.Dropout(config["dropout"]),
        nn.Linear(config["head_dim"], 3),
    )


def _check_input(x: Tensor) -> None:
    if x.ndim != 5 or tuple(x.shape[1:]) != (2, 2, 64, 64):
        raise ValueError(
            "Expected [batch, 2 scales, 2 channels, 64, 64]; "
            f"received {tuple(x.shape)}"
        )


class DualScaleM2(nn.Module):
    """Archived M2 architecture: shared CNN, per-scale GAP, 128-to-32 head."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            *_encoder(config["channels"]), nn.AdaptiveAvgPool2d(1)
        )
        self.head = _classification_head(2 * config["channels"][-1], config)

    def forward(self, x: Tensor) -> Tensor:
        _check_input(x)
        batch = x.shape[0]
        z = self.encoder(x.reshape(batch * 2, 2, 64, 64))
        return self.head(z.reshape(batch, 2 * self.config["channels"][-1]))


def _position_encoding(grid: int, dimension: int) -> Tensor:
    """Fixed 2D sine/cosine positions, ordered by row then column."""
    quarter = dimension // 4
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(quarter, dtype=torch.float32) / quarter
    )
    y, x = torch.meshgrid(
        torch.arange(grid, dtype=torch.float32),
        torch.arange(grid, dtype=torch.float32),
        indexing="ij",
    )
    y_angles = y.reshape(-1, 1) * frequencies
    x_angles = x.reshape(-1, 1) * frequencies
    return torch.cat(
        [y_angles.sin(), y_angles.cos(), x_angles.sin(), x_angles.cos()], dim=-1
    )


class SpatialTokenBlock(nn.Module):
    """Pre-normalized FFN; optional pre-normalized spatial self-attention."""

    def __init__(self, config: dict[str, Any], attention: bool) -> None:
        super().__init__()
        dimension = config["token_dim"]
        dropout = config["dropout"]
        self.attention_enabled = attention
        if attention:
            self.attention_norm = nn.LayerNorm(dimension)
            self.attention = nn.MultiheadAttention(
                embed_dim=dimension,
                num_heads=config["attention_heads"],
                dropout=dropout,
                batch_first=True,
            )
            self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, config["ffn_dim"]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(config["ffn_dim"], dimension),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        if self.attention_enabled:
            normalized = self.attention_norm(tokens)
            attended, _ = self.attention(
                normalized, normalized, normalized, need_weights=False
            )
            tokens = tokens + self.attention_dropout(attended)
        return tokens + self.ffn(self.ffn_norm(tokens))


class SpatialDualScaleCNN(nn.Module):
    """Create spatial tokens before pooling them away; retain order in the head.

    No global average pooling is applied before token creation. Both scales use
    the same CNN and token projection. Position codes and learned scale codes
    are added to the real spatial feature grid. The flattened ordered tokens
    retain spatial sensitivity, including when attention is disabled.
    """

    def __init__(self, config: dict[str, Any], attention: bool) -> None:
        super().__init__()
        self.config = config
        dimension = config["token_dim"]
        grid = config["spatial_grid"]
        self.encoder = _encoder(config["channels"])
        # Fixed divisible grids have the same bins as adaptive pooling. Use
        # regular average pooling because adaptive pooling's CUDA backward
        # does not support strict deterministic algorithms for spatial grids.
        factor = 8 // grid
        self.spatial_pool = nn.AvgPool2d(kernel_size=factor, stride=factor)
        self.token_projection: nn.Module = (
            nn.Identity() if config["channels"][-1] == dimension
            else nn.Linear(config["channels"][-1], dimension)
        )
        self.register_buffer(
            "position_encoding", _position_encoding(grid, dimension), persistent=True
        )
        self.scale_embedding = nn.Embedding(2, dimension)
        self.blocks = nn.Sequential(*[
            SpatialTokenBlock(config, attention=attention)
            for _ in range(config["transformer_layers"])
        ])
        self.head = _classification_head(2 * grid * grid * dimension, config)

    def forward(self, x: Tensor) -> Tensor:
        _check_input(x)
        batch = x.shape[0]
        features = self.spatial_pool(self.encoder(x.reshape(batch * 2, 2, 64, 64)))
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.token_projection(tokens)
        tokens = tokens.reshape(batch, 2, -1, self.config["token_dim"])
        tokens = (
            tokens + self.position_encoding[None, None, :, :]
            + self.scale_embedding.weight[None, :, None, :]
        )
        tokens = tokens.reshape(batch, -1, self.config["token_dim"])
        tokens = self.blocks(tokens)
        return self.head(tokens.flatten(1))


def build_model(name: str, config: Mapping[str, Any] | None = None) -> nn.Module:
    """Build one of the three primary neural models or the optional ablation."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; choose one of {MODEL_NAMES}")
    settings = _settings(config)
    if name == "m2":
        return DualScaleM2(settings)
    return SpatialDualScaleCNN(settings, attention=name == "cnn_transformer")


def backbone_spec(name: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return JSON-ready architecture details without changing the random seed.

    These counts allow the report to disclose the extra parameters of attention
    and the larger spatial readout relative to the archived GAP model.
    """
    settings = _settings(config)
    with torch.random.fork_rng(devices=[]):
        model = build_model(name, settings)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    spec: dict[str, Any] = {
        "name": name,
        "input_shape": [2, 2, 64, 64],
        "num_classes": 3,
        "config": settings,
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "parameter_groups": {
            group: sum(parameter.numel() for parameter in module.parameters())
            for group, module in model.named_children()
        },
        "shared_encoder_between_scales": True,
        "convolution_blocks": "Conv3x3 -> GroupNorm(4) -> ReLU -> MaxPool2x2",
        "pooling": "global_average_1x1" if name == "m2" else "spatial_grid_average",
        "spatial_tokens": 0 if name == "m2" else 2 * settings["spatial_grid"] ** 2,
        "position_encoding": None if name == "m2" else "fixed_2d_sine_cosine",
        "scale_encoding": None if name == "m2" else "learned_scale_embedding",
        "attention": name == "cnn_transformer",
        "readout": "concatenated_scale_GAP" if name == "m2" else "flatten_ordered_tokens",
        "m2_reproduction_scope": "architecture_only_with_default_config",
        "comparison_note": (
            "Attention adds parameters; the four-model comparison alone does not "
            "establish a parameter-matched causal effect of attention."
        ),
    }
    return spec


# Paired H0-H3 target-aware heads.
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
    """Initialize paired models from the common backbone builder.

    Seeding immediately before this function gives matching original weights
    for every head/input variant. Expansion does not consume additional CPU
    RNG draws, so dropout starts from the same RNG state across head versions.
    """
    if backbone_name not in BACKBONES:
        raise ValueError(f"Unknown backbone {backbone_name!r}; choose from {BACKBONES}")
    if head_version not in HEAD_VERSIONS:
        raise ValueError(f"Unknown head version {head_version!r}; choose from {HEAD_VERSIONS}")
    original = build_model(backbone_name, model_config)
    return ControlledModel(backbone_name, original, head_version)


def model_spec(
    backbone_name: str,
    model_config: Mapping[str, Any] | None = None,
    head_version: str = "H0",
) -> dict[str, Any]:
    """JSON-ready specifications; input variants do not change parameter counts."""
    original = backbone_spec(backbone_name, model_config)
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
