"""Small neural models for the restarted Task 1 comparison.

All models take ``[batch, 2 scales, 2 channels, 64, 64]`` tensors. Channel 0 is
the normalized contact signal; channel 1 is the valid-pixel fraction. The data
pipeline owns all normalization, split decisions, and augmentation.

``m2`` reproduces the *architecture* of the archived dual-scale CNN, with its
default configuration, but does not claim to reproduce its numerical results:
the restarted experiment uses a new dataset protocol and training runs.
``m2plus`` preserves a spatial grid and applies independent token FFNs.
``cnn_transformer`` adds attention between those same spatial tokens. Their
parameter counts are reported explicitly; they are not perfectly matched, so
this comparison alone is not a strict causal proof of attention's benefit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


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


def model_spec(name: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
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
