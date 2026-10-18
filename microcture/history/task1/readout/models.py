"""M2+/H3 with a fixed target gate applied before any token projection.

O0 uses the frozen M2+ readout. O1 changes only feature-map weights; it adds
no parameter or state tensor and leaves positional/scale codes untouched.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from microcture.task1.models import build_model as build_original_model
from microcture.task1_controlled.models import ControlledModel, model_spec as controlled_spec

VARIANTS = ("O0", "O1")
BETA = 0.5


def target_gate(mask: Tensor, grid: int = 4) -> Tensor:
    """Area-average a 64² target mask; mean-one gate, or identity if empty."""
    if mask.ndim != 4 or mask.shape[1:] != (1, 64, 64) or grid != 4:
        raise ValueError("Target gate requires [N,1,64,64] and the frozen 4x4 grid")
    if not torch.isfinite(mask).all() or (mask < 0).any():
        raise ValueError("Target masks must be finite and nonnegative")
    pooled = F.avg_pool2d(mask, kernel_size=16, stride=16)
    mean = pooled.mean(dim=(-2, -1), keepdim=True)
    safe_mean = torch.where(mean > 0, mean, torch.ones_like(mean))
    mixed = (1 - BETA) + BETA * pooled / safe_mean
    return torch.where(mean > 0, mixed, torch.ones_like(mixed))


class TargetReadoutModel(ControlledModel):
    def __init__(self, original, variant: str):
        if variant not in VARIANTS:
            raise ValueError(f"Choose {VARIANTS}")
        super().__init__("m2plus", original, "H3")
        self.readout_variant = variant

    def spatial_readout(self, x: Tensor) -> Tensor:
        if x.ndim != 5 or tuple(x.shape[1:]) != (2, 3, 64, 64) or x.shape[0] < 1:
            raise ValueError("Expected nonempty [N,2,3,64,64] input")
        batch = x.shape[0]
        features = self.encoder(x.reshape(batch * 2, 3, 64, 64))
        features = self.backbone.spatial_pool(features)
        if self.readout_variant == "O1":
            mask = x[:, :, 2:3].reshape(batch * 2, 1, 64, 64)
            features = features * target_gate(mask).to(features.dtype)
        tokens = self.backbone.token_projection(features.flatten(2).transpose(1, 2))
        tokens = tokens.reshape(batch, 2, -1, self.config["token_dim"])
        tokens = (tokens + self.backbone.position_encoding[None, None]
                  + self.backbone.scale_embedding.weight[None, :, None])
        tokens = self.backbone.blocks(tokens.reshape(batch, -1, self.config["token_dim"]))
        return tokens.flatten(1)


def build_model(variant: str, model_config: dict | None = None) -> TargetReadoutModel:
    original = build_original_model("m2plus", model_config)
    if original.config["spatial_grid"] != 4:
        raise ValueError("This control freezes the spatial grid at 4x4")
    return TargetReadoutModel(original, variant)


def model_spec(variant: str, model_config: dict | None = None) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"Choose {VARIANTS}")
    return {**controlled_spec("m2plus", model_config, "H3"), "name": variant,
            "backbone": "m2plus", "experiment_family": "single_factor_target_spatial_readout",
            "readout_variant": variant, "gate_beta": BETA if variant == "O1" else 0.0,
            "gate_parameters": 0, "gate_position": "after CNN spatial_pool; before projection/position/scale/FFN",
            "gate_formula": "(1-beta)+beta*m/mean(m); empty-mask gate=1",
            "parameter_matched_to_O0": True}
