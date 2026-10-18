"""Preregistered inner-only checkpoint selection; epoch zero is excluded.

Selection reads only weighted inner-validation CE. The full forty-epoch fit
still completes, and both its final checkpoint and inner-CE checkpoint are
retained. Outer scores, CHID F1 and training acceptance never choose weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from microcture.task1_controlled.evidence import json_signature


@dataclass
class InnerCESelector:
    """Strict improvement keeps the earliest epoch when CE ties exactly."""

    training_epochs: int = 40
    selected_epoch: int | None = None
    minimum_ce: float | None = None

    def consider(self, epoch: int, weighted_validation_ce: float) -> bool:
        if type(epoch) is not int or not 0 <= epoch <= self.training_epochs:
            raise ValueError("Selection epoch must lie in the registered training budget")
        value = float(weighted_validation_ce)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Weighted inner-validation CE must be finite and nonnegative")
        if epoch == 0:
            return False
        if self.minimum_ce is None or value < self.minimum_ce:
            self.minimum_ce = value
            self.selected_epoch = epoch
            return True
        return False


def choose_inner_ce_epoch(history: list[dict], training_epochs: int = 40) -> int:
    """Validate the complete trajectory and select from epochs 1..40 only."""
    if [row.get("epoch") for row in history] != list(range(training_epochs + 1)):
        raise ValueError("Checkpoint selection requires the complete contiguous registered trajectory")
    selector = InnerCESelector(training_epochs)
    for row in history:
        selector.consider(row["epoch"], row["weighted_validation_ce"])
    if selector.selected_epoch is None:
        raise ValueError("No trained epoch is available for inner-CE selection")
    return selector.selected_epoch


def selection_signature(config: dict) -> str:
    """Bind the choice to its frozen rule, independently of all outer scores."""
    return json_signature(config["selection"])


def endpoint_selection_records(history: list[dict], config: dict) -> dict[str, dict]:
    epochs = int(config["training"]["epochs"])
    selected = choose_inner_ce_epoch(history, epochs)
    signature = selection_signature(config)
    return {
        policy: {
            "policy": policy,
            "selected_epoch": epoch,
            "selection_signature": signature,
            "metric": "weighted_validation_ce" if policy == "inner_ce" else None,
            "weighted_train_ce": history[epoch]["weighted_train_ce"],
            "weighted_validation_ce": history[epoch]["weighted_validation_ce"],
            "selection_scope": "inner training/validation only; no outer scores or CHID F1",
            "full_training_epochs": epochs,
            "early_stopping": False,
            "refit": False,
        }
        for policy, epoch in (("final40", epochs), ("inner_ce", selected))
    }
