"""Formal target-aware inputs using a frozen inner-training fit.

Constructing outer tensors is deliberately a separate operation. Callers must
finish the fixed checkpoint before invoking :func:`prepare_outer`; neither
validation nor outer data can fit a normalizer or select a checkpoint here.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from microcture.task1 import data as original
from microcture.task1_controlled import data as controlled


CONDITIONS = ("I0", "G0", "I3")


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_frozen_partition(dataset: dict, frozen_fold: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check all three partitions using IDs, spatial groups and window guards."""
    indices = []
    for split in ("train", "validation", "test"):
        ids = frozen_fold.get(f"{split}_ids")
        _require(isinstance(ids, list) and bool(ids), f"Missing frozen {split} IDs")
        indices.append(original._indices(dataset, ids))
    train, validation, test = indices
    for first, second in ((train, validation), (train, test), (validation, test)):
        original._assert_disjoint(dataset, first, second)
    combined = np.concatenate(indices)
    _require(len(combined) == len(dataset["structure_id"])
             and np.array_equal(np.sort(combined), np.arange(len(dataset["structure_id"]))),
             "Frozen train/validation/test do not partition all dataset structures")
    return train, validation, test


def _check_preprocessing(dataset: dict, frozen_fold: dict, preprocessing: dict) -> None:
    _check_frozen_partition(dataset, frozen_fold)
    _require(preprocessing.get("dataset_signature") == dataset["signature"],
             "Formal preprocessing uses a different dataset")
    recorded = preprocessing.get("signature")
    _require(recorded == controlled._signature({key: value for key, value in preprocessing.items()
                                               if key != "signature"}),
             "Formal target preprocessing signature mismatch")
    _require(preprocessing.get("frozen_fold_signature") == controlled._signature(frozen_fold),
             "Formal preprocessing uses a different frozen fold")
    _require(preprocessing.get("fold") == frozen_fold.get("fold"),
             "Formal preprocessing fold index differs")
    _require(set(preprocessing.get("train_ids", [])) == set(frozen_fold["train_ids"])
             and set(preprocessing.get("validation_ids", [])) == set(frozen_fold["validation_ids"]),
             "Formal preprocessing inner partition differs from frozen fold")
    signal = preprocessing.get("signal_preprocessing")
    _require(isinstance(signal, dict) and signal == frozen_fold["preprocessing"],
             "Formal signal preprocessing differs from the frozen inner-training fit")
    _require(preprocessing.get("frozen_signal_preprocessing_signature") == signal.get("signature"),
             "Formal frozen signal signature differs")
    _require(preprocessing.get("signal_statistics_refitted") is False
             and preprocessing.get("splits_refitted") is False
             and preprocessing.get("outer_images_used") is False,
             "Formal preprocessing must retain frozen inner-only fitting")
    scaler = preprocessing.get("length_scaler", {})
    _require(np.isfinite(scaler.get("mean", np.nan))
             and np.isfinite(scaler.get("std", np.nan)) and scaler.get("std", 0) > 0,
             "Invalid fitted target-length scaler")
    _require(scaler.get("fitted_unique_structures") == len(frozen_fold["train_ids"])
             and scaler.get("fit_scope") == "frozen inner-training structures only",
             "Target-length scaler must use unique inner-training structures only")


def prepare_inner(dataset: dict, frozen_fold: dict) -> tuple[dict, dict, dict]:
    """Reuse frozen signal fit and fit length on unique inner-training IDs."""
    _check_frozen_partition(dataset, frozen_fold)
    train, validation, preprocessing = controlled.prepare_controlled_fold(dataset, frozen_fold)
    _check_preprocessing(dataset, frozen_fold, preprocessing)
    return train, validation, preprocessing


def prepare_outer(dataset: dict, frozen_fold: dict, preprocessing: dict) -> dict:
    """Transform the outer fold using existing fits; never fit any statistic.

    ``preprocessing`` is the exact object returned by :func:`prepare_inner`.
    This function is called only after checkpoint selection is fixed by the
    runner; it has no access to fitting or model-selection operations.
    """
    _check_preprocessing(dataset, frozen_fold, preprocessing)
    signal = original.transform_heldout(dataset, frozen_fold["test_ids"],
                                        preprocessing["signal_preprocessing"])
    lookup = controlled._annotation_lookup(dataset)
    result = controlled._add_target_features(dataset, signal, lookup, preprocessing["length_scaler"])
    _require(set(result["structure_id"]) == set(frozen_fold["test_ids"]),
             "Outer transformed IDs differ from frozen test partition")
    return result


def apply_condition(samples: dict, name: str) -> dict:
    """Copy the input and implement same-shape I0/G0/I3 ablations.

    I0 removes both target features. G0 removes both contact and validity
    channels, so its remaining inputs contain only the supplied geometry.
    Labels and metadata are never consumed to construct an ablation.
    """
    _require(name in CONDITIONS, f"Unknown formal condition: {name}; choose {CONDITIONS}")
    n = len(samples["y"])
    _require(samples["x"].shape == (n, 2, 3, 64, 64) and samples["length"].shape == (n, 1),
             "Unexpected formal target-aware input layout")
    result = {**samples, "x": samples["x"].copy(), "length": samples["length"].copy(),
              "input_condition": name, "input_variant": name}
    if name == "I0":
        result["x"][:, :, 2] = 0
        result["length"][:] = 0
    elif name == "G0":
        result["x"][:, :, :2] = 0
    _require(np.isfinite(result["x"]).all() and np.isfinite(result["length"]).all(),
             "Nonfinite formal condition inputs")
    return result
