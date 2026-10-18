"""Target geometry for frozen inner-fold diagnostics.

The original task-one signal extraction is reused without fitting new OE
curves, scalers, or splits. Only training and inner-validation structures are
materialised. Target masks describe the supplied interval; they contain no
class, ID, or source-sheet encoding.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np

from microcture.task1 import data as original


INPUT_VARIANTS = ("I0", "I1", "I2", "I3")


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _signature(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def target_masks_for_interval(
    start_bp: float, end_bp: float, center_bin: int, *, n_bins: int,
    chrom_length: int, resolution_bp: int = 100,
    window_bins: tuple[int, int] | list[int] = (64, 256), output_size: int = 64,
) -> np.ndarray:
    """Return ``[2 scales, 64, 64]`` interval-area fractions.

    For each coarse bin, coverage is its base-pair overlap with ``[start,end)``
    divided by the nominal 100 bp. The target matrix is the outer product of
    these fractions. Area pooling averages all pixels, independently of signal
    availability. Thus large-view 4x4 pooling gives exact covered-area fractions.

    The interval must be a non-wrapping half-open interval on the chromosome;
    windows themselves wrap around the origin. A future inference caller must
    supply a candidate interval and its anchor, rather than a class label.
    """
    _require(np.isfinite(start_bp) and np.isfinite(end_bp)
             and 0 <= start_bp < end_bp <= chrom_length, "Invalid target interval")
    _require(type(center_bin) in (int, np.int64, np.int32) and 0 <= center_bin < n_bins,
             "Invalid target center bin")
    _require(resolution_bp == 100 and output_size == 64 and list(window_bins) == [64, 256],
             "Frozen diagnostics require 100-bp bins, 64/256-bin windows, and 64x64 outputs")
    _require(n_bins == (chrom_length + resolution_bp - 1) // resolution_bp,
             "Chromosome length and circular bin count differ")
    masks = []
    for width in window_bins:
        bins = (int(center_bin) + np.arange(-width // 2, width // 2)) % n_bins
        starts = bins * resolution_bp
        ends = np.minimum(starts + resolution_bp, chrom_length)
        overlap_bp = np.maximum(0.0, np.minimum(ends, end_bp) - np.maximum(starts, start_bp))
        axis_fraction = overlap_bp / resolution_bp
        # The mean of the outer-product block equals the outer product of
        # each axis block's mean: this is 2-D area pooling, not mask dilation.
        factor = width // output_size
        pooled_axis = axis_fraction.reshape(output_size, factor).mean(axis=1)
        masks.append(np.outer(pooled_axis, pooled_axis).astype(np.float32))
    return np.stack(masks)


def _annotation_lookup(dataset: dict) -> dict[str, dict]:
    rows = dataset["metadata"]["structures"]
    lookup = {row["structure_id"]: row for row in rows}
    _require(len(lookup) == len(rows), "Duplicate target structure IDs")
    _require(set(dataset["structure_id"]) == set(lookup), "Target annotations and cached structures differ")
    return lookup


def _interval(row: dict, chrom_length: int) -> tuple[float, float]:
    start, end = float(row["start"]), float(row["end"])
    _require(np.isfinite(start) and np.isfinite(end) and 0 <= start < end <= chrom_length,
             "Invalid target annotation interval")
    return start, end


def _fit_length_scaler(dataset: dict, training: np.ndarray, lookup: dict[str, dict]) -> dict:
    # One observation per independent structure, not one per WT replicate.
    lengths = np.array([_interval(lookup[str(dataset["structure_id"][index])],
                                  int(dataset["metadata"]["chrom_length"]))
                        for index in training], dtype=np.float64)
    log_length = np.log1p((lengths[:, 1] - lengths[:, 0]) / 100.0)
    mean, raw_std = float(log_length.mean()), float(log_length.std())
    return {"mean": mean, "std": raw_std if raw_std >= 1e-6 else 1.0, "raw_std": raw_std,
            "constant_training_lengths": raw_std < 1e-6,
            "constant_scale_rule": "use std=1 if training log-length std < 1e-6",
            "fitted_unique_structures": len(training),
            "formula": "log1p((end_bp-start_bp)/100)",
            "fit_scope": "frozen inner-training structures only"}


def _add_target_features(dataset: dict, samples: dict, lookup: dict[str, dict], scaler: dict) -> dict:
    config = dataset["config"]
    n = len(samples["structure_id"])
    _require(samples["x"].shape == (n, 2, 2, 64, 64), "Unexpected frozen signal tensor layout")
    indices = {str(sid): index for index, sid in enumerate(dataset["structure_id"])}
    masks, intervals, lengths = [], [], []
    for sid in samples["structure_id"]:
        row = lookup[str(sid)]
        start, end = _interval(row, int(dataset["metadata"]["chrom_length"]))
        index = indices[str(sid)]
        mask = target_masks_for_interval(start, end, int(dataset["center_bin"][index]),
                                        n_bins=int(dataset["n_bins"]),
                                        chrom_length=int(dataset["metadata"]["chrom_length"]),
                                        resolution_bp=config["resolution_bp"],
                                        window_bins=config["window_bins"], output_size=config["output_size"])
        masks.append(mask)
        intervals.append([start, end])
        lengths.append(end - start)
    masks_array = np.stack(masks)
    length_bp = np.array(lengths, dtype=np.float64)
    normalised_length = ((np.log1p(length_bp / 100.0) - scaler["mean"]) / scaler["std"])
    result = {**samples,
              "x": np.concatenate((samples["x"], masks_array[:, :, None]), axis=2),
              "length": normalised_length[:, None].astype(np.float32),
              "length_bp": length_bp, "target_interval_bp": np.array(intervals, dtype=np.float64)}
    _require(np.isfinite(result["x"]).all() and np.isfinite(result["length"]).all(),
             "Nonfinite controlled model inputs")
    return result


def prepare_controlled_fold(dataset: dict, frozen_fold: dict) -> tuple[dict, dict, dict]:
    """Reuse a frozen fold and fit only the new training-length standardisation.

    Returned ``x`` is ``[N,2 scales,3 channels,64,64]`` with signal, validity,
    target interval fractions; ``length`` is ``[N,1]``. No outer-holdout image is
    extracted and no original fitted statistic or split is recomputed.
    """
    signal_fit = frozen_fold["preprocessing"]
    _require(signal_fit["dataset_signature"] == dataset["signature"], "Frozen signal fit uses a different dataset")
    recorded = signal_fit["signature"]
    _require(recorded == original._json_hash({key: value for key, value in signal_fit.items() if key != "signature"}),
             "Frozen signal preprocessing signature mismatch")
    training = original._indices(dataset, frozen_fold["train_ids"])
    validation = original._indices(dataset, frozen_fold["validation_ids"])
    _require(set(frozen_fold["train_ids"]) == set(signal_fit["train_ids"]),
             "Frozen fold training IDs and fitted signal training IDs differ")
    original._assert_disjoint(dataset, training, validation)
    outer_ids = set(frozen_fold.get("test_ids", []))
    _require(not outer_ids.intersection(frozen_fold["train_ids"])
             and not outer_ids.intersection(frozen_fold["validation_ids"]),
             "Outer-holdout IDs appear in a diagnostic training/validation subset")
    config = dataset["config"]
    _require(config["resolution_bp"] == 100 and config["window_bins"] == [64, 256]
             and config["output_size"] == 64, "Controlled inputs must preserve the frozen signal protocol")
    lookup = _annotation_lookup(dataset)
    length_scaler = _fit_length_scaler(dataset, training, lookup)
    # These helpers extract only the explicit inner-training IDs. In particular
    # original.prepare_fold (which would refit OE/scalers) is never invoked.
    training_signal = original._apply_scalers(
        original._extract_inputs(dataset, training, signal_fit["expected"]), signal_fit["scalers"])
    validation_signal = original.transform_heldout(dataset, frozen_fold["validation_ids"], signal_fit)
    train_samples = _add_target_features(dataset, training_signal, lookup, length_scaler)
    validation_samples = _add_target_features(dataset, validation_signal, lookup, length_scaler)
    preprocessing = {
        "protocol": "frozen_inner_target_geometry_v1", "fold": frozen_fold.get("fold"),
        "dataset_signature": dataset["signature"], "frozen_fold_signature": _signature(frozen_fold),
        "frozen_signal_preprocessing_signature": recorded,
        "signal_preprocessing": copy.deepcopy(signal_fit), "length_scaler": length_scaler,
        "train_ids": dataset["structure_id"][training].tolist(),
        "validation_ids": dataset["structure_id"][validation].tolist(),
        "outer_images_used": False, "signal_statistics_refitted": False, "splits_refitted": False,
        "coordinate_status": dataset["metadata"].get("coordinate_status", ["provisional"]),
        "tensor_layout": "N,scale(6.4kb/25.6kb),channel(signal/valid_fraction/target_interval_fraction),64,64",
        "length_layout": "N,1", "target_mask_formula": "outer_product(bin_interval_overlap_bp/100), then area mean pooling",
        "target_mask_independent_of_validity": True,
        "inference_requirements": "Supply the target interval [start,end) and its anchor; target geometry contains no class/ID/source encoding.",
        "augmentation_rule": "Reverse both spatial axes of all three channels together; keep length unchanged.",
        "input_variants": {"I0": "zero target mask and zero length", "I1": "target mask with zero length",
                           "I2": "zero target mask with length", "I3": "target mask and length"},
    }
    preprocessing["signature"] = _signature(preprocessing)
    return train_samples, validation_samples, preprocessing


def apply_input_variant(samples: dict, name: str) -> dict:
    """Return a copied input with the same shape and preserved signal channels."""
    _require(name in INPUT_VARIANTS, f"Unknown input variant: {name}; choose {INPUT_VARIANTS}")
    n = len(samples["y"])
    _require(samples["x"].shape == (n, 2, 3, 64, 64) and samples["length"].shape == (n, 1),
             "Unexpected controlled input layout")
    result = {**samples, "x": samples["x"].copy(), "length": samples["length"].copy(), "input_variant": name}
    if name in ("I0", "I2"):
        result["x"][:, :, 2] = 0
    if name in ("I0", "I1"):
        result["length"][:] = 0
    return result
