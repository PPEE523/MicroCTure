"""One-factor ROI alignment on the frozen 100-bp contact source.

R0 is the exact original I3 input. R1 replaces only the first image scale by
the full supplied interval in relative coordinates. Frozen distance expected
values, the second image scale, target length and partitions stay unchanged.
The first R1 signal scaler is fitted on inner-training images only.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from microcture.task1 import data as original
from microcture.task1_controlled import data as controlled
from microcture.task1_formal import data as formal

ALIGNMENTS = ("R0", "R1")
PROTOCOL = "task1_roi_coordinate_alignment_v1"


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def roi_area_weights(start_bp: float, end_bp: float, source_bin_starts: np.ndarray,
                     resolution_bp: int = 100, output_size: int = 64,
                     chrom_length: int | None = None) -> np.ndarray:
    """Exact overlap/output-width weights; rows are relative interval bins.

    Source starts may be in circular-window order. Coordinate intervals remain
    nonwrapping, as in the preserved annotation table. Partial chromosome bins
    are clipped physically rather than extrapolated across its end.
    """
    starts = np.asarray(source_bin_starts, dtype=np.float64)
    _require(starts.ndim == 1 and len(starts) > 0 and np.isfinite(starts).all(),
             "Expected finite one-dimensional source bin coordinates")
    _require(len(np.unique(starts)) == len(starts), "Duplicate source bins")
    _require(np.isfinite(start_bp) and np.isfinite(end_bp) and 0 <= start_bp < end_bp,
             "Invalid nonwrapping target interval")
    _require(isinstance(resolution_bp, int) and resolution_bp > 0 and
             isinstance(output_size, int) and output_size > 0, "Invalid resolution/canvas")
    ends = starts + resolution_bp
    if chrom_length is not None:
        _require(end_bp <= chrom_length and np.all((starts >= 0) & (starts < chrom_length)),
                 "Target/source lies outside chromosome")
        ends = np.minimum(ends, chrom_length)
    edges = np.linspace(start_bp, end_bp, output_size + 1, dtype=np.float64)
    overlaps = np.maximum(0.0, np.minimum(edges[1:, None], ends[None, :])
                          - np.maximum(edges[:-1, None], starts[None, :]))
    return overlaps / ((end_bp - start_bp) / output_size)


def resample_roi(signal: np.ndarray, validity: np.ndarray,
                 source_bin_starts: np.ndarray, start_bp: float, end_bp: float,
                 resolution_bp: int = 100, output_size: int = 64,
                 chrom_length: int | None = None) -> tuple[np.ndarray, dict]:
    """Area-resample source log(OE), weighting exclusively by validity.

    Signal, validity and target geometry have separate meanings. A missing
    source contact contributes zero validity, never a measured zero signal.
    The mask records physical target coverage independent of data availability.
    """
    signal = np.asarray(signal, dtype=np.float64)
    validity = np.asarray(validity, dtype=np.float64)
    n = len(source_bin_starts)
    _require(signal.shape == validity.shape == (n, n), "Source matrices must match bin coordinates")
    _require(np.isfinite(signal).all() and np.isfinite(validity).all(), "Nonfinite source")
    _require(np.all((validity >= 0) & (validity <= 1)), "Validity outside [0,1]")
    _require(np.allclose(signal, signal.T, atol=1e-10, rtol=1e-10) and
             np.allclose(validity, validity.T, atol=1e-10, rtol=1e-10), "Source must be symmetric")
    weights = roi_area_weights(start_bp, end_bp, source_bin_starts, resolution_bp,
                               output_size, chrom_length)
    active = np.any(weights > 0, axis=0)
    _require(bool(active.any()), "Target has no supplied source bins")
    # Excluding zero columns lowers work without changing the area formula.
    a = weights[:, active]
    local_signal = signal[np.ix_(active, active)]
    local_validity = validity[np.ix_(active, active)]
    area = a @ local_validity @ a.T
    total = a @ (local_signal * local_validity) @ a.T
    mean = np.divide(total, area, out=np.zeros_like(total), where=area > 0)
    axis_coverage = a.sum(axis=1)
    mask = np.outer(axis_coverage, axis_coverage)
    _require(np.all(axis_coverage <= 1 + 1e-10) and np.all(area <= 1 + 1e-10),
             "Source bins overlap or area weights are invalid")
    # Exact complete interval coverage is represented by a constant one mask;
    # fractional validity and the diagonal exclusion remain separate channels.
    mask[np.abs(mask - 1) <= 1e-10] = 1
    channels = np.stack((mean, np.clip(area, 0, 1), np.clip(mask, 0, 1))).astype(np.float32)
    count = int(active.sum())
    mode = "upsampling" if count < output_size else "downsampling" if count > output_size else "equal_bin_count"
    facts = {
        "native_bin_count": count,
        "source_resolution_bp": resolution_bp,
        "output_size": output_size,
        "output_bin_width_bp": float((end_bp - start_bp) / output_size),
        "resampling_mode": mode,
        "target_physical_coverage_fraction": float(mask.mean()),
        "target_validity_fraction": float(area.mean()),
        "target_source_resolution_refined": False,
        "signal_formula": "A@(log1p_OE*valid)@A.T/(A@valid@A.T); zero when invalid",
    }
    return channels, facts


def _native_source(dataset: dict, structure_index: int, replicate_index: int,
                   expected_records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct true cached 100-bp source, before the original 4x4 pooling."""
    bins = dataset["window_footprints"][structure_index]
    n_bins = dataset["n_bins"]
    difference = (bins[None, :] - bins[:, None]) % n_bins
    distance = np.minimum(difference, n_bins - difference)
    start = np.where(difference <= n_bins // 2, bins[:, None], bins[None, :])
    matrix = np.asarray(dataset["bands"][replicate_index][start, distance], dtype=np.float64)
    occupancy = dataset["bin_occupancy"][bins]
    exposure = occupancy[:, None] * occupancy[None, :]
    supported = (distance >= 2) & (exposure > 0)
    good = np.where(supported, matrix, 0).sum(axis=0) > 0
    valid = exposure * (good[:, None] & good[None, :] & supported)
    expected = np.asarray(expected_records[replicate_index]["expected"], dtype=np.float64)
    denominator = expected[distance] * exposure
    oe = np.divide(matrix, denominator, out=np.zeros_like(matrix),
                   where=(valid > 0) & (denominator > 0))
    return np.log1p(oe), valid, bins * int(dataset["config"]["resolution_bp"])


def _replace_first_scale(dataset: dict, samples: dict, base_preprocessing: dict) -> dict:
    _require(dataset["config"]["resolution_bp"] == 100 and
             dataset["config"]["output_size"] == 64 and
             dataset["config"]["window_bins"] == [64, 256], "ROI control requires the frozen 100-bp protocol")
    original_x = samples["x"]
    _require(original_x.shape == (len(samples["y"]), 2, 3, 64, 64), "Expected frozen I3 input layout")
    out = {**samples, "x": original_x.copy()}
    lookup = {str(sid): i for i, sid in enumerate(dataset["structure_id"])}
    replicate_lookup = {sample["replicate"]: i for i, sample in enumerate(dataset["metadata"]["samples"])}
    facts = []
    for row, (sid, replicate) in enumerate(zip(samples["structure_id"], samples["replicate"])):
        index = lookup[str(sid)]
        annotation = dataset["metadata"]["structures"][index]
        signal, validity, starts = _native_source(dataset, index, replicate_lookup[int(replicate)],
                                                  base_preprocessing["signal_preprocessing"]["expected"])
        channels, record = resample_roi(signal, validity, starts, int(annotation["start"]),
                                       int(annotation["end"]), chrom_length=dataset["metadata"]["chrom_length"])
        _require(record["target_physical_coverage_fraction"] >= 1 - 1e-8,
                 f"ROI outside frozen source footprint: {sid}")
        _require(np.any(channels[1] > 0), f"Completely missing ROI: {sid}, WT{replicate}")
        out["x"][row, 0] = channels
        facts.append({"structure_id": str(sid), "replicate": int(replicate), **record})
    out["roi_resampling"] = facts
    _require(np.array_equal(out["x"][:, 1], original_x[:, 1]), "Second scale changed during ROI alignment")
    for field in ("length", "y", "structure_id", "replicate", "sample_id", "group_id"):
        _require(np.array_equal(out[field], samples[field]), f"Frozen metadata changed: {field}")
    return out


def _standardize_first(samples: dict, scaler: dict) -> dict:
    values, valid = samples["x"][:, 0, 0], samples["x"][:, 0, 1] > 0
    samples["x"][:, 0, 0] = np.where(valid, (values - scaler["mean"]) / scaler["std"], 0)
    _require(np.isfinite(samples["x"]).all(), "Nonfinite ROI-aligned input")
    return samples


def _preprocessing(base: dict, alignment: str, scaler: dict | None) -> dict:
    value = {
        "protocol": PROTOCOL, "alignment": alignment, "base_preprocessing": base,
        "base_preprocessing_signature": base["signature"],
        "roi_first_signal_scaler": scaler,
        "roi_source": "unchanged frozen 100-bp count band; native 256x256 log1p(OE) before pooling",
        "distance_expected_refitted": False, "length_scaler_refitted": False,
        "second_scale_byte_identical": True, "roi_first_scaler_fit_scope":
            "frozen inner-training replicate images only" if alignment == "R1" else "unchanged original I3 fit",
        "dataset_signature": base["dataset_signature"], "frozen_fold_signature": base["frozen_fold_signature"],
    }
    value["signature"] = controlled._signature(value)
    return value


def prepare_roi_inner(dataset: dict, frozen_fold: dict, alignment: str) -> tuple[dict, dict, dict]:
    """Reuse the frozen fold; R1 fits only its replacement first-scale scaler."""
    _require(alignment in ALIGNMENTS, f"Choose alignment from {ALIGNMENTS}")
    train, validation, base = formal.prepare_inner(dataset, frozen_fold)
    if alignment == "R0":
        return train, validation, _preprocessing(base, alignment, None)
    train = _replace_first_scale(dataset, train, base)
    validation = _replace_first_scale(dataset, validation, base)
    valid = train["x"][:, 0, 1] > 0
    values = train["x"][:, 0, 0][valid].astype(np.float64)
    _require(len(values) > 0, "No valid ROI training pixels")
    scaler = {"mean": float(values.mean()), "std": max(float(values.std()), 1e-6),
              "fitted_valid_pooled_pixels": int(len(values)),
              "train_ids": base["train_ids"], "fit_scope": "frozen inner-training replicate images only"}
    return _standardize_first(train, scaler), _standardize_first(validation, scaler), _preprocessing(base, alignment, scaler)


def prepare_roi_outer(dataset: dict, frozen_fold: dict, preprocessing: dict, alignment: str) -> dict:
    """Transform an outer partition with existing fits, without fitting."""
    _require(alignment in ALIGNMENTS and preprocessing["alignment"] == alignment, "ROI alignment differs from fitted preprocessing")
    _require(preprocessing["signature"] == controlled._signature({k: v for k, v in preprocessing.items() if k != "signature"}),
             "ROI preprocessing signature changed")
    _require(preprocessing["protocol"] == PROTOCOL, "Unknown ROI preprocessing protocol")
    base = preprocessing["base_preprocessing"]
    _require(base["signature"] == preprocessing["base_preprocessing_signature"], "Frozen base preprocessing changed")
    samples = formal.prepare_outer(dataset, frozen_fold, base)
    if alignment == "R0":
        _require(preprocessing["roi_first_signal_scaler"] is None, "R0 cannot refit its scaler")
        return samples
    scaler = preprocessing["roi_first_signal_scaler"]
    _require(scaler["fit_scope"] == "frozen inner-training replicate images only" and
             scaler["train_ids"] == base["train_ids"] and np.isfinite(scaler["mean"]) and
             np.isfinite(scaler["std"]) and scaler["std"] > 0, "Invalid inner-training ROI scaler")
    return _standardize_first(_replace_first_scale(dataset, samples, base), scaler)
