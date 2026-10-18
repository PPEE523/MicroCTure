#!/usr/bin/env python3
"""User-run, label-blinded inspection of retained contact counts and model inputs.

``check`` reads small metadata, annotation/config sources and file statistics;
it does not open a numeric contact band. ``build`` extracts local windows from
read-only memory maps and renders evidence, without training, inference, fitting
or changing any scientific result. Real builds belong to the project user.
"""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

sys.dont_write_bytecode = True
ROOT = _layout_root
sys.path.insert(0, str(ROOT))

import numpy as np

from microcture.task1 import data as original
from microcture.task1_controlled import data as controlled
from microcture.task1_formal import data as formal
from microcture.task1_formal.protocol import (
    _package_signature, _validate_cache_metadata, json_signature, sha256,
    source_signature,
)

PROTOCOL = "task1_blind_morphology_evidence_v1"
DEFAULT_REVIEW = "reports/task1_morphology_review_v1"
DEFAULT_OUTPUT = "reports/task1_morphology_evidence_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def read_json(path: Path) -> dict:
    require(path.stat().st_size <= 2 * 1024 * 1024, f"Not small metadata: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def safe_output(value: str | Path) -> Path:
    supplied = Path(value)
    unresolved = supplied if supplied.is_absolute() else ROOT / supplied
    for component in (unresolved, *unresolved.parents):
        require(not component.is_symlink(), "Evidence output or an ancestor is a symlink")
        if component == ROOT:
            break
    output = unresolved.resolve()
    require(ROOT / "reports" in output.parents, "Evidence output must be below project reports/")
    protected = [ROOT / "data", ROOT / "results", ROOT / "evidence", ROOT / "micro-c数据",
                 ROOT / "reports/task1_controlled_review_v1", ROOT / DEFAULT_REVIEW,
                 ROOT / "reports/task1_formal_visuals_v1"]
    for tree in (ROOT / "results", ROOT / "reports"):
        if tree.exists():
            protected += [marker.parent for marker in tree.rglob("FROZEN.json")]
    for path in protected:
        path = path.resolve()
        require(output != path and path not in output.parents and output not in path.parents,
                f"Evidence output overlaps protected evidence: {path}")
    return output


def read_reveal_key(review_dir: Path, metadata: dict) -> list[dict]:
    path = review_dir / "reveal_key.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(bool(rows) and {"case_id", "structure_id"} <= set(rows[0]),
            "reveal_key.csv must contain case_id and structure_id")
    expected = {row["structure_id"] for row in metadata["structures"]}
    require({row["structure_id"] for row in rows} == expected and len(rows) == len(expected),
            "Reveal key must register every dataset structure exactly once")
    require(len({row["case_id"] for row in rows}) == len(rows), "Duplicate blind case IDs")
    forbidden = set(metadata["label_order"])
    for row in rows:
        case_id = row["case_id"]
        require(case_id and all(char.isalnum() or char in "-_" for char in case_id),
                "Unsafe blind case_id")
        require(not any(label.lower() in case_id.lower() for label in forbidden),
                "Blind case_id contains a class name")
        require(case_id != row["structure_id"], "Blind case_id exposes a structure ID")
    return [{"case_id": row["case_id"], "structure_id": row["structure_id"]}
            for row in sorted(rows, key=lambda row: row["case_id"])]


def selected_ids(path: Path, registered: list[dict]) -> set[str]:
    """A CSV may specify structure_id or case_id; plain files contain one ID/line."""
    require(path.is_file(), f"Missing registered case list: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        content = stream.read()
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    require(bool(lines), "Case list is empty")
    if "structure_id" in lines[0].split(",") or "case_id" in lines[0].split(","):
        rows = list(csv.DictReader(content.splitlines()))
        key = "structure_id" if "structure_id" in rows[0] else "case_id"
        requested = [row[key].strip() for row in rows]
    else:
        requested = lines
    reveal = {row["case_id"]: row["structure_id"] for row in registered}
    known = {row["structure_id"] for row in registered}
    result = [reveal.get(value, value) for value in requested]
    require(bool(result) and len(result) == len(set(result)), "Empty or duplicate selected cases")
    require(set(result) <= known, "Case list contains an unregistered structure/case ID")
    return set(result)


def metadata_check(run_dir: Path, review_dir: Path, case_list: Path) -> dict:
    """Validate provenance and fits without opening a count band or reading pixels."""
    manifest_path = run_dir / "manifest.json"
    manifest = read_json(manifest_path)
    require(json_signature({k: v for k, v in manifest.items() if k != "experiment_signature"})
            == manifest["experiment_signature"], "Formal manifest signature changed")
    require(source_signature(ROOT) == manifest["source_signature"], "Trained formal source changed")
    for package, entry in (("task1", "run_task1_comparison.py"),
                           ("task1_controlled", "run_task1_controlled.py")):
        require(_package_signature(ROOT, package, entry) == manifest["dependency_source_signature"][package],
                f"Trained dependency source changed: {package}")
    cache_dir = project_path(manifest["config"]["cache_dir"])
    metadata, _ = _validate_cache_metadata(ROOT, cache_dir, manifest["dataset_signature"])
    registered = read_reveal_key(review_dir, metadata)
    selection = selected_ids(case_list, registered)
    folds, outer_assignment = {}, {}
    skeleton = {"signature": metadata["signature"], "config": metadata["config"],
                "structure_id": np.array([row["structure_id"] for row in metadata["structures"]]),
                "group_id": np.array(metadata["group_id"]), "metadata": metadata,
                "n_bins": metadata["n_bins"]}
    width = max(metadata["config"]["window_bins"])
    skeleton["window_footprints"] = (np.array(metadata["center_bin"])[:, None]
                                     + np.arange(-width // 2, width // 2)) % metadata["n_bins"]
    for number in manifest["config"]["folds"]:
        record = read_json(run_dir / "folds" / f"fold_{number}.json")
        require(json_signature({k: v for k, v in record.items() if k != "signature"}) == record["signature"],
                f"Formal fold {number} signature changed")
        frozen, preprocessing = record["frozen_fold"], record["preprocessing"]
        require(json_signature(frozen) == manifest["fold_signatures"][str(number)],
                "Frozen original fold changed")
        require({key: frozen[key] for key in ("train_ids", "validation_ids", "test_ids")}
                == manifest["fold_ids"][str(number)], "Frozen partition changed")
        formal._check_preprocessing(skeleton, frozen, preprocessing)
        for sid in frozen["test_ids"]:
            require(sid not in outer_assignment, "Structure occurs in multiple outer folds")
            outer_assignment[sid] = number
        folds[number] = record
    require(set(outer_assignment) == {row["structure_id"] for row in registered},
            "Outer folds do not register every unique structure")
    return {"manifest": manifest, "metadata": metadata, "cache_dir": cache_dir,
            "registered": registered, "selected": selection, "folds": folds,
            "outer_assignment": outer_assignment,
            "small_input_hashes": {str(manifest_path): sha256(manifest_path),
                str(cache_dir / "metadata.json"): sha256(cache_dir / "metadata.json"),
                str(review_dir / "reveal_key.csv"): sha256(review_dir / "reveal_key.csv"),
                str(case_list): sha256(case_list)}}


def interval_geometry(start_bp: float, end_bp: float, bins: np.ndarray,
                      chrom_length: int, factor: int = 1) -> dict:
    """Exact half-open coverage and visible boundaries, including origin wrapping."""
    starts = bins.astype(float) * 100
    ends = np.minimum(starts + 100, chrom_length)
    overlaps = np.maximum(0, np.minimum(ends, end_bp) - np.maximum(starts, start_bp))
    axis = overlaps / 100
    segments = []
    for index in np.flatnonzero(overlaps > 0):
        left = index + (max(starts[index], start_bp) - starts[index]) / 100
        right = index + (min(ends[index], end_bp) - starts[index]) / 100
        if segments and abs(segments[-1][1] - left) < 1e-9:
            segments[-1][1] = float(right)
        else:
            segments.append([float(left), float(right)])
    visible_bp = float(overlaps.sum())
    return {"axis": axis, "native_mask": np.outer(axis, axis),
            "visible_segments": [[left / factor, right / factor] for left, right in segments],
            "interval_coverage_fraction": visible_bp / (end_bp - start_bp),
            "visible_target_bp": visible_bp,
            "cropped_target_bp": max(0.0, end_bp - start_bp - visible_bp)}


def extract_case(dataset: dict, index: int, preprocessing: dict) -> tuple[list[dict], dict]:
    """Reconstruct native stages and bind exact pooled tensors to the existing extractor.

    Local coverage is computed on the 256-bin context *before* taking the
    64-bin centre, exactly as in the training source. No summary statistic is fit.
    """
    config, n_bins = dataset["config"], dataset["n_bins"]
    require(config["resolution_bp"] == 100 and config["window_bins"] == [64, 256]
            and config["output_size"] == 64, "Evidence requires the trained 100bp/two-scale protocol")
    row = dataset["metadata"]["structures"][index]
    start_bp, end_bp = float(row["start"]), float(row["end"])
    bins = dataset["window_footprints"][index]
    difference = (bins[None, :] - bins[:, None]) % n_bins
    distance = np.minimum(difference, n_bins - difference)
    band_start = np.where(difference <= n_bins // 2, bins[:, None], bins[None, :])
    occupancy = dataset["bin_occupancy"][bins]
    exposure = occupancy[:, None] * occupancy[None, :]
    supported = (distance >= 2) & (exposure > 0)
    signal_fit = preprocessing["signal_preprocessing"]
    # This read-only transformation uses already frozen train-only parameters.
    exact = original._apply_scalers(original._extract_inputs(dataset, np.array([index]),
                                     signal_fit["expected"]), signal_fit["scalers"])
    exact_masks = controlled.target_masks_for_interval(
        start_bp, end_bp, int(dataset["center_bin"][index]), n_bins=n_bins,
        chrom_length=int(dataset["metadata"]["chrom_length"]))
    records = []
    for replicate_index, band in enumerate(dataset["bands"]):
        counts = np.asarray(band[band_start, distance])
        expected = np.asarray(signal_fit["expected"][replicate_index]["expected"])
        good = np.where(supported, counts, 0).sum(axis=0) > 0
        valid = exposure * (good[:, None] & good[None, :] & supported)
        denominator = expected[distance] * exposure
        oe = np.divide(counts, denominator, out=np.zeros_like(counts),
                       where=(valid > 0) & (denominator > 0))
        log_oe = np.log1p(oe)
        for scale, width in enumerate(config["window_bins"]):
            offset = (256 - width) // 2
            window = slice(offset, offset + width)
            factor = width // 64
            weight = valid[window, window]
            shape = (64, factor, 64, factor)
            amount = weight.reshape(shape).sum(axis=(1, 3))
            total = (log_oe[window, window] * weight).reshape(shape).sum(axis=(1, 3))
            pooled = np.divide(total, amount, out=np.zeros_like(total), where=amount > 0)
            scaler = signal_fit["scalers"][scale]
            standard = np.where(amount > 0, (pooled - scaler["mean"]) / scaler["std"], 0)
            require(np.allclose(standard.astype(np.float32), exact["x"][replicate_index, scale, 0],
                                rtol=2e-6, atol=2e-6), "Evidence differs from the exact model signal input")
            geometry = interval_geometry(start_bp, end_bp, bins[window],
                                         int(dataset["metadata"]["chrom_length"]), factor)
            native_mask = geometry["native_mask"]
            mask = native_mask.reshape(shape).mean(axis=(1, 3))
            require(np.allclose(mask, exact_masks[scale], rtol=0, atol=1e-7),
                    "Evidence differs from the exact target mask input")
            target_area = float(native_mask.sum())
            target_valid_fraction = (float((native_mask * weight).sum()) / target_area
                                     if target_area > 0 else None)
            records.append({"replicate": int(dataset["replicate"][replicate_index]),
                "width": width, "factor": factor, "raw_counts": counts[window, window],
                "native_log1p_oe": log_oe[window, window], "native_valid": weight,
                "pooled_log1p_oe": pooled,
                "model_signal": exact["x"][replicate_index, scale, 0],
                "model_valid": exact["x"][replicate_index, scale, 1],
                "native_mask": native_mask, "model_mask": exact_masks[scale],
                "native_segments": [[a * factor, b * factor] for a, b in geometry["visible_segments"]],
                "model_segments": geometry["visible_segments"],
                "interval_coverage_fraction": geometry["interval_coverage_fraction"],
                "cropped_target_bp": geometry["cropped_target_bp"],
                "visible_target_bp": geometry["visible_target_bp"],
                "target_contact_valid_fraction": target_valid_fraction,
                "window_contact_valid_fraction": float(weight.mean()),
                "raw_count_sum_displayed_matrix": float(counts[window, window].sum()),
                "window_wraps_origin": bool(np.any(np.diff(bins[window]) < 0)),
                "fully_missing_genomic_bins": int((~good[window]).sum()),
                "terminal_partial_bins": int((occupancy[window] == 0).sum())})
    scaler = preprocessing["length_scaler"]
    length_bp = end_bp - start_bp
    return records, {"target_length_bp": length_bp,
                     "target_start_relative_anchor_bp": start_bp - int(dataset["center_bin"][index]) * 100,
                     "target_end_relative_anchor_bp": end_bp - int(dataset["center_bin"][index]) * 100,
                     "anchor_definition": "floor(annotation center bp / 100) * 100; relative zero is this coarse bin start",
                     "length_input_standardized": float((np.log1p(length_bp / 100)
                                                         - scaler["mean"]) / scaler["std"]),
                     "coordinate_status": row["coordinate_status"],
                     "exact_model_signal_and_mask_verified": True}


def display_limits(*, raw_count_cap: float = 100, oe_cap: float = 20,
                   z_cap: float = 3) -> dict:
    """Fixed display caps across every case, replicate and scale; never fit them."""
    raw_count_cap, oe_cap, z_cap = map(float, (raw_count_cap, oe_cap, z_cap))
    require(all(np.isfinite(value) and value > 0 for value in (raw_count_cap, oe_cap, z_cap)),
            "Display caps must be finite and positive")
    return {"raw_count_cap": raw_count_cap, "oe_cap": oe_cap,
            "raw_log1p_count_vmax": float(np.log1p(raw_count_cap)),
            "log1p_oe_vmax": float(np.log1p(oe_cap)),
            "standardized_abs_vmax": z_cap,
            "display_rule": "fixed global display caps across all cases, WT/scales; display only, no biological threshold or input value changed"}


def saturation_fractions(record: dict, limits: dict) -> dict:
    """Disclose clipped display fractions, conditional on valid pixels for OE/z."""
    def fraction(array, threshold, valid=None):
        values = np.asarray(array).ravel() if valid is None else np.asarray(array)[valid > 0]
        return float(np.mean(values >= threshold)) if values.size else None
    return {"raw_log1p_count": fraction(np.log1p(record["raw_counts"]), limits["raw_log1p_count_vmax"]),
            "native_log1p_oe": fraction(record["native_log1p_oe"], limits["log1p_oe_vmax"], record["native_valid"]),
            "pooled_log1p_oe": fraction(record["pooled_log1p_oe"], limits["log1p_oe_vmax"], record["model_valid"]),
            "absolute_model_z": fraction(abs(record["model_signal"]), limits["standardized_abs_vmax"], record["model_valid"])}


def _draw_boundaries(ax, segments: list[list[float]], size: int) -> None:
    for first, last in segments:
        for boundary in (first, last):
            if 0 < boundary < size:
                ax.axvline(boundary, color="#35d0d0", linewidth=0.7, linestyle="--", alpha=0.9)
                ax.axhline(boundary, color="#35d0d0", linewidth=0.7, linestyle="--", alpha=0.9)


def _imshow(ax, array: np.ndarray, *, row: dict, native: bool, title: str,
            vmin: float, vmax: float, cmap: str, mask: np.ndarray | None = None):
    import matplotlib.pyplot as plt
    size = array.shape[0]
    values = np.ma.masked_where(mask == 0, array) if mask is not None else array
    color_map = plt.get_cmap(cmap).with_extremes(bad="#dedede")
    im = ax.imshow(values, origin="lower", interpolation="nearest", extent=(0, size, 0, size),
                   vmin=vmin, vmax=vmax, cmap=color_map)
    _draw_boundaries(ax, row["native_segments"] if native else row["model_segments"], size)
    ax.set_title(title, fontsize=8.5)
    ticks = [0, size / 2, size]
    half_kb = row["width"] * 0.1 / 2
    ax.set_xticks(ticks, [f"{-half_kb:g}", "0", f"{half_kb:g}"], fontsize=7)
    ax.set_yticks(ticks, [f"{-half_kb:g}", "0", f"{half_kb:g}"], fontsize=7)
    return im


def render_case(case_id: str, records: list[dict], facts: dict, destination: Path,
                *, dpi: int = 150, limits: dict | None = None) -> dict:
    """Save two blinded pages. Original labels/IDs/predictions never reach plots."""
    import matplotlib.pyplot as plt
    limits = display_limits() if limits is None else limits
    records = sorted(records, key=lambda row: (row["replicate"], row["width"]))
    fig, axes = plt.subplots(4, 5, figsize=(15.8, 12.1), constrained_layout=True)
    for row_index, record in enumerate(records):
        titles = ["100 bp log1p(count), unmasked", "100 bp log1p(OE), before pooling",
                  "64 px log1p(OE), validity-weighted pooling", "64 px exact model signal, train z-score",
                  "64 px exact model validity"]
        arrays = [np.log1p(record["raw_counts"]), record["native_log1p_oe"],
                  record["pooled_log1p_oe"], record["model_signal"], record["model_valid"]]
        ranges = [(0, limits["raw_log1p_count_vmax"]), (0, limits["log1p_oe_vmax"]),
                  (0, limits["log1p_oe_vmax"]),
                  (-limits["standardized_abs_vmax"], limits["standardized_abs_vmax"]), (0, 1)]
        for column, (array, title, (low, high)) in enumerate(zip(arrays, titles, ranges)):
            ax = axes[row_index, column]
            invalid = record["native_valid"] if column == 1 else record["model_valid"] if column in (2, 3) else None
            image = _imshow(ax, array, row=record, native=column < 2, title=title,
                            vmin=low, vmax=high, cmap="RdBu_r" if column == 3 else "magma" if column < 4 else "Greys",
                            mask=invalid)
            fig.colorbar(image, ax=ax, fraction=0.043, pad=0.02).ax.tick_params(labelsize=6)
        coverage = record["interval_coverage_fraction"] * 100
        axes[row_index, 0].set_ylabel(f"WT {record['replicate']} | {record['width'] * .1:g} kb\n"
                                     f"Target visible: {coverage:.1f}%\nAnchor-relative kb", fontsize=8)
    fig.suptitle(f"{case_id} | Independent WT views, two scales, exact preprocessing stages\n"
                 "Retained 100 bp counts were aggregated from a 10 bp source; 10-to-100 bp loss is not evaluated here.",
                 fontsize=11)
    fig.supxlabel("Anchor-relative kb; cyan dashes = supplied interval boundaries. Gray masked pixels are excluded from model signal.\n"
                  "Display caps are fixed across all cases and are not biological thresholds; no label, prediction, or attribution map is shown.", fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"contact_evidence.{suffix}", dpi=dpi, metadata={"Creator": PROTOCOL})
    plt.close(fig)
    fig, axes = plt.subplots(4, 4, figsize=(13.0, 12.1), constrained_layout=True)
    for row_index, record in enumerate(records):
        for column, (array, native, title) in enumerate((
                (record["native_mask"], True, "100 bp supplied target interval mask"),
                (record["model_mask"], False, "64 px exact supplied target mask"),
                (record["native_valid"], True, "100 bp contact validity"),
                (record["model_valid"], False, "64 px exact contact validity"))):
            ax = axes[row_index, column]
            image = _imshow(ax, array, row=record, native=native, title=title,
                            vmin=0, vmax=1, cmap="viridis" if column < 2 else "Greys")
            fig.colorbar(image, ax=ax, fraction=0.043, pad=0.02).ax.tick_params(labelsize=6)
        valid = record["target_contact_valid_fraction"]
        valid_text = f"{valid * 100:.1f}%" if valid is not None else "no visible target"
        axes[row_index, 0].set_ylabel(f"WT {record['replicate']} | {record['width'] * .1:g} kb\n"
                                     f"Target visible: {record['interval_coverage_fraction'] * 100:.1f}%\n"
                                     f"Target contact validity: {valid_text}", fontsize=8)
    fig.suptitle(f"{case_id} | Supplied geometry and missingness, separate from contact morphology\n"
                 f"Target length {facts['target_length_bp'] / 1000:g} kb; frozen inner-train length input {facts['length_input_standardized']:.3f}; "
                 f"coordinates: {facts['coordinate_status']}", fontsize=10)
    fig.supxlabel("Mask and length describe the supplied candidate; they are not evidence of biological morphology.\n"
                  "Windows wrap around the chromosome origin; the supplied interval is non-wrapping [start,end). Distance 0/1 and the terminal partial bin are excluded from model validity.", fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"geometry_and_validity.{suffix}", dpi=dpi, metadata={"Creator": PROTOCOL})
    plt.close(fig)
    measurements = []
    for record in records:
        measurements.append({**{key: record[key] for key in (
            "replicate", "width", "factor", "interval_coverage_fraction", "cropped_target_bp",
            "visible_target_bp", "target_contact_valid_fraction", "window_contact_valid_fraction",
            "raw_count_sum_displayed_matrix", "window_wraps_origin", "fully_missing_genomic_bins",
            "terminal_partial_bins")},
            "display_saturation_fraction": saturation_fractions(record, limits)})
    payload = {"case_id": case_id, **facts, "display_limits": limits, "measurements": measurements,
               "raw_count_definition": "100bp aggregate of source 10bp upper-storage counts, mirrored off-diagonal for this displayed window; not original highest resolution",
               "fit_scope": "frozen inner-training signal/OE/length parameters from this structure's outer fold; no refitting",
               "label_blinding": "No structure ID, true label, prediction or source-sheet information in this directory"}
    write_json(destination / "evidence.json", payload)
    files = [{"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(destination.iterdir())]
    write_json(destination / "ARTIFACTS.json", {"protocol": PROTOCOL, "case_id": case_id,
                                               "status": "complete", "files": files})
    return payload


def verify_case(path: Path, case_id: str) -> dict:
    artifacts = read_json(path / "ARTIFACTS.json")
    require(artifacts.get("case_id") == case_id and artifacts.get("status") == "complete",
            f"Invalid completed evidence: {case_id}")
    expected = {"contact_evidence.png", "contact_evidence.pdf", "geometry_and_validity.png",
                "geometry_and_validity.pdf", "evidence.json"}
    require({item["path"] for item in artifacts["files"]} == expected,
            f"Case file inventory changed: {case_id}")
    require({file.name for file in path.iterdir()} == expected | {"ARTIFACTS.json"},
            f"Unindexed files in blind evidence: {case_id}")
    for item in artifacts["files"]:
        file = path / item["path"]
        require(file.is_file() and file.stat().st_size == item["size_bytes"]
                and sha256(file) == item["sha256"], f"Case evidence changed: {case_id}/{item['path']}")
    return read_json(path / "evidence.json")


def build(check: dict, output: Path, *, resume: bool, dpi: int,
          limits: dict | None = None) -> dict:
    limits = display_limits() if limits is None else limits
    selected = [row for row in check["registered"] if row["structure_id"] in check["selected"]]
    protocol = {"protocol": PROTOCOL, "formal_experiment_signature": check["manifest"]["experiment_signature"],
                "dataset_signature": check["metadata"]["signature"], "dpi": dpi,
                "global_display_limits": limits,
                "builder_sha256": sha256(Path(__file__).resolve()),
                "reveal_key_sha256": check["small_input_hashes"][next(key for key in check["small_input_hashes"]
                                                                     if key.endswith("reveal_key.csv"))],
                "all_registered_case_ids": [row["case_id"] for row in check["registered"]],
                "fits_refitted": False, "inference_run": False}
    signature = json_signature(protocol)
    if output.exists():
        require(resume, "Output exists; use --resume to verify existing panels or choose a new output")
        existing = read_json(output / "manifest.json")
        require(existing.get("evidence_signature") == signature
                and {k: v for k, v in existing.items() if k != "evidence_signature"} == protocol,
                "Evidence protocol, source, DPI or reveal key changed; use a new output")
    else:
        output.mkdir(parents=True)
        (output / "blind").mkdir()
        write_json(output / "manifest.json", {**protocol, "evidence_signature": signature})
        (output / "README.md").write_text(
            "# Blinded morphology evidence\n\n"
            "Review only `blind/` before recording morphology decisions. The label/ID reveal key belongs to the separate review directory. "
            "These are two unaveraged WT views under the frozen fold-specific preprocessing, not model attributions. "
            "Geometry and length are supplied candidate information, not independent morphology evidence.\n\n"
            "Contact counts are retained 100 bp aggregates of original 10 bp counts. The large window is compressed to 64 pixels after log1p(OE), "
            "using valid-pixel weighted pooling. Display caps are fixed across all cases and do not change inputs. Target coverage, validity and display saturation fractions are numerical in each case's evidence.json. "
            "The total raw-count sum displayed mirrors off-diagonal pairs, so it is not the upper-storage audit sum.\n\n"
            "No statistical fit, model inference, training or modification of original scientific artifacts occurs. "
            "Previously inspected folds remain development evidence, not a new blind test.\n", encoding="utf-8")
    dataset = original.load_dataset(check["cache_dir"])
    lookup = {str(sid): index for index, sid in enumerate(dataset["structure_id"])}
    for sequence, item in enumerate(selected, 1):
        start = time.monotonic()
        case_id, sid = item["case_id"], item["structure_id"]
        destination = output / "blind" / case_id
        if destination.exists():
            verify_case(destination, case_id)
            print(f"[{sequence}/{len(selected)}] {case_id} verified existing", flush=True)
            continue
        preprocessing = check["folds"][check["outer_assignment"][sid]]["preprocessing"]
        records, facts = extract_case(dataset, lookup[sid], preprocessing)
        with tempfile.TemporaryDirectory(prefix="microcture-morphology-case-", dir="/tmp") as temporary:
            staged = Path(temporary) / case_id
            staged.mkdir()
            render_case(case_id, records, facts, staged, dpi=dpi, limits=limits)
            require(not destination.exists(), "Concurrent evidence build reached the same case")
            shutil.move(str(staged), str(destination))
        print(f"[{sequence}/{len(selected)}] {case_id} complete ({time.monotonic() - start:.1f}s)", flush=True)
    completed = [case_id for case_id in protocol["all_registered_case_ids"]
                 if (output / "blind" / case_id / "ARTIFACTS.json").is_file()]
    status = {"protocol": PROTOCOL, "evidence_signature": signature,
              "registered_cases": len(check["registered"]), "requested_cases": len(selected),
              "completed_case_ids": completed, "completed_cases": len(completed),
              "status": "complete" if len(completed) == len(check["registered"]) else "partial",
              "all_cases_remain_registered": True, "labels_or_predictions_in_blind_panels": False}
    write_json(output / "status.json", status)
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "build"))
    parser.add_argument("--run-dir", default="results/task1_formal_v1")
    parser.add_argument("--review-dir", default=DEFAULT_REVIEW)
    parser.add_argument("--case-list", help="CSV structure_id/case_id column or one ID per line; default review-dir/cases.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true", help="Verify and reuse existing completed evidence without overwriting it")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--raw-count-cap", type=float, default=100,
                        help="Fixed across all cases; only limits displayed log1p(count), not input counts")
    parser.add_argument("--oe-cap", type=float, default=20,
                        help="Fixed across all cases; only limits displayed log1p(OE), not input OE")
    parser.add_argument("--z-cap", type=float, default=3,
                        help="Fixed symmetric exact model signal display range, not a biological threshold")
    args = parser.parse_args(argv)
    try:
        require(80 <= args.dpi <= 300, "Choose a DPI between 80 and 300")
        limits = display_limits(raw_count_cap=args.raw_count_cap, oe_cap=args.oe_cap, z_cap=args.z_cap)
        review_dir = project_path(args.review_dir)
        case_list = project_path(args.case_list) if args.case_list else review_dir / "cases.csv"
        output = safe_output(args.output)
        checked = metadata_check(project_path(args.run_dir), review_dir, case_list)
        if args.command == "check":
            print(json.dumps({"status": "ready", "registered_cases": len(checked["registered"]),
                "requested_cases": len(checked["selected"]), "output": str(output),
                "coordinate_status": checked["metadata"]["coordinate_status"],
                "formal_experiment_signature": checked["manifest"]["experiment_signature"],
                "contact_band_pixels_read": 0, "fitting_or_inference": False,
                "outputs_per_case": "2 blinded pages, each PNG+PDF, numeric evidence.json and integrity ARTIFACTS.json",
                "raw_resolution_note": "retained 100bp aggregate from source 10bp; no 10bp detail claim",
                "global_display_limits": limits,
                "build_run_by": "project user; CPU only, no GPU required"}, ensure_ascii=False, indent=2))
            return 0
        os.environ["MPLBACKEND"] = "Agg"
        with tempfile.TemporaryDirectory(prefix="microcture-morphology-mpl-", dir="/tmp") as temporary:
            os.environ["MPLCONFIGDIR"] = temporary
            status = build(checked, output, resume=args.resume, dpi=args.dpi, limits=limits)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, KeyError, FileNotFoundError, OSError) as error:
        print(f"Morphology evidence refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
