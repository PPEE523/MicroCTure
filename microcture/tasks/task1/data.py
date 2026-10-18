"""Shared Task 1 data: circular counts, spatial folds and target-aware inputs.

All normalization is fitted on inner training only; outer tensors are constructed
separately after the fixed endpoint. Experiment-specific input removals live in
ablation.py. Historical source modules are not imported."""
from __future__ import annotations

from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any
import warnings

import h5py
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from microcture.layout import PROJECT_ROOT, resolve

from .training import reject_retained_overlap


# Circular count cache and spatial folds.
LABEL_ORDER = ["OPCID", "CHIN", "CHID"]
FORMAT_VERSION = 2
DEFAULTS = {
    "data_config": "configs/data.json", "annotations": "data/processed/structures.csv",
    "resolution_bp": 100, "window_bins": [64, 256], "output_size": 64,
    "guard_bins": 10, "folds": 5, "split_seed": 20261010,
    "split_search_trials": 2000, "min_class_per_fold": 2,
    "inner_validation_fraction": 0.2, "chunk_size": 2_000_000,
    "normalization": "oe", "train_augmentation": "reverse",
    "allow_provisional_coordinates": True,
}


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _config(config: dict) -> dict:
    result = {**DEFAULTS, **config}
    widths = result["window_bins"]
    _require(result["resolution_bp"] == 100, "This protocol supports 100-bp coarse bins only")
    _require(len(widths) == 2 and all(isinstance(v, int) and v > 0 and v % 2 == 0 for v in widths),
             "window_bins must contain two positive even widths")
    _require(widths == sorted(widths) and len(set(widths)) == 2, "Order scales from small to large")
    _require(all(v % result["output_size"] == 0 for v in widths), "Each window must divide output_size")
    _require(result["normalization"] == "oe", "The comparison protocol uses train-only OE normalization")
    _require(result["train_augmentation"] in ("reverse", "none"), "Unknown train augmentation")
    _require(result["guard_bins"] >= 0 and result["folds"] >= 2, "Invalid guard or fold count")
    _require(result["split_search_trials"] > 0 and result["chunk_size"] > 0, "Search/chunk size must be positive")
    _require(0 < result["inner_validation_fraction"] < 1, "Invalid validation fraction")
    _require(result.get("label_order", LABEL_ORDER) == LABEL_ORDER, f"label_order must be {LABEL_ORDER}")
    return result


def _group_windows(centers: np.ndarray, n_bins: int, width: int, guard: int) -> tuple[np.ndarray, np.ndarray]:
    """Union overlapping/guard-adjacent windows, including the circular origin."""
    parent = np.arange(len(centers))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for i in range(len(centers)):
        for j in range(i):
            gap = abs(int(centers[i]) - int(centers[j]))
            if min(gap, n_bins - gap) < width + guard:
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(len(centers))]
    mapping = {value: i for i, value in enumerate(sorted(set(roots)))}
    groups = np.array([mapping[value] for value in roots], dtype=np.int64)
    footprints = (centers[:, None] + np.arange(-width // 2, width // 2)) % n_bins
    return groups, footprints.astype(np.int64)


def _occupancy(dataset: dict, indices: np.ndarray) -> np.ndarray:
    result = np.zeros(dataset["n_bins"], dtype=bool)
    if len(indices):
        result[dataset["window_footprints"][indices].ravel()] = True
    return result


def _assert_disjoint(dataset: dict, first: np.ndarray, second: np.ndarray) -> None:
    _require(len(first) and len(second), "Empty training or held-out split")
    _require(not np.intersect1d(first, second).size, "Structures occur in both splits")
    _require(not np.intersect1d(dataset["group_id"][first], dataset["group_id"][second]).size,
             "Spatial groups occur in both splits")
    first_bins, second_bins = _occupancy(dataset, first), _occupancy(dataset, second)
    for shift in range(-int(dataset["config"]["guard_bins"]), int(dataset["config"]["guard_bins"]) + 1):
        _require(not np.any(first_bins & np.roll(second_bins, shift)),
                 f"Cross-split window/guard overlap at coarse-bin shift {shift}")


def _group_distribution(y: np.ndarray, groups: np.ndarray) -> list[dict]:
    return [{"group_id": int(group), "counts": dict(zip(LABEL_ORDER,
             np.bincount(y[groups == group], minlength=3).astype(int).tolist()))}
            for group in np.unique(groups)]


def _make_folds(dataset: dict, config: dict) -> list[dict]:
    y, groups = dataset["y"], dataset["group_id"]
    n_folds, minimum = int(config["folds"]), int(config["min_class_per_fold"])
    _require(len(np.unique(groups)) >= n_folds, "Too few spatial groups for outer cross-validation")
    best, best_score, selected_seed = None, float("inf"), None
    target = np.bincount(y, minlength=3) / n_folds
    for trial in range(int(config["split_search_trials"])):
        seed = int(config["split_seed"]) + trial
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        candidate = list(splitter.split(np.zeros(len(y)), y, groups))
        counts = np.array([np.bincount(y[test], minlength=3) for _, test in candidate])
        if np.any(counts < minimum):
            continue
        score = float(np.square((counts - target) / np.maximum(target, 1)).mean())
        score += float(np.var(counts.sum(axis=1) / (len(y) / n_folds))) * 0.1
        if score < best_score:
            best, best_score, selected_seed = candidate, score, seed
    _require(best is not None,
             "No feasible grouped outer split with at least " + str(minimum) +
             " examples/class/fold. Do not fall back to overlapping windows. Group counts: " +
             json.dumps(_group_distribution(y, groups), ensure_ascii=False))
    folds = []
    for index, (train, test) in enumerate(best):
        _assert_disjoint(dataset, train, test)
        folds.append({"fold": index, "train_ids": dataset["structure_id"][train].tolist(),
                      "test_ids": dataset["structure_id"][test].tolist(),
                      "train_counts": dict(zip(LABEL_ORDER, np.bincount(y[train], minlength=3).tolist())),
                      "test_counts": dict(zip(LABEL_ORDER, np.bincount(y[test], minlength=3).tolist())),
                      "selected_split_seed": selected_seed, "balance_score": best_score})
    return folds


def _raw_band(path: Path, destination: Path, data_config: dict, config: dict) -> dict:
    """Store each upper coarse pixel once, using the shorter circular arc.

    Off-diagonal entries are mirrored only when extracting windows. Coarse
    distance 0 retains the original upper-storage sum; coarse distances 0 and 1
    never enter OE or model input, so intra-coarse-bin mirroring is immaterial.
    """
    resolution = int(config["resolution_bp"])
    length, raw_resolution = int(data_config["chrom_length"]), int(data_config["bin_size"])
    _require(resolution % raw_resolution == 0, "Raw bins do not divide coarse resolution")
    factor = resolution // raw_resolution
    n_bins = (length + resolution - 1) // resolution
    max_distance = max(config["window_bins"])
    band = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float64,
                                    shape=(n_bins, max_distance + 1))
    band[:] = 0
    raw_sum, retained_sum, observed_nnz = 0, 0, 0
    with h5py.File(path, "r") as handle:
        attrs = handle.attrs
        storage = attrs.get("storage-mode", "")
        if isinstance(storage, bytes):
            storage = storage.decode()
        _require(storage == "symmetric-upper", f"Unsupported storage mode: {storage}")
        _require(int(attrs["bin-size"]) == raw_resolution, "Raw resolution mismatch")
        names = [value.decode() if isinstance(value, bytes) else str(value)
                 for value in handle["chroms/name"][:]]
        _require(names == [data_config["chrom"]], "The protocol requires the configured single chromosome")
        _require(handle["chroms/length"][:].tolist() == [length], "Chromosome length mismatch")
        raw_n_bins = (length + raw_resolution - 1) // raw_resolution
        starts = handle["bins/start"][:]
        ends = handle["bins/end"][:]
        _require(np.array_equal(starts, np.arange(raw_n_bins) * raw_resolution), "Raw bin starts mismatch")
        _require(np.array_equal(ends, np.minimum(starts + raw_resolution, length)), "Raw bin ends mismatch")
        _require(np.all(handle["bins/chrom"][:] == 0), "Unexpected chromosome bin")
        pixels = handle["pixels"]
        nnz = len(pixels["count"])
        _require(nnz == int(attrs["nnz"]), "Raw nnz metadata mismatch")
        _require(all(len(pixels[key]) == nnz for key in ("bin1_id", "bin2_id")), "Pixel columns differ in length")
        for offset in range(0, nnz, int(config["chunk_size"])):
            end = min(offset + int(config["chunk_size"]), nnz)
            a = pixels["bin1_id"][offset:end].astype(np.int64)
            b = pixels["bin2_id"][offset:end].astype(np.int64)
            values = pixels["count"][offset:end]
            _require(np.all((a >= 0) & (a <= b) & (b < raw_n_bins)), "Invalid upper-storage pixel coordinates")
            _require(np.isfinite(values).all() and np.all(values >= 0) and np.all(values == np.floor(values)),
                     "Counts must be finite nonnegative integers")
            raw_sum += int(values.sum(dtype=np.int64))
            observed_nnz += len(values)
            a //= factor
            b //= factor
            difference = b - a
            distance = np.minimum(difference, n_bins - difference)
            keep = distance <= max_distance
            start = np.where(difference <= n_bins // 2, a, b)
            np.add.at(band, (start[keep], distance[keep]), values[keep])
            retained_sum += int(values[keep].sum(dtype=np.int64))
            print(f"  {path.name}: {end:,}/{nnz:,} stored pixels", flush=True)
        metadata_sum = int(attrs["sum"])
        _require(raw_sum == metadata_sum, f"Full stored count sum mismatch: {raw_sum} vs {metadata_sum}")
    band.flush()
    _require(float(band.sum(dtype=np.float64)) == retained_sum, "Coarse-band stored count sum mismatch")
    del band
    return {"full_stored_count_sum": raw_sum, "metadata_stored_count_sum": metadata_sum,
            "band_stored_count_sum": retained_sum, "stored_nnz": observed_nnz,
            "band_shape": [n_bins, max_distance + 1], "band_dtype": "float64",
            "band_size_bytes": destination.stat().st_size,
            "band_mtime_ns": destination.stat().st_mtime_ns,
            "raw_path": str(path), "raw_size_bytes": path.stat().st_size,
            "raw_mtime_ns": path.stat().st_mtime_ns}


def guard_cache_output(root: Path, output: Path) -> Path:
    """Permit a new cache version without entering code, raw data or saved evidence."""
    root = Path(root).resolve()
    path = resolve(root, output)
    if any(ancestor.is_symlink() for ancestor in (path, *path.parents)):
        raise ValueError("Prepare output cannot pass through symbolic links")
    path = path.resolve()
    cache_root = root / "data/task1/cache"
    if not path.is_relative_to(cache_root) or len(path.relative_to(cache_root).parts) != 1:
        raise ValueError("Prepare output must be data/task1/cache/<new_version>")
    reject_retained_overlap(root, path, kinds=("cache", "result", "raw"))
    if path.exists():
        raise FileExistsError(f"Preserve existing cache; choose a new version: {path}")
    for marker in (root / "data").rglob("FROZEN.json"):
        protected = marker.parent
        if path == protected or protected in path.parents or path in protected.parents:
            raise ValueError(f"Prepare output overlaps frozen evidence: {protected}")
    return path


def prepare_dataset(config: dict, root: Path, output: Path, *, data_root: Path | None = None) -> dict:
    """Build a new cache; refuse every existing destination, including old data."""
    root = Path(root).resolve()
    output = guard_cache_output(root, output)
    raw_root = resolve(root, data_root).resolve() if data_root is not None else root
    config = _config(config)
    _require(not output.exists(), f"Destination already exists; load it or choose a fresh directory: {output}")
    data_path = resolve(root, config["data_config"])
    annotation_path = resolve(root, config["annotations"])
    data_config = json.loads(data_path.read_text(encoding="utf-8"))
    annotation_source = resolve(raw_root, data_config["annotations"]["path"])
    _require(annotation_source.is_file(), f"Missing original annotation workbook: {annotation_source}")
    input_hashes = {str(path.relative_to(root) if path.is_relative_to(root) else path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (data_path, annotation_path)}
    workbook_provenance = {"path": str(annotation_source), "sha256": hashlib.sha256(annotation_source.read_bytes()).hexdigest()}
    with annotation_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected_counts = data_config["annotations"]["expected_counts"]
    _require(dict(Counter(row["type"] for row in rows)) == expected_counts,
             f"Annotation class counts differ from source specification: {Counter(row['type'] for row in rows)}")
    ids = [row["structure_id"] for row in rows]
    _require(len(ids) == len(set(ids)), "Duplicate structure IDs")
    _require(all(row["coordinate_system"] == "0-based-half-open" for row in rows), "Unknown coordinate convention")
    statuses = sorted({row["coordinate_status"] for row in rows})
    _require(config["allow_provisional_coordinates"] or statuses == ["verified"],
             "Coordinates are provisional; set allow_provisional_coordinates explicitly or verify the source")
    length, resolution = int(data_config["chrom_length"]), int(config["resolution_bp"])
    for row in rows:
        _require(row["chrom"] == data_config["chrom"], f"Chromosome mismatch: {row['structure_id']}")
        _require(0 <= int(row["start"]) <= float(row["center"]) < int(row["end"]) <= length,
                 f"Invalid coordinates: {row['structure_id']}")
    n_bins = (length + resolution - 1) // resolution
    _require(2 * (max(config["window_bins"]) + config["guard_bins"]) < n_bins, "Window too large for circular chromosome")
    centers = np.array([int(float(row["center"]) // resolution) for row in rows], dtype=np.int64)
    groups, footprints = _group_windows(centers, n_bins, max(config["window_bins"]), config["guard_bins"])
    skeleton = {"structure_id": np.array(ids), "y": np.array([LABEL_ORDER.index(row["type"]) for row in rows]),
                "group_id": groups, "window_footprints": footprints, "n_bins": n_bins, "config": config}
    print(f"Grouping {len(rows)} structures into {len(np.unique(groups))} spatial groups", flush=True)
    folds = _make_folds(skeleton, config)
    samples = data_config["samples"]
    _require(len(samples) == 2 and all(sample.get("condition") == "WT_37C" for sample in samples),
             "This comparison requires the two WT_37C replicates")
    _require(len({sample['sample_id'] for sample in samples}) == 2 and
             len({sample['replicate'] for sample in samples}) == 2, "Duplicate replicate identifiers")
    for sample in samples:
        _require(Path(sample["sample_id"]).name == sample["sample_id"], "Unsafe sample identifier")
        _require(resolve(raw_root, sample["path"]).is_file(), f"Missing raw matrix: {sample['path']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    try:
        sample_records = []
        for sample in samples:
            name = f"{sample['sample_id']}.band.npy"
            print(f"Streaming fresh circular band: {sample['sample_id']}", flush=True)
            audit = _raw_band(resolve(raw_root, sample["path"]), output / name, data_config, config)
            sample_records.append({**sample, **audit, "band_file": name})
        occupancy = (np.arange(n_bins) * resolution + resolution) <= length
        metadata = {"format_version": FORMAT_VERSION, "project_root": str(root), "config": config,
                    "label_order": LABEL_ORDER, "structures": rows, "group_id": groups.tolist(),
                    "center_bin": centers.tolist(), "n_bins": n_bins, "chrom_length": length,
                    "samples": sample_records, "coordinate_status": statuses,
                    "bin_occupancy": {"full_bins": int(occupancy.sum()),
                                      "terminal_source_fraction": float((length % resolution or resolution) / resolution),
                                      "terminal_partial_bin_masked": bool(length % resolution)},
                    "class_counts": dict(Counter(row["type"] for row in rows)),
                    "groups": _group_distribution(skeleton["y"], groups),
                    "input_sha256": input_hashes, "annotation_workbook": workbook_provenance,
                    "data_root": str(raw_root),
                    "evaluation_scope": "exploratory grouped cross-validation; previously inspected data are not a new blind test",
                    "notes": ["Two replicates of a structure always remain in the same spatial split.",
                              "Circular coarse-bin distance treats the terminal partial bin as one position; a terminal partial bin is fully masked in background opportunities and inputs.",
                              "Upper-storage coarse distance zero is stored once for count audits and excluded from model inputs along with distance one.",
                              "No legacy transform, image, window cache, or split is reused.",
                              "No fitted normalization is stored in this raw cache; each training fold fits its own parameters."]}
        metadata["signature"] = _json_hash({"metadata": metadata, "folds": folds})
        _write_json(output / "metadata.json", metadata)
        _write_json(output / "folds.json", folds)
    except BaseException:
        # This directory was created here and did not exist on entry.
        shutil.rmtree(output)
        raise
    return load_dataset(output)


def load_dataset(path: Path, *, source_root: Path | None = None, data_root: Path | None = None) -> dict:
    """Open immutable fresh bands and structure-level arrays without pixel scans."""
    path = resolve(PROJECT_ROOT, path).resolve()
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    folds = json.loads((path / "folds.json").read_text(encoding="utf-8"))
    _require(metadata["format_version"] == FORMAT_VERSION, "Unsupported task-one cache format")
    recorded = metadata["signature"]
    unsigned = {key: value for key, value in metadata.items() if key != "signature"}
    _require(recorded == _json_hash({"metadata": unsigned, "folds": folds}), "Cache metadata/fold signature mismatch")
    project_root = (resolve(PROJECT_ROOT, source_root).resolve() if source_root is not None
                    else resolve(PROJECT_ROOT, metadata["project_root"]))
    _require(project_root.is_dir(),
             f"Recorded project root is unavailable: {project_root}. Source provenance cannot be checked after moving this cache; prepare a fresh cache in the current project.")
    for relative_path, expected_hash in metadata["input_sha256"].items():
        source = resolve(project_root, relative_path)
        if not source.is_file() and data_root is not None:
            source = resolve(resolve(PROJECT_ROOT, data_root), relative_path)
        _require(source.is_file(), f"Recorded annotation/config source is unavailable: {source}; prepare a fresh cache")
        _require(hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash,
                 f"Recorded annotation/config source changed: {source}; prepare a fresh cache")
    config, rows = metadata["config"], metadata["structures"]
    n_bins, length = int(metadata["n_bins"]), int(metadata["chrom_length"])
    centers = np.array(metadata["center_bin"], dtype=np.int64)
    widths = config["window_bins"]
    footprints = (centers[:, None] + np.arange(-max(widths) // 2, max(widths) // 2)) % n_bins
    bands = []
    for sample in metadata["samples"]:
        band_path = path / sample["band_file"]
        band_stat = band_path.stat()
        _require(band_stat.st_size == sample["band_size_bytes"] and band_stat.st_mtime_ns == sample["band_mtime_ns"],
                 f"Cached numeric band changed: {band_path}; prepare a fresh cache")
        raw_path = (resolve(resolve(PROJECT_ROOT, data_root), sample["path"]) if data_root is not None
                    else resolve(PROJECT_ROOT, sample["raw_path"]))
        if raw_path.exists():
            raw_stat = raw_path.stat()
            _require(raw_path.is_file() and raw_stat.st_size == sample["raw_size_bytes"]
                     and raw_stat.st_mtime_ns == sample["raw_mtime_ns"],
                     f"Raw matrix source changed: {raw_path}; prepare a fresh cache")
        else:
            warnings.warn(f"Raw matrix source is unavailable: {raw_path}. Using the verified cached band; raw source size/time cannot be rechecked.",
                          RuntimeWarning, stacklevel=2)
        band = np.load(band_path, mmap_mode="r", allow_pickle=False)
        _require(list(band.shape) == sample["band_shape"] and band.dtype == np.float64, "Band shape/dtype mismatch")
        bands.append(band)
    return {"path": path, "metadata": metadata, "config": config, "folds": folds,
            "signature": recorded, "label_order": metadata["label_order"], "n_bins": n_bins,
            "structure_id": np.array([row["structure_id"] for row in rows]),
            "y": np.array([LABEL_ORDER.index(row["type"]) for row in rows], dtype=np.int64),
            "group_id": np.array(metadata["group_id"], dtype=np.int64), "center_bin": centers,
            "window_footprints": footprints.astype(np.int64), "bands": bands,
            "replicate": np.array([sample["replicate"] for sample in metadata["samples"]]),
            "bin_occupancy": ((np.arange(n_bins) + 1) * config["resolution_bp"] <= length).astype(np.float64)}


def _indices(dataset: dict, ids: list[str]) -> np.ndarray:
    ids = list(ids)
    _require(len(ids) == len(set(ids)), "Duplicate requested structure IDs")
    lookup = {value: i for i, value in enumerate(dataset["structure_id"])}
    missing = set(ids) - set(lookup)
    _require(not missing, f"Unknown structure IDs: {sorted(missing)}")
    # Canonical order makes repeated preprocessing fits and signatures identical.
    return np.array(sorted(lookup[value] for value in ids), dtype=np.int64)


def split_train_validation(dataset: dict, outer_train_ids: list[str], config: dict, seed: int) -> tuple[list[str], list[str]]:
    """Choose one reproducible grouped inner holdout using labels/positions only."""
    config = _config(config)
    indices = _indices(dataset, outer_train_ids)
    y, groups = dataset["y"][indices], dataset["group_id"][indices]
    fraction, minimum = config["inner_validation_fraction"], config["min_class_per_fold"]
    n_splits = max(2, int(round(1 / fraction)))
    _require(len(np.unique(groups)) >= n_splits, "Too few outer-training groups for inner validation")
    target = np.bincount(y, minlength=3) * fraction
    best, score = None, float("inf")
    for trial in range(int(config["split_search_trials"])):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(seed) + trial)
        for train, validation in splitter.split(np.zeros(len(y)), y, groups):
            counts = np.bincount(y[validation], minlength=3)
            if np.any(counts < minimum) or np.any(np.bincount(y[train], minlength=3) < minimum):
                continue
            current = float(np.square((counts - target) / np.maximum(target, 1)).mean())
            current += 0.1 * ((len(validation) - len(y) * fraction) / (len(y) * fraction)) ** 2
            if current < score:
                best, score = (indices[train], indices[validation]), current
    _require(best is not None, "No feasible disjoint inner validation split; group distribution: " +
             json.dumps(_group_distribution(y, groups), ensure_ascii=False))
    _assert_disjoint(dataset, *best)
    return dataset["structure_id"][best[0]].tolist(), dataset["structure_id"][best[1]].tolist()


def _fit_expected(dataset: dict, training: np.ndarray) -> list[dict]:
    exposure = dataset["bin_occupancy"]
    train_bins = _occupancy(dataset, training) & (exposure > 0)
    expected_records = []
    max_distance = max(dataset["config"]["window_bins"])
    for band in dataset["bands"]:
        sums, opportunities = np.zeros(max_distance + 1), np.zeros(max_distance + 1)
        for distance in range(2, max_distance + 1):
            both = train_bins & np.roll(train_bins, -distance)
            weights = exposure * np.roll(exposure, -distance)
            opportunities[distance] = weights[both].sum(dtype=np.float64)
            sums[distance] = band[:, distance][both].sum(dtype=np.float64)
        expected = np.divide(sums, opportunities, out=np.zeros_like(sums), where=opportunities > 0)
        # A width-W image uses distances only through W-1. Distance W is kept
        # for an audit but must not reject an otherwise valid single window.
        _require(np.all(expected[2:max_distance] > 0),
                 "Training-only distance background has zero support/counts for a distance used in model inputs; inspect input coverage")
        expected_records.append({"expected": expected.tolist(), "stored_count_sum_by_distance": sums.tolist(),
                                 "zero_inclusive_pair_opportunities": opportunities.tolist(),
                                 "training_footprint_bins": int(train_bins.sum())})
    return expected_records


def _extract_inputs(dataset: dict, indices: np.ndarray, expected_records: list[dict]) -> dict:
    config = dataset["config"]
    widths, size, n_bins = config["window_bins"], int(config["output_size"]), dataset["n_bins"]
    max_width = max(widths)
    values, labels, ids, groups, replicates, sample_ids = [], [], [], [], [], []
    for replicate_index, band in enumerate(dataset["bands"]):
        expected = np.array(expected_records[replicate_index]["expected"])
        for index in indices:
            bins = dataset["window_footprints"][index]
            difference = (bins[None, :] - bins[:, None]) % n_bins
            distance = np.minimum(difference, n_bins - difference)
            start = np.where(difference <= n_bins // 2, bins[:, None], bins[None, :])
            matrix = np.asarray(band[start, distance])
            occupancy = dataset["bin_occupancy"][bins]
            exposure = occupancy[:, None] * occupancy[None, :]
            supported = (distance >= 2) & (exposure > 0)
            # Coverage is a local per-example missing-data rule, not a fitted
            # threshold or a cross-sample statistic from the held-out subset.
            good = np.where(supported, matrix, 0).sum(axis=0) > 0
            valid = exposure * (good[:, None] & good[None, :] & supported)
            denominator = expected[distance] * exposure
            oe = np.divide(matrix, denominator, out=np.zeros_like(matrix), where=(valid > 0) & (denominator > 0))
            signal = np.log1p(oe)
            scales = []
            for width in widths:
                left = (max_width - width) // 2
                window = slice(left, left + width)
                weight = valid[window, window]
                factor = width // size
                pooled_shape = (size, factor, size, factor)
                amount = weight.reshape(pooled_shape).sum(axis=(1, 3))
                total = (signal[window, window] * weight).reshape(pooled_shape).sum(axis=(1, 3))
                mean = np.divide(total, amount, out=np.zeros_like(total), where=amount > 0)
                scales.append(np.stack((mean, amount / (factor * factor))).astype(np.float32))
            values.append(np.stack(scales))
            _require(any(np.any(scale[1] > 0) for scale in scales),
                     f"Completely missing example: {dataset['structure_id'][index]}, replicate {replicate_index}")
            labels.append(dataset["y"][index])
            ids.append(dataset["structure_id"][index])
            groups.append(dataset["group_id"][index])
            replicates.append(dataset["replicate"][replicate_index])
            sample_ids.append(dataset["metadata"]["samples"][replicate_index]["sample_id"])
    return {"x": np.stack(values), "y": np.array(labels, dtype=np.int64),
            "structure_id": np.array(ids), "group_id": np.array(groups, dtype=np.int64),
            "replicate": np.array(replicates), "sample_id": np.array(sample_ids)}


def _apply_scalers(samples: dict, scalers: list[dict]) -> dict:
    for scale, parameters in enumerate(scalers):
        values = samples["x"][:, scale, 0]
        valid = samples["x"][:, scale, 1] > 0
        samples["x"][:, scale, 0] = np.where(valid, (values - parameters["mean"]) / parameters["std"], 0)
    _require(np.isfinite(samples["x"]).all(), "Nonfinite transformed model inputs")
    return samples


def prepare_fold(dataset: dict, train_ids: list[str], validation_ids: list[str], config: dict) -> tuple[dict, dict, dict]:
    """Fit on train footprints and pooled train pixels; transform both subsets.

    Inputs have shape (examples, 2 scales, 2 channels, 64, 64), where each
    example is one replicate of one structure. The training loop may reverse
    both image axes together during training; this function creates no copies.
    """
    config = _config(config)
    for key in ("resolution_bp", "window_bins", "output_size", "guard_bins", "normalization"):
        _require(config[key] == dataset["config"][key], f"Cached protocol mismatch for {key}")
    train, heldout = _indices(dataset, train_ids), _indices(dataset, validation_ids)
    _assert_disjoint(dataset, train, heldout)
    expected = _fit_expected(dataset, train)
    training = _extract_inputs(dataset, train, expected)
    scalers = []
    for scale in range(len(config["window_bins"])):
        valid = training["x"][:, scale, 1] > 0
        values = training["x"][:, scale, 0][valid].astype(np.float64)
        _require(len(values) > 0, f"No valid training pixels at scale {scale}")
        scalers.append({"mean": float(values.mean()), "std": max(float(values.std()), 1e-6),
                        "fitted_valid_pooled_pixels": int(len(values))})
    preprocessing = {"dataset_signature": dataset["signature"],
                     "train_ids": dataset["structure_id"][train].tolist(),
                     "expected": expected, "scalers": scalers,
                     "fit_scope": "training structure footprints and training replicate images only",
                     "tensor_layout": "N,scale(6.4kb/25.6kb),channel(log1p_OE/valid_fraction),64,64",
                     "training_augmentation": config["train_augmentation"]}
    preprocessing["signature"] = _json_hash(preprocessing)
    validation = transform_heldout(dataset, validation_ids, preprocessing)
    return _apply_scalers(training, scalers), validation, preprocessing


def transform_heldout(dataset: dict, ids: list[str], preprocessing: dict) -> dict:
    """Reuse an inner-training fit for its validation or outer holdout set."""
    _require(preprocessing["dataset_signature"] == dataset["signature"], "Preprocessing/cache signature mismatch")
    recorded = preprocessing["signature"]
    _require(recorded == _json_hash({key: value for key, value in preprocessing.items() if key != "signature"}),
             "Preprocessing signature mismatch")
    train, heldout = _indices(dataset, preprocessing["train_ids"]), _indices(dataset, ids)
    _assert_disjoint(dataset, train, heldout)
    return _apply_scalers(_extract_inputs(dataset, heldout, preprocessing["expected"]), preprocessing["scalers"])


# Supplied target intervals and inner-only geometry fitting.
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
    _require(recorded == _json_hash({key: value for key, value in signal_fit.items() if key != "signature"}),
             "Frozen signal preprocessing signature mismatch")
    training = _indices(dataset, frozen_fold["train_ids"])
    validation = _indices(dataset, frozen_fold["validation_ids"])
    _require(set(frozen_fold["train_ids"]) == set(signal_fit["train_ids"]),
             "Frozen fold training IDs and fitted signal training IDs differ")
    _assert_disjoint(dataset, training, validation)
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
    # prepare_fold (which would refit OE/scalers) is never invoked.
    training_signal = _apply_scalers(
        _extract_inputs(dataset, training, signal_fit["expected"]), signal_fit["scalers"])
    validation_signal = transform_heldout(dataset, frozen_fold["validation_ids"], signal_fit)
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


# Formal partition and transform verification.
def _check_frozen_partition(dataset: dict, frozen_fold: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check all three partitions using IDs, spatial groups and window guards."""
    indices = []
    for split in ("train", "validation", "test"):
        ids = frozen_fold.get(f"{split}_ids")
        _require(isinstance(ids, list) and bool(ids), f"Missing frozen {split} IDs")
        indices.append(_indices(dataset, ids))
    train, validation, test = indices
    for first, second in ((train, validation), (train, test), (validation, test)):
        _assert_disjoint(dataset, first, second)
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
    _require(recorded == _signature({key: value for key, value in preprocessing.items()
                                               if key != "signature"}),
             "Formal target preprocessing signature mismatch")
    _require(preprocessing.get("frozen_fold_signature") == _signature(frozen_fold),
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
    train, validation, preprocessing = prepare_controlled_fold(dataset, frozen_fold)
    _check_preprocessing(dataset, frozen_fold, preprocessing)
    return train, validation, preprocessing


def prepare_outer(dataset: dict, frozen_fold: dict, preprocessing: dict) -> dict:
    """Transform the outer fold using existing fits; never fit any statistic.

    ``preprocessing`` is the exact object returned by :func:`prepare_inner`.
    This function is called only after checkpoint selection is fixed by the
    runner; it has no access to fitting or model-selection operations.
    """
    _check_preprocessing(dataset, frozen_fold, preprocessing)
    signal = transform_heldout(dataset, frozen_fold["test_ids"],
                                        preprocessing["signal_preprocessing"])
    lookup = _annotation_lookup(dataset)
    result = _add_target_features(dataset, signal, lookup, preprocessing["length_scaler"])
    _require(set(result["structure_id"]) == set(frozen_fold["test_ids"]),
             "Outer transformed IDs differ from frozen test partition")
    return result
