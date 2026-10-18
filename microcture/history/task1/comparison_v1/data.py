"""Fresh task-one data and spatially grouped splits; never read legacy caches.

The persistent cache is an unnormalised circular distance band for each WT
replicate. All observed/expected estimates and signal scalers are fitted using
only the supplied training structure footprints. Preparing it scans raw pixels
once and is deliberately a user-run operation.
"""
from __future__ import annotations

from collections import Counter
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


def prepare_dataset(config: dict, root: Path, output: Path) -> dict:
    """Build a new cache; refuse every existing destination, including old data."""
    root, output = Path(root).resolve(), Path(output).resolve()
    config = _config(config)
    _require(not output.exists(), f"Destination already exists; load it or choose a fresh directory: {output}")
    data_path, annotation_path = root / config["data_config"], root / config["annotations"]
    data_config = json.loads(data_path.read_text(encoding="utf-8"))
    annotation_source = root / data_config["annotations"]["path"]
    _require(annotation_source.is_file(), f"Missing original annotation workbook: {annotation_source}")
    input_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (data_path, annotation_path, annotation_source)}
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
        _require((root / sample["path"]).is_file(), f"Missing raw matrix: {sample['path']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    try:
        sample_records = []
        for sample in samples:
            name = f"{sample['sample_id']}.band.npy"
            print(f"Streaming fresh circular band: {sample['sample_id']}", flush=True)
            audit = _raw_band(root / sample["path"], output / name, data_config, config)
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
                    "input_sha256": input_hashes,
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


def load_dataset(path: Path) -> dict:
    """Open immutable fresh bands and structure-level arrays without pixel scans."""
    path = Path(path).resolve()
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    folds = json.loads((path / "folds.json").read_text(encoding="utf-8"))
    _require(metadata["format_version"] == FORMAT_VERSION, "Unsupported task-one cache format")
    recorded = metadata["signature"]
    unsigned = {key: value for key, value in metadata.items() if key != "signature"}
    _require(recorded == _json_hash({"metadata": unsigned, "folds": folds}), "Cache metadata/fold signature mismatch")
    project_root = Path(metadata["project_root"])
    _require(project_root.is_dir(),
             f"Recorded project root is unavailable: {project_root}. Source provenance cannot be checked after moving this cache; prepare a fresh cache in the current project.")
    for relative_path, expected_hash in metadata["input_sha256"].items():
        source = project_root / relative_path
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
        raw_path = Path(sample["raw_path"])
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
