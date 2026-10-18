"""Audit saved unique-structure errors and prepare a blinded morphology register.

Only JSON/CSV annotations and saved predictions are read. No contact matrices,
model weights, preprocessing, fitting or inference are loaded or performed.
Geometry measures describe interval occupancy, not measured contact validity.
"""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import sys

sys.dont_write_bytecode = True
ROOT = _layout_root
CLASSES = ("OPCID", "CHIN", "CHID")
MODELS = ("m2", "m2plus", "cnn_transformer")
SEEDS = (42, 43, 44)
REVIEW_FIELDS = {
    "review_status": "pending", "reviewer_id": "", "review_date": "",
    "stage1_completed_before_reveal": "unknown", "candidate_morphology_resolvable": "unknown",
    "within_target_distinct_units": "unknown", "boundary_evidence": "unknown",
    "stripe_evidence": "unknown", "cluster_evidence": "unknown",
    "target_alignment_adequate": "unknown", "context_window_adequate": "unknown",
    "WT_replicate_morphology_agreement": "unknown", "reviewer_confidence": "unknown",
    "observation_notes": "", "stage2_label_assessment": "unknown",
    "stage2_boundary_assessment": "unknown", "stage2_adjudication": "pending",
}


def json_signature(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty register: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"Inconsistent CSV fields: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def interval_geometry(start: int, end: int, center_bin: int, *, chrom_length: int,
                      n_bins: int, resolution: int = 100,
                      window_bins: tuple[int, int] = (64, 256), output_size: int = 64) -> dict:
    """Exact circular-window overlap with the frozen interval-mask convention.

    Mean pooling of an outer-product mask equals the outer product of pooled
    axis fractions. ``effective_output_axis_bins`` is fractional ROI occupancy;
    it is not the number of bins with valid or visible biological signal.
    """
    if (not 0 <= start < end <= chrom_length or not 0 <= center_bin < n_bins
            or n_bins != (chrom_length + resolution - 1) // resolution
            or list(window_bins) != [64, 256] or resolution != 100 or output_size != 64):
        raise ValueError("Invalid interval or a non-frozen input-window convention")
    result = {"length_bp": end - start}
    for prefix, width in zip(("fine", "coarse"), window_bins):
        indices = [(center_bin + offset) % n_bins for offset in range(-width // 2, width // 2)]
        overlap = [max(0, min(min((index + 1) * resolution, chrom_length), end)
                       - max(index * resolution, start)) for index in indices]
        factor = width // output_size
        pooled = [sum(overlap[i:i + factor]) / (factor * resolution)
                  for i in range(0, width, factor)]
        covered = sum(overlap)
        result.update({f"{prefix}_window_nominal_bp": width * resolution,
                       f"{prefix}_output_bin_bp": resolution * factor,
                       f"{prefix}_ROI_covered_bp": covered,
                       f"{prefix}_ROI_coverage_fraction": covered / (end - start),
                       f"{prefix}_ROI_clipped": int(covered < end - start),
                       f"{prefix}_ROI_nonzero_native_axis_bins": sum(value > 0 for value in overlap),
                       f"{prefix}_ROI_nonzero_output_axis_bins": sum(value > 0 for value in pooled),
                       f"{prefix}_ROI_effective_output_axis_bins": sum(pooled),
                       f"{prefix}_ROI_effective_output_area_bins": sum(pooled) ** 2,
                       f"{prefix}_window_wraps_origin": int(indices != sorted(indices))})
    return result


def case_flags(true: int, predictions: dict[str, list[int]]) -> dict:
    """Unique-case flags; stable means all three seeds, never majority voting."""
    if set(predictions) != set(MODELS) or any(len(values) != len(SEEDS) for values in predictions.values()):
        raise ValueError("Each backbone must have the same three complete seeds")
    if true not in range(3) or any(value not in range(3) for values in predictions.values() for value in values):
        raise ValueError("Invalid class")
    values = predictions["m2plus"]
    return {"m2plus_stable_correct": int(all(value == true for value in values)),
            "m2plus_stable_wrong": int(all(value != true for value in values)),
            "m2plus_any_wrong": int(any(value != true for value in values)),
            "m2plus_seed_unanimous": int(len(set(values)) == 1),
            "m2plus_CHID_stable_FN": int(true == 2 and all(value != 2 for value in values)),
            "m2plus_CHID_any_FN": int(true == 2 and any(value != 2 for value in values)),
            "m2plus_CHID_to_CHIN_stable": int(true == 2 and all(value == 1 for value in values)),
            "m2plus_CHID_stable_FP": int(true != 2 and all(value == 2 for value in values)),
            "m2plus_CHID_any_FP": int(true != 2 and any(value == 2 for value in values)),
            "m2plus_CHIN_to_CHID_stable": int(true == 1 and all(value == 2 for value in values)),
            "m2plus_CHIN_to_CHID_any": int(true == 1 and any(value == 2 for value in values)),
            "same_seed_cross_model_disagreement_count": sum(len({predictions[model][i] for model in MODELS}) > 1
                                                          for i in range(len(SEEDS))),
            "all_models_all_seeds_correct": int(all(value == true for values in predictions.values() for value in values)),
            "all_models_all_seeds_wrong": int(all(value != true for values in predictions.values() for value in values)),
            "all_models_all_seeds_same_prediction": int(len({value for values in predictions.values() for value in values}) == 1)}


def length_stratum(length: int) -> str:
    for limit, name in ((3200, "00_le_3.2kb"), (6400, "01_3.2_to_6.4kb"),
                        (12800, "02_6.4_to_12.8kb"), (25600, "03_12.8_to_25.6kb")):
        if length <= limit:
            return name
    return "04_gt_25.6kb"


def roi_stratum(value: int) -> str:
    for limit, name in ((8, "00_le_8_bins"), (16, "01_9_to_16_bins"),
                        (32, "02_17_to_32_bins")):
        if value <= limit:
            return name
    return "03_33_to_64_bins"


def _verify_indexed_csv(directory: Path, manifest: dict, name: str) -> list[dict]:
    index_path = directory / "ARTIFACTS.json"
    index = read_json(index_path)
    if json_signature(index["files"]) != index.get("signature"):
        raise ValueError(f"Artifact index signature changed: {index_path}")
    if index["experiment_signature"] != manifest["experiment_signature"]:
        raise ValueError(f"Artifact index experiment differs: {index_path}")
    job = index["job"]
    if (job["condition"] != directory.parents[2].name or job["model"] != directory.parents[1].name
            or f"seed_{job['seed']}" != directory.parent.name or f"fold_{job['fold']}" != directory.name):
        raise ValueError(f"Artifact index job differs from its directory: {index_path}")
    matched = [row for row in index["files"] if row["name"] == name]
    path = directory / name
    if len(matched) != 1 or path.is_symlink() or path.stat().st_size != matched[0]["size_bytes"] or file_sha256(path) != matched[0]["sha256"]:
        raise ValueError(f"Sealed numerical file changed: {path}")
    return csv_rows(path)


def _read_saved(run_dir: Path) -> tuple[dict, dict, dict, dict]:
    manifest = read_json(run_dir / "manifest.json")
    if json_signature({k: v for k, v in manifest.items() if k != "experiment_signature"}) != manifest["experiment_signature"]:
        raise ValueError("Formal manifest signature changed")
    if manifest["config"]["seeds"] != list(SEEDS) or manifest["config"]["folds"] != list(range(5)):
        raise ValueError("This audit requires the frozen five folds and seeds 42/43/44")
    if manifest["label_order"] != list(CLASSES):
        raise ValueError("Label order differs from the frozen protocol")
    cache_path = Path(manifest["config"]["cache_dir"])
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    metadata_path = cache_path / "metadata.json"
    metadata = read_json(metadata_path)
    cached_folds = json.loads((cache_path / "folds.json").read_text(encoding="utf-8"))
    unsigned_metadata = {key: value for key, value in metadata.items() if key != "signature"}
    if (metadata["signature"] != manifest["dataset_signature"]
            or json_signature({"metadata": unsigned_metadata, "folds": cached_folds}) != manifest["dataset_signature"]):
        raise ValueError("Cached metadata/fold content signature differs")
    for relative, expected in metadata["input_sha256"].items():
        path = Path(relative)
        if not path.is_absolute():
            path = ROOT / path
        if file_sha256(path) != expected:
            raise ValueError(f"Frozen annotation/config source changed: {path}")
    structures = metadata["structures"]
    if not (len(structures) == len(metadata["center_bin"]) == len(metadata["group_id"]) == manifest["n_structures"]):
        raise ValueError("Incomplete frozen structure metadata")
    identities = {}
    test_fold = {}
    for fold in range(5):
        entry = read_json(run_dir / "folds" / f"fold_{fold}.json")
        frozen = entry["frozen_fold"]
        if (json_signature({k: v for k, v in entry.items() if k != "signature"}) != entry["signature"]
                or json_signature(frozen) != manifest["fold_signatures"][str(fold)]
                or {key: frozen[key] for key in ("train_ids", "validation_ids", "test_ids")} != manifest["fold_ids"][str(fold)]):
            raise ValueError(f"Frozen fold changed: {fold}")
        for sid in frozen["test_ids"]:
            if sid in test_fold:
                raise ValueError("Structure occurs in more than one outer fold")
            test_fold[sid] = fold
    for row, group, center in zip(structures, metadata["group_id"], metadata["center_bin"]):
        sid = row["structure_id"]
        if sid in identities or sid not in test_fold or row["type"] not in CLASSES:
            raise ValueError("Repeated or missing frozen structure identity")
        identities[sid] = {"annotation": row, "group_id": int(group), "center_bin": int(center),
                           "fold": test_fold[sid], "y_true": CLASSES.index(row["type"])}
    if set(test_fold) != set(identities):
        raise ValueError("Outer folds and metadata structure sets differ")
    predictions, source_files = {}, []
    for model in MODELS:
        for seed in SEEDS:
            oof_dir = run_dir / "oof" / "I3" / model / f"seed_{seed}"
            status = read_json(oof_dir / "status.json")
            if (status["status"] != "complete" or status.get("oof_current") is not True
                    or (status["model"], status["condition"], status["seed"]) != (model, "I3", seed)):
                raise ValueError(f"Incomplete or incorrect OOF: {oof_dir}")
            for key in ("experiment_signature", "dataset_signature", "source_signature"):
                if status[key] != manifest[key]:
                    raise ValueError(f"OOF signature differs: {oof_dir}")
            rows = csv_rows(oof_dir / "predictions.csv")
            by_id = {row["structure_id"]: row for row in rows}
            if len(by_id) != len(rows) or set(by_id) != set(identities):
                raise ValueError(f"OOF does not cover all unique structures: {oof_dir}")
            originals = {}
            for fold in range(5):
                directory = run_dir / "runs" / "I3" / model / f"seed_{seed}" / f"fold_{fold}"
                for original in _verify_indexed_csv(directory, manifest, "outer_predictions.csv"):
                    sid = original["structure_id"]
                    if sid in originals or identities.get(sid, {}).get("fold") != fold:
                        raise ValueError("Sealed fold predictions contain invalid or duplicate identities")
                    originals[sid] = original
            if set(originals) != set(identities):
                raise ValueError("Sealed outer predictions are incomplete")
            for sid, row in by_id.items():
                identity = identities[sid]
                if (int(row["y_true"]), int(row["group_id"]), int(row["fold"])) != (identity["y_true"], identity["group_id"], identity["fold"]):
                    raise ValueError(f"OOF frozen identity differs: {sid}")
                if any(row.get(field) != value for field, value in originals[sid].items()):
                    raise ValueError(f"OOF does not match its sealed original outer CSV: {sid}")
                probs = [float(row[f"p_{name}"]) for name in CLASSES]
                if (any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs)
                        or not math.isclose(sum(probs), 1., abs_tol=2e-6)
                        or int(row["y_pred"]) != max(range(3), key=probs.__getitem__)
                        or int(row["replicate_count"]) != 2):
                    raise ValueError(f"Invalid saved probability/replicate evidence: {sid}")
            predictions[(model, seed)] = by_id
            source_files.append({"path": str(oof_dir / "predictions.csv"),
                                 "sha256": file_sha256(oof_dir / "predictions.csv")})
    group_folds = {}
    for row in identities.values():
        old = group_folds.setdefault(row["group_id"], row["fold"])
        if old != row["fold"]:
            raise ValueError("A frozen spatial group occurs in multiple outer folds")
    provenance = {"experiment_signature": manifest["experiment_signature"],
                  "source_signature": manifest["source_signature"], "dataset_signature": manifest["dataset_signature"],
                  "metadata_path": str(metadata_path), "metadata_sha256": file_sha256(metadata_path),
                  "metadata_content_signature_verified": True,
                  "manifest_sha256": file_sha256(run_dir / "manifest.json"),
                  "verified_complete_I3_oof_files": len(predictions), "verified_sealed_outer_CSVs": len(predictions) * 5,
                  "n_unique_structures": len(identities), "n_spatial_groups": len(group_folds),
                  "coordinate_status": metadata.get("coordinate_status"), "prediction_sources": source_files,
                  "verification_scope": "Manifest/fold signatures, numerical CSV artifact seals, OOF-to-outer identity/probability equality; no model-weight reading or image processing."}
    return manifest, metadata, identities, {"predictions": predictions, "provenance": provenance}


def _build_cases(metadata: dict, identity: dict, predictions: dict) -> list[dict]:
    opaque_order = sorted(identity, key=lambda sid: hashlib.sha256((metadata["signature"] + "|" + sid).encode()).hexdigest())
    anonymous = {sid: f"case_{i:04d}" for i, sid in enumerate(opaque_order, start=1)}
    cases = []
    for sid in opaque_order:
        item, source = identity[sid], identity[sid]["annotation"]
        start, end = int(source["start"]), int(source["end"])
        values = {model: [int(predictions[(model, seed)][sid]["y_pred"]) for seed in SEEDS] for model in MODELS}
        geometry = interval_geometry(start, end, item["center_bin"], chrom_length=metadata["chrom_length"],
                                     n_bins=metadata["n_bins"], resolution=metadata["config"]["resolution_bp"],
                                     window_bins=tuple(metadata["config"]["window_bins"]),
                                     output_size=metadata["config"]["output_size"])
        row = {"case_id": anonymous[sid], "structure_id": sid, "annotation_type": source["type"],
               "chrom": source["chrom"], "start_bp": start, "end_bp": end,
               "center_bin": item["center_bin"], "group_id": item["group_id"], "fold": item["fold"],
               "coordinate_system": source["coordinate_system"], "coordinate_status": source["coordinate_status"],
               **geometry, "length_stratum": length_stratum(end - start),
               "coarse_ROI_nonzero_axis_bin_stratum": roi_stratum(geometry["coarse_ROI_nonzero_output_axis_bins"]),
               "same_center_other_structure_count": sum(other != sid and value["center_bin"] == item["center_bin"] for other, value in identity.items()),
               "overlapping_other_structure_count": sum(other != sid and max(start, int(value["annotation"]["start"])) < min(end, int(value["annotation"]["end"])) for other, value in identity.items()),
               "overlapping_CHIN_annotation_count": sum(other != sid and value["annotation"]["type"] == "CHIN" and max(start, int(value["annotation"]["start"])) < min(end, int(value["annotation"]["end"])) for other, value in identity.items()),
               "overlapping_CHID_annotation_count": sum(other != sid and value["annotation"]["type"] == "CHID" and max(start, int(value["annotation"]["start"])) < min(end, int(value["annotation"]["end"])) for other, value in identity.items()),
               **case_flags(item["y_true"], values)}
        for model in MODELS:
            for seed in SEEDS:
                saved = predictions[(model, seed)][sid]
                row[f"{model}_seed_{seed}_prediction"] = CLASSES[int(saved["y_pred"])]
                row[f"{model}_seed_{seed}_p_CHID"] = float(saved["p_CHID"])
                row[f"{model}_seed_{seed}_WT_disagree"] = int(saved["replicate_disagree"])
            row[f"{model}_correct_seed_count"] = sum(value == item["y_true"] for value in values[model])
            row[f"{model}_stable_wrong"] = int(all(value != item["y_true"] for value in values[model]))
        row.update({"signal_validity_in_ROI": "unknown", "biological_morphology_resolvable": "unknown",
                    "annotation_correctness": "unknown", "boundary_correctness": "unknown"})
        cases.append(row)
    return cases


def _association_rows(cases: list[dict]) -> list[dict]:
    result = []
    fields = ("length_stratum", "coarse_ROI_nonzero_axis_bin_stratum", "fine_ROI_clipped", "coarse_ROI_clipped")
    for field in fields:
        for label in CLASSES:
            candidates = [row for row in cases if row["annotation_type"] == label]
            for stratum in sorted({str(row[field]) for row in candidates}):
                members = [row for row in candidates if str(row[field]) == stratum]
                result.append({"geometry_feature": field, "stratum": stratum, "annotation_type": label,
                               "n_unique_structures": len(members), "n_spatial_groups": len({row["group_id"] for row in members}),
                               "m2plus_stable_correct": sum(row["m2plus_stable_correct"] for row in members),
                               "m2plus_stable_wrong": sum(row["m2plus_stable_wrong"] for row in members),
                               "m2plus_any_wrong": sum(row["m2plus_any_wrong"] for row in members),
                               "m2plus_CHID_stable_FN": sum(row["m2plus_CHID_stable_FN"] for row in members),
                               "m2plus_CHID_stable_FP": sum(row["m2plus_CHID_stable_FP"] for row in members),
                               "length_bp_median": median(row["length_bp"] for row in members),
                               "stable_wrong_rate": sum(row["m2plus_stable_wrong"] for row in members) / len(members)})
    return result


def _group_rows(cases: list[dict]) -> list[dict]:
    rows = []
    for group in sorted({row["group_id"] for row in cases}):
        members = [row for row in cases if row["group_id"] == group]
        folds = {row["fold"] for row in members}
        if len(folds) != 1:
            raise ValueError("Group split across folds")
        rows.append({"group_id": group, "fold": next(iter(folds)), "n_unique_structures": len(members),
                     **{f"{label}_support": sum(row["annotation_type"] == label for row in members) for label in CLASSES},
                     "m2plus_stable_wrong": sum(row["m2plus_stable_wrong"] for row in members),
                     "m2plus_CHID_stable_FN": sum(row["m2plus_CHID_stable_FN"] for row in members),
                     "m2plus_CHIN_to_CHID_stable": sum(row["m2plus_CHIN_to_CHID_stable"] for row in members),
                     "cross_model_disagreement_cases": sum(row["same_seed_cross_model_disagreement_count"] > 0 for row in members)})
    return rows


def _aggregate(cases: list[dict], predictions: dict) -> dict:
    result = {"n_unique_structures": len(cases), "class_support": dict(Counter(row["annotation_type"] for row in cases)),
              "M2plus_unique_case_counts": {key: sum(row[key] for row in cases) for key in case_flags(0, {model: [0] * 3 for model in MODELS})},
              "M2plus_stable_CHID_FN_ids": [row["structure_id"] for row in cases if row["m2plus_CHID_stable_FN"]],
              "M2plus_stable_CHIN_to_CHID_ids": [row["structure_id"] for row in cases if row["m2plus_CHIN_to_CHID_stable"]],
              "cross_model_disagreement_unique_cases": sum(row["same_seed_cross_model_disagreement_count"] > 0 for row in cases),
              "models": {}, "geometry_by_annotation": {}, "CHID_by_frozen_fold": {}}
    for model in MODELS:
        stable_miss = [row for row in cases if row["annotation_type"] == "CHID" and row[f"{model}_correct_seed_count"] == 0]
        seed_stats = {}
        for seed in SEEDS:
            rows = list(predictions[(model, seed)].values())
            seed_stats[str(seed)] = {"CHID_TP": sum(int(row["y_true"]) == 2 and int(row["y_pred"]) == 2 for row in rows),
                                     "CHID_FN": sum(int(row["y_true"]) == 2 and int(row["y_pred"]) != 2 for row in rows),
                                     "CHID_FP": sum(int(row["y_true"]) != 2 and int(row["y_pred"]) == 2 for row in rows),
                                     "CHIN_to_CHID": sum(int(row["y_true"]) == 1 and int(row["y_pred"]) == 2 for row in rows),
                                     "CHID_to_CHIN": sum(int(row["y_true"]) == 2 and int(row["y_pred"]) == 1 for row in rows)}
        result["models"][model] = {"per_seed": seed_stats, "stable_CHID_FN_ids": [row["structure_id"] for row in stable_miss],
                                  "stable_wrong_unique_cases": sum(row[f"{model}_stable_wrong"] for row in cases),
                                  "all_seeds_correct_unique_cases": sum(row[f"{model}_correct_seed_count"] == 3 for row in cases)}
    for label in CLASSES:
        members = [row for row in cases if row["annotation_type"] == label]
        correct = [row for row in members if row["m2plus_stable_correct"]]
        wrong = [row for row in members if row["m2plus_stable_wrong"]]
        result["geometry_by_annotation"][label] = {"support": len(members),
            "fine_window_clipped": sum(row["fine_ROI_clipped"] for row in members),
            "coarse_window_clipped": sum(row["coarse_ROI_clipped"] for row in members),
            "length_bp_median": median(row["length_bp"] for row in members),
            "stable_correct_length_bp_median": median(row["length_bp"] for row in correct) if correct else None,
            "stable_wrong_length_bp_median": median(row["length_bp"] for row in wrong) if wrong else None,
            "coarse_nonzero_output_axis_bins_median": median(row["coarse_ROI_nonzero_output_axis_bins"] for row in members),
            "coarse_nonzero_axis_bins_range": [min(row["coarse_ROI_nonzero_output_axis_bins"] for row in members), max(row["coarse_ROI_nonzero_output_axis_bins"] for row in members)],
            "n_stable_wrong": len(wrong), "n_stable_correct": len(correct)}
    for fold in range(5):
        members = [row for row in cases if row["annotation_type"] == "CHID" and row["fold"] == fold]
        result["CHID_by_frozen_fold"][str(fold)] = {
            "support": len(members), "length_le_6400bp": sum(row["length_bp"] <= 6400 for row in members),
            "M2plus_stable_FN": sum(row["m2plus_CHID_stable_FN"] for row in members),
            "M2plus_any_FN": sum(row["m2plus_CHID_any_FN"] for row in members)}
    return result


def _blind_cases(cases: list[dict]) -> list[dict]:
    # Never include original structure IDs, annotation names, source-sheet tags,
    # model confidence, error strata, priority or label-derived overlapping counts.
    fields = ("case_id", "chrom", "start_bp", "end_bp", "center_bin", "length_bp",
              "group_id", "fold", "coordinate_system", "coordinate_status")
    return [{**{field: row[field] for field in fields}, **REVIEW_FIELDS} for row in cases]


def assert_pending_register(existing: list[dict], cases: list[dict]) -> None:
    """Only an untouched, identity-identical template may be regenerated."""
    expected = {row["case_id"]: row for row in _blind_cases(cases)}
    if len(existing) != len(expected) or {row["case_id"] for row in existing} != set(expected):
        raise ValueError("Existing blinded register differs; choose a new output directory")
    for row in existing:
        original = expected[row["case_id"]]
        if any(str(row.get(field, "")) != str(value) for field, value in original.items()):
            raise ValueError("Existing review observations or identity fields must be preserved; choose a new output directory")


def priority_cases(cases: list[dict]) -> tuple[list[dict], dict]:
    """All CHID and stable CHID false positives plus matched correct references.

    Each anchor gets one stable-correct reference of each annotation class,
    where available. Match group first, then frozen fold, then log-length;
    references can be reused and are never ranked by model confidence.
    Priority and match labels stay out of the blinded register and figures.
    """
    chosen: dict[str, dict] = {}
    matches = []

    def include(row: dict, reason: str) -> None:
        entry = chosen.setdefault(row["structure_id"], {
            "case_id": row["case_id"], "structure_id": row["structure_id"],
            "selection_reasons": [], "group_id": row["group_id"], "fold": row["fold"]})
        if reason not in entry["selection_reasons"]:
            entry["selection_reasons"].append(reason)

    anchors = [row for row in cases if row["annotation_type"] == "CHID" or row["m2plus_CHID_stable_FP"]]
    for anchor in anchors:
        include(anchor, "all_CHID" if anchor["annotation_type"] == "CHID" else "stable_CHID_false_positive")
        for label in CLASSES:
            candidates = [row for row in cases if row["annotation_type"] == label
                          and row["m2plus_stable_correct"] and row["structure_id"] != anchor["structure_id"]]
            if not candidates:
                matches.append({"anchor": anchor["structure_id"], "reference_class": label,
                                "reference": None, "match_scope": "unavailable"})
                continue
            def rank(row):
                scope = 0 if row["group_id"] == anchor["group_id"] else (1 if row["fold"] == anchor["fold"] else 2)
                return scope, abs(math.log(row["length_bp"] / anchor["length_bp"])), row["case_id"]
            reference = min(candidates, key=rank)
            scope = ("same_group", "same_fold", "other_fold")[rank(reference)[0]]
            include(reference, "matched_stable_correct_reference")
            matches.append({"anchor": anchor["structure_id"], "reference_class": label,
                            "reference": reference["structure_id"], "match_scope": scope,
                            "absolute_log_length_ratio": rank(reference)[1]})
    rows = [{**row, "selection_reasons": "|".join(row["selection_reasons"])}
            for row in sorted(chosen.values(), key=lambda row: row["case_id"])]
    return rows, {"n_priority_cases": len(rows), "n_anchors": len(anchors),
                  "n_all_CHID": sum(row["annotation_type"] == "CHID" for row in anchors),
                  "n_stable_CHID_false_positive_anchors": sum(row["m2plus_CHID_stable_FP"] for row in anchors),
                  "matching_rule": "One M2+ all-three-seed-correct reference of each class per anchor, group then fold then absolute log-length-ratio; reuse allowed; no confidence ranking.",
                  "match_scope_counts": dict(Counter(row["match_scope"] for row in matches)),
                  "matches": matches,
                  "limitations": "Same-group matches can be unavailable; fold and cross-fold fallback are disclosed. This is an engineering inspection priority, not a statistical sampling design or a new test cohort."}


def audit_errors(run_dir: Path, output_dir: Path) -> dict:
    """Read saved evidence, write a new all-case review bundle, return its summary."""
    run_dir, output_dir = run_dir.resolve(), output_dir.resolve()
    if run_dir == output_dir or run_dir in output_dir.parents:
        raise ValueError("Review output must be separate from the scientific run")
    protected = (ROOT / "results", ROOT / "evidence", ROOT / "data", ROOT / "reports/task1_controlled_review_v1")
    if any(output_dir == path or path in output_dir.parents for path in protected):
        raise ValueError("Review output cannot overwrite scientific/frozen evidence or data")
    manifest, metadata, identities, saved = _read_saved(run_dir)
    cases = _build_cases(metadata, identities, saved["predictions"])
    aggregate = _aggregate(cases, saved["predictions"])
    associations, groups = _association_rows(cases), _group_rows(cases)
    priority, priority_summary = priority_cases(cases)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Preserve completed human observations: regeneration may not silently reset
    # even one edited response, reviewer ID or note.
    blind_path = output_dir / "blind_cases.csv"
    if blind_path.exists():
        assert_pending_register(csv_rows(blind_path), cases)
    write_csv(output_dir / "cases.csv", cases)
    write_csv(blind_path, _blind_cases(cases))
    write_csv(output_dir / "reveal_key.csv", [{"case_id": row["case_id"], "structure_id": row["structure_id"],
              "annotation_type": row["annotation_type"], **{f"{model}_seed_{seed}_prediction": row[f"{model}_seed_{seed}_prediction"]
              for model in MODELS for seed in SEEDS}} for row in cases])
    write_csv(output_dir / "geometry_associations.csv", associations)
    write_csv(output_dir / "group_review.csv", groups)
    write_csv(output_dir / "priority_cases.csv", priority)
    result = {"protocol": "task1_saved_errors_morphology_review_v1", "status": "numeric_audit_complete_morphology_pending",
              "source_run": str(run_dir), "provenance": saved["provenance"], "statistics": aggregate,
              "priority_review": priority_summary,
              "review_register": {"total_cases": len(cases), "pending_morphology_reviews": len(cases),
                                 "review_fields": REVIEW_FIELDS, "stage1": "Hide cases.csv/reveal_key.csv and original labels/predictions; inspect target interval and WT replicates first.",
                                 "stage2": "After recording stage1 observations, reveal labels/predictions; assess annotation, boundary and failure cause with evidence.",
                                 "not_inferred": ["biological_morphology_resolvable", "annotation_correctness", "boundary_correctness", "signal_validity_in_ROI"]},
              "geometry_convention": {"resolution_bp": 100, "window_bp": [6400, 25600], "output_size": 64,
                                       "pooled_bin_bp": [100, 400], "target_interval": "0-based half-open non-wrapping", "window_wrap": "circular chromosome",
                                       "effective_output_axis_bins": "sum of fractional interval occupancy; no contact signal loaded",
                                       "effective_output_area_bins": "squared axis occupancy, exact outer-product ROI mask area",
                                       "association_scope": "Descriptive within annotation class and frozen spatial group; no causal attribution or bin-wise biological visibility claim."},
              "selection_scope": "All unique structures, original five folds and all three seeds retained. These already-seen OOF errors are developmental evidence, not a new blind holdout."}
    report = _markdown_report(result, cases, associations)
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "analysis.md").write_text(report, encoding="utf-8")
    return result


def _markdown_report(result: dict, cases: list[dict], associations: list[dict]) -> str:
    stats = result["statistics"]
    counts = stats["M2plus_unique_case_counts"]
    lines = ["# 形态证据与系统性错误：数值审计及待审登记", "",
             "本页只审计保存的标签、区间与预测；尚未检查真实接触图，不能判定生物学形态可辨识性或标注正确性。",
             f"覆盖全部 **{len(cases)} 个唯一结构、49个原空间组、5个冻结外层折、seeds42/43/44**。9个I3完整OOF CSV已逐行对应45个封存外层CSV；WT重复先按原协议聚合。", "",
             "## 系统性错误", "",
             "“稳定”指三个种子都发生；统计单位是唯一结构，三个种子不作为三个独立生物样本。", "",
             "| 模型 | 三种子均正确 | 三种子均错误 | 稳定CHID漏检 |", "| --- | ---: | ---: | ---: |"]
    for model in MODELS:
        item = stats["models"][model]
        lines.append(f"| {model} | {item['all_seeds_correct_unique_cases']} | {item['stable_wrong_unique_cases']} | {len(item['stable_CHID_FN_ids'])}/26 |")
    lines += ["", f"M2+稳定CHID漏检：{', '.join(stats['M2plus_stable_CHID_FN_ids'])}。",
              f"M2+稳定CHIN→CHID误报：{', '.join(stats['M2plus_stable_CHIN_to_CHID_ids']) or '无'}。",
              f"M2+稳定CHID误报共{counts['m2plus_CHID_stable_FP']}个；CHIN→CHID为其中{counts['m2plus_CHIN_to_CHID_stable']}个。",
              f"至少一个同种子跨模型分歧的结构共{stats['cross_model_disagreement_unique_cases']}个；9次预测全部正确的参考结构{counts['all_models_all_seeds_correct']}个。", "",
              "| 模型 / seed | CHID TP | CHID FN | CHID FP | CHID→CHIN | CHIN→CHID |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for model in MODELS:
        for seed in SEEDS:
            item = stats["models"][model]["per_seed"][str(seed)]
            lines.append(f"| {model} / {seed} | {item['CHID_TP']} | {item['CHID_FN']} | {item['CHID_FP']} | {item['CHID_to_CHIN']} | {item['CHIN_to_CHID']} |")
    lines += ["", "## 几何关联与输入覆盖", "",
              "精细视野6.4kb、100bp/输出轴bin；粗视野25.6kb、400bp/输出轴bin。下表ROI bins仅指目标掩码几何覆盖，与接触计数、有效像素、可见形态分辨率是不同量。", "",
              "| 标注 | 数量 | 精细视野截断目标 | 粗视野截断目标 | 目标长度中位数kb | M2+稳定正确长度中位数kb | M2+稳定错误长度中位数kb |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label in CLASSES:
        item = stats["geometry_by_annotation"][label]
        def kb(value):
            return f"{value / 1000:.2f}" if value is not None else "无"
        lines.append(f"| {label} | {item['support']} | {item['fine_window_clipped']} | {item['coarse_window_clipped']} | {kb(item['length_bp_median'])} | {kb(item['stable_correct_length_bp_median'])} | {kb(item['stable_wrong_length_bp_median'])} |")
    lines += ["", "| CHID长度分层 | 唯一结构 | M2+稳定漏检 | M2+任一种子漏检 | 空间组数 |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in associations:
        if row["geometry_feature"] == "length_stratum" and row["annotation_type"] == "CHID":
            lines.append(f"| {row['stratum']} | {row['n_unique_structures']} | {row['m2plus_CHID_stable_FN']} | {row['m2plus_any_wrong']} | {row['n_spatial_groups']} |")
    lines += ["", "| 冻结fold | CHID数量 | ≤6.4kb数量 | M2+稳定漏检 | M2+任一种子漏检 |", "| --- | ---: | ---: | ---: | ---: |"]
    for fold, row in stats["CHID_by_frozen_fold"].items():
        lines.append(f"| {fold} | {row['support']} | {row['length_le_6400bp']} | {row['M2plus_stable_FN']} | {row['M2plus_any_FN']} |")
    lines += ["", "六个稳定漏检集中于fold0/4，但fold1/2也有短CHID且没有稳定漏检。这提醒我们同时核对空间组与图像证据，不能只把错误归因于短目标。"]
    lines += ["", "| 稳定CHID漏检 | group / fold | 长度kb | 精细目标覆盖 | 粗视野非零轴bin | 粗视野等效轴bin |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in sorted(cases, key=lambda row: int(row["structure_id"].split("_")[-1])):
        if row["m2plus_CHID_stable_FN"]:
            lines.append(f"| {row['structure_id']} | {row['group_id']} / {row['fold']} | {row['length_bp']/1000:.2f} | {row['fine_ROI_coverage_fraction']:.3f} | {row['coarse_ROI_nonzero_output_axis_bins']} | {row['coarse_ROI_effective_output_axis_bins']:.2f} |")
    lines += ["", "长度、窗口覆盖与ROI bins与标注和空间位置相互关联，且各分层样本量小；这些是描述性关联，不能证明长度导致错误或判断应放大/缩小视野。几何审核不是接触图形态审核。", "",
              "## 双阶段人工审核", "",
              "1. 第一阶段只用 `blind_cases.csv` 和匿名目标接触图，隐藏 `cases.csv`、`reveal_key.csv`、原始结构ID、标签和预测。逐条记录三类形态证据、目标与可见结构的对应、边界、上下文及WT重复一致性。每个未知字段保持 unknown，直到确有观察。",
              "2. 第一阶段完成后揭示标签与预测，再登记标签定义、边界、分辨率、上下文和模型错误是否得到图像证据支持。不能用预测置信度代替可辨识性，也不能因多模型一致就认定标注有误。",
              "3. 对全部344例保留登记；优先批次同时纳入稳定漏检、稳定误报、跨模型分歧和相同长度/空间位置的正确参考，审核者不见类别或优先级。只检查六个漏检不足以验证可靠识别。",
              "4. 当前坐标为 provisional。明确确认标注坐标系统和目标定义，在复评前版本化；历史结果和本登记不回写改标。", "",
              "## 文件与验收", "",
              "- `cases.csv`：344例完整数值审核表（揭示后使用）。",
              "- `blind_cases.csv`：344例匿名随机顺序待审表；全部形态与标注判断仍未完成。",
              "- `reveal_key.csv`：隔离保存的匿名ID→原结构/标签/预测映射。",
              "- `geometry_associations.csv`：按标注类分别统计长度、粗ROI bins和窗口截断关联。",
              "- `group_review.csv`：全部49空间组的原fold、类支持与稳定错误计数。",
              f"- `priority_cases.csv`：工程优先面板{result['priority_review']['n_priority_cases']}例，包含全部26个CHID、全部稳定CHID误报，以及每个锚点按原group→fold→log长度匹配的三类稳定正确参照；优先类型不进入盲图。",
              "- `summary.json`：来源哈希、统计口径和未知字段。", "",
              "数值阶段成功判据：结构344/344、空间组49/49、冻结身份全部一致、9个OOF与45个封存CSV逐行一致，稳定错误与正确参考都登记。形态阶段尚未验收，必须补齐独立观察和揭示后裁决。", "",
              "上述已看过的OOF只支持开发诊断。根据这些错误选输入/优化方案后，原外层结果不再作为全新盲测；下一次正式复评须明确开发/最终评估边界，并统一输入、选择与训练规则。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/task1_formal_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/task1_morphology_review_v1")
    args = parser.parse_args()
    result = audit_errors(args.run_dir, args.output_dir)
    print(json.dumps({"status": result["status"], "n_unique_structures": result["statistics"]["n_unique_structures"],
                      "stable_CHID_FN": result["statistics"]["M2plus_stable_CHID_FN_ids"],
                      "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
