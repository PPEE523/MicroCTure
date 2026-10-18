#!/usr/bin/env python3
"""Lightweight Task 1 audit: annotations, cache metadata and saved results only.

No raw pixel scan, image preprocessing, model fitting or inference is performed.
Cross-class collisions are computed using the center-only input protocol. They
are representation conflicts, not a declaration that the original labels are
biologically wrong. Use --strict to fail a preflight with such conflicts.
"""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = _layout_root
CLASS_NAMES = ("OPCID", "CHIN", "CHID")


def read_annotations(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("start", "end", "center", "length_bp"):
            row[key] = float(row[key])
    if len({row["structure_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate structure IDs in annotations")
    return rows


def annotation_findings(rows: list[dict], resolution: int = 100,
                        window_bins: tuple[int, int] = (64, 256)) -> dict:
    """Derive object-scope and deterministic image-identity conflicts."""
    by_center: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_center[(row["chrom"], int(row["center"] // resolution))].append(row)
    conflicts = []
    for (chrom, center_bin), members in by_center.items():
        if len({row["type"] for row in members}) > 1:
            conflicts.append({"chrom": chrom, "center_bin": center_bin,
                              "structures": [{key: row[key] for key in
                                              ("structure_id", "type", "start", "end", "center", "length_bp")}
                                             for row in members],
                              "reason": "same center bin creates the same signal/mask at both scales for each replicate; object interval is absent from model input"})
    distribution = {}
    for kind in CLASS_NAMES:
        members = [row for row in rows if row["type"] == kind]
        lengths = sorted(row["length_bp"] for row in members)
        count = len(lengths)
        median = (lengths[count // 2] if count % 2 else
                  (lengths[count // 2 - 1] + lengths[count // 2]) / 2) if count else None
        distribution[kind] = {"n": count, "min_length_bp": min(lengths) if lengths else None,
                              "median_length_bp": median, "max_length_bp": max(lengths) if lengths else None,
                              "fully_visible": {str(width * resolution): sum(
                                  row["start"] >= int(row["center"] // resolution) * resolution - width * resolution / 2
                                  and row["end"] <= int(row["center"] // resolution) * resolution + width * resolution / 2
                                  for row in members) for width in window_bins}}
    chid_details = []
    chin_rows = [row for row in rows if row["type"] == "CHIN"]
    for chid in (row for row in rows if row["type"] == "CHID"):
        candidates = [row for row in chin_rows if row["chrom"] == chid["chrom"]]
        overlap = [row for row in candidates if min(row["end"], chid["end"]) > max(row["start"], chid["start"])]
        contained = [row for row in candidates if row["start"] >= chid["start"] and row["end"] <= chid["end"]]
        nearest = min(candidates, key=lambda row: abs(row["center"] - chid["center"])) if candidates else None
        delta = abs(int(nearest["center"] // resolution) - int(chid["center"] // resolution)) if nearest else None
        chid_details.append({"structure_id": chid["structure_id"], "length_bp": chid["length_bp"],
                             "overlapping_chin": [row["structure_id"] for row in overlap],
                             "contained_chin": [row["structure_id"] for row in contained],
                             "nearest_chin": nearest["structure_id"] if nearest else None,
                             "nearest_center_distance_bp": abs(nearest["center"] - chid["center"]) if nearest else None,
                             "same_input_center": delta == 0 if nearest else None})
    return {"class_distribution": distribution, "cross_class_input_conflicts": conflicts,
            "chid_context": chid_details,
            "overlap_summary": {"chid_with_overlapping_chin": sum(bool(row["overlapping_chin"]) for row in chid_details),
                                "distinct_overlapping_chin": len({sid for row in chid_details for sid in row["overlapping_chin"]}),
                                "distinct_contained_chin": len({sid for row in chid_details for sid in row["contained_chin"]}),
                                "chid_nearest_chin_within_2kb": sum(row["nearest_center_distance_bp"] is not None and
                                                                       row["nearest_center_distance_bp"] <= 2000 for row in chid_details)}}


def collect_audit(root: Path, cache_dir: Path, run_dir: Path) -> dict:
    annotations = root / "data/processed/structures.csv"
    rows = read_annotations(annotations)
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    report = {"scope": "small annotation/cache metadata and existing predictions/metrics only; no training or pixel scans",
              "dataset_signature": metadata["signature"], "coordinate_status": metadata["coordinate_status"],
              "input_sha256_current": {}, "source_checks": [],
              **annotation_findings(rows, config["resolution_bp"], tuple(config["window_bins"]))}
    for relative, expected in metadata["input_sha256"].items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        report["input_sha256_current"][relative] = actual
        report["source_checks"].append({"path": relative, "matches_prepared_cache": actual == expected})
    group_by_id = dict(zip((row["structure_id"] for row in metadata["structures"]), metadata["group_id"]))
    report["class_spatial_groups"] = {kind: len({group_by_id[row["structure_id"]] for row in rows if row["type"] == kind})
                                      for kind in CLASS_NAMES}
    report["fold_counts"] = []
    for path in sorted((run_dir / "folds").glob("fold_*.json")):
        fold = json.loads(path.read_text(encoding="utf-8"))
        report["fold_counts"].append({"fold": fold["fold"], **{
            key: {"counts": dict(Counter(sid.split("_", 1)[0] for sid in fold[key])),
                  "chid_groups": len({group_by_id[sid] for sid in fold[key] if sid.startswith("CHID_")})}
            for key in ("train_ids", "validation_ids", "test_ids")}})
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        report["comparison_status"] = summary["status"]
        report["model_chid_scores"] = [{key: row[key] for key in
                                        ("model", "status", "macro_f1_mean", "CHID_precision_mean", "CHID_recall_mean", "CHID_f1_mean")}
                                       for row in summary["comparison"]]
    collapsed = []
    for path in sorted((run_dir / "runs").glob("*/seed_*/fold_*/metrics.json")):
        metric = json.loads(path.read_text(encoding="utf-8"))
        confusion = metric["validation"]["confusion_matrix"]
        predicted_counts = [sum(row[column] for row in confusion) for column in range(3)]
        total = sum(predicted_counts)
        if total and max(predicted_counts) / total >= 0.95:
            collapsed.append({"model": metric["model"], "seed": metric["seed"], "fold": metric["fold"],
                              "validation_predictions": dict(zip(CLASS_NAMES, predicted_counts)),
                              "validation_macro_f1": metric["validation"]["macro_f1"],
                              "selected_epoch": metric["selection"].get("selected_epoch"),
                              "selected_training_loss": metric["selection"].get("best_training_loss"),
                              "outer_chid_recall": metric["test"]["per_class"]["CHID"]["recall"]})
    report["validation_prediction_collapses_95pct"] = collapsed
    report["interpretation_limits"] = [
        "Cross-class identical inputs prove a center-only single-label representation conflict, not incorrect biological source labels.",
        "Two exact collisions alone cannot explain the whole CHID error rate.",
        "Prediction collapse is already present in inner validation; separate optimization checks are needed alongside data redesign.",
        "No controlled retraining was done, so the contribution of resolution, masks, pooling, limited examples or learning rate is not quantified.",
        "Source coordinates remain provisional; this audit does not verify the author's coordinate convention."]
    report["status"] = "representation_conflicts_found" if report["cross_class_input_conflicts"] else "review_required"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache/task1_comparison_v1")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/task1_comparison_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/task1_chid_data_diagnosis.json")
    parser.add_argument("--strict", action="store_true", help="Exit 2 when cross-class identical-input conflicts are found")
    args = parser.parse_args()
    report = collect_audit(ROOT, args.cache_dir.resolve(), args.run_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Audit saved: {args.output}")
    print(f"Status: {report['status']}; identical-input conflict groups: {len(report['cross_class_input_conflicts'])}")
    for row in report.get("model_chid_scores", []):
        print(f"{row['model']}: CHID precision={row['CHID_precision_mean']:.3f}, recall={row['CHID_recall_mean']:.3f}, F1={row['CHID_f1_mean']:.3f}")
    print(f"Inner-validation collapses (>=95% one predicted class): {len(report['validation_prediction_collapses_95pct'])}")
    return 2 if args.strict and report["cross_class_input_conflicts"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
