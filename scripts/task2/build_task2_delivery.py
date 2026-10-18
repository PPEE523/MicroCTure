#!/usr/bin/env python3
"""Plot saved task-two evidence; never train, rescan, or turn missing data into zero.

Run from the existing project: conda run --no-capture-output -n microcture \
    python -B scripts/build_task2_delivery.py
All inputs remain read-only. Outputs use existing submission directories.
"""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from centering_status import aggregate, as_bool, normalize_summary

ROOT = _layout_root
LEGACY = _layout_resolve(ROOT / "data/legacy_experiments/results")
PORTABLE = _layout_resolve(ROOT / "submission/task2/centering")
OUT = _layout_resolve(ROOT / "submission/figures")
BLUE, GREEN, ORANGE, GRAY = "#2b5797", "#1c886b", "#d37d25", "#d9dee5"
SOURCES: dict[str, dict] = {}


def source(path: Path) -> Path:
    path = _layout_resolve(path)
    data = path.read_bytes()
    SOURCES[path.relative_to(ROOT).as_posix()] = {
        "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)
    }
    return path


def read_json(path: Path):
    return json.loads(source(path).read_text())


def read_csv(path: Path):
    with source(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_evidence():
    report = read_json(_layout_resolve(LEGACY / "task2_v2/report.json"))
    methods = read_csv(_layout_resolve(LEGACY / "task2_v2/method_summary.csv"))
    clusters = read_csv(_layout_resolve(LEGACY / "task2_v2/clusters.csv"))
    disjoint = read_csv(_layout_resolve(LEGACY / "task2_v2/unannotated_supported_disjoint.csv"))
    review = read_json(_layout_resolve(LEGACY / "task2_novelty_review/report.json"))
    morphology = read_json(_layout_resolve(LEGACY / "task2_morphology_audit/report.json"))
    clean = [r for r in clusters if int(r["annotated_members"]) == 0]
    supported_clean = [r for r in clean if int(r["supported_members"]) >= 5]
    passing = [r for r in clusters if as_bool(r["stable_candidate_new_cluster"]) is True]
    runs = {(r["method"], r["k"], r["seed"]) for r in clusters}
    assert len(methods) == len(report["methods"]) == 14
    assert len(runs) == report["cluster_runs"] == 168
    assert len(disjoint) == report["unannotated_supported_disjoint"] == 21
    assert len(passing) == report["all_passing_cluster_configurations"] == 0
    for row, saved in zip(methods, report["methods"]):
        assert row["method"] == saved["method"]
        for field in ("candidates", "unannotated", "replicate_supported", "unannotated_supported"):
            assert int(row[field]) == saved[field]

    cohorts = {}
    for name in ("primary", "supplement"):
        saved = read_json(PORTABLE / name / "report.json")
        rows = read_csv(PORTABLE / name / "candidate_summary.csv")
        variants = read_csv(PORTABLE / name / "candidate_variants.csv")
        rows = [normalize_summary(r, saved["calibration_windows"], 10) for r in rows]
        summary = aggregate(rows)
        for key, value in summary.items():
            assert saved[key] == value, (name, key, saved[key], value)
        assessed = [r for r in rows if r["assessment_status"] == "assessed"]
        cohorts[name] = {
            **summary,
            "calibration_windows": saved["calibration_windows"],
            "minimum_calibration_windows": 10,
            "rows": rows,
            "metric_records": len(variants),
            "assessed_metric_records": sum(r["assessment_status"] == "assessed" for r in variants),
            "not_evaluable_metric_records": sum(r["assessment_status"] != "assessed" for r in variants),
            "assessed_subset_only": {
                "candidates": len(assessed),
                "any_two_metric_anomaly": (sum(r["original_outside_variants"] + r["boundary_outside_variants"] > 0 for r in assessed) if assessed else None),
                "recentered_two_metric_anomaly": (sum(r["boundary_outside_variants"] > 0 for r in assessed) if assessed else None),
                "robust_recentered_anomalies": (sum(r["robust_recentered_anomaly"] for r in assessed) if assessed else None),
            },
        }
    return {
        "discovery": {
            "scan_windows": report["scan_windows"],
            "eligible_windows": report["eligible_windows"],
            "methods": methods,
            "candidate_budget_per_method": 189,
            "selected_union_windows": report["unique_selected_windows"],
            "unannotated_replicated_union": report["unannotated_supported_union"],
            "unannotated_replicated_disjoint": len(disjoint),
            "cluster_runs": len(runs),
            "nonempty_cluster_records": len(clusters),
            "unannotated_cluster_records": len(clean),
            "unannotated_with_five_supported_members": len(supported_clean),
            "max_supported_members_in_unannotated_cluster": max(int(r["supported_members"]) for r in clean),
            "stable_unannotated_cluster_records": sum(float(r["min_seed_jaccard"]) >= 0.6 for r in clean),
            "passing_cluster_records": len(passing),
            "primary_passing_clusters": report["primary_candidate_new_clusters"],
            "rules": {
                "replication": "correlation > max(0.5, matched-control correlation 95th percentile); at least 10 controls",
                "cluster": "no annotated members; at least 5 replication-supported members; minimum best-match Jaccard across other seeds >= 0.6",
                "clustering": "14 ranking methods x K in [4,6,8,12] x seeds [42,43,44]; fixed candidate budget 189 per method",
            },
        },
        "shape_review": {
            "candidates": review["candidates"],
            "same_scale_known_references": review["same_scale_known_references"],
            "pca_outside_both": review["pca_outside_both"],
            "shape_outside_both": review["shape_outside_both"],
            "two_metric_distinct_priority": review["distinct_shape_priority"],
            "stable_distinct_morphologies_confirmed": morphology["stable_distinct_morphologies_confirmed"],
            "residual_replication_range": [morphology["marginal_residual_correlation_min"], morphology["marginal_residual_correlation_max"]],
        },
        "centering": {
            "candidate_windows": 21,
            "fixed_variants_per_candidate": 54,
            "variant_grid": "2 anchors x 3 widths (6.4/12.8/25.6 kb) x 3 shifts (-800/0/+800 bp) x 3 masks (none/10% seed17/10% seed29)",
            "distance_metrics": ["PCA distance", "numerical shape distance"],
            "reference_windows": 108,
            "reproducible_boundaries": 18,
            "robust_rule": "reproducible boundary; complete QC-passing grid; both metrics exceed known thresholds in both WT repeats in >=80% of variants at EVERY recentered scale",
            "cohorts": cohorts,
        },
    }


def plot_discovery(evidence):
    discovery = evidence["discovery"]
    methods = discovery["methods"]
    fig, (ax, gates) = plt.subplots(1, 2, figsize=(14.2, 6.2), gridspec_kw={"width_ratios": [1.8, 1.0]})
    y = np.arange(len(methods))
    unannotated = np.array([int(r["unannotated"]) for r in methods])
    replicated = np.array([int(r["unannotated_supported"]) for r in methods])
    ax.barh(y, unannotated, color="#d9e3f2", label="Unannotated selected windows")
    ax.barh(y, replicated, color=GREEN, label="Unannotated + replicate supported")
    for yi, total, passed in zip(y, unannotated, replicated):
        ax.text(total + 1, yi, f"{passed} / {total}", va="center", fontsize=9)
    ax.set_yticks(y, [r["method"].replace("_", " ") for r in methods])
    ax.invert_yaxis()
    ax.set_xlim(0, 111)
    ax.set_xlabel("Windows; every method selects the same 189-window budget")
    ax.set_title("A  Candidate replication by method", loc="left", fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.01, -0.15), fontsize=8, frameon=False)
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    labels = ["Nonempty cluster records", "No annotated members", "+ at least 5 supported", "+ stable across seeds"]
    counts = [discovery["nonempty_cluster_records"], discovery["unannotated_cluster_records"], discovery["unannotated_with_five_supported_members"], discovery["passing_cluster_records"]]
    gy = np.arange(4)
    gates.barh(gy, np.log10(np.array(counts) + 1), color=[BLUE, ORANGE, GREEN, GREEN])
    for yi, val in zip(gy, counts):
        gates.text(np.log10(val + 1) + 0.07, yi, f"{val:,}", va="center", fontweight="bold", fontsize=11)
    gates.set_yticks(gy, labels, fontsize=9)
    gates.invert_yaxis()
    gates.set_xticks([0, 1, 2, 3], ["0", "9", "99", "999"])
    gates.set_xlim(0, 3.65)
    gates.set_xlabel("Cluster records, log10(count + 1)")
    gates.set_title("B  Predeclared gates", loc="left", fontweight="bold")
    fig.text(0.60, 0.115, "168 runs reuse windows; records are not independent.\nThe 11 unannotated records have at most 2 supported\nmembers each; none meets the stability criterion.", fontsize=8.5, va="top", linespacing=1.4)
    fig.suptitle("Task 2 | Reproducible windows do not yet form a new structure cluster", x=0.025, ha="left", fontweight="bold", fontsize=15)
    fig.text(0.025, 0.02, "Saved V2 evidence: 5,804 scanned windows; 5,445 QC eligible. Previously viewed WT data: exploratory, not external confirmation.", fontsize=9, color="#444444")
    fig.subplots_adjust(left=0.1, right=0.96, top=0.85, bottom=0.29, wspace=0.66)
    fig.savefig(OUT / "task2_delivery_discovery.png", dpi=180, facecolor="white")
    plt.close(fig)


def plot_centering(evidence):
    centering = evidence["centering"]
    primary = centering["cohorts"]["primary"]
    supplement = centering["cohorts"]["supplement"]
    rows = supplement["rows"]
    data = np.array([[float(r[f"{anchor}_outside_fraction"]) if r.get(f"{anchor}_outside_fraction") is not None else np.nan for anchor in ("original", "boundary")] for r in rows])
    fig, (ax, status) = plt.subplots(1, 2, figsize=(12.7, 9.6), gridspec_kw={"width_ratios": [1.0, 1.18]})
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad(GRAY)
    ax.imshow(data, cmap=cmap, norm=Normalize(0, 1), aspect="auto")
    for i in range(len(rows)):
        for j in range(2):
            val = data[i, j]
            ax.text(j, i, "NA" if not np.isfinite(val) else f"{val:.1%}", va="center", ha="center", fontsize=9)
    ax.set_yticks(np.arange(len(rows)), [r["entity_id"] for r in rows], fontsize=8.5)
    ax.set_xticks([0, 1], ["Original center\n27 variants", "Boundary center\n27 variants"])
    ax.set_title("A  Two-metric anomaly fraction (supplement)", loc="left", fontsize=11, fontweight="bold", pad=12)
    ax.legend(handles=[Patch(facecolor=GRAY, label="NA: incomplete quality-passing grid")], loc="upper left", bbox_to_anchor=(-0.05, -0.06), frameon=False, fontsize=8.5)
    status.axis("off")
    status.set_title("B  Calibration and decision status", loc="left", fontsize=11, fontweight="bold", pad=12)
    lines = [
        (0.98, "PRIMARY FIXED COHORT", BLUE, 12),
        (0.93, "8 calibration windows < required 10", "#333333", 11),
        (0.88, f"{primary['not_evaluable_candidates']} / 21 candidates: NA", ORANGE, 16),
        (0.82, "Missing calibration cannot be reported as zero anomalies.", "#444444", 9),
        (0.72, "EXPLORATORY SUPPLEMENT", BLUE, 12),
        (0.67, "17 calibration windows, from already viewed WT regions", "#333333", 10),
        (0.61, "19 assessed + 2 NA (quality failures)", "#333333", 12),
        (0.55, "0 / 19 assessed candidates meets robust anomaly rule", GREEN, 11),
        (0.49, "21-candidate complete-grid total remains NA", ORANGE, 11),
        (0.39, "MATCHED, FIXED PERTURBATIONS", BLUE, 12),
        (0.34, "2 anchors x 3 scales x 3 shifts x 3 masks = 54", "#333333", 11),
        (0.28, "6.4 / 12.8 / 25.6 kb; shifts -800 / 0 / +800 bp", "#333333", 10),
        (0.22, "108 known references receive the same treatment.", "#333333", 10),
        (0.16, "Robustness requires >=80% at EACH recentered scale,", "#444444", 9),
        (0.12, "both metrics in both repeats, plus a reproducible boundary.", "#444444", 9),
    ]
    for y, text, color, size in lines:
        status.text(0, y, text, transform=status.transAxes, fontsize=size, color=color, va="top", fontweight="bold" if size >= 12 else "normal")
    fig.suptitle("Task 2 | Window sensitivity: distinguish negative results from unavailable evidence", x=0.025, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.025, 0.025, "Color range is fixed to 0-100%. Individual exceptional crops do not establish a robust new morphology. No independent biological confirmation.", fontsize=9, color="#444444")
    fig.subplots_adjust(left=0.14, right=0.97, top=0.91, bottom=0.13, wspace=0.25)
    fig.savefig(OUT / "task2_delivery_centering.png", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    assert OUT.is_dir() and (_layout_resolve(ROOT / "submission")).is_dir()
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    evidence = load_evidence()
    plot_discovery(evidence)
    plot_centering(evidence)
    summary = {
        "schema_version": 1,
        "action": "saved_evidence_reanalysis",
        "scientific_status": "Discovery pipeline and sensitivity audit completed; no rule-passing new cluster confirmed. Primary centering endpoint is not evaluable.",
        "training_performed": False,
        "raw_contact_pixels_read": False,
        "genome_rescan_performed": False,
        "independent_external_confirmation": False,
        "limitations": [
            "WT repeats were already viewed during prior development; all subsequent analyses remain exploratory.",
            "Unannotated windows are not a biological definition of a new class; annotation incompleteness and provisional coordinates remain.",
            "1205 cluster records reuse windows and are not 1205 independent statistical tests or discoveries.",
            "Primary centering calibration has 8 windows below fixed minimum 10: all decisions are null/NA.",
            "Supplement reuses known windows already inspected; 2 of 21 candidates fail complete-grid quality, so full-cohort endpoints remain null/NA.",
            "Saved historical manifests are not asserted to match current source; this script hashes the saved inputs it actually reads.",
            "A distance outlier or a robust anomaly would still require biological validation before claiming a new structure type.",
        ],
        **evidence,
        "source_files": SOURCES,
        "figures": ["submission/figures/task2_delivery_discovery.png", "submission/figures/task2_delivery_centering.png"],
    }
    for relative in summary["figures"]:
        path = _layout_resolve(ROOT / relative)
        summary.setdefault("output_files", {})[relative] = {
            "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        }
    output = _layout_resolve(ROOT / "submission/task2_delivery_summary.json")
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"summary": str(output.relative_to(ROOT)), "figures": summary["figures"],
                      "cluster_records": evidence["discovery"]["nonempty_cluster_records"],
                      "passing_clusters": evidence["discovery"]["passing_cluster_records"],
                      "primary_assessment": evidence["centering"]["cohorts"]["primary"]["assessment_status"],
                      "supplement_assessed": evidence["centering"]["cohorts"]["supplement"]["assessed_candidates"]}))


if __name__ == "__main__":
    main()
