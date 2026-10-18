#!/usr/bin/env python3
"""Verify saved formal evidence and export figures without fitting or inference.

Scientific results are read only. Derived reports live in a separate reports/
directory; Matplotlib's disposable cache lives in a temporary /tmp directory.
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
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = _layout_root
sys.path.insert(0, str(ROOT))

import numpy as np

from microcture.task1.evaluation import _read_prediction_csv
from microcture.task1_formal.evaluation import (
    _baseline_outer_summary, _fold_ids, _guard_run, _identity, _load_job,
    _neural_summary, _same_score, _seed_agreement,
)
from microcture.task1_formal.protocol import (
    _package_signature, _validate_cache_metadata, job_directory, json_signature,
    planned_jobs, sha256, source_signature,
)
from microcture.task1_formal.runner import assert_resume


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_saved_results(run_dir: Path) -> tuple[dict, dict, dict]:
    """Check seals, recompute acceptance and bind OOF back to saved fold CSVs."""
    _guard_run(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    signature = manifest["experiment_signature"]
    if json_signature({k: v for k, v in manifest.items() if k != "experiment_signature"}) != signature:
        raise ValueError("Formal manifest signature changed")
    if source_signature(ROOT) != manifest["source_signature"]:
        raise ValueError("Formal training source differs from the recorded source")
    for package, entry in (("task1", "run_task1_comparison.py"),
                           ("task1_controlled", "run_task1_controlled.py")):
        if _package_signature(ROOT, package, entry) != manifest["dependency_source_signature"][package]:
            raise ValueError(f"Recorded {package} dependency source changed")
    config = manifest["config"]
    metadata, _ = _validate_cache_metadata(ROOT, project_path(config["cache_dir"]),
                                           manifest["dataset_signature"])
    identity = _identity(metadata)
    folds = _fold_ids(manifest, identity, run_dir)
    for fold in config["folds"]:
        saved = read_json(run_dir / "folds" / f"fold_{fold}.json")
        if (json_signature({k: v for k, v in saved.items() if k != "signature"}) != saved["signature"]
                or {key: saved["frozen_fold"][key] for key in ("train_ids", "validation_ids", "test_ids")} != manifest["fold_ids"][str(fold)]
                or json_signature(saved["frozen_fold"]) != manifest["fold_signatures"][str(fold)]):
            raise ValueError(f"Fold {fold} signature or frozen partition changed")
    summary = read_json(run_dir / "summary.json")
    for key in ("experiment_signature", "source_signature", "dataset_signature"):
        if summary[key] != manifest[key]:
            raise ValueError(f"Saved summary {key} differs from the manifest")
    if summary["n_structures"] != len(identity):
        raise ValueError("Summary structure count differs from frozen metadata")
    support = Counter(row["type"] for row in identity.values())
    if summary["class_support"] != dict(support) or summary["class_names"] != manifest["label_order"]:
        raise ValueError("Summary class order/support differs from frozen metadata")
    jobs = planned_jobs(config)
    records = []
    for job in jobs:
        directory = job_directory(run_dir, job)
        if assert_resume(directory, job, manifest) is None:
            raise ValueError(f"Missing planned job: {directory}")
        record = _load_job(run_dir, job, manifest, identity, folds)
        if record["status"] != "completed":
            raise ValueError("This report requires all registered jobs; failed jobs must be reported separately")
        records.append(record)
    if (summary["status"] != "complete" or summary["expected_jobs"] != len(jobs)
            or summary["completed_jobs"] != len(records)
            or any(summary[key] for key in ("pending_jobs", "invalid_jobs",
                                            "numerical_failure_jobs", "execution_failure_jobs"))):
        raise ValueError("Summary status/counts disagree with the verified saved jobs")
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        job = record["job"]
        grouped.setdefault((job["condition"], job["model"]), []).append(record)
    if set(summary["models"]) != {f"{condition}/{model}" for condition, model in grouped}:
        raise ValueError("Summary model coverage differs from the registered jobs")
    oof_count = 0
    for (condition, model), cases in grouped.items():
        detail = summary["models"][f"{condition}/{model}"]
        expected_seeds = sorted({record["job"]["seed"] for record in cases})
        if set(detail["seeds"]) != {str(seed) for seed in expected_seeds}:
            raise ValueError(f"{condition}/{model}: expected seed identities differ")
        oof = {}
        for seed in expected_seeds:
            directory = run_dir / "oof" / condition / model / f"seed_{seed}"
            status = read_json(directory / "status.json")
            saved_metrics = read_json(directory / "metrics.json")
            for document in (status, saved_metrics):
                if document.get("status") != "complete" or document.get("oof_current") is not True:
                    raise ValueError(f"Stale or incomplete OOF: {directory}")
                for key in ("experiment_signature", "source_signature", "dataset_signature"):
                    if document[key] != manifest[key]:
                        raise ValueError(f"OOF {key} changed: {directory}")
                if (document["condition"], document["model"], document["seed"]) != (condition, model, seed):
                    raise ValueError(f"OOF identity changed: {directory}")
            prediction = _read_prediction_csv(directory / "predictions.csv")
            if set(prediction["structure_id"]) != set(identity):
                raise ValueError(f"OOF does not cover every unique structure: {directory}")
            expected_rows = {row["structure_id"]: (row, record["job"]["fold"])
                             for record in cases if record["job"]["seed"] == seed
                             for row in record["predictions"]["rows"]}
            for row in prediction["rows"]:
                original, fold = expected_rows[row["structure_id"]]
                if int(row["fold"]) != fold:
                    raise ValueError(f"OOF outer fold changed: {row['structure_id']}")
                for field in ("y_true", "y_pred", "group_id", "replicate_count", "replicate_disagree"):
                    if str(row[field]) != str(original[field]):
                        raise ValueError(f"OOF {field} changed: {row['structure_id']}")
                for field in ("p_OPCID", "p_CHIN", "p_CHID"):
                    if not np.isclose(float(row[field]), float(original[field]), rtol=0, atol=1e-10):
                        raise ValueError(f"OOF probability changed: {row['structure_id']}")
                if row["replicate_probabilities_json"] != original["replicate_probabilities_json"]:
                    raise ValueError(f"OOF replicate evidence changed: {row['structure_id']}")
            _same_score(saved_metrics, prediction["metrics"], str(directory))
            _same_score(detail["seeds"][str(seed)]["oof"], prediction["metrics"], str(directory))
            oof[seed] = {"by_id": {row["structure_id"]: row for row in prediction["rows"]}}
            oof_count += 1
        if _seed_agreement(expected_seeds, oof, identity) != detail["seed_prediction_agreement"]:
            raise ValueError(f"{condition}/{model}: seed agreement differs from saved predictions")
        if cases[0]["job"]["kind"] == "neural":
            if _neural_summary(cases, config) != detail["learning_diagnostics"]:
                raise ValueError(f"{condition}/{model}: saved learning/event aggregation changed")
        elif _baseline_outer_summary(cases, config) != detail["outer_prediction_events"]:
            raise ValueError(f"{condition}/{model}: baseline event aggregation changed")
        rows = [row for row in summary["comparison"] if (row["condition"], row["model"]) == (condition, model)]
        if len(rows) != 1 or rows[0]["status"] != "complete":
            raise ValueError(f"{condition}/{model}: missing or repeated complete comparison row")
        aggregate = rows[0]
        if aggregate["complete_seeds"] != len(expected_seeds) or aggregate["expected_seeds"] != len(expected_seeds):
            raise ValueError(f"{condition}/{model}: comparison seed denominator changed")
        fields = [(name, name) for name in ("macro_f1", "accuracy", "balanced_accuracy")]
        fields += [(f"{cls}_{metric}", (cls, metric)) for cls in summary["class_names"]
                   for metric in ("precision", "recall", "f1")]
        for field, location in fields:
            scores = [detail["seeds"][str(seed)]["oof"] for seed in expected_seeds]
            values = [score[location] if isinstance(location, str) else score["per_class"][location[0]][location[1]]
                      for score in scores]
            if not np.isclose(aggregate[field + "_mean"], np.mean(values), rtol=0, atol=1e-12):
                raise ValueError(f"{condition}/{model}: comparison {field} mean changed")
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
            stored_sd = aggregate[field + "_seed_sample_std"]
            if (sd is None and stored_sd is not None) or (sd is not None and (stored_sd is None or not np.isclose(stored_sd, sd, rtol=0, atol=1e-12))):
                raise ValueError(f"{condition}/{model}: comparison {field} sample SD changed")
    inputs = [run_dir / "manifest.json", run_dir / "summary.json"]
    inputs.extend(sorted((run_dir / "oof").rglob("*.csv")))
    inputs.extend(sorted((run_dir / "oof").rglob("*.json")))
    inputs.extend(sorted((run_dir / "runs").rglob("ARTIFACTS.json")))
    hashes = {str(path.relative_to(run_dir)): sha256(path) for path in inputs}
    verification = {
        "experiment_signature": signature, "source_signature": manifest["source_signature"],
        "dataset_signature": manifest["dataset_signature"],
        "verified_jobs": len(records), "verified_neural_histories": sum(job["kind"] == "neural" for job in jobs),
        "verified_complete_oof": oof_count, "n_structures_per_oof": len(identity),
        "checks": ["scientific ARTIFACTS file sizes and SHA-256", "unchanged trained source",
                   "frozen spatial folds and dataset metadata", "complete epoch trajectories and recomputed learning acceptance",
                   "OOF probabilities, biological replicates and folds agree with sealed job predictions",
                   "OOF confusion/precision/recall/F1 recomputed", "seed agreement and event counts recomputed"],
        "saved_input_sha256": hashes,
        "execution": "read saved evidence only; no fit, preprocessing, or inference",
    }
    return manifest, summary, verification


def guard_report_output(run_dir: Path, output_dir: Path, signature: str) -> Path:
    for path in (output_dir, *output_dir.parents):
        if path.is_symlink():
            raise ValueError(f"Report output through a symlink is forbidden: {path}")
    path = output_dir.resolve()
    reports = ROOT / "reports"
    if path == reports or reports not in path.parents or run_dir == path or run_dir in path.parents:
        raise ValueError("Figures must use a separate subdirectory of project reports/")
    for protected in (reports / "task1_controlled_review_v1",):
        if path == protected or protected in path.parents or path in protected.parents:
            raise ValueError("Output overlaps frozen controlled reports")
    if path.exists():
        if any(child.is_symlink() for child in path.rglob("*")):
            raise ValueError("Symbolic links inside report output are forbidden")
        marker = path / "analysis_summary.json"
        if any(path.iterdir()) and not marker.is_file():
            raise ValueError("Existing directory has no analysis provenance; use a new --output-dir")
        if marker.is_file() and read_json(marker).get("experiment_signature") != signature:
            raise ValueError("Existing report belongs to a different experiment; use a new --output-dir")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/task1_formal_v1")
    parser.add_argument("--output-dir", default="reports/task1_formal_visuals_v1")
    args = parser.parse_args(argv)
    run_dir = project_path(args.run_dir)
    print("Verifying saved formal results (read only)...", flush=True)
    manifest, summary, verification = verify_saved_results(run_dir)
    output_dir = guard_report_output(run_dir.resolve(), project_path(args.output_dir), manifest["experiment_signature"])
    with tempfile.TemporaryDirectory(prefix="microcture-formal-viz-", dir="/tmp") as cache:
        os.environ["MPLBACKEND"] = "Agg"
        os.environ["MPLCONFIGDIR"] = cache
        from scripts.task1_formal_performance_plots import plot_performance
        from scripts.task1_formal_stability_plots import plot_stability
        from scripts.task1_formal_analysis_report import write_analysis_report
        import matplotlib

        verification["plotting_source_sha256"] = {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (Path(__file__), ROOT / "scripts/task1_formal_performance_plots.py",
                         ROOT / "scripts/task1_formal_stability_plots.py",
                         ROOT / "scripts/task1_formal_analysis_report.py")
        }
        verification["plotting_environment"] = {"matplotlib": matplotlib.__version__, "numpy": np.__version__}

        print(f"Verified {verification['verified_jobs']} jobs and {verification['verified_complete_oof']} complete OOF results. Exporting figures...", flush=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        # A provenance marker makes an interrupted render safely repeatable.
        marker = output_dir / "analysis_summary.json"
        if not marker.exists():
            marker.write_text(json.dumps({"status": "rendering", "experiment_signature": manifest["experiment_signature"]}, indent=2) + "\n", encoding="utf-8")
        figures_dir = output_dir / "figures"
        figures = plot_performance(run_dir, summary, figures_dir)
        figures.extend(plot_stability(run_dir, summary, figures_dir))
        for name, checksum in verification["saved_input_sha256"].items():
            if sha256(run_dir / name) != checksum:
                raise ValueError(f"Saved input changed during plotting: {name}")
        verification["saved_inputs_unchanged_after_plotting"] = True
        write_analysis_report(run_dir, summary, output_dir, figures, verification)
    print(json.dumps({"status": "complete", "figures": len(figures),
                      "report": str(output_dir / "analysis.md"),
                      "scientific_results_modified": False}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
