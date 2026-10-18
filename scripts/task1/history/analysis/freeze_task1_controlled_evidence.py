#!/usr/bin/env python3
"""Freeze completed controlled evidence in place, or verify its SHA-256 index.

This standalone utility uses only the standard library. It never trains,
preprocesses, deletes artifacts, or copies raw data/count caches.
"""
from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys

sys.dont_write_bytecode = True
ROOT = _layout_root
FROZEN_INSTRUCTION = "Use independent directories for formal experiments and future reports; never resume, overwrite or delete this controlled evidence"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_signature(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def inside_root(root: Path, path: Path) -> Path:
    """Reject symbolic links before resolving paths, including ancestor links."""
    path = path if path.is_absolute() else root / path
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Evidence paths must stay inside project root: {path}") from error
    current = root
    for part in relative.parts:
        if part in ("..", "."):
            raise ValueError(f"Relative traversal is forbidden: {path}")
        current /= part
        if current.is_symlink():
            raise ValueError(f"Cannot freeze or verify a symbolic-link artifact: {current}")
    return path


def tree(root: Path, directory: Path) -> tuple[list[Path], list[Path]]:
    directory = inside_root(root, directory)
    if not directory.is_dir():
        raise ValueError(f"Missing artifact directory: {directory}")
    files, directories = [], [directory]
    for path in sorted(directory.rglob("*")):
        inside_root(root, path)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            directories.append(path)
        else:
            raise ValueError(f"Unsupported artifact type: {path}")
    return files, directories


def source_paths(root: Path) -> list[Path]:
    paths = sorted((root / "microcture/task1_controlled").glob("*.py"))
    if not paths:
        raise ValueError("Missing controlled source package")
    paths.append(root / "scripts/run_task1_controlled.py")
    for path in paths:
        inside_root(root, path)
        if not path.is_file():
            raise ValueError(f"Missing controlled source: {path}")
    return paths


def controlled_source_signature(root: Path) -> str:
    return json_signature({str(path.relative_to(root)): sha256(path) for path in source_paths(root)})


def planned_jobs(config: dict) -> list[dict]:
    settings = config["input_stage"]
    cases = config["head_stage"]["cases"]
    if settings["variants"] != ["I0", "I1", "I2", "I3"] or set(settings["geometry_baselines"]) != {"length_only", "mask_only", "mask_length"}:
        raise ValueError("Freeze expects four preregistered input controls and three geometry baselines")
    if config["head_stage"]["versions"] != ["H0", "H1", "H2", "H3"] or len(cases) != 6:
        raise ValueError("Freeze expects all four heads on all six preregistered cases")
    base_lr = config["training"]["learning_rate"]
    common = {"stage": "input", "fold": settings["fold"], "seed": settings["seed"], "head": "H0", "backbone": "m2", "learning_rate": base_lr}
    jobs = [{**common, "name": name, "input_variant": name, "kind": "neural"} for name in settings["variants"]]
    jobs += [{**common, "name": name, "input_variant": "I3", "kind": "geometry"} for name in settings["geometry_baselines"]]
    for case, specification in cases.items():
        common = {"stage": "heads", "case": case, **specification, "input_variant": "I3", "kind": "neural"}
        jobs += [{**common, "name": head, "head": head,
                  "learning_rate": config["head_stage"]["H3_learning_rate"] if head == "H3" else base_lr}
                 for head in config["head_stage"]["versions"]]
    return jobs


def verify_completed(root: Path, run_dir: Path, report_dir: Path) -> dict:
    """Check saved metadata and the independently generated report before chmod."""
    tree(root, run_dir)
    tree(root, report_dir)
    manifest = read_json(run_dir / "manifest.json")
    expected_experiment = json_signature({key: value for key, value in manifest.items() if key != "experiment_signature"})
    if manifest["experiment_signature"] != expected_experiment:
        raise ValueError("Controlled experiment manifest signature mismatch")
    current_source = controlled_source_signature(root)
    if manifest["source_signature"] != current_source:
        raise ValueError("Current controlled source differs from the trained manifest")
    config = manifest["config"]
    if read_json(inside_root(root, root / "configs/task1_controlled.json")) != config:
        raise ValueError("Controlled configuration differs from the trained manifest")
    old_index = inside_root(root, Path(config["evidence_dir"]) / "freeze_manifest.json")
    if sha256(old_index) != manifest["frozen_evidence_sha256"]:
        raise ValueError("Original failed-evidence index differs from the controlled manifest")
    old = read_json(old_index)
    if old["dataset_signature"] != manifest["dataset_signature"] or old["run_signature"] != manifest["original_run_signature"]:
        raise ValueError("Original dataset/run evidence signature mismatch")
    report = read_json(report_dir / "summary.json")
    if report["experiment_signature"] != expected_experiment or report["source_signature"] != current_source:
        raise ValueError("Derived report source/experiment signature mismatch")
    if report.get("source_signature_unchanged") is not True or report.get("frozen_metadata_and_inner_ids_verified") is not True:
        raise ValueError("Derived report did not verify the frozen metadata/source")
    aggregate = report["head_aggregate"]
    if report.get("pending") or aggregate.get("comparison_status") != "fullcoverage" or aggregate.get("decision_ready") is not True:
        raise ValueError("A full-coverage independently verified report is required")
    input_rows = {row["name"]: row for row in report["input_review"]}
    head_rows = {(row["case"], row["head"]): row for row in report["case_head_review"]}
    if len(input_rows) != 7 or len(head_rows) != 24 or len(report["input_review"]) != 7 or len(report["case_head_review"]) != 24:
        raise ValueError("Report must contain exactly 7 input and 24 head records")
    pass_counts = {head: 0 for head in ("H0", "H1", "H2", "H3")}
    acceptance_records = []
    for job in planned_jobs(config):
        directory = run_dir / "input" / job["name"] if job["stage"] == "input" else run_dir / "heads" / job["case"] / job["name"]
        path = directory / "metrics.json"
        if not path.is_file():
            raise ValueError(f"Missing completed case: {directory}")
        metrics = read_json(path)
        if metrics.get("status") != "completed" or metrics.get("job") != job or metrics.get("experiment_signature") != expected_experiment:
            raise ValueError(f"Incomplete or inconsistent job: {directory}")
        for filename in ("protocol.json", "validation_predictions.csv"):
            if not (directory / filename).is_file():
                raise ValueError(f"Missing completed artifact: {directory / filename}")
        protocol = read_json(directory / "protocol.json")
        if protocol.get("job") != job or protocol.get("training") != config["training"] or protocol.get("acceptance") != config["acceptance"]:
            raise ValueError(f"Stored protocol differs from the preregistered manifest: {directory}")
        row = input_rows.get(job["name"]) if job["stage"] == "input" else head_rows.get((job["case"], job["head"]))
        if row is None or row.get("status") != "completed":
            raise ValueError(f"Report omitted a completed job: {directory}")
        if job["kind"] == "neural":
            for filename in ("history.json", "history.csv", "initialization.json", "final.pt"):
                if not (directory / filename).is_file():
                    raise ValueError(f"Missing complete neural evidence: {directory / filename}")
            history = read_json(directory / "history.json")["epochs"]
            if [item["epoch"] for item in history] != list(range(config["training"]["epochs"] + 1)):
                raise ValueError(f"Incomplete diagnostic trace: {directory}")
            if history[-1]["weighted_train_ce"] != metrics["final_train_ce"] or history[0]["weighted_train_ce"] != metrics["initial_train_ce"]:
                raise ValueError(f"Trace endpoint mismatch: {directory}")
            for split in ("train", "validation"):
                if history[-1][f"{split}_diagnostics"] != metrics[f"{split}_diagnostics"]:
                    raise ValueError(f"Trace final diagnostic mismatch: {directory}")
            accepted = metrics["acceptance"]["accepted"]
            if row.get("accepted") is not accepted or row.get("final_train_ce") != metrics["final_train_ce"]:
                raise ValueError(f"Report acceptance/loss mismatch: {directory}")
            if job["stage"] == "heads":
                if row.get("history_acceptance_agrees") is not True:
                    raise ValueError(f"Independent acceptance verification is absent: {directory}")
                if row.get("failed_checks") != metrics["acceptance"]["primary"]["failed_checks"]:
                    raise ValueError(f"Report failure evidence differs from saved acceptance: {directory}")
                pass_counts[job["head"]] += int(accepted)
            acceptance_records.append({"job": job, "accepted": accepted,
                                       "failed_checks": metrics["acceptance"]["primary"]["failed_checks"]})
    for head, count in pass_counts.items():
        summary = aggregate["head_summaries"][head]
        if summary["completed_cases"] != 6 or summary["missing_cases"] != 0 or summary["numerical_failure_cases"] != 0:
            raise ValueError(f"Incomplete aggregate head coverage: {head}")
        if summary["strict_accepted"]["numerator"] != count or summary["strict_accepted"]["denominator"] != 6:
            raise ValueError(f"Aggregated acceptance differs from original metrics: {head}")
    return {"manifest": manifest, "report": report, "strict_pass_counts": pass_counts,
            "acceptance_records": acceptance_records, "old_index": old_index}


def _read_only(paths: list[Path]) -> None:
    for path in paths:
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)


def freeze_evidence(root: Path, evidence_dir: Path, run_dir: Path | None = None,
                    report_dir: Path | None = None, decision_paths: list[Path] | None = None,
                    freeze_date: str = "2026-10-12") -> dict:
    root = root.resolve()
    evidence_dir = inside_root(root, evidence_dir)
    run_dir = inside_root(root, run_dir or root / "results/task1_controlled_v1")
    report_dir = inside_root(root, report_dir or root / "reports/task1_controlled_review_v1")
    if (evidence_dir / "freeze_manifest.json").exists():
        return verify_evidence(root, evidence_dir)
    if evidence_dir.exists():
        raise ValueError("Evidence destination exists without a completed freeze; preserve it and choose a fresh destination")
    if any(first == second or first in second.parents or second in first.parents
           for first, second in ((run_dir, report_dir), (run_dir, evidence_dir), (report_dir, evidence_dir))):
        raise ValueError("Run, report and evidence directories must be separate")
    if (run_dir / "FROZEN.json").exists() or (report_dir / "FROZEN.json").exists():
        raise ValueError("Existing freeze marker cannot be overwritten")
    checked = verify_completed(root, run_dir, report_dir)
    manifest = checked["manifest"]
    snapshots = source_paths(root) + [root / relative for relative in (
        "scripts/report_task1_controlled.py", "scripts/task1_controlled_reporting.py",
        "configs/task1_controlled.json", "environment.yml", "environment.lock.yml")]
    decisions = [inside_root(root, path) for path in decision_paths or []]
    for path in snapshots + decisions:
        inside_root(root, path)
        if not path.is_file():
            raise ValueError(f"Missing requested source/decision snapshot: {path}")
    snapshots = sorted(set(snapshots + decisions))
    # All checks finish before creating a destination or changing permissions.
    evidence_dir.mkdir(parents=True)
    for path in snapshots:
        destination = evidence_dir / "source_snapshot" / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    trees = [run_dir, report_dir, evidence_dir]
    files, directories = [], [evidence_dir]
    for directory in trees:
        current_files, current_directories = tree(root, directory)
        files += current_files
        directories += current_directories
    records = [{"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
               for path in sorted(files)]
    old_index = checked["old_index"]
    freeze = {"status": "frozen_controlled_experiment", "freeze_date": freeze_date,
              "run_dir": str(run_dir.relative_to(root)), "report_dir": str(report_dir.relative_to(root)),
              "source_signature": manifest["source_signature"], "experiment_signature": manifest["experiment_signature"],
              "dataset_signature": manifest["dataset_signature"], "original_run_signature": manifest["original_run_signature"],
              "completed_jobs": {"input": 7, "heads": 24}, "strict_pass_counts": checked["strict_pass_counts"],
              "strict_pass_denominator_per_head": 6, "acceptance_records": checked["acceptance_records"],
              "candidate_recipe": {"input": "I3", "head_architecture": "LayerNorm -> Linear -> GELU -> Dropout -> Linear",
                                   "head_version": "H3", "learning_rate": manifest["config"]["head_stage"]["H3_learning_rate"],
                                   "supported_by_all_six_cases": checked["strict_pass_counts"]["H3"] == 6,
                                   "scope": "Candidate numerical-stability recipe on six deliberately selected failed development cases; not a recognition-performance milestone or independent generalization estimate",
                                   "H1_lower_learning_rate_comparison": "untested"},
              "report_recommendation": checked["report"]["head_aggregate"]["candidate_recommendation"],
              "decision_snapshot_paths": [str(path.relative_to(root)) for path in decisions],
              "old_evidence_reference": {"path": str(old_index.relative_to(root)), "sha256": sha256(old_index),
                                         "cache_dir": manifest["config"]["cache_dir"], "copies_created": False},
              "raw_data_policy": "Raw matrices, annotations and retained numeric data stay at their original paths; no copies or deletions",
              "scope": "All input/head traces, checkpoints, acceptance failures, predictions and plots plus independently derived report frozen in place; one source/config/environment/decision snapshot",
              "frozen_trees": [str(path.relative_to(root)) for path in trees],
              "directories": sorted({str(path.relative_to(root)) for path in directories}), "files": records}
    manifest_path = evidence_dir / "freeze_manifest.json"
    write_json(manifest_path, freeze)
    marker = {"status": "frozen_controlled_experiment", "evidence_manifest": str(manifest_path.relative_to(root)),
              "evidence_manifest_sha256": sha256(manifest_path),
              "instruction": FROZEN_INSTRUCTION}
    marker_paths = [run_dir / "FROZEN.json", report_dir / "FROZEN.json"]
    for path in marker_paths:
        write_json(path, marker)
    _read_only(files + marker_paths + [manifest_path])
    _read_only(sorted(set(directories), key=lambda path: len(path.parts), reverse=True))
    return verify_evidence(root, evidence_dir)


def verify_evidence(root: Path, evidence_dir: Path) -> dict:
    root = root.resolve()
    evidence_dir = inside_root(root, evidence_dir)
    manifest_path = inside_root(root, evidence_dir / "freeze_manifest.json")
    freeze = read_json(manifest_path)
    if freeze.get("status") != "frozen_controlled_experiment":
        raise ValueError("Unexpected freeze index kind")
    errors = []
    expected = {record["path"]: record for record in freeze["files"]}
    if len(expected) != len(freeze["files"]):
        errors.append("Duplicate artifact records")
    for relative, record in expected.items():
        path = inside_root(root, Path(relative))
        if not path.is_file() or path.stat().st_size != record["size_bytes"]:
            errors.append(f"Missing/size changed: {relative}")
        elif sha256(path) != record["sha256"]:
            errors.append(f"Hash changed: {relative}")
        if path.exists() and path.stat().st_mode & 0o222:
            errors.append(f"Write permission restored: {relative}")
    marker_paths = [inside_root(root, Path(freeze[key])) / "FROZEN.json" for key in ("run_dir", "report_dir")]
    expected_marker_paths = {str(path.relative_to(root)) for path in marker_paths}
    discovered = set()
    discovered_directories = {str(evidence_dir.relative_to(root))}
    for relative in freeze["frozen_trees"]:
        files, directories = tree(root, Path(relative))
        discovered.update(str(path.relative_to(root)) for path in files)
        discovered_directories.update(str(path.relative_to(root)) for path in directories)
    if discovered != set(expected) | expected_marker_paths | {str(manifest_path.relative_to(root))}:
        errors.append("Frozen tree inventory changed (unexpected or missing artifact)")
    if discovered_directories != set(freeze["directories"]):
        errors.append("Frozen directory inventory changed")
    for relative in freeze["directories"]:
        path = inside_root(root, Path(relative))
        if not path.is_dir() or path.stat().st_mode & 0o222:
            errors.append(f"Directory missing or writable: {relative}")
    manifest_hash = sha256(manifest_path)
    expected_marker = {"status": freeze["status"], "evidence_manifest": str(manifest_path.relative_to(root)),
                       "evidence_manifest_sha256": manifest_hash, "instruction": FROZEN_INSTRUCTION}
    for path in marker_paths:
        inside_root(root, path)
        if not path.is_file():
            errors.append(f"Missing freeze marker: {path}")
            continue
        marker = read_json(path)
        if marker != expected_marker:
            errors.append(f"Freeze manifest differs from marker: {path}")
        if path.stat().st_mode & 0o222:
            errors.append(f"Freeze marker writable: {path}")
    if manifest_path.stat().st_mode & 0o222:
        errors.append("Freeze manifest writable")
    old_index = inside_root(root, Path(freeze["old_evidence_reference"]["path"]))
    if not old_index.is_file() or sha256(old_index) != freeze["old_evidence_reference"]["sha256"]:
        errors.append("Original failed-evidence reference changed")
    if errors:
        raise ValueError("Frozen controlled evidence verification failed: " + "; ".join(errors))
    return {"status": "verified", "files": len(freeze["files"]),
            "bytes_hashed": sum(record["size_bytes"] for record in freeze["files"]),
            "evidence_manifest": str(manifest_path), "evidence_manifest_sha256": manifest_hash,
            "experiment_signature": freeze["experiment_signature"], "strict_pass_counts": freeze["strict_pass_counts"],
            "scope": freeze["candidate_recipe"]["scope"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-dir", type=Path, default=Path("results/task1_controlled_v1"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/task1_controlled_review_v1"))
    parser.add_argument("--evidence-dir", type=Path, default=Path("evidence/task1_controlled_v1"))
    parser.add_argument("--decision-path", type=Path, action="append", default=[], help="Repeat for the protocol decision JSON/Markdown to snapshot")
    parser.add_argument("--freeze-date", default="2026-10-12")
    args = parser.parse_args()
    if args.command == "verify":
        result = verify_evidence(args.root, args.evidence_dir)
    else:
        result = freeze_evidence(args.root, args.evidence_dir, args.run_dir, args.report_dir, args.decision_path, args.freeze_date)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
