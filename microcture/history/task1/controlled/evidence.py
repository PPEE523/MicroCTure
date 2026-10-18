"""Freeze evidence in place and preserve the exact original source snapshot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import stat


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_signature(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file())


def _read_only(paths: list[Path]) -> None:
    for path in paths:
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)


def freeze_failed_experiment(root: Path, evidence_dir: Path) -> dict:
    root, evidence_dir = root.resolve(), evidence_dir.resolve()
    manifest_path = evidence_dir / "freeze_manifest.json"
    if manifest_path.exists():
        return verify_evidence(root, evidence_dir)
    if evidence_dir.exists():
        raise ValueError("Evidence destination already exists without a completed freeze; choose a fresh directory")
    run_dir = root / "results/task1_comparison_v1"
    cache_dir = root / "data/cache/task1_comparison_v1"
    original_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or summary.get("errors"):
        raise ValueError("Expected the complete four-model failed comparison, with no evidence errors")
    source_paths = sorted((root / "microcture/task1").glob("*.py"))
    source_paths.append(root / "scripts/run_task1_comparison.py")
    current_source = json_signature({str(path.relative_to(root)): sha256(path) for path in source_paths})
    if current_source != original_manifest["source_signature"]:
        raise ValueError("Current original source differs from the trained manifest; cannot claim an exact source freeze")
    snapshot_paths = source_paths + [root / relative for relative in (
        "configs/task1_comparison.json", "configs/data.json", "environment.yml",
        "environment.lock.yml", "AGENTS.md", "docs/TASK1_COMPARISON.md",
        "data/processed/structures.csv", "scripts/audit_task1_data.py")]
    evidence_dir.mkdir(parents=True)
    try:
        for path in snapshot_paths:
            destination = evidence_dir / "source_snapshot" / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        diagnostic_paths = sorted((root / "reports").glob("task1_chid*"))
        files = [path for path in _files(run_dir) if path.name != "FROZEN.json"]
        files += _files(cache_dir) + diagnostic_paths + _files(evidence_dir / "source_snapshot")
        records = []
        for path in files:
            if path.is_symlink():
                raise ValueError(f"Cannot freeze a symbolic-link artifact: {path}")
            records.append({"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size,
                            "sha256": sha256(path)})
        freeze = {"status": "frozen_failed_experiment", "freeze_date": "2026-10-12",
                  "run_dir": str(run_dir.relative_to(root)), "cache_dir": str(cache_dir.relative_to(root)),
                  "source_signature": current_source, "dataset_signature": original_manifest["dataset_signature"],
                  "run_signature": original_manifest["run_signature"], "files": records,
                  "original_inputs": json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))["input_sha256"],
                  "raw_data": "Raw matrices and original annotations remain at their original paths; no raw-data copies or deletions.",
                  "scope": "50 fits and all OOF/report/checkpoint artifacts frozen in place; cache counts hashed; exact source/config/environment snapshot retained",
                  "interpretation": "Complete execution with unresolved target ambiguity and failed training cells; not an achieved recognition milestone."}
        manifest_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        marker = {"status": "frozen_failed_experiment", "evidence_manifest": str(manifest_path.relative_to(root)),
                  "evidence_manifest_sha256": sha256(manifest_path),
                  "instruction": "Use a new controlled result directory. Do not resume, resummarize, overwrite or delete this failed evidence."}
        (run_dir / "FROZEN.json").write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        _read_only(files + [manifest_path, run_dir / "FROZEN.json"])
        directories = [run_dir, cache_dir, evidence_dir]
        directories += [path for parent in (run_dir, cache_dir, evidence_dir) for path in parent.rglob("*") if path.is_dir()]
        _read_only(directories)
    except BaseException:
        # Never remove source evidence on failure. Incomplete destination is
        # explicit and will not be silently reused on a later invocation.
        raise
    return verify_evidence(root, evidence_dir)


def verify_evidence(root: Path, evidence_dir: Path) -> dict:
    root, evidence_dir = root.resolve(), evidence_dir.resolve()
    manifest_path = evidence_dir / "freeze_manifest.json"
    freeze = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    for record in freeze["files"]:
        path = root / record["path"]
        if not path.is_file() or path.stat().st_size != record["size_bytes"]:
            errors.append(f"Missing/size changed: {record['path']}")
        elif sha256(path) != record["sha256"]:
            errors.append(f"Hash changed: {record['path']}")
    marker_path = root / freeze["run_dir"] / "FROZEN.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker["evidence_manifest_sha256"] != sha256(manifest_path):
        errors.append("Freeze manifest differs from the marker")
    if errors:
        raise ValueError("Frozen evidence verification failed: " + "; ".join(errors))
    return {"status": "verified", "files": len(freeze["files"]),
            "bytes_hashed": sum(record["size_bytes"] for record in freeze["files"]),
            "evidence_manifest": str(manifest_path), "evidence_manifest_sha256": sha256(manifest_path),
            "run_signature": freeze["run_signature"], "dataset_signature": freeze["dataset_signature"]}
