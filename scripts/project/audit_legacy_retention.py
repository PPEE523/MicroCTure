#!/usr/bin/env python3
"""Check Git main's published snapshot and retained task2–6 numerical bytes.

Reads Git objects and preserved files. Does not check out code, import training
modules, alter old results, restore checkpoints or execute scientific pipelines.
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
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = _layout_root
OUT = _layout_resolve(ROOT / "reports/legacy_retention_review_v1")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    revision = git("rev-parse", "origin/main").decode().strip()
    cleanup = json.loads((ROOT / "report_materials/cleanup_manifest.json").read_text())
    mapping = {row["original_path"]: row for row in cleanup["files"]}
    public_manifest = json.loads(git("show", f"{revision}:showcase/results/manifest.json"))
    inventory = []
    for row in public_manifest["files"]:
        blob = git("show", f"{revision}:showcase/results/{row['file']}")
        if len(blob) != row["bytes"] or hashlib.sha256(blob).hexdigest() != row["sha256"]:
            raise ValueError(f"Published snapshot differs: {row['file']}")
        old = mapping.get(row["source"], {})
        local = old.get("retained_path")
        status = "not_retained_locally; available_in_Git"
        if local and _layout_resolve(ROOT / local).is_file():
            if sha(_layout_resolve(ROOT / local)) != row["sha256"]:
                raise ValueError(f"Retained public output differs: {local}")
            status = "Git_and_retained_file_exact"
        inventory.append({"scope": "published_snapshot", "original_path": row["source"], "retained_path": local or "",
                          "size_bytes": len(blob), "sha256": row["sha256"], "status": status})
    numerical = [r for r in cleanup["files"] if re.match(r"^results/task[2-6](?:/|_)", r["original_path"])
                 and r.get("retained_path") and Path(r["retained_path"]).suffix.lower() in {".csv", ".json", ".npz"}]
    for row in numerical:
        path = _layout_resolve(ROOT / row["retained_path"])
        if ROOT not in path.resolve().parents or path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing/unsafe retained numerical file: {path}")
        if path.stat().st_size != row["size_bytes"] or sha(path) != row["sha256"]:
            raise ValueError(f"Retained numerical bytes changed: {path}")
        inventory.append({"scope": "retained_numerical", "original_path": row["original_path"],
                          "retained_path": row["retained_path"], "size_bytes": row["size_bytes"], "sha256": row["sha256"],
                          "status": "exact_cleanup_snapshot; historical_failed_versions_are_not_final_results"})
    summary = {
        "status": "passed", "review_date": "2026-10-14", "remote": git("remote", "get-url", "origin").decode().strip(),
        "remote_main": revision, "local_main": git("rev-parse", "main").decode().strip(),
        "current_branch": git("branch", "--show-current").decode().strip(), "current_HEAD": git("rev-parse", "HEAD").decode().strip(),
        "worktree_porcelain_entries": len(git("status", "--porcelain").decode().splitlines()),
        "published_snapshot_files": len(public_manifest["files"]), "published_snapshot_bytes": public_manifest["total_bytes"],
        "published_also_preserved_locally": sum(r["status"] == "Git_and_retained_file_exact" for r in inventory),
        "retained_non_task1_numerical_files": len(numerical), "retained_non_task1_numerical_bytes": sum(r["size_bytes"] for r in numerical),
        "original_cleanup_manifest_sha256": sha(ROOT / "report_materials/cleanup_manifest.json"),
        "audit_script_sha256": sha(Path(__file__)),
        "scope": "Byte/provenance checks, not biological validation. Includes historical superseded/failed numerical files only as preservation evidence. No raw genome scan, training, checkpoint restoration or branch checkout.",
        "scientific_review": "See analysis.md for independent saved-array/CSV recalculations and claim-specific limitations; old verification passed fields remain historical."
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "preservation_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0])); writer.writeheader(); writer.writerows(inventory)
    (OUT / "audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
