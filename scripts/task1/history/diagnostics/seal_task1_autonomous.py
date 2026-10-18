#!/usr/bin/env python3
"""Seal completed autonomous studies in place, or verify every recorded byte.

No copies of weights, data or plots are made. No runner FROZEN marker is added:
the signed parent protocols also use their output guard during read-only checks.
Instead all inventoried files become read-only and the independent inventory
detects changes, additions and removals in every declared evidence directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEST = ROOT / "evidence/task1_autonomous_v1"
STUDIES = (
    ("task1_roi_v1", "task1_roi_review_v1", "task1_roi", "task1_roi", "completed_jobs", 75, "complete_oof", 15),
    ("task1_native_roi_v2", "task1_native_roi_review_v1", "task1_native_roi_v2", "task1_native_roi_v2", "completed_jobs", 90, "complete_oof", 18),
    ("task1_shape_baseline_v2", "task1_shape_baseline_review_v2", "task1_shape_baseline_v2", "task1_shape_baseline_v2", "completed_fits", 40, "complete_OOF", 8),
    ("task1_dropout_v1", "task1_dropout_review_v1", "task1_dropout_v1", "task1_dropout", "completed_jobs", 60, "complete_oof", 12),
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe(path):
    require(".." not in path.parts, "Parent traversal is forbidden")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlink evidence is forbidden")
    path = path.resolve()
    require(ROOT.resolve() in path.parents and path != ROOT.resolve(), "Evidence must remain inside this project")
    return path


def verify_scientific_tree(run, expected):
    """Recheck seals now; a stale audit status cannot bless changed fit files."""
    manifest = json.loads((run / "manifest.json").read_text())
    require(signature({k: v for k, v in manifest.items() if k != "experiment_signature"}) == manifest["experiment_signature"],
            f"Manifest experiment signature differs: {run}")
    indices = sorted((run / "runs").rglob("ARTIFACTS.json"))
    require(len(indices) == expected, f"Wrong current fit/derived index count: {run}")
    for p in indices:
        index = json.loads(p.read_text())
        require(index["experiment_signature"] == manifest["experiment_signature"] and
                index["signature"] == signature(index["files"]), f"Index signature differs: {p}")
        names = set()
        for row in index["files"]:
            name = row["name"]
            require(Path(name).name == name and name not in names and name != "ARTIFACTS.json", "Unsafe/duplicate indexed name")
            names.add(name)
            actual = safe(p.parent / name)
            require(actual.is_file() and actual.stat().st_size == row["size_bytes"] and digest(actual) == row["sha256"],
                    f"Changed scientific file: {actual}")
        require({v.name for v in p.parent.iterdir()} == names | {"ARTIFACTS.json"}, f"Index contents differ: {p.parent}")
        metrics = json.loads((p.parent / "metrics.json").read_text())
        require(metrics["job"] == index["job"], f"Job identity differs: {p}")
        if "status" in metrics:
            require(metrics["status"] == "completed", f"Incomplete retained fit: {p}")
    return manifest


def selected_files(directories, files):
    chosen = set()
    for relative in directories:
        directory = safe(ROOT / relative)
        require(directory.is_dir(), f"Missing evidence directory: {relative}")
        for p in directory.rglob("*"):
            require(not p.is_symlink(), f"Symlink inside evidence: {p}")
            if p.is_file():
                require(p.name not in (".run.lock", ".training.lock"), f"Active run lock remains: {p}")
                if "__pycache__" in p.parts or p.suffix == ".pyc":
                    continue
                chosen.add(p)
    for relative in files:
        p = safe(ROOT / relative)
        require(p.is_file(), f"Missing evidence file: {relative}")
        chosen.add(p)
    return sorted(chosen)


def collect_plan():
    directories, files, studies = [], [], []
    for run, report, package, entry, fits_key, count, oof_key, oof_count in STUDIES:
        summary = json.loads((ROOT / "results" / run / "summary.json").read_text())
        require(summary["status"] == "complete" and not summary.get("errors"), f"Incomplete study: {run}")
        require(summary[fits_key] == count and summary[oof_key] == oof_count, f"Wrong study coverage: {run}")
        check = json.loads((ROOT / "reports" / report / "integrity.json").read_text())
        require(check["status"] == "passed", f"Independent audit not passed: {run}")
        manifest = verify_scientific_tree(ROOT / "results" / run, count)
        require(json.loads((ROOT / f"configs/{entry}.json").read_text()) == manifest["config"], "Current registered config differs")
        require(manifest["source_signature"] == summary["source_signature"] == check["source_signature"] and
                manifest["experiment_signature"] == summary["experiment_signature"] == check["experiment_signature"], "Audit/run provenance differs")
        sources = sorted((ROOT / "microcture" / package).glob("*.py")) + [ROOT / f"scripts/run_{entry}.py"]
        require(signature({str(p.relative_to(ROOT)): digest(p) for p in sources}) == manifest["source_signature"], "Current trained source changed")
        for dep, expected_signature in manifest.get("dependency_source_signature", {}).items():
            dep_sources = sorted((ROOT / "microcture" / dep).glob("*.py"))
            dep_entry = ROOT / ("scripts/run_task1_comparison.py" if dep == "task1" else f"scripts/run_{dep}.py")
            require(dep_sources and dep_entry.is_file() and signature({str(p.relative_to(ROOT)): digest(p) for p in dep_sources + [dep_entry]}) == expected_signature,
                    f"Frozen dependency source changed: {dep}")
        require((ROOT / "reports" / report / "analysis.md").is_file(), f"Missing analysis: {report}")
        directories.extend([f"results/{run}", f"reports/{report}", f"microcture/{package}"])
        files.extend([f"scripts/run_{entry}.py", f"configs/{entry}.json"])
        studies.append({"run_dir": f"results/{run}", "actual_new_fits": count, "complete_model_OOF": oof_count,
                        "source_signature": summary["source_signature"], "experiment_signature": summary["experiment_signature"]})
    fusion = json.loads((ROOT / "results/task1_fixed_fusion_v1/summary.json").read_text())
    require(fusion["status"] == "complete" and fusion["derived_fold_jobs"] == 80 and fusion["main_OOF"] == 12 and fusion["pooled_OOF"] == 4,
            "Fixed fusion coverage incomplete")
    require(json.loads((ROOT / "reports/task1_fixed_fusion_review_v1/integrity.json").read_text())["status"] == "passed", "Fusion audit not passed")
    fusion_manifest = verify_scientific_tree(ROOT / "results/task1_fixed_fusion_v1", 80)
    fusion_check = json.loads((ROOT / "reports/task1_fixed_fusion_review_v1/integrity.json").read_text())
    require(all(fusion[k] == fusion_manifest[k] == fusion_check[k] for k in ("source_signature", "experiment_signature")), "Fusion provenance differs")
    require(json.loads((ROOT / "configs/task1_fixed_fusion.json").read_text()) == fusion_manifest["config"], "Fusion config differs")
    require(digest(ROOT / "scripts/evaluate_task1_fixed_fusion.py") == fusion_manifest["source_signature"], "Fusion source changed")
    directories.extend(["results/task1_fixed_fusion_v1", "reports/task1_fixed_fusion_review_v1",
                        "reports/task1_autonomous_v1", "reports/task1_autonomous_review_v1", "data/task1_native_roi_v2"])
    files.extend(["scripts/evaluate_task1_fixed_fusion.py", "configs/task1_fixed_fusion.json", "scripts/seal_task1_autonomous.py",
                  "docs/TASK1_AUTONOMOUS_PROTOCOL.md", "environment.yml", "environment.lock.yml", "data/processed/structures.csv",
                  "scripts/probe_native_roi_feasibility.py", "tests/test_native_roi_feasibility.py",
                  "scripts/summarize_task1_case_persistence.py"])
    # Dependency code and pre-fit failures remain evidence, with zero extra fits.
    for package in ("task1", "task1_controlled", "task1_formal", "task1_readout", "task1_reevaluation",
                    "task1_native_roi", "task1_shape_baseline"):
        directories.append(f"microcture/{package}")
        stem = "task1_comparison" if package == "task1" else package
        files.extend([f"scripts/run_{stem}.py", f"configs/{stem}.json"])
    directories.append("reports/task1_shape_baseline_review_v1")
    files.extend(str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("test_task1_*.py"))
    # Supporting audit, plot and diagnosis code is reusable report evidence.
    files.extend(str(p.relative_to(ROOT)) for p in (ROOT / "scripts").glob("*.py")
                 if p.name.startswith(("audit_task1_native_roi", "plot_task1_native", "audit_task1_roi_review", "plot_task1_roi_review",
                                       "report_task1_shape_baseline", "report_task1_fixed_fusion", "audit_task1_dropout", "plot_task1_dropout",
                                       "probe_task1_local_coverage", "summarize_task1_autonomous")))
    return {"directories": sorted(set(directories)), "files": sorted(set(files)), "studies": studies,
            "actual_new_fits": sum(s["actual_new_fits"] for s in studies),
            "complete_new_model_OOF": sum(s["complete_model_OOF"] for s in studies),
            "fixed_fusion": {"actual_new_fits": 0, "derived_fold_records": 80, "main_OOF": 12, "pooled_diagnostic_OOF": 4}}


def inventory(paths):
    return [{"path": str(p.relative_to(ROOT)), "size_bytes": p.stat().st_size, "sha256": digest(p)} for p in paths]


def fresh_audits():
    """Revalidate the entire job matrix, OOF joins and science before sealing.

    These existing audit functions only read scientific inputs and update their
    separate report JSON/CSV. They do not train, preprocess contacts or infer.
    """
    for run, _, _, _, fits_key, count, oof_key, oof_count in STUDIES:
        p = ROOT / "results" / run / "summary.json"
        require(p.is_file(), f"Study has not completed: {run}")
        s = json.loads(p.read_text())
        require(s["status"] == "complete" and s[fits_key] == count and s[oof_key] == oof_count and not s.get("errors"),
                f"Refuse sealing incomplete study: {run}")
    from scripts.audit_task1_roi_review import audit as roi_audit
    from scripts.audit_task1_native_roi import build_report as native_audit
    from scripts.report_task1_shape_baseline import audit as shape_audit
    from scripts.report_task1_fixed_fusion import audit as fusion_audit
    from scripts.audit_task1_dropout import build_report as dropout_audit
    roi_audit(ROOT / "results/task1_roi_v1", ROOT / "reports/task1_roi_review_v1")
    native_audit(ROOT / "results/task1_native_roi_v2", ROOT / "reports/task1_native_roi_review_v1")
    shape_audit(ROOT / "results/task1_shape_baseline_v2", ROOT / "reports/task1_shape_baseline_review_v2")
    fusion_audit()
    dropout_audit(ROOT / "results/task1_dropout_v1", ROOT / "reports/task1_dropout_review_v1")


def verify():
    manifest = json.loads((DEST / "freeze_manifest.json").read_text())
    require(signature({k: v for k, v in manifest.items() if k != "manifest_signature"}) == manifest["manifest_signature"], "Manifest metadata signature differs")
    require(manifest["protocol"] == "task1_autonomous_in_place_evidence_v1" and manifest["actual_new_fits"] == 265 and
            manifest["complete_new_model_OOF"] == 53 and manifest["fixed_fusion"]["actual_new_fits"] == 0, "Unexpected study scope/counts")
    require(not (stat.S_IMODE((DEST / "freeze_manifest.json").stat().st_mode) & 0o222), "Freeze manifest became writable")
    require(signature(manifest["inventory"]) == manifest["inventory_signature"], "Inventory signature differs")
    current = inventory(selected_files(manifest["directories"], manifest["files"]))
    require(current == manifest["inventory"], "Scientific evidence changed, was added or removed")
    require(all(not (stat.S_IMODE((ROOT / r["path"]).stat().st_mode) & 0o222) for r in current), "Some sealed files became writable")
    return {"status": "verified", "files": len(current), "bytes": sum(r["size_bytes"] for r in current),
            "actual_new_fits": manifest["actual_new_fits"], "inventory_signature": manifest["inventory_signature"]}


def seal():
    if (DEST / "freeze_manifest.json").exists():
        return verify()
    require(not DEST.exists(), "Partial evidence destination retained; do not silently overwrite")
    fresh_audits()
    plan = collect_plan()
    paths = selected_files(plan["directories"], plan["files"])
    records = inventory(paths)
    manifest = {"protocol": "task1_autonomous_in_place_evidence_v1", "date": "2026-10-14", **plan,
                "inventory": records, "inventory_signature": signature(records),
                "scope": "Developmental evidence only; no independent biological certification. Raw data and annotations unchanged; no files copied or deleted."}
    manifest["preseal_validation"] = "All five independent matrix/OOF/scientific audits rerun, then every ARTIFACT byte, source/config and dependency signature rechecked before snapshot"
    manifest["manifest_signature"] = signature(manifest)
    DEST.mkdir(parents=True)
    with (DEST / "freeze_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    for p in (*paths, DEST / "freeze_manifest.json"):
        p.chmod(stat.S_IMODE(p.stat().st_mode) & ~0o222)
    return verify()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal", "verify"))
    args = parser.parse_args()
    print(json.dumps(seal() if args.command == "seal" else verify(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
