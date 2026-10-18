"""Build or verify the six-task delivery from saved results; never fit or scan contacts."""
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
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import sys
import time

ROOT = _layout_root
BUILDERS = ("build_task1_delivery.py", "build_task2_delivery.py",
            "build_task34_delivery.py", "build_task56_delivery.py")
SUMMARIES = ("task1_delivery_summary.json", "task2_delivery_summary.json",
             "task34_delivery_summary.json", "task56_delivery_summary.json")
MANIFEST = ROOT / "submission/results_delivery_manifest.json"
REPORTS = tuple(f"docs/TASK{number}.md" for number in range(1, 7))
TASK2_REVIEW = "submission/task2/verification/v2"
TASK2_VALIDATION = "submission/task2/validation/frozen21_v1"
TASK3_ATLAS = "submission/task3/atlas/v1"
TASK5_ANALYSIS = "submission/task5/paired_analysis/v1"


def task5_analysis_extension():
    directory = ROOT / TASK5_ANALYSIS
    if not directory.exists():
        return None
    from microcture.tasks.task5.analysis import verify
    verify(directory)
    return {"directory": TASK5_ANALYSIS, "figures": [TASK5_ANALYSIS + "/" + name for name in
            ("paired_improvement.png", "distance_stratification.png", "final_summary.png")]}


def task3_atlas_extension():
    directory = ROOT / TASK3_ATLAS
    if not directory.exists():
        return None
    from microcture.tasks.task3.atlas import verify
    verify(directory)
    return {"directory": TASK3_ATLAS, "figures": [TASK3_ATLAS + "/" + name for name in
            ("whole_genome.png", "representative_region.png")]}


def task2_extension():
    directory = ROOT / TASK2_REVIEW
    if not directory.exists():
        return None
    from microcture.tasks.task2.delivery import verify_snapshot
    verify_snapshot(directory)
    return {"directory": TASK2_REVIEW, "figures": [TASK2_REVIEW + "/model_diagnostics.png"]}


def task2_validation_extension():
    directory=ROOT/TASK2_VALIDATION
    if not directory.exists():
        return None
    from microcture.tasks.task2.validation_delivery import verify_snapshot
    verify_snapshot(directory)
    return {"directory":TASK2_VALIDATION,"figures":[TASK2_VALIDATION+'/'+name for name in
            ('candidate_support.png','known_retrieval.png','morphology_status.png')]}


def local_path(name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Invalid local path: {name}")
    path = _layout_resolve(ROOT.joinpath(*relative.parts))
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError(f"External path: {name}")
    return path


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_records():
    one, two, three, five = [read_json(ROOT / "submission" / name) for name in SUMMARIES]
    records = list(one["sources"])
    records += [{"path": name, **record} for name, record in two["source_files"].items()]
    records += list(three["inputs"].values()) + [three["source"]]
    records += five["source_files"]
    if task2_extension():
        records += read_json(ROOT / TASK2_REVIEW / "review.json")["sources"]
    if task2_validation_extension():
        records += read_json(ROOT/TASK2_VALIDATION/'cohort_frozen.json')['input_files']
    if task3_atlas_extension():
        records += read_json(ROOT/TASK3_ATLAS/'sources.json')['files']
    if task5_analysis_extension():
        records += read_json(ROOT/TASK5_ANALYSIS/'sources.json')['files']
    merged = {}
    for row in records:
        name = row["path"]
        if name in merged and merged[name] != row["sha256"]:
            raise ValueError(f"Conflicting saved source hashes: {name}")
        merged[name] = row["sha256"]
    return [{"path": name, "sha256": value} for name, value in sorted(merged.items())]


def verify(require_local_inputs=False):
    manifest = read_json(MANIFEST)
    errors = []
    for row in manifest["files"]:
        path = local_path(row["path"])
        if not path.is_file() or sha256(path) != row["sha256"]:
            errors.append(f"Missing or changed delivery file: {row['path']}")
        elif path.stat().st_size != row["bytes"]:
            errors.append(f"Incorrect delivery size: {row['path']}")
        if path.suffix == ".png" and path.is_file():
            with path.open("rb") as stream:
                header = stream.read(24)
            if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or min(struct.unpack(">II", header[16:24])) < 256:
                errors.append(f"Invalid or tiny PNG: {row['path']}")
    checked, absent = 0, []
    for row in manifest["saved_inputs"]:
        path = local_path(row["path"])
        if not path.is_file():
            absent.append(row["path"])
            if require_local_inputs:
                errors.append(f"Missing local recomputation input: {row['path']}")
        elif sha256(path) != row["sha256"]:
            errors.append(f"Changed recomputation input: {row['path']}")
        else:
            checked += 1
    reports = manifest.get("reports", [manifest["report"]] if "report" in manifest else [])
    if manifest.get("schema_version") == 2 and tuple(reports) != REPORTS:
        errors.append("The delivery must reference all six canonical task documents")
    images = []
    indexed = {row["path"] for row in manifest["files"]}
    for name in reports:
        document = local_path(name)
        if not document.is_file():
            errors.append(f"Missing task document: {name}")
            continue
        targets = re.findall(r"!\[[^\]]*\]\(([^\n)]+)\)", document.read_text())
        if not targets:
            errors.append(f"Task document has no result figures: {name}")
        images.extend(targets)
        for target in targets:
            dest = (document.parent / target).resolve()
            if not dest.is_relative_to(ROOT) or not dest.is_file() or str(dest.relative_to(ROOT)) not in indexed:
                errors.append(f"Unindexed or missing report image: {name} -> {target}")
    extension = manifest.get("extensions", {}).get("task2_review")
    validation = manifest.get("extensions", {}).get("task2_validation")
    atlas = manifest.get("extensions", {}).get("task3_atlas")
    task5_analysis = manifest.get("extensions", {}).get("task5_analysis")
    expected_images = 12 + (len(extension["figures"]) if extension else 0) + (len(validation['figures']) if validation else 0)
    if atlas:
        from microcture.tasks.task3.atlas import verify as verify_atlas
        verify_atlas(local_path(atlas["directory"]), require_local_inputs=require_local_inputs)
        expected_images += len(atlas["figures"]) - 1  # replaces the old combined task-3 image
    if task5_analysis:
        from microcture.tasks.task5.analysis import verify as verify_task5_analysis
        verify_task5_analysis(local_path(task5_analysis['directory']),root=ROOT,require_local_inputs=require_local_inputs)
        expected_images += len(task5_analysis['figures'])
    if extension:
        from microcture.tasks.task2.delivery import verify_snapshot
        verify_snapshot(local_path(extension["directory"]), require_local_inputs=require_local_inputs)
    if validation:
        from microcture.tasks.task2.validation_delivery import verify_snapshot as verify_validation_snapshot
        verify_validation_snapshot(local_path(validation['directory']),root=ROOT,require_local_inputs=require_local_inputs)
    if len(images) != expected_images:
        errors.append(f"Expected {expected_images} indexed report figures, found {len(images)}")
    if errors:
        raise SystemExit("\n".join(errors))
    result = {"status": "verified", "tasks": 6, "new_figures": manifest["new_figures"],
              "report_figures": len(images), "delivery_files": len(manifest["files"]),
              "saved_input_hashes_checked": checked, "local_inputs_absent": len(absent),
              "training_performed": False, "raw_contact_scan_performed": False}
    print(json.dumps(result, ensure_ascii=False))
    return result


def protected_indices():
    protected = (ROOT / "submission/manifest.json", *sorted((ROOT / "evidence").glob("*/freeze_manifest.json")))
    return {str(p.relative_to(ROOT)): sha256(p) for p in protected}


def index_delivery(runs, before, *, action):
    for name, expected in before.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Protected evidence index changed: {name}")
    inputs = source_records()
    for row in inputs:
        path = local_path(row["path"])
        if path.exists() and sha256(path) != row["sha256"]:
            raise ValueError(f"Saved recomputation input changed: {row['path']}")
    paths = {ROOT / "README.md", ROOT / "docs/README.md", Path(__file__), ROOT / "run.py",
             _layout_resolve(ROOT / "scripts/check_submission_evidence.py"), _layout_resolve(ROOT / "scripts/review_saved_task56.py"),
             ROOT / "microcture/layout.py", ROOT / "configs/layout.json"}
    paths.update(ROOT / name for name in REPORTS)
    paths.update(_layout_resolve(ROOT / "scripts" / name) for name in BUILDERS)
    paths.update(ROOT / "submission" / name for name in SUMMARIES)
    figures = sorted((ROOT / "submission/figures").glob("*delivery*.png"))
    if len(figures) != 11:
        raise ValueError(f"Expected 11 generated figures, found {len(figures)}")
    paths.update(figures)
    paths.add(ROOT / "showcase/results/task2_morphology_audit/comparisons/w64_c43040.png")
    extension = task2_extension()
    if extension:
        paths.update((ROOT / TASK2_REVIEW).iterdir())
        paths.update(ROOT / name for name in (
            "scripts/task2/review_task2.py", "microcture/tasks/task2/review.py",
            "microcture/tasks/task2/diagnostics.py", "microcture/tasks/task2/delivery.py", "site/index.html"))
    validation = task2_validation_extension()
    if validation:
        paths.update((ROOT/TASK2_VALIDATION).iterdir())
        paths.update(ROOT/name for name in ('microcture/tasks/task2/validation.py','microcture/tasks/task2/validation_delivery.py',
                     'scripts/task2/validate_candidates.py','scripts/task2/publish_validation.py',
                     'configs/task2/verification/candidate_evidence_v1.json'))
    atlas = task3_atlas_extension()
    if atlas:
        paths.update(p for p in (ROOT / TASK3_ATLAS).rglob("*") if p.is_file())
        paths.update(ROOT / name for name in ("microcture/tasks/task3/atlas.py",
                     "microcture/tasks/task3/assets/atlas.js", "microcture/tasks/task3/assets/atlas.css",
                     "scripts/task3/build_atlas.py", "configs/task3/atlas/atlas_v1.json",
                     "tests/test_task3_atlas.py"))
    task5_analysis = task5_analysis_extension()
    if task5_analysis:
        paths.update(p for p in (ROOT/TASK5_ANALYSIS).rglob('*') if p.is_file())
        paths.update(ROOT/name for name in ('microcture/tasks/task5/analysis.py','microcture/tasks/task5/__init__.py',
                     'scripts/task5/analyze_saved.py','configs/task5/paired_analysis/paired_v1.json',
                     'tests/test_task5_analysis.py'))
    extensions={}
    if extension:extensions['task2_review']=extension
    if validation:extensions['task2_validation']=validation
    if atlas:extensions['task3_atlas']=atlas
    if task5_analysis:extensions['task5_analysis']=task5_analysis
    value = {"schema_version": 2, "date": "2026-10-16" if validation else "2026-10-15", "tasks": 6,
             "reports": list(REPORTS), "navigation": "README.md", "new_figures": 11 + (1 if extension else 0) + (3 if validation else 0) + (2 if atlas else 0) + (3 if task5_analysis else 0),
             "extensions": extensions,
             "training_performed": False, "raw_contact_scan_performed": False,
             "scientific_interpretation": "Retained numerical experiments plus new descriptive reanalysis; not all biological targets attained",
             "last_action": action, "builds": runs, "protected_evidence_indices": before,
             "saved_inputs": inputs,
             "files": [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                        "sha256": sha256(p)} for p in sorted(paths)]}
    MANIFEST.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    verify(require_local_inputs=action == "build")


def build():
    before = protected_indices()
    runs = []
    for name in BUILDERS:
        started = time.monotonic()
        subprocess.run([sys.executable, "-B", str(_layout_resolve(ROOT / "scripts" / name))], cwd=ROOT, check=True)
        runs.append({"script": "scripts/" + name, "returncode": 0,
                     "seconds": round(time.monotonic() - started, 3)})
    index_delivery(runs, before, action="build")


def refresh():
    """Refresh documentation/code hashes while requiring unchanged saved results."""
    previous = read_json(MANIFEST)
    for row in previous["files"]:
        if row["path"].startswith(("submission/", "showcase/")):
            path = local_path(row["path"])
            if not path.is_file() or sha256(path) != row["sha256"]:
                raise ValueError(f"Cannot refresh changed numerical evidence or figure: {row['path']}; rebuild explicitly")
    before = previous["protected_evidence_indices"]
    index_delivery(previous["builds"], before, action="documentation_index_refresh")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify", "refresh"))
    parser.add_argument("--require-local-inputs", action="store_true",
                        help="verify: require retained local arrays as well as portable delivery files")
    args = parser.parse_args()
    if args.action == "build":
        build()
    elif args.action == "refresh":
        refresh()
    else:
        verify(args.require_local_inputs)


if __name__ == "__main__":
    main()
