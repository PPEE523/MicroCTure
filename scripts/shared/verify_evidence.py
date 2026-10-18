"""Verify original frozen bytes after an explicit project-layout relocation.

Old indices remain unchanged. Paths in those indices are logical provenance
names; microcture.layout resolves their current physical locations. No training,
report regeneration, chmod, index rewriting or evidence mutation is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from microcture.layout import PROJECT_ROOT, logical, resolve


INDICES = {
    "failed": "evidence/task1_failed_v1/freeze_manifest.json",
    "controlled": "evidence/task1_controlled_v1/freeze_manifest.json",
    "autonomous": "evidence/task1_autonomous_v1/freeze_manifest.json",
}
CONTROLLED_INSTRUCTION = (
    "Use independent directories for formal experiments and future reports; "
    "never resume, overwrite or delete this controlled evidence"
)
FAILED_INSTRUCTION = (
    "Use a new controlled result directory. Do not resume, resummarize, overwrite "
    "or delete this failed evidence."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def relative_name(value):
    name = str(value)
    path = PurePosixPath(name)
    require(bool(name) and not path.is_absolute() and ".." not in path.parts
            and "\\" not in name and path.parts, f"Unsafe logical path: {name}")
    require(path.as_posix() == name, f"Noncanonical logical path: {name}")
    return name


def safe_physical(path, root):
    """Reject aliases before resolving an on-disk path or opening its bytes."""
    path, root = Path(path), Path(root)
    require(path.is_absolute(), f"Physical evidence path must be absolute: {path}")
    require(".." not in path.parts, f"Parent traversal is forbidden: {path}")
    require(path != root and path.is_relative_to(root), f"Evidence escapes project root: {path}")
    current = path
    while True:
        require(not current.is_symlink(), f"Symlink evidence is forbidden: {current}")
        if current.parent == current:
            break
        current = current.parent
    require(path.resolve().is_relative_to(root.resolve()), f"Evidence escapes project root: {path}")
    return path


def path_for(name, root):
    return safe_physical(resolve(relative_name(name), root=root), root)


def name_for(path, root):
    value = Path(str(logical(path, root=root)))
    if value.is_absolute():
        value = value.relative_to(root)
    return relative_name(value.as_posix())


def readonly(path):
    require(not (stat.S_IMODE(path.stat().st_mode) & 0o222),
            f"Frozen evidence became writable: {path}")


def read_index(name, root):
    path = path_for(name, root)
    require(path.is_file(), f"Missing freeze manifest: {name}")
    readonly(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Invalid freeze manifest: {name}")
    return path, value


def verify_records(records, root):
    require(isinstance(records, list), "Frozen inventory must be a list")
    expected, physical = {}, set()
    for row in records:
        name = relative_name(row["path"])
        require(name not in expected, f"Duplicate frozen inventory path: {name}")
        path = path_for(name, root)
        require(path not in physical, f"Multiple logical files resolve to one physical artifact: {path}")
        expected[name] = row
        physical.add(path)
        require(path.is_file() and path.stat().st_size == row["size_bytes"],
                f"Frozen file missing or size changed: {name}")
        require(digest(path) == row["sha256"], f"Frozen file hash changed: {name}")
        readonly(path)
        require(name_for(path, root) == name, f"Layout mapping does not round-trip: {name}")
    return expected


def tree(name, root, *, ignore_bytecode=False):
    directory = path_for(name, root)
    require(directory.is_dir(), f"Missing frozen directory: {name}")
    files, directories = set(), {relative_name(name)}
    for path in sorted(directory.rglob("*")):
        safe_physical(path, root)
        if ignore_bytecode and ("__pycache__" in path.parts or path.suffix == ".pyc"):
            continue
        current_name = name_for(path, root)
        require(path_for(current_name, root) == path, f"Ambiguous reverse layout mapping: {path}")
        if path.is_file():
            files.add(current_name)
        elif path.is_dir():
            directories.add(current_name)
        else:
            raise ValueError(f"Unsupported frozen artifact type: {path}")
    return files, directories


def verify_marker(name, index_name, manifest_hash, status, instruction, root):
    path = path_for(name, root)
    require(path.is_file(), f"Missing freeze marker: {name}")
    readonly(path)
    expected = {"status": status, "evidence_manifest": index_name,
                "evidence_manifest_sha256": manifest_hash, "instruction": instruction}
    require(json.loads(path.read_text(encoding="utf-8")) == expected,
            f"Freeze marker differs from original manifest: {name}")


def outcome(kind, manifest_path, records):
    return {"status": "verified", "kind": kind, "files": len(records),
            "bytes_hashed": sum(row["size_bytes"] for row in records),
            "original_manifest_sha256": digest(manifest_path),
            "original_logical_paths_preserved": True, "training_launched": False}


def verify_failed(root=PROJECT_ROOT):
    root = Path(root).absolute()
    index_name = INDICES["failed"]
    manifest_path, freeze = read_index(index_name, root)
    require(freeze.get("status") == "frozen_failed_experiment", "Unexpected failed freeze status")
    expected = verify_records(freeze["files"], root)
    marker = relative_name(freeze["run_dir"]) + "/FROZEN.json"
    verify_marker(marker, index_name, digest(manifest_path), freeze["status"], FAILED_INSTRUCTION, root)
    # The original verifier checked every indexed file. Also reject additions
    # in the three fully frozen trees without declaring diagnostic siblings
    # outside those trees to be part of this historical inventory.
    roots = [freeze["run_dir"], freeze["cache_dir"], str(PurePosixPath(index_name).parent)]
    actual, directories = set(), set()
    for name in roots:
        files, dirs = tree(name, root)
        actual.update(files)
        directories.update(dirs)
    indexed_in_trees = {name for name in expected
                        if any(PurePosixPath(name).is_relative_to(PurePosixPath(prefix)) for prefix in roots)}
    require(actual == indexed_in_trees | {marker, index_name},
            "Frozen failed tree inventory changed (added or missing artifact)")
    for name in directories:
        readonly(path_for(name, root))
    return outcome("failed", manifest_path, freeze["files"])


def verify_controlled(root=PROJECT_ROOT):
    root = Path(root).absolute()
    index_name = INDICES["controlled"]
    manifest_path, freeze = read_index(index_name, root)
    require(freeze.get("status") == "frozen_controlled_experiment", "Unexpected controlled freeze status")
    expected = verify_records(freeze["files"], root)
    markers = {relative_name(freeze[key]) + "/FROZEN.json" for key in ("run_dir", "report_dir")}
    actual, directories = set(), {str(PurePosixPath(index_name).parent)}
    for name in freeze["frozen_trees"]:
        files, dirs = tree(name, root)
        actual.update(files)
        directories.update(dirs)
    require(actual == set(expected) | markers | {index_name},
            "Frozen controlled tree inventory changed (added or missing artifact)")
    require(directories == set(freeze["directories"]), "Frozen controlled directory inventory changed")
    for name in directories:
        readonly(path_for(name, root))
    for name in markers:
        verify_marker(name, index_name, digest(manifest_path), freeze["status"], CONTROLLED_INSTRUCTION, root)
    old = freeze["old_evidence_reference"]
    previous_path = path_for(old["path"], root)
    require(previous_path.is_file() and digest(previous_path) == old["sha256"],
            "Original failed-evidence reference changed")
    return outcome("controlled", manifest_path, freeze["files"])


def verify_autonomous(root=PROJECT_ROOT):
    root = Path(root).absolute()
    index_name = INDICES["autonomous"]
    manifest_path, freeze = read_index(index_name, root)
    unsigned = {key: value for key, value in freeze.items() if key != "manifest_signature"}
    require(signature(unsigned) == freeze["manifest_signature"], "Freeze manifest metadata signature differs")
    require(freeze["protocol"] == "task1_autonomous_in_place_evidence_v1"
            and freeze["actual_new_fits"] == 265 and freeze["complete_new_model_OOF"] == 53
            and freeze["fixed_fusion"]["actual_new_fits"] == 0, "Unexpected autonomous scope/counts")
    require(signature(freeze["inventory"]) == freeze["inventory_signature"], "Inventory signature differs")
    expected = verify_records(freeze["inventory"], root)
    actual = set()
    for name in freeze["directories"]:
        files, _ = tree(name, root, ignore_bytecode=True)
        actual.update(files)
    for name in freeze["files"]:
        path = path_for(name, root)
        require(path.is_file(), f"Missing explicitly frozen file: {name}")
        actual.add(relative_name(name))
    require(actual == set(expected), "Frozen autonomous tree inventory changed (added or missing artifact)")
    require(not any(PurePosixPath(name).name in (".run.lock", ".training.lock") for name in actual),
            "Active run lock remains in frozen evidence")
    result = outcome("autonomous", manifest_path, freeze["inventory"])
    result.update(actual_new_fits=freeze["actual_new_fits"], inventory_signature=freeze["inventory_signature"])
    return result


def verify_all(root=PROJECT_ROOT):
    checks = [verify_failed(root), verify_controlled(root), verify_autonomous(root)]
    return {"status": "verified", "checks": checks, "training_launched": False,
            "indices_rewritten": False, "symlinks_used": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="verify all three original indices (default)")
    group.add_argument("--kind", choices=tuple(INDICES))
    args = parser.parse_args(argv)
    function = {"failed": verify_failed, "controlled": verify_controlled,
                "autonomous": verify_autonomous}.get(args.kind, verify_all)
    print(json.dumps(function(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
