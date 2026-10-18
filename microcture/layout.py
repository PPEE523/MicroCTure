"""Resolve historical evidence names to the organized project layout.

Original manifests keep their original paths and signatures. Only this explicit
mapping changes physical locations; scientific bytes are never rewritten.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def mappings(root=PROJECT_ROOT):
    path = Path(root) / "configs/layout.json"
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("moves", [])
    old, new = set(), set()
    for row in rows:
        for key in ("old", "new"):
            item = Path(row[key])
            if item.is_absolute() or ".." in item.parts or not item.parts:
                raise ValueError(f"Unsafe layout path: {row[key]}")
        if row["old"] in old or row["new"] in new:
            raise ValueError("Duplicate layout source or destination")
        old.add(row["old"])
        new.add(row["new"])
    return rows


def _relative(path, root):
    path, root = Path(path), Path(os.path.abspath(root))
    if not path.is_absolute():
        if ".." in path.parts:
            raise ValueError("Parent traversal is forbidden in project paths")
        return path, root
    path = Path(os.path.normpath(path))
    try:
        return path.relative_to(root), root
    except ValueError:
        return None, path


def resolve(path, value=None, *, root=PROJECT_ROOT):
    """Accept resolve(path, root=...) or resolve(root, path); preserve symlink information."""
    if value is not None:
        root, path = path, value
    relative, base = _relative(path, root)
    if relative is None:
        return base
    rows = mappings(base)
    for row in sorted(rows, key=lambda r: len(Path(r["old"]).parts), reverse=True):
        old = Path(row["old"])
        if relative == old or old in relative.parents:
            return base / row["new"] / relative.relative_to(old)
    # Old task-two-to-six code used results/taskN while the retained inputs
    # were under data/legacy_experiments/results/taskN. Recognize that source
    # spelling without redirecting a new task/experiment output directory.
    if relative.parts and relative.parts[0] == "results":
        if any(relative == Path(r["new"]) or Path(r["new"]) in relative.parents for r in rows):
            return base / relative
        historical = Path("data/legacy_experiments") / relative
        for row in sorted(rows, key=lambda r: len(Path(r["old"]).parts), reverse=True):
            old = Path(row["old"])
            if historical == old or old in historical.parents:
                suffix = historical.relative_to(old)
                target = base / row["new"]
                if not suffix.parts or (target / suffix.parts[0]).exists():
                    return target / suffix
    return base / relative


def logical(path, *, root=PROJECT_ROOT):
    """Return a project-relative original name for migrated evidence."""
    relative, base = _relative(path, root)
    if relative is None:
        raise ValueError("Evidence lies outside the project")
    for row in sorted(mappings(base), key=lambda r: len(Path(r["new"]).parts), reverse=True):
        new = Path(row["new"])
        if relative == new or new in relative.parents:
            return Path(row["old"]) / relative.relative_to(new)
    return relative


def bootstrap():
    """Expose project and organized script libraries for direct CLI execution."""
    import sys
    paths = [PROJECT_ROOT]
    paths.extend(p for p in (PROJECT_ROOT / "scripts").rglob("*") if p.is_dir() and "__pycache__" not in p.parts)
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
