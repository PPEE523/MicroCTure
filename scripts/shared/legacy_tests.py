"""Temporary original-path hard links for testing immutable historical code.

Production imports use microcture.tasks. This context exposes only relocated
source/config files to the old fixture suite, then removes every created link
and directory. It never aliases data, results, reports, documents or indices.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys

from microcture.layout import PROJECT_ROOT, mappings, resolve


def missing_dropout_references(root=PROJECT_ROOT):
    """List absent local ROI evidence; no replacement scientific data is made."""
    root = Path(root).absolute()
    reference = resolve("results/task1_roi_v1", root=root)
    fit = Path("runs/R1/I3/m2plus/seed_42/fold_0")
    required = [Path("manifest.json"), Path("folds/R1/fold_0.json")]
    required.extend(fit / name for name in ("metrics.json", "protocol.json", "ARTIFACTS.json"))
    index = reference / fit / "ARTIFACTS.json"
    if index.is_file():
        for row in json.loads(index.read_text())["files"]:
            if Path(row["name"]).name != row["name"]:
                raise ValueError("Unsafe saved dropout reference artifact name")
            required.append(fit / row["name"])
    return sorted({str((reference / name).relative_to(root))
                   for name in required if not (reference / name).is_file()})


def _safe(path, root):
    if ".." in path.parts or path == root or not path.is_relative_to(root):
        raise ValueError(f"Legacy test path escapes project: {path}")
    for current in (path, *path.parents):
        if current.is_symlink():
            raise ValueError(f"Legacy tests forbid symbolic links: {current}")
    return path


@contextmanager
def legacy_test_paths(root=PROJECT_ROOT):
    """Expose exact historical source bytes for one unit-suite invocation."""
    root = Path(root).absolute()
    files, directories = [], []
    previous_env, previous_flag = os.environ.get("PYTHONDONTWRITEBYTECODE"), sys.dont_write_bytecode
    try:
        os.environ["PYTHONDONTWRITEBYTECODE"], sys.dont_write_bytecode = "1", True
        for row in mappings(root):
            kind, old = row.get("kind"), Path(row["old"])
            if kind not in ("source", "config"):
                continue
            allowed = ("microcture", "scripts") if kind == "source" else ("configs",)
            if not old.parts or old.parts[0] not in allowed:
                raise ValueError(f"Unsupported historical {kind} prefix: {old}")
            source = _safe(root / row["new"], root)
            if not source.exists():
                raise ValueError(f"Missing historical {kind} source: {source}")
            candidates = sorted(source.rglob("*")) if source.is_dir() else [source]
            for current in candidates:
                _safe(current, root)
                if not current.is_file() or "__pycache__" in current.parts:
                    continue
                suffixes = (".py",) if kind == "source" else (".json", ".yml", ".yaml")
                if current.suffix not in suffixes:
                    continue
                relative = current.relative_to(source) if source.is_dir() else Path()
                target = _safe(root / old / relative, root)
                if target.exists():
                    if not target.is_file() or hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(current.read_bytes()).digest():
                        raise FileExistsError(f"Refuse to replace an existing legacy test path: {target}")
                    continue
                missing, parent = [], target.parent
                while not parent.exists():
                    _safe(parent, root)
                    missing.append(parent)
                    parent = parent.parent
                for directory in reversed(missing):
                    directory.mkdir()
                    directories.append(directory)
                os.link(current, target, follow_symlinks=False)
                files.append(target)
        yield
    finally:
        try:
            for path in reversed(files):
                path.unlink(missing_ok=True)
            for path in reversed(directories):
                path.rmdir()
        finally:
            sys.dont_write_bytecode = previous_flag
            if previous_env is None:
                os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
            else:
                os.environ["PYTHONDONTWRITEBYTECODE"] = previous_env


@contextmanager
def legacy_test_fixtures():
    """Adapt reference locations for the immutable dropout fixture module.

    No scientific configuration, assertion, model or global pathlib method is
    replaced. Temporary external fixture roots pass through layout unchanged.
    """
    import unittest
    from unittest.mock import patch
    with ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if name not in ("test_task1_dropout", "tests.test_task1_dropout"):
                continue
            missing = missing_dropout_references(module.ROOT)
            if missing:
                reason = ("Requires local frozen ROI reference evidence; the portable source package "
                          "does not include it. Missing: " + ", ".join(missing))
                stack.enter_context(patch.object(module.DropoutProtocolTests, "setUp",
                                                side_effect=unittest.SkipTest(reason)))
                print("Local-only dropout fixtures skipped: " + reason, file=sys.stderr)
                continue
            experiment = module.E
            project_path = experiment.project_path
            job_directory = experiment.reference_roi.job_directory
            stack.enter_context(patch.object(experiment, "project_path",
                side_effect=lambda root, value, original=project_path: resolve(original(root, value))))
            stack.enter_context(patch.object(experiment.reference_roi, "job_directory",
                side_effect=lambda directory, job, original=job_directory: original(resolve(directory), job)))
            def reference_preprocessing(root, config, fold_number, preprocessing, original=experiment):
                path = resolve(root / config["reference_roi_run_dir"]) / "folds" / "R1" / f"fold_{fold_number}.json"
                original.require(json.loads(path.read_text())["preprocessing"] == preprocessing,
                                 "D5 preprocessing differs from frozen D2 before training")
            stack.enter_context(patch.object(experiment, "assert_reference_preprocessing",
                                            side_effect=reference_preprocessing))
        yield


def main(argv=None):
    """Run discovered unit fixtures with temporary source and location support."""
    import argparse
    import unittest
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modules", nargs="*", help="optional unittest module names")
    args = parser.parse_args(argv)
    with legacy_test_paths():
        loader = unittest.defaultTestLoader
        suite = loader.loadTestsFromNames(args.modules) if args.modules else loader.discover("tests")
        with legacy_test_fixtures():
            result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
