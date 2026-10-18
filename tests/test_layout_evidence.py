"""Relocated evidence fixtures; never alter retained scientific project files."""
from __future__ import annotations

import json
import os
import hashlib
import contextlib
import io
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.shared import verify_evidence as verifier
from scripts.shared import legacy_tests
from scripts.shared.legacy_tests import legacy_test_paths, missing_dropout_references


class RelocatedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="microcture-layout-evidence-", dir="/tmp")
        self.root = Path(self.temp.name)
        self.mapping = {}
        self.addCleanup(patch.stopall)
        patch.object(verifier, "resolve", self.resolve).start()
        patch.object(verifier, "logical", self.logical).start()

    def tearDown(self):
        for path in [self.root, *self.root.rglob("*")]:
            if not path.is_symlink():
                path.chmod(stat.S_IMODE(path.stat().st_mode) | (0o700 if path.is_dir() else 0o600))
        self.temp.cleanup()

    def resolve(self, name, *, root):
        path = Path(name)
        if path.is_absolute():
            path = path.relative_to(root)
        for old, new in sorted(self.mapping.items(), key=lambda pair: len(pair[0]), reverse=True):
            if path.is_relative_to(old):
                return Path(root) / new / path.relative_to(old)
        return Path(root) / path

    def logical(self, path, *, root):
        relative = Path(path).relative_to(root)
        for old, new in sorted(self.mapping.items(), key=lambda pair: len(pair[1]), reverse=True):
            if relative.is_relative_to(new):
                return (Path(old) / relative.relative_to(new)).as_posix()
        return relative.as_posix()

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content) + "\n" if isinstance(content, dict) else content)
        return path

    def record(self, name):
        path = self.root / name
        return {"path": name, "size_bytes": path.stat().st_size, "sha256": verifier.digest(path)}

    def freeze(self, *names, directories=False):
        for name in names:
            path = self.root / name
            candidates = [path, *path.rglob("*")] if path.is_dir() else [path]
            for item in candidates:
                if item.is_file() or directories:
                    item.chmod(stat.S_IMODE(item.stat().st_mode) & ~0o222)

    def relocate(self, old, new):
        destination = self.root / new
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = self.root / old
        original_mode = stat.S_IMODE(source.stat().st_mode)
        if source.is_dir():
            source.chmod(original_mode | 0o200)
        source.rename(destination)
        destination.chmod(original_mode)
        self.mapping[old] = new

    def autonomous(self):
        run = "results/task1_roi_v1"
        self.write(run + "/metrics.json", '{"status":"complete"}\n')
        self.write("scripts/seal_task1_autonomous.py", "# original immutable seal\n")
        rows = [self.record(run + "/metrics.json"), self.record("scripts/seal_task1_autonomous.py")]
        value = {"protocol": "task1_autonomous_in_place_evidence_v1", "actual_new_fits": 265,
                 "complete_new_model_OOF": 53, "fixed_fusion": {"actual_new_fits": 0},
                 "directories": [run], "files": ["scripts/seal_task1_autonomous.py"],
                 "inventory": rows, "inventory_signature": verifier.signature(rows)}
        value["manifest_signature"] = verifier.signature(value)
        self.write(verifier.INDICES["autonomous"], value)
        self.freeze(run, "scripts/seal_task1_autonomous.py", verifier.INDICES["autonomous"])
        return run

    def controlled(self):
        run, report = "results/task1_controlled_v1", "reports/task1_controlled_review_v1"
        evidence = "evidence/task1_controlled_v1"
        self.write(run + "/metrics.json", "{\"status\":\"complete\"}\n")
        self.write(report + "/summary.json", "{\"status\":\"passed\"}\n")
        self.write(evidence + "/source_snapshot/config.json", "{}\n")
        old_index = verifier.INDICES["failed"]
        self.write(old_index, {"fixture": "prior reference"})
        rows = [self.record(name) for name in (run + "/metrics.json", report + "/summary.json",
                                             evidence + "/source_snapshot/config.json")]
        names = [run, report, evidence]
        directories = sorted({str(path.relative_to(self.root))
                              for name in names for path in [self.root / name, *(self.root / name).rglob("*")]
                              if path.is_dir()})
        value = {"status": "frozen_controlled_experiment", "run_dir": run, "report_dir": report,
                 "frozen_trees": names, "directories": directories, "files": rows,
                 "old_evidence_reference": {"path": old_index, "sha256": verifier.digest(self.root / old_index)}}
        index_name = verifier.INDICES["controlled"]
        path = self.write(index_name, value)
        marker = {"status": value["status"], "evidence_manifest": index_name,
                  "evidence_manifest_sha256": verifier.digest(path), "instruction": verifier.CONTROLLED_INSTRUCTION}
        self.write(run + "/FROZEN.json", marker)
        self.write(report + "/FROZEN.json", marker)
        self.freeze(*names, directories=True)
        return run, report

    def test_relocation_preserves_old_manifest_bytes_and_seal_source(self):
        run = self.autonomous()
        index = self.root / verifier.INDICES["autonomous"]
        before = index.read_bytes()
        self.relocate(run, "results/task1/ablation/roi_v1")
        self.relocate("scripts/seal_task1_autonomous.py", "scripts/task1/history/seal_task1_autonomous.py")
        self.assertEqual(verifier.verify_autonomous(self.root)["files"], 2)
        self.assertEqual(index.read_bytes(), before)
        self.assertFalse((self.root / run).exists())

    def test_changed_bytes_are_rejected_after_relocation(self):
        run = self.autonomous()
        self.relocate(run, "results/task1/ablation/roi_v1")
        path = self.resolve(run + "/metrics.json", root=self.root)
        path.chmod(0o644)
        path.write_text('{"status":"damaged!"}\n')
        path.chmod(0o444)
        with self.assertRaisesRegex(ValueError, "changed"):
            verifier.verify_autonomous(self.root)

    def test_added_file_is_rejected(self):
        run = self.autonomous()
        self.relocate(run, "results/task1/ablation/roi_v1")
        (self.resolve(run, root=self.root) / "unexpected.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            verifier.verify_autonomous(self.root)

    def test_missing_file_is_rejected(self):
        run = self.autonomous()
        self.relocate(run, "results/task1/ablation/roi_v1")
        self.resolve(run + "/metrics.json", root=self.root).unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            verifier.verify_autonomous(self.root)

    def test_writable_file_is_rejected(self):
        run = self.autonomous()
        (self.root / run / "metrics.json").chmod(0o644)
        with self.assertRaisesRegex(ValueError, "writable"):
            verifier.verify_autonomous(self.root)

    def test_symlink_file_is_rejected(self):
        run = self.autonomous()
        path = self.root / run / "metrics.json"
        copy = self.root / "same-bytes.json"
        copy.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(copy)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            verifier.verify_autonomous(self.root)

    def test_symlink_directory_is_rejected(self):
        run = self.autonomous()
        original = self.root / run
        target = self.root / "moved-result"
        original.rename(target)
        original.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            verifier.verify_autonomous(self.root)

    def test_manifest_tampering_is_rejected(self):
        self.autonomous()
        path = self.root / verifier.INDICES["autonomous"]
        value = json.loads(path.read_text())
        value["actual_new_fits"] = 266
        path.chmod(0o644)
        path.write_text(json.dumps(value))
        path.chmod(0o444)
        with self.assertRaisesRegex(ValueError, "metadata signature"):
            verifier.verify_autonomous(self.root)

    def test_controlled_relocation_keeps_exact_directory_inventory_and_marker(self):
        run, report = self.controlled()
        self.relocate(run, "results/task1/ablation/controlled_v1")
        self.relocate(report, "reports/task1/ablation/controlled_v1")
        self.assertEqual(verifier.verify_controlled(self.root)["files"], 3)
        destination = self.resolve(run, root=self.root)
        destination.chmod(0o755)
        (destination / "extra_empty_directory").mkdir()
        destination.chmod(0o555)
        with self.assertRaisesRegex(ValueError, "directory inventory changed"):
            verifier.verify_controlled(self.root)

    def test_changed_controlled_marker_is_rejected(self):
        run, _ = self.controlled()
        marker = self.root / run / "FROZEN.json"
        value = json.loads(marker.read_text())
        value["evidence_manifest"] = "evidence/new-index.json"
        marker.chmod(0o644)
        marker.write_text(json.dumps(value))
        marker.chmod(0o444)
        with self.assertRaisesRegex(ValueError, "marker differs"):
            verifier.verify_controlled(self.root)

    def test_unsafe_names_and_mapping_aliases_are_rejected(self):
        run = self.autonomous()
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            verifier.path_for("../outside", self.root)
        self.mapping["scripts/seal_task1_autonomous.py"] = run + "/metrics.json"
        with self.assertRaisesRegex(ValueError, "Multiple logical|round-trip"):
            verifier.verify_autonomous(self.root)

    def test_failed_tree_relocation_and_added_file_rejection(self):
        run, cache = "results/task1_comparison_v1", "data/cache/task1_comparison_v1"
        index_name = verifier.INDICES["failed"]
        evidence = str(Path(index_name).parent)
        names = [run + "/metrics.json", cache + "/metadata.json", evidence + "/source_snapshot/code.py"]
        for name in names:
            self.write(name, "{}\n")
        value = {"status": "frozen_failed_experiment", "run_dir": run, "cache_dir": cache,
                 "files": [self.record(name) for name in names]}
        path = self.write(index_name, value)
        marker = {"status": value["status"], "evidence_manifest": index_name,
                  "evidence_manifest_sha256": verifier.digest(path), "instruction": verifier.FAILED_INSTRUCTION}
        self.write(run + "/FROZEN.json", marker)
        self.freeze(run, cache, evidence, directories=True)
        self.relocate(run, "results/task1/model_comparison/failed_v1")
        self.relocate(cache, "data/task1/cache/failed_v1")
        self.assertEqual(verifier.verify_failed(self.root)["files"], 3)
        destination = self.resolve(cache, root=self.root)
        destination.chmod(0o755)
        (destination / "extra.json").write_text("{}")
        destination.chmod(0o555)
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            verifier.verify_failed(self.root)

    def legacy_mapping(self, old="microcture/old_package", new="microcture/history/package", kind="source"):
        self.write("configs/layout.json", {"moves": [{"old": old, "new": new, "kind": kind}]})
        return old, new

    def test_legacy_links_are_same_bytes_and_cleaned(self):
        old, new = self.legacy_mapping()
        source = self.write(new + "/__init__.py", "VALUE = 7\n")
        source.chmod(0o444)
        before = os.environ.get("PYTHONDONTWRITEBYTECODE")
        with legacy_test_paths(self.root):
            target = self.root / old / "__init__.py"
            self.assertEqual(target.stat().st_ino, source.stat().st_ino)
            self.assertFalse(target.is_symlink())
            self.assertEqual(target.stat().st_mode & 0o222, 0)
            self.assertEqual(os.environ["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertFalse((self.root / old).exists())
        self.assertTrue(source.is_file())
        self.assertEqual(os.environ.get("PYTHONDONTWRITEBYTECODE"), before)

    def test_legacy_links_are_cleaned_after_test_exception(self):
        old, new = self.legacy_mapping()
        self.write(new + "/__init__.py", "VALUE = 7\n")
        with self.assertRaisesRegex(RuntimeError, "fixture error"):
            with legacy_test_paths(self.root):
                raise RuntimeError("fixture error")
        self.assertFalse((self.root / old).exists())

    def test_legacy_context_never_overwrites_existing_paths(self):
        old, new = self.legacy_mapping()
        self.write(new + "/__init__.py", "VALUE = 7\n")
        target = self.write(old + "/__init__.py", "VALUE = 8\n")
        with self.assertRaisesRegex(FileExistsError, "Refuse to replace"):
            with legacy_test_paths(self.root):
                pass
        self.assertEqual(target.read_text(), "VALUE = 8\n")

    def test_legacy_context_does_not_alias_numeric_data(self):
        old, new = self.legacy_mapping("results/old", "results/task1/experiment", "result")
        source = self.write(new + "/metrics.json", "{}\n")
        with legacy_test_paths(self.root):
            self.assertFalse((self.root / old).exists())
        self.assertTrue(source.is_file())

    def test_legacy_context_rejects_source_aliases_into_data(self):
        _, new = self.legacy_mapping("data/unsafe", "microcture/history/package", "source")
        self.write(new + "/__init__.py", "VALUE = 7\n")
        with self.assertRaisesRegex(ValueError, "Unsupported historical"):
            with legacy_test_paths(self.root):
                pass

    def test_portable_missing_roi_inputs_are_listed_without_creating_files(self):
        missing = missing_dropout_references(self.root)
        self.assertIn("results/task1_roi_v1/manifest.json", missing)
        self.assertIn("results/task1_roi_v1/folds/R1/fold_0.json", missing)
        self.assertFalse((self.root / "results").exists())

    def test_roi_presence_checks_include_indexed_fit_files(self):
        run = "results/task1/ablation/target_roi/v1"
        self.write("configs/layout.json", {"moves": [
            {"old": "results/task1_roi_v1", "new": run, "kind": "result"}]})
        fit = run + "/runs/R1/I3/m2plus/seed_42/fold_0/"
        for name in (run + "/manifest.json", run + "/folds/R1/fold_0.json",
                     fit + "metrics.json", fit + "protocol.json"):
            self.write(name, "{}\n")
        self.write(fit + "ARTIFACTS.json", {"files": [{"name": "final.pt"}]})
        self.assertEqual(missing_dropout_references(self.root), [fit + "final.pt"])
        self.write(fit + "final.pt", "fixture weight bytes\n")
        self.assertEqual(missing_dropout_references(self.root), [])

    def test_missing_roi_marks_fixture_skipped_with_an_explicit_reason(self):
        fixture = type("DropoutProtocolTests", (unittest.TestCase,), {
            "setUp": lambda self: self.fail("Unavailable reference must not be read"),
            "test_reference": lambda self: self.fail("Unavailable reference must not run")})
        module = SimpleNamespace(ROOT=self.root, DropoutProtocolTests=fixture)
        result = unittest.TestResult()
        with patch.dict("sys.modules", {"tests.test_task1_dropout": module}), \
             contextlib.redirect_stderr(io.StringIO()) as output:
            with legacy_tests.legacy_test_fixtures():
                fixture("test_reference").run(result)
        self.assertEqual(len(result.skipped), 1)
        self.assertIn("Requires local frozen ROI reference", result.skipped[0][1])
        self.assertIn("portable source package", output.getvalue())
        self.assertFalse(result.errors or result.failures)

    def test_complete_roi_does_not_skip_the_original_fixture_setup(self):
        called = []
        fixture = type("DropoutProtocolTests", (unittest.TestCase,), {
            "setUp": lambda self: called.append("original setup"), "test_reference": lambda self: None})
        experiment = SimpleNamespace(project_path=lambda root, name: root / name,
            reference_roi=SimpleNamespace(job_directory=lambda directory, job: directory),
            assert_reference_preprocessing=lambda *args: None)
        module = SimpleNamespace(ROOT=self.root, DropoutProtocolTests=fixture, E=experiment)
        result = unittest.TestResult()
        with patch.dict("sys.modules", {"tests.test_task1_dropout": module}), \
             patch.object(legacy_tests, "missing_dropout_references", return_value=[]):
            with legacy_tests.legacy_test_fixtures():
                fixture("test_reference").run(result)
        self.assertEqual(called, ["original setup"])
        self.assertFalse(result.skipped or result.errors or result.failures)

    def test_frozen_dropout_test_bytes_remain_original(self):
        path = Path(__file__).resolve().parents[1] / "tests/test_task1_dropout.py"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "c40290a15c7599211f7926dd75d7f5492d6eb363d4ac185f4441fba0ef5d531f")


if __name__ == "__main__":
    unittest.main()
