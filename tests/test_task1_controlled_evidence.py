"""Synthetic evidence inventory checks; no real experiments are read or altered."""
import json
from pathlib import Path
import stat
import tempfile
import unittest

from microcture.task1_controlled.evidence import _read_only, sha256, verify_evidence


class FrozenEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="microcture-evidence-test-", dir="/tmp")
        self.root = Path(self.temporary.name)
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()
        self.run = self.root / "run"
        self.run.mkdir()
        self.artifact = self.run / "history.csv"
        self.artifact.write_bytes(b"123456")
        self.manifest = self.evidence / "freeze_manifest.json"
        self.manifest.write_text(json.dumps({"run_dir": "run", "files": [
            {"path": "run/history.csv", "size_bytes": 6, "sha256": sha256(self.artifact)}],
            "run_signature": "synthetic-run", "dataset_signature": "synthetic-data"}), encoding="utf-8")
        (self.run / "FROZEN.json").write_text(json.dumps({"evidence_manifest_sha256": sha256(self.manifest)}), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_valid_inventory_preserves_content_and_read_only_permission(self):
        _read_only([self.artifact])
        result = verify_evidence(self.root, self.evidence)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["files"], 1)
        self.assertEqual(self.artifact.read_bytes(), b"123456")
        self.assertEqual(stat.S_IMODE(self.artifact.stat().st_mode) & 0o222, 0)

    def test_equal_size_artifact_change_is_detected(self):
        self.artifact.write_bytes(b"123457")
        with self.assertRaisesRegex(ValueError, "Hash changed"):
            verify_evidence(self.root, self.evidence)

    def test_freeze_manifest_change_is_detected_by_marker(self):
        value = json.loads(self.manifest.read_text())
        value["run_signature"] = "changed"
        self.manifest.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            verify_evidence(self.root, self.evidence)


if __name__ == "__main__":
    unittest.main()
