"""Evidence inventory fixtures: no real training or project-file mutation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json

from scripts import seal_task1_autonomous as seal


class AutonomousSealTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dest = self.root / "evidence"
        self.dest.mkdir()
        self.result = self.root / "results" / "study"
        self.result.mkdir(parents=True)
        self.file = self.result / "metrics.json"
        self.file.write_text('{"status":"complete"}\n')
        self.addCleanup(patch.stopall)
        patch.object(seal, "ROOT", self.root).start()
        patch.object(seal, "DEST", self.dest).start()
        records = seal.inventory([self.file])
        self.manifest = {
            "protocol": "task1_autonomous_in_place_evidence_v1", "actual_new_fits": 265,
            "complete_new_model_OOF": 53, "fixed_fusion": {"actual_new_fits": 0},
            "inventory": records, "inventory_signature": seal.signature(records),
            "directories": ["results/study"], "files": []}
        self.manifest["manifest_signature"] = seal.signature(self.manifest)
        self.manifest_path = self.dest / "freeze_manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.manifest_path.chmod(0o444)
        self.file.chmod(0o444)

    def test_verify_and_reject_changed_bytes(self):
        self.assertEqual(seal.verify()["files"], 1)
        self.file.chmod(0o644)
        self.file.write_text('{"status":"failed"}\n')
        self.file.chmod(0o444)
        with self.assertRaisesRegex(ValueError, "changed"):
            seal.verify()

    def test_added_and_missing_files_cannot_disappear(self):
        extra = self.result / "unindexed.json"
        extra.write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed"):
            seal.verify()
        extra.unlink()
        self.file.unlink()
        with self.assertRaisesRegex(ValueError, "changed"):
            seal.verify()

    def test_writable_file_is_not_read_only_evidence(self):
        self.file.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "writable"):
            seal.verify()

    def test_symlink_is_not_followed(self):
        (self.result / "alias").symlink_to(self.file)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            seal.verify()

    def test_parent_traversal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "traversal"):
            seal.safe(self.root / ".." / "outside")

    def test_metadata_and_manifest_permissions_are_checked(self):
        self.manifest_path.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "writable"):
            seal.verify()
        self.manifest["actual_new_fits"] = 999
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.manifest_path.chmod(0o444)
        with self.assertRaisesRegex(ValueError, "metadata"):
            seal.verify()


if __name__ == "__main__":
    unittest.main()
