"""Exclusive copy publication on independent filesystems; tiny owned fixtures."""
from __future__ import annotations
import errno
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from microcture.task1_native_roi_v2 import data as D


class CrossFilesystemPublication(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="microcture-native-v2-publish-test-", dir="/tmp")
        self.root = Path(self.directory.name)
        self.stages = []
        real_mkstemp = tempfile.mkstemp

        def factory(*args, **kwargs):
            descriptor, name = real_mkstemp(prefix="microcture-native-v2-expected-fixture-", suffix=".json", dir="/tmp")
            self.stages.append(Path(name))
            return descriptor, name

        self.factory = patch.object(D.tempfile, "mkstemp", side_effect=factory)
        self.marker_patch = patch.object(D, "PUBLICATION_FAILURE_DIR", self.root / "forensics")
        self.factory.start(); self.marker_patch.start()

    def tearDown(self):
        self.factory.stop(); self.marker_patch.stop()
        for path in self.stages:
            path.unlink(missing_ok=True)
        self.directory.cleanup()

    def test_cross_device_hardlink_not_required_and_temporary_removed_on_success(self):
        value = {"status": "complete", "binding": {"fixture": "cross filesystem"}, "numeric": [1, 2, 3]}
        destination = self.root / "expected/fold_0.json"
        with patch.object(D.os, "link", side_effect=OSError(errno.EXDEV, "different filesystems")) as link:
            D._write_new_json(destination, value)
        link.assert_not_called()
        self.assertEqual(json.loads(destination.read_text()), value)
        self.assertTrue(all(not path.exists() for path in self.stages))

    def test_existing_scientific_file_is_never_overwritten(self):
        destination = self.root / "expected.json"
        original = b'{"original": true}\n'
        destination.write_bytes(original)
        value = {"status": "complete", "binding": {"fixture": "duplicate"}}
        with self.assertRaises(FileExistsError):
            D._write_new_json(destination, value)
        self.assertEqual(destination.read_bytes(), original)
        self.assertEqual(json.loads(self.stages[-1].read_text()), value)
        marker = next((self.root / "forensics").glob("*.failure.json"))
        self.assertEqual(json.loads(marker.read_text())["destination"], str(destination.resolve()))

    def test_partial_destination_and_full_numerical_json_both_retained_and_retry_blocked(self):
        dataset, fold, sources = {"signature": "native-publication-fixture"}, {"fold": 0}, []
        binding = {"fixture": "partial publication"}
        value = {"status": "complete", "binding": binding, "records": [{"expected": [1, 2, 3]}]}
        cache = self.root / "expected"
        destination = cache / D.data_source_signature()[:12] / "fold_0.json"

        def partial_copy(source, target):
            target.write(b'{"status":')
            target.flush()
            raise OSError(errno.ENOSPC, "mock full destination filesystem")

        with patch("shutil.copyfileobj", side_effect=partial_copy), self.assertRaises(OSError):
            D._write_new_json(destination, value)
        partial_bytes = destination.read_bytes()
        self.assertEqual(partial_bytes, b'{"status":')
        self.assertEqual(json.loads(self.stages[-1].read_text()), value)
        with patch.object(D, "native_source_records", return_value=sources), \
             patch.object(D, "_expected_binding", return_value=binding), \
             patch.object(D, "training_native_mask") as fitting:
            with self.assertRaisesRegex(ValueError, "publication failure"):
                D.prepare_shared_expected(dataset, fold, cache_dir=cache)
        fitting.assert_not_called()
        self.assertEqual(destination.read_bytes(), partial_bytes)

    def test_unpublished_full_stage_binding_blocks_retry_without_destination(self):
        binding = {"fixture": "full staged normalizer"}
        descriptor, name = D.tempfile.mkstemp()
        with D.os.fdopen(descriptor, "w") as stream:
            json.dump({"status": "complete", "binding": binding}, stream)
        with patch.object(D, "native_source_records", return_value=[]), \
             patch.object(D, "_expected_binding", return_value=binding), \
             patch.object(D, "training_native_mask") as fitting:
            with self.assertRaisesRegex(ValueError, "Unpublished.*no automatic retry"):
                D.prepare_shared_expected({"signature": "fixture"}, {"fold": 1}, cache_dir=self.root / "cache")
        fitting.assert_not_called()
        self.assertTrue(Path(name).exists())


if __name__ == "__main__":
    unittest.main()
