"""The retired generator must fail before reading contacts or changing evidence."""
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from microcture.layout import bootstrap, resolve

bootstrap()
import prepare_dataset


class LegacyDatasetRetiredTest(unittest.TestCase):
    def test_main_preserves_existing_cache_and_shared_split_tables(self):
        with TemporaryDirectory(prefix="microcture-retired-dataset-", dir="/tmp") as folder:
            root = Path(folder)
            old_names = ["data/task1/cache/legacy_classification_v1/train.npz",
                         "data/task5/splits/classification_split.csv",
                         "data/task5/splits/classification_examples.csv"]
            for name in old_names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((name + "\nretained evidence\n").encode())
            before = {str(p.relative_to(root)): p.read_bytes()
                      for p in root.rglob("*") if p.is_file()}
            with (mock.patch.object(prepare_dataset, "ROOT", root),
                  mock.patch.object(prepare_dataset, "_layout_resolve") as lookup,
                  mock.patch.object(prepare_dataset, "fit_expected") as scan,
                  mock.patch.object(prepare_dataset, "aggregate_local") as pixels,
                  mock.patch.object(prepare_dataset, "write_csv") as csv,
                  mock.patch.object(prepare_dataset.np, "savez_compressed") as cache):
                with self.assertRaisesRegex(RuntimeError, "retired.*prepare_task1"):
                    prepare_dataset.main()
                for operation in (lookup, scan, pixels, csv, cache):
                    operation.assert_not_called()
            after = {str(p.relative_to(root)): p.read_bytes()
                     for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)

    def test_direct_cli_reports_the_replacement_and_fails_without_outputs(self):
        with TemporaryDirectory(prefix="microcture-retired-cli-", dir="/tmp") as folder:
            result = subprocess.run(
                [sys.executable, "-B", str(resolve("scripts/prepare_dataset.py"))],
                cwd=folder, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Legacy dataset generation is retired", result.stderr)
            self.assertIn("scripts/task1/prepare_task1.py", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
