"""Exercise migrated input names and fresh output paths in isolated workspaces."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from microcture.layout import logical, resolve


class LayoutPathsTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="microcture-layout-paths-", dir="/tmp")
        self.addCleanup(temporary.cleanup)
        self.parent = Path(temporary.name)
        self.root = self.parent / "project"
        (self.root / "configs").mkdir(parents=True)
        self.rows = [
            {"old": "legacy/source", "new": "data/task1/cache/model_v1"},
            {"old": "legacy/source/special", "new": "data/task1/cache/special_v1"},
            {"old": "legacy/deep", "new": "data/task1/cache/model_v1/nested"},
            {"old": "data/legacy_experiments/results/task5",
             "new": "results/task5/super_resolution/v1"},
        ]
        self.write_mapping(self.rows)

    def write_mapping(self, rows):
        (self.root / "configs/layout.json").write_text(
            json.dumps({"schema_version": 1, "moves": rows}), encoding="utf-8")

    def make_file(self, relative, text="saved evidence\n"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_old_names_and_both_call_styles_reach_the_same_saved_bytes(self):
        saved = self.make_file("data/task1/cache/model_v1/fold/metrics.json", "{\"score\": 1}\n")
        old = "legacy/source/fold/metrics.json"
        self.assertEqual(resolve(old, root=self.root), saved)
        self.assertEqual(resolve(self.root / old, root=self.root), saved)
        self.assertEqual(resolve(self.root, old), saved)
        self.assertEqual(resolve(old, root=self.root).read_bytes(), saved.read_bytes())
        self.assertEqual(logical(saved, root=self.root), Path(old))

    def test_more_specific_source_and_destination_names_win(self):
        self.assertEqual(resolve("legacy/source/special/window.npz", root=self.root),
                         self.root / "data/task1/cache/special_v1/window.npz")
        self.assertEqual(logical("data/task1/cache/model_v1/nested/oof.csv", root=self.root),
                         Path("legacy/deep/oof.csv"))
        # A directory whose name merely starts with the old name is unrelated.
        self.assertEqual(resolve("legacy/source_extra/window.npz", root=self.root),
                         self.root / "legacy/source_extra/window.npz")

    def test_short_results_alias_reads_migrated_evidence(self):
        saved = self.make_file("results/task5/super_resolution/v1/metrics.csv", "psnr\n24\n")
        alias = resolve("results/task5/metrics.csv", root=self.root)
        self.assertEqual(alias, saved)
        self.assertEqual(alias.read_text(), "psnr\n24\n")
        self.assertEqual(resolve("data/legacy_experiments/results/task5/metrics.csv", root=self.root), saved)

    def test_new_experiment_output_and_canonical_input_are_not_redirected(self):
        saved = self.make_file("results/task5/super_resolution/v1/metrics.csv")
        new = self.root / "results/task5/ablation/new_v2"
        self.assertEqual(resolve(new, root=self.root), new)
        self.assertFalse(new.exists())
        self.assertEqual(resolve(saved, root=self.root), saved)
        # Future versions of an existing experiment also keep their new name.
        next_version = self.root / "results/task5/super_resolution/v2"
        self.assertEqual(resolve(next_version, root=self.root), next_version)

    def test_absolute_normalization_keeps_paths_in_the_correct_workspace(self):
        absolute = self.root / "unused/../legacy/source/fold.json"
        self.assertEqual(resolve(absolute, root=self.root),
                         self.root / "data/task1/cache/model_v1/fold.json")
        outside = self.parent / "outside/unused/../evidence.json"
        self.assertEqual(resolve(outside, root=self.root), self.parent / "outside/evidence.json")
        with self.assertRaisesRegex(ValueError, "outside"):
            logical(outside, root=self.root)

    def test_relative_parent_traversal_is_rejected_before_lookup(self):
        for name in ("../outside.json", "legacy/../source/item.json"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "traversal"):
                    resolve(name, root=self.root)
                with self.assertRaisesRegex(ValueError, "traversal"):
                    logical(name, root=self.root)

    def test_resolver_preserves_symlinks_for_callers_to_validate(self):
        outside = self.parent / "outside.csv"
        outside.write_text("external\n")
        target = self.root / "results/task5/super_resolution/v1/linked.csv"
        target.parent.mkdir(parents=True)
        target.symlink_to(outside)
        alias = resolve("results/task5/linked.csv", root=self.root)
        self.assertEqual(alias, target)
        self.assertTrue(alias.is_symlink())
        self.assertNotEqual(alias, outside)
        self.assertEqual(logical(alias, root=self.root),
                         Path("data/legacy_experiments/results/task5/linked.csv"))

    def test_duplicate_source_or_destination_invalidates_the_mapping(self):
        for duplicate in (
                {"old": "legacy/source", "new": "data/task1/cache/other"},
                {"old": "legacy/other", "new": "data/task1/cache/model_v1"}):
            with self.subTest(duplicate=duplicate):
                self.write_mapping(self.rows + [duplicate])
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    resolve("unrelated.txt", root=self.root)
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    logical("unrelated.txt", root=self.root)

    def test_unsafe_mapping_paths_are_rejected(self):
        for key, value in (("old", "../escape"), ("new", "/tmp/escape"),
                           ("old", ""), ("new", "data/../escape")):
            with self.subTest(key=key, value=value):
                row = {"old": "input", "new": "data/input"}
                row[key] = value
                self.write_mapping([row])
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    resolve("input/item.json", root=self.root)

    def test_workspace_without_mapping_retains_plain_project_paths(self):
        (self.root / "configs/layout.json").unlink()
        self.assertEqual(resolve("data/task2/cache/new_v1", root=self.root),
                         self.root / "data/task2/cache/new_v1")
        self.assertEqual(logical("data/task2/cache/new_v1", root=self.root),
                         Path("data/task2/cache/new_v1"))


if __name__ == "__main__":
    unittest.main()
