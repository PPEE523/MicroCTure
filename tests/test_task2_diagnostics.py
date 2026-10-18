"""Saved-table diagnostics reject broken comparisons without fitting models."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from microcture.tasks.task2.diagnostics import load_diagnostics, plot_diagnostics


class Task2DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="microcture-task2-diagnostics-", dir="/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.seeds = [42, 43, 44]
        self.models = [f"{family}_{seed}" for family in ("ae", "dae") for seed in self.seeds]
        self.methods = ["shape", "density"]
        for family in ("ae", "dae"):
            for seed in self.seeds:
                self.methods.extend((f"{family}_{seed}", f"{family}_fusion_{seed}"))
        self.config = {"seeds": self.seeds, "random_references": 20,
                       "full_candidate_quotas": {"64": 140, "256": 49},
                       "test_candidate_quotas": {"64": 20, "256": 7},
                       "ae_max_epochs": 40, "clusters": [4, 6, 8, 12],
                       "clustering_seeds": self.seeds,
                       "min_new_cluster_members": 5, "cluster_min_jaccard": .6}
        self.report = {"config": self.config, "methods": [{"method": method} for method in self.methods],
                       "test_known_counts": {"OPCID": 9, "CHIN": 45, "CHID": 5}}
        (self.directory / "report.json").write_text(json.dumps(self.report))
        recall = []
        for domain, totals, windows, covered in (("full", (68, 250, 26), 189, 21504), ("test", (9, 45, 5), 27, 3072)):
            for position, method in enumerate(self.methods + [f"random_{seed}" for seed in range(20)]):
                for kind, total in zip(("OPCID", "CHIN", "CHID"), totals):
                    recalled = position % (total + 1)
                    recall.append({"domain": domain, "method": method, "type": kind,
                                   "recalled": recalled, "total": total, "recall": recalled / total,
                                   "windows": windows, "covered_bins": covered})
        self.write("known_recall.csv", recall)
        self.write("quality_correlations.csv", [
            {"model": name, "quality_metric": metric, "test_spearman": value}
            for name in self.models for metric, value in (("mean_oe", -.584), ("valid_fraction", .08), ("zero_fraction", -.22))])
        self.write("training_summary.csv", [
            {"model": name, "family": name.split("_")[0], "seed": int(name.split("_")[1]),
             "epochs": 40, "best_epoch": 39, "train_windows": 2000,
             "validation_windows": 873, "validation_loss": .03, "test_loss": .032}
            for name in self.models])
        self.write("clusters.csv", [
            {"method": method, "k": k, "seed": seed, "cluster": 0,
             "members": 10, "annotated_members": 0, "supported_members": 2,
             "min_seed_jaccard": .7, "passes_original_rule": False, "stable_candidate_new_cluster": False}
            for method in self.methods for k in self.config["clusters"] for seed in self.seeds])

    def read(self, filename):
        with (self.directory / filename).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def write(self, filename, rows):
        with (self.directory / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_complete_method_and_quality_grids_have_actual_source_hashes(self):
        result = load_diagnostics(self.directory)
        self.assertEqual(len(result["test_retrieval_rows"]), 14 * 3)
        self.assertEqual(len(result["score_quality_rows"]), 6 * 3)
        self.assertEqual(len(result["cluster_gate_rows"]), 14 * 4 * 3)
        self.assertFalse(result["training_performed"])
        self.assertFalse(result["independent_external_confirmation"])
        self.assertEqual(len(result["sources"]), 5)
        for source in result["sources"]:
            payload = Path(source["path"]).read_bytes()
            self.assertEqual(source["bytes"], len(payload))
            self.assertEqual(source["sha256"], hashlib.sha256(payload).hexdigest())

    def test_denominator_must_match_other_methods_and_random_references(self):
        rows = self.read("known_recall.csv")
        target = next(row for row in rows if row["domain"] == "test" and row["method"] == "random_19" and row["type"] == "OPCID")
        target["total"] = "10"
        target["recall"] = str(int(target["recalled"]) / 10)
        self.write("known_recall.csv", rows)
        with self.assertRaisesRegex(ValueError, "denominator differs"):
            load_diagnostics(self.directory)

    def test_candidate_count_budget_cannot_change_for_one_method(self):
        rows = self.read("known_recall.csv")
        rows[0]["windows"] = "190"
        self.write("known_recall.csv", rows)
        with self.assertRaisesRegex(ValueError, "Candidate budget differs"):
            load_diagnostics(self.directory)

    def test_same_number_of_windows_still_requires_same_scale_coverage(self):
        rows = self.read("known_recall.csv")
        rows[0]["covered_bins"] = "21505"
        self.write("known_recall.csv", rows)
        with self.assertRaisesRegex(ValueError, "Candidate budget differs"):
            load_diagnostics(self.directory)

    def test_missing_random_reference_cannot_silently_narrow_range(self):
        rows = self.read("known_recall.csv")
        self.write("known_recall.csv", rows[:-1])
        with self.assertRaisesRegex(ValueError, "grid is incomplete"):
            load_diagnostics(self.directory)

    def test_random_range_is_observed_range_and_signed_correlations_are_preserved(self):
        result = load_diagnostics(self.directory)
        row = next(row for row in result["test_retrieval_rows"] if row["method"] == "shape" and row["type"] == "OPCID")
        counts = [(14 + index) % 10 for index in range(20)]
        self.assertEqual(row["random_references"], 20)
        self.assertAlmostEqual(row["random_recalled_mean"], sum(counts) / 20)
        self.assertAlmostEqual(row["random_recall_mean"], sum(counts) / (20 * 9))
        self.assertAlmostEqual(row["random_recall_min"], min(counts) / 9)
        self.assertAlmostEqual(row["random_recall_max"], max(counts) / 9)
        self.assertIn("not a confidence interval", row["random_range_interpretation"])
        self.assertEqual(result["score_quality_rows"][0]["test_spearman"], -.584)

    def test_stable_cluster_flag_cannot_bypass_member_gate(self):
        rows = self.read("clusters.csv")
        rows[0]["stable_candidate_new_cluster"] = "True"
        self.write("clusters.csv", rows)
        with self.assertRaisesRegex(ValueError, "stable cluster rule differs"):
            load_diagnostics(self.directory)

    def test_cluster_identity_duplicates_are_rejected(self):
        rows = self.read("clusters.csv")
        self.write("clusters.csv", rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "duplicate cluster"):
            load_diagnostics(self.directory)

    def test_incomplete_quality_grid_is_not_a_smaller_comparison(self):
        rows = self.read("quality_correlations.csv")
        self.write("quality_correlations.csv", rows[:-1])
        with self.assertRaisesRegex(ValueError, "Score-quality.*incomplete"):
            load_diagnostics(self.directory)

    def test_nonfinite_correlations_cannot_be_plotted_as_zero(self):
        rows = self.read("quality_correlations.csv")
        rows[0]["test_spearman"] = "nan"
        self.write("quality_correlations.csv", rows)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            load_diagnostics(self.directory)

    def test_three_panel_plot_is_generated_without_model_imports_or_training(self):
        data = load_diagnostics(self.directory)
        output = self.directory / "figures/diagnostics.png"
        before = {source["path"]: Path(source["path"]).read_bytes() for source in data["sources"]}
        self.assertEqual(plot_diagnostics(data, output), output)
        payload = output.read_bytes()
        self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", payload[16:24])
        self.assertGreater(width, 256)
        self.assertGreater(height, 256)
        for name, original in before.items():
            self.assertEqual(Path(name).read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
