"""Mutate small bundled centering fixtures; no original workspace or fitting."""
from __future__ import annotations

import copy
import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_submission_evidence import signature, verify_centering_snapshot, verify_submission


class Task2SubmissionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="microcture-task2-evidence-", dir="/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.submission = Path(self.temporary.name) / "submission"
        shutil.copytree(ROOT / "submission/task2/centering", self.submission / "task2/centering")
        self.manifest = json.loads((ROOT / "submission/manifest.json").read_text())
        self.specification = copy.deepcopy(self.manifest["centering_checks"])
        self.base = self.submission / self.specification["directory"]

    def read_rows(self, name, filename):
        with (self.base / name / filename).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def write_rows(self, name, filename, rows):
        with (self.base / name / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def check(self):
        return verify_centering_snapshot(self.submission, self.specification)

    def test_portable_saved_decisions_keep_na_distinct_from_zero(self):
        result = self.check()
        self.assertEqual(result["cohorts"]["primary"]["not_evaluable_candidates"], 21)
        self.assertEqual(result["cohorts"]["supplement"]["assessed_candidates"], 19)
        self.assertEqual(result["cohorts"]["supplement"]["not_evaluable_candidates"], 2)
        for cohort in result["cohorts"].values():
            self.assertEqual(cohort["metric_rows"], 21 * 54 * 2)
            self.assertIsNone(cohort["whole_cohort_anomaly_counts"])
        self.assertFalse(result["independent_confirmation"])

    def test_primary_na_cannot_be_serialized_as_false(self):
        rows = self.read_rows("primary", "candidate_variants.csv")
        rows[0]["outside_both"] = "False"
        self.write_rows("primary", "candidate_variants.csv", rows)
        with self.assertRaisesRegex(ValueError, "NA must not become a negative"):
            self.check()

    def test_primary_summary_na_cannot_be_zero(self):
        rows = self.read_rows("primary", "candidate_summary.csv")
        rows[0]["original_outside_variants"] = "0"
        self.write_rows("primary", "candidate_summary.csv", rows)
        with self.assertRaisesRegex(ValueError, "NA must not become a negative"):
            self.check()

    def test_two_quality_failed_candidates_cannot_be_dropped(self):
        rows = self.read_rows("supplement", "candidate_summary.csv")
        retained = [row for row in rows if row["assessment_status"] == "assessed"]
        self.assertEqual(len(rows) - len(retained), 2)
        self.write_rows("supplement", "candidate_summary.csv", retained)
        with self.assertRaisesRegex(ValueError, "candidate summaries are missing"):
            self.check()

    def test_duplicate_variant_cannot_replace_a_missing_variant(self):
        rows = self.read_rows("primary", "candidate_variants.csv")
        rows[1] = dict(rows[0])
        self.write_rows("primary", "candidate_variants.csv", rows)
        with self.assertRaisesRegex(ValueError, "variant grid is duplicate or incomplete"):
            self.check()

    def test_missing_variant_is_rejected(self):
        rows = self.read_rows("supplement", "candidate_variants.csv")
        self.write_rows("supplement", "candidate_variants.csv", rows[:-1])
        with self.assertRaisesRegex(ValueError, "variant grid is duplicate or incomplete"):
            self.check()

    def test_finite_distance_decision_is_recomputed(self):
        rows = self.read_rows("supplement", "candidate_variants.csv")
        row = next(row for row in rows if row["assessment_status"] == "assessed")
        row["outside_both"] = "True" if row["outside_both"] == "False" else "False"
        self.write_rows("supplement", "candidate_variants.csv", rows)
        with self.assertRaisesRegex(ValueError, "Centering decision differs"):
            self.check()

    def test_partial_cohort_cannot_claim_zero_robust_anomalies(self):
        path = self.base / "supplement/report.json"
        report = json.loads(path.read_text())
        report["robust_recentered_anomalies"] = 0
        path.write_text(json.dumps(report))
        path = self.base / "review.json"
        review = json.loads(path.read_text())
        review["results"]["supplement"] = report
        path.write_text(json.dumps(review))
        with self.assertRaisesRegex(ValueError, "aggregate must remain null"):
            self.check()

    def test_viewed_supplement_cannot_become_independent_confirmation(self):
        self.specification["independent_confirmation"] = True
        with self.assertRaisesRegex(ValueError, "cannot become independent confirmation"):
            self.check()

    def test_original_source_option_does_not_mislocate_clone_retained_reports(self):
        manifest = {"format_version": 1, "scientific_boundaries": self.manifest["scientific_boundaries"],
                    "centering_checks": self.specification,
                    "provenance_groups": {"retained_centering_review": self.manifest["provenance_groups"]["retained_centering_review"]},
                    "files": [r for r in self.manifest["files"] if r["path"].startswith("task2/")]}
        self.assertEqual(len(manifest["files"]), 9)
        self.assertEqual({r["source_kind"] for r in manifest["files"]}, {"retained_centering_review"})
        self.assertEqual({r["source_scope"] for r in manifest["files"]}, {"retained_clone", "original_workspace"})
        manifest["total_bytes"] = sum(row["bytes"] for row in manifest["files"])
        manifest["content_signature"] = signature(manifest)
        (self.submission / "manifest.json").write_text(json.dumps(manifest))
        result = verify_submission(self.submission, source_root=Path(self.temporary.name) / "absent-workspace")
        self.assertEqual(result["packaged_files"], 9)
        self.assertEqual(result["original_source_files_checked"], 0)
        self.assertEqual(result["status"], "passed")


if __name__ == "__main__":
    unittest.main()
