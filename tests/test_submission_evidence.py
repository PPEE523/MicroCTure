"""Small portable-package fixtures; no scientific workspace or training needed."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_submission_evidence import _score, sha256, signature, verify_submission


class SubmissionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="microcture-submission-test-", dir="/tmp")
        self.root = Path(self.temporary.name)
        self.submission = self.root / "delivery/submission"
        self.submission.mkdir(parents=True)
        payload = self.submission / "table.csv"
        payload.write_text("model,score\nm2plus,0.8\n", encoding="utf-8")
        self.source = self.root / "scientific-source"
        (self.source / "results").mkdir(parents=True)
        (self.source / "results/table.csv").write_bytes(payload.read_bytes())
        self.manifest = {
            "format_version": 1,
            "scientific_boundaries": {"independent_blind_test": False, "new_expert_truth_available": False,
                                      "independent_new_samples_available": False,
                                      "reliable_three_class_morphology_established": False},
            "files": [{"path": "table.csv", "bytes": payload.stat().st_size, "sha256": sha256(payload),
                       "source_kind": "copied_scientific_evidence", "source_path": "results/table.csv"}],
            "total_bytes": payload.stat().st_size,
        }
        self.write_manifest()

    def tearDown(self):
        self.temporary.cleanup()

    def write_manifest(self):
        value = {key: entry for key, entry in self.manifest.items() if key != "content_signature"}
        value["content_signature"] = signature(value)
        (self.submission / "manifest.json").write_text(json.dumps(value), encoding="utf-8")

    def test_portable_verification_does_not_require_original_workspace(self):
        (self.source / "results/table.csv").unlink()
        result = verify_submission(self.submission)
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["original_workspace_required"])
        self.assertEqual(result["original_source_files_checked"], 0)

    def test_optional_original_source_verification(self):
        result = verify_submission(self.submission, source_root=self.source)
        self.assertEqual(result["original_source_files_checked"], 1)
        (self.source / "results/table.csv").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source evidence differs"):
            verify_submission(self.submission, source_root=self.source)

    def test_copied_evidence_hash_detects_changed_bytes(self):
        path = self.submission / "table.csv"
        path.write_bytes(b"X" * path.stat().st_size)
        with self.assertRaisesRegex(ValueError, "Packaged evidence differs"):
            verify_submission(self.submission)

    def test_manifest_signature_detects_unsanctioned_index_edit(self):
        path = self.submission / "manifest.json"
        value = json.loads(path.read_text())
        value["total_bytes"] += 1
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "signature differs"):
            verify_submission(self.submission)

    def test_duplicate_manifest_paths_are_rejected(self):
        self.manifest["files"].append(dict(self.manifest["files"][0]))
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "unique"):
            verify_submission(self.submission)

    def test_path_traversal_is_rejected(self):
        self.manifest["files"][0]["path"] = "../outside.csv"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Unsafe package path"):
            verify_submission(self.submission)

    def test_unindexed_file_is_rejected(self):
        (self.submission / "unindexed.cache").write_text("not part of coursework")
        with self.assertRaisesRegex(ValueError, "Unindexed"):
            verify_submission(self.submission)

    def write_delivery_extension(self):
        repository = self.submission.parent
        report = repository / "docs/RESULTS_DELIVERY.md"
        report.parent.mkdir()
        report.write_text("Saved-result analysis; no new biological claim.\n")
        summary = self.submission / "delivery_summary.json"
        summary.write_text('{"new_training": false}\n')
        index = {"schema_version": 1, "tasks": 6, "training_performed": False,
                 "raw_contact_scan_performed": False, "report": "docs/RESULTS_DELIVERY.md",
                 "files": [{"path": str(p.relative_to(repository)), "bytes": p.stat().st_size,
                            "sha256": sha256(p)} for p in (report, summary)]}
        path = self.submission / "results_delivery_manifest.json"
        path.write_text(json.dumps(index))
        return index, path, summary

    def test_separate_report_index_preserves_frozen_science_manifest(self):
        core_bytes = (self.submission / "manifest.json").read_bytes()
        self.write_delivery_extension()
        self.assertEqual(verify_submission(self.submission)["status"], "passed")
        self.assertEqual((self.submission / "manifest.json").read_bytes(), core_bytes)

    def test_separate_report_tampering_is_detected(self):
        _, _, summary = self.write_delivery_extension()
        summary.write_text("changed")
        with self.assertRaisesRegex(ValueError, "Supplemental delivery differs"):
            verify_submission(self.submission)

    def test_separate_report_cannot_reindex_frozen_evidence(self):
        index, path, _ = self.write_delivery_extension()
        core = self.submission / "table.csv"
        index["files"].append({"path": "submission/table.csv", "bytes": core.stat().st_size,
                               "sha256": sha256(core)})
        path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "cannot replace the core"):
            verify_submission(self.submission)

    def test_separate_report_cannot_hide_path_traversal(self):
        index, path, _ = self.write_delivery_extension()
        index["files"][-1]["path"] = "../outside"
        path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "Unsafe package path"):
            verify_submission(self.submission)

    def test_separate_report_preserves_no_training_boundary(self):
        index, path, _ = self.write_delivery_extension()
        index["training_performed"] = True
        path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "evidence boundaries differ"):
            verify_submission(self.submission)

    def write_six_task_extension(self):
        index, path, summary = self.write_delivery_extension()
        repository = self.submission.parent
        reports = [f"docs/TASK{number}.md" for number in range(1, 7)]
        records = []
        for name in reports:
            document = repository / name
            document.write_text("# Task result, methods and visualization\n")
            records.append({"path": name, "bytes": document.stat().st_size,
                            "sha256": sha256(document)})
        index.update(schema_version=2, reports=reports, files=records + [index["files"][-1]])
        index.pop("report")
        path.write_text(json.dumps(index))
        return index, path

    def test_six_task_documents_replace_single_report_without_changing_core(self):
        original = (self.submission / "manifest.json").read_bytes()
        self.write_six_task_extension()
        self.assertEqual(verify_submission(self.submission)["status"], "passed")
        self.assertEqual((self.submission / "manifest.json").read_bytes(), original)

    def test_six_task_index_rejects_omitting_one_document(self):
        index, path = self.write_six_task_extension()
        index["reports"].pop()
        path.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "evidence boundaries differ"):
            verify_submission(self.submission)

    def test_six_task_index_checks_document_hashes(self):
        self.write_six_task_extension()
        (self.submission.parent / "docs/TASK4.md").write_text("changed claim")
        with self.assertRaisesRegex(ValueError, "Supplemental delivery differs"):
            verify_submission(self.submission)

    def test_hidden_directory_symlink_is_rejected(self):
        (self.submission / "hidden-raw-cache").symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlinks"):
            verify_submission(self.submission)

    def test_known_scientific_boundary_cannot_be_upgraded(self):
        self.manifest["scientific_boundaries"]["reliable_three_class_morphology_established"] = True
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "scientific evidence boundaries"):
            verify_submission(self.submission)

    def test_retained_legacy_files_are_checked_without_modifying_them(self):
        legacy = self.submission.parent / "showcase/results/task2/summary.csv"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("retained,1\n")
        original = legacy.read_bytes()
        self.manifest["retained_legacy_files"] = [{"path": "showcase/results/task2/summary.csv",
                                                  "bytes": legacy.stat().st_size, "sha256": sha256(legacy)}]
        self.write_manifest()
        result = verify_submission(self.submission)
        self.assertEqual(result["retained_legacy_files"], 1)
        self.assertEqual(legacy.read_bytes(), original)
        legacy.write_text("retained,0\n")
        with self.assertRaisesRegex(ValueError, "legacy showcase differs"):
            verify_submission(self.submission)

    def test_constant_classifier_remains_in_fixed_three_class_metrics(self):
        result = _score([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        self.assertEqual(result["n_structures"], 3)
        self.assertAlmostEqual(result["macro_f1"], 1 / 6)
        self.assertEqual(result["per_class"]["CHID"]["f1"], 0.)

    def test_derived_demo_files_do_not_require_a_false_original_source(self):
        self.manifest["files"][0]["source_kind"] = "derived_delivery_demo"
        self.manifest["files"][0]["source_path"] = "submission/demo/task1/input.npz"
        self.write_manifest()
        result = verify_submission(self.submission, source_root=self.root / "absent-workspace")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["original_source_files_checked"], 0)


if __name__ == "__main__":
    unittest.main()
