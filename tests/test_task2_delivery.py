"""Task-two evidence joins, portable snapshots and read-only command boundaries."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from microcture.layout import bootstrap
bootstrap()
from microcture.tasks.task2 import delivery
from scripts.task2 import review_task2 as cli


class Task2DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="microcture-task2-delivery-", dir="/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.candidate = {"window_id": "w64_c32", "start_bp": "0", "window_length_bp": "6400",
                          "correlation": ".8", "cutoff": ".6"}

    def table(self, path, rows):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def centering(self, name, count, quality=True, candidate="w64_c32"):
        directory = self.root / "centering" / name
        self.table(directory / "candidate_summary.csv", [
            {"entity_id": candidate, "all_variants_quality_pass": quality,
             "calibration_sufficient": count >= 10, "original_outside_variants": 0,
             "boundary_outside_variants": 0, "robust_recentered_anomaly": False}])
        entities = [{"entity_id": candidate, "kind": "candidate", "original_center_bin": 32,
                     "type": "", "calibration_member": False}]
        entities.extend({"entity_id": f"known_{index}", "kind": "known", "original_center_bin": 100 + index,
                         "type": "CHIN", "calibration_member": True} for index in range(count))
        self.table(directory / "entities.csv", entities)
        (directory / "report.json").write_text(json.dumps({"candidates": 1, "calibration_windows": count}))
        return directory

    def snapshot(self):
        directory = self.root / "submission/task2/verification/test_v1"
        directory.mkdir(parents=True)
        source = self.root / "results/task2/discovery/v2/input.csv"
        source.parent.mkdir(parents=True)
        source.write_text("window_id,value\nw64_c32,1\n")
        code = self.root / "microcture/tasks/task2/delivery.py"
        code.parent.mkdir(parents=True)
        code.write_text('"""Test implementation provenance."""\n')
        summary = {"status": "saved_table_consistency_passed", "training_performed": False,
                   "raw_pixels_read": False, "independent_confirmation": False,
                   "sources": [self.record(source, relative=True)],
                   "code_sources": [self.record(code, relative=True)]}
        (directory / "review.json").write_text(json.dumps(summary))
        for name in ("candidate_evidence.csv", "retrieval_diagnostics.csv", "score_quality.csv"):
            (directory / name).write_text("id,value\nfixed,0\n")
        (directory / "index.html").write_text("<!doctype html><p>NA is not zero.</p>")
        (directory / "model_diagnostics.png").write_bytes(b"small fixture, no actual rendering needed")
        self.reindex(directory)
        return directory, source, code

    def record(self, path, relative=False):
        path = Path(path)
        payload = path.read_bytes()
        return {"path": path.relative_to(self.root).as_posix() if relative else path.name,
                "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

    def reindex(self, directory):
        rows = [self.record(path) for path in sorted(directory.iterdir())
                if path.is_file() and path.name != "ARTIFACTS.json"]
        (directory / "ARTIFACTS.json").write_text(json.dumps({"schema_version": 1, "files": rows}))

    def test_missing_supplementary_evidence_stays_na(self):
        rows, overview, sources = delivery.candidate_catalogue([self.candidate])
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["pca_outside_both"])
        self.assertIsNone(rows[0]["shape_outside_both"])
        self.assertIsNone(rows[0]["residual_replication"])
        self.assertIsNone(rows[0]["primary_robust_anomaly"])
        self.assertEqual(rows[0]["primary_assessment_status"], "not_evaluable")
        self.assertEqual(overview["novelty"]["status"], "unavailable")
        self.assertEqual(sources, [])

    def test_an_old_candidate_cohort_cannot_supply_new_candidate_metrics(self):
        novelty, morphology = self.root / "novelty", self.root / "morphology"
        self.table(novelty / "candidate_review.csv", [{"window_id": "w64_c999", "pca_outside_both": True}])
        self.table(morphology / "candidate_review_table.csv", [{"window_id": "w64_c999", "residual_replication": .99}])
        self.centering("primary", 10, candidate="w64_c999")
        rows, overview, sources = delivery.candidate_catalogue(
            [self.candidate], novelty_dir=novelty, morphology_dir=morphology, centering_dir=self.root / "centering")
        self.assertIsNone(rows[0]["pca_outside_both"])
        self.assertIsNone(rows[0]["residual_replication"])
        self.assertIsNone(rows[0]["primary_robust_anomaly"])
        for name in ("novelty", "morphology"):
            self.assertEqual(overview[name]["status"], "unavailable_candidate_identity_mismatch")
        self.assertEqual(overview["centering"]["primary"]["status"], "unavailable_candidate_identity_mismatch")
        self.assertEqual(len(sources), 5)

    def test_same_candidate_name_with_different_coordinates_is_rejected(self):
        self.table(self.root / "novelty/candidate_review.csv", [
            {"window_id": "w64_c32", "start_bp": 100, "window_length_bp": 6400}])
        with self.assertRaisesRegex(ValueError, "candidate coordinates differ"):
            delivery.candidate_catalogue([self.candidate], novelty_dir=self.root / "novelty")

    def test_legacy_false_is_normalized_to_na_for_low_calibration_and_quality_failure(self):
        self.centering("primary", 8)
        self.centering("supplement", 10, quality=False)
        rows, overview, _ = delivery.candidate_catalogue([self.candidate], centering_dir=self.root / "centering")
        for name in ("primary", "supplement"):
            self.assertEqual(rows[0][name + "_assessment_status"], "not_evaluable")
            self.assertIsNone(rows[0][name + "_robust_anomaly"])
            cohort = overview["centering"][name]
            self.assertEqual(cohort["status"], "matched_saved_cohort")
            self.assertIsNone(cohort["aggregate"]["robust_recentered_anomalies"])
            self.assertEqual(cohort["aggregate"]["not_evaluable_candidates"], 1)
            self.assertEqual(cohort["calibration_type_counts"], {"CHIN": 8 if name == "primary" else 10})
        self.assertEqual(rows[0]["primary_assessment_reason"], "insufficient_calibration")

    def test_centering_calibrator_entities_must_match_reported_count(self):
        directory = self.centering("primary", 8)
        (directory / "report.json").write_text(json.dumps({"candidates": 1, "calibration_windows": 9}))
        with self.assertRaisesRegex(ValueError, "calibration membership differs"):
            delivery.candidate_catalogue([self.candidate], centering_dir=self.root / "centering")

    def test_intact_compact_snapshot_checks_local_input_hashes_and_code(self):
        directory, _, _ = self.snapshot()
        result = delivery.verify_snapshot(directory, root=self.root, require_local_inputs=True)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["snapshot_files"], 6)
        self.assertEqual(result["local_input_hashes_checked"], 1)
        self.assertEqual(result["local_inputs_absent"], 0)

    def test_snapshot_artifact_tampering_is_rejected(self):
        directory, _, _ = self.snapshot()
        (directory / "candidate_evidence.csv").write_text("id,value\nchanged,1\n")
        with self.assertRaisesRegex(ValueError, "Changed task-two review artifact"):
            delivery.verify_snapshot(directory, root=self.root)

    def test_missing_snapshot_artifact_is_rejected(self):
        directory, _, _ = self.snapshot()
        (directory / "model_diagnostics.png").unlink()
        with self.assertRaisesRegex(ValueError, "Changed task-two review artifact"):
            delivery.verify_snapshot(directory, root=self.root)

    def test_unindexed_snapshot_additions_are_rejected(self):
        directory, _, _ = self.snapshot()
        (directory / "unindexed.csv").write_text("not part of the review\n")
        with self.assertRaisesRegex(ValueError, "inventory is incomplete or contains additions"):
            delivery.verify_snapshot(directory, root=self.root)

    def test_missing_local_inputs_are_reported_and_only_strict_mode_requires_them(self):
        directory, source, _ = self.snapshot()
        source.unlink()
        result = delivery.verify_snapshot(directory, root=self.root)
        self.assertEqual(result["local_inputs_absent"], 1)
        self.assertEqual(result["local_input_hashes_checked"], 0)
        with self.assertRaisesRegex(FileNotFoundError, "Missing local review source"):
            delivery.verify_snapshot(directory, root=self.root, require_local_inputs=True)

    def test_present_but_changed_local_input_is_never_a_missing_input(self):
        directory, source, _ = self.snapshot()
        source.write_text("changed original table\n")
        with self.assertRaisesRegex(ValueError, "Changed saved review source"):
            delivery.verify_snapshot(directory, root=self.root)

    def test_missing_implementation_is_rejected_even_in_portable_mode(self):
        directory, _, code = self.snapshot()
        code.unlink()
        with self.assertRaisesRegex(ValueError, "Changed review implementation"):
            delivery.verify_snapshot(directory, root=self.root)

    def test_snapshot_cannot_claim_independent_confirmation(self):
        directory, _, _ = self.snapshot()
        summary = json.loads((directory / "review.json").read_text())
        summary["independent_confirmation"] = True
        (directory / "review.json").write_text(json.dumps(summary))
        self.reindex(directory)
        with self.assertRaisesRegex(ValueError, "cannot claim training or independent confirmation"):
            delivery.verify_snapshot(directory, root=self.root)

    def test_public_output_is_one_new_version_and_cannot_overwrite(self):
        version = self.root / "submission/task2/verification/new_v1"
        with mock.patch.object(cli, "ROOT", self.root):
            self.assertEqual(cli.public_output(version), version)
            self.assertFalse(version.exists())
            version.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                cli.public_output(version)

    def test_public_output_rejects_outside_and_deep_nested_paths(self):
        with mock.patch.object(cli, "ROOT", self.root):
            for path in (self.root / "results/task2/review/v1", self.root / "submission/task2/verification/a/b"):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, "Public review must use"):
                    cli.public_output(path)

    def test_public_output_rejects_symlinks_even_when_target_is_inside_project(self):
        allowed = self.root / "submission/task2/verification"
        allowed.mkdir(parents=True)
        (allowed / "internal_target").mkdir()
        (allowed / "internal_alias").symlink_to(allowed / "internal_target", target_is_directory=True)
        with mock.patch.object(cli, "ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "cannot pass through a symlink"):
                cli.public_output(allowed / "internal_alias/new_version")

    def cli_fixture(self):
        configuration = self.root / "configs/shared/data.json"
        configuration.parent.mkdir(parents=True)
        configuration.write_text(json.dumps({"chrom_length": 10000}))
        code = self.root / "microcture/tasks/task2"
        code.mkdir(parents=True)
        for name in ("review.py", "diagnostics.py", "delivery.py"):
            (code / name).write_text("# fixture provenance\n")
        return {"summary": {}, "checks": {}, "reproducibility": {},
                "disjoint_candidate_rows": [], "sources": []}

    def test_cli_check_never_writes_output(self):
        audit = self.cli_fixture()
        before = {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with mock.patch.object(cli, "ROOT", self.root), \
             mock.patch.object(cli, "audit_discovery", return_value=audit), \
             mock.patch.object(cli, "load_diagnostics", return_value={"sources": [], "test_retrieval_rows": [], "score_quality_rows": []}), \
             mock.patch.object(cli, "candidate_catalogue", return_value=([], {}, [])), \
             mock.patch.object(cli, "project_sources", return_value=[]), \
             mock.patch.object(cli, "write_snapshot") as snapshot, \
             mock.patch.object(cli, "write_csv") as output_csv, \
             mock.patch.object(cli, "fresh_report_path") as fresh_report, \
             mock.patch.object(cli, "public_output") as public, \
             redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(cli.main(["check", "--data-root", str(self.root), "--experiment-dir", str(self.root / "results/task2/discovery/v2")]), 0)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "saved_table_consistency_passed")
            for helper in (snapshot, output_csv, fresh_report, public):
                helper.assert_not_called()
        after = {str(path): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_cli_check_rejects_output_arguments_before_reading_evidence(self):
        with mock.patch.object(cli, "ROOT", self.root), \
             mock.patch.object(cli, "audit_discovery") as audit, \
             redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                cli.main(["check", "--data-root", str(self.root), "--report-dir", str(self.root / "reports/task2/review/v1")])
            self.assertEqual(error.exception.code, 2)
            audit.assert_not_called()
        self.assertFalse((self.root / "reports").exists())

    def test_preflight_inspects_headers_without_creating_output_or_loading_contacts(self):
        configuration = self.root / "configs/shared/data.json"
        configuration.parent.mkdir(parents=True)
        configuration.write_text(json.dumps({"chrom_length": 10000, "samples": [{"sample_id": "WT1"}, {"sample_id": "WT2"}]}))
        config = self.root / "config.json"
        config.write_text(json.dumps({"window_bins": [64, 256], "seeds": [42, 43, 44], "clusters": [4, 6, 8, 12], "clustering_seeds": [42, 43, 44]}))
        self.table(self.root / "scan.csv", [{"window_id": "w64_c32"}])
        self.table(self.root / "data/shared/annotations/structures.csv", [{"structure_id": "known_1"}])
        args = SimpleNamespace(config=config, output_dir=self.root / "results/task2/discovery/future_v1",
                               scan_file=self.root / "scan.csv", band_cache_dir=self.root / "existing_cache")
        with mock.patch.object(cli, "ROOT", self.root), \
             mock.patch.object(cli, "fresh_output_path", return_value=args.output_dir), \
             mock.patch("contact_band.inspect_band_cache", return_value={"status": "ready"}) as inspect, \
             mock.patch("discovery_v2_core.validate_config", side_effect=lambda value: value), \
             mock.patch("discover_task2_v2.validate_scan_identity") as identity, \
             mock.patch("discover_task2_v2.load_band", side_effect=AssertionError("Preflight must never read contact pixels")), \
             mock.patch("discovery_v2_core.fit_model", side_effect=AssertionError("Preflight must never fit models")):
            result = cli.preflight(args)
            self.assertEqual(inspect.call_count, 2)
            identity.assert_called_once()
            self.assertEqual(result["status"], "metadata_preflight_passed")
            self.assertEqual(result["neural_fits"], 6)
            self.assertEqual(result["ranking_methods"], 14)
            self.assertEqual(result["cluster_runs"], 168)
            self.assertFalse(result["training_performed"])
            self.assertFalse(result["output_created"])
        self.assertFalse(args.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
