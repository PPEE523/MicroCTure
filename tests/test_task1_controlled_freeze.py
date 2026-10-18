"""Standard-library fixtures for controlled evidence freezing; no model execution."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/freeze_task1_controlled_evidence.py"
SPEC = importlib.util.spec_from_file_location("controlled_freeze", SCRIPT)
freeze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze)


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    else:
        path.write_text(value, encoding="utf-8")


def fixture(root: Path, accepted: bool = False) -> tuple[Path, Path, Path]:
    config = {"input_stage": {"variants": ["I0", "I1", "I2", "I3"],
                              "geometry_baselines": ["length_only", "mask_only", "mask_length"],
                              "fold": 2, "seed": 42},
              "head_stage": {"versions": ["H0", "H1", "H2", "H3"], "H3_learning_rate": 1e-4,
                             "cases": {f"case_{index}": {"backbone": "m2plus", "fold": index % 5, "seed": 42}
                                       for index in range(6)}},
              "training": {"epochs": 2, "learning_rate": 1e-3},
              "acceptance": {"CHID_F1_threshold": None},
              "evidence_dir": "evidence/task1_failed_v1", "cache_dir": "data/cache/task1_comparison_v1"}
    for relative in ("microcture/task1_controlled/__init__.py", "scripts/run_task1_controlled.py",
                     "scripts/report_task1_controlled.py", "scripts/task1_controlled_reporting.py"):
        write(root / relative, "# hand-written fixture source\n")
    for relative in ("environment.yml", "environment.lock.yml"):
        write(root / relative, "name: microcture\n")
    write(root / "configs/task1_controlled.json", config)
    old_index = root / "evidence/task1_failed_v1/freeze_manifest.json"
    write(old_index, {"dataset_signature": "fixture-dataset", "run_signature": "fixture-old-run"})
    run_dir, report_dir, evidence_dir = [root / relative for relative in (
        "results/task1_controlled_v1", "reports/task1_controlled_review_v1", "evidence/task1_controlled_v1")]
    manifest = {"config": config, "source_signature": freeze.controlled_source_signature(root),
                "dataset_signature": "fixture-dataset", "original_run_signature": "fixture-old-run",
                "frozen_evidence_sha256": freeze.sha256(old_index)}
    manifest["experiment_signature"] = freeze.json_signature(manifest)
    write(run_dir / "manifest.json", manifest)
    inputs, heads = [], []
    for job in freeze.planned_jobs(config):
        directory = run_dir / "input" / job["name"] if job["stage"] == "input" else run_dir / "heads" / job["case"] / job["name"]
        metric = {"status": "completed", "job": job, "experiment_signature": manifest["experiment_signature"]}
        row = {"status": "completed", "name": job["name"]}
        write(directory / "protocol.json", {"job": job, "training": config["training"], "acceptance": config["acceptance"]})
        write(directory / "validation_predictions.csv", "structure_id,y_true,y_pred\nexample,1,1\n")
        if job["kind"] == "neural":
            diag = {"prediction_counts": [0, 1, 0]}
            metric.update({"initial_train_ce": 1.1, "final_train_ce": 1.0,
                           "train_diagnostics": diag, "validation_diagnostics": diag,
                           "acceptance": {"accepted": accepted, "primary": {"failed_checks": [] if accepted else ["encoder_gradient_finite_effective"]}}})
            row.update({"accepted": accepted, "final_train_ce": 1.0})
            history = [{"epoch": epoch, "weighted_train_ce": 1.1 if epoch == 0 else 1.0,
                        "train_diagnostics": diag, "validation_diagnostics": diag} for epoch in range(3)]
            write(directory / "history.json", {"epochs": history})
            write(directory / "history.csv", "epoch,weighted_train_ce\n0,1.1\n1,1.0\n2,1.0\n")
            write(directory / "initialization.json", {"fixture": True})
            write(directory / "final.pt", "fixture checkpoint\n")
            write(directory / "failure_trace.png", "fixture scientific failure evidence\n")
        write(directory / "metrics.json", metric)
        if job["stage"] == "input":
            inputs.append(row)
        else:
            row.update({"case": job["case"], "head": job["head"], "history_acceptance_agrees": True,
                        "failed_checks": metric["acceptance"]["primary"]["failed_checks"]})
            heads.append(row)
    report = {"experiment_signature": manifest["experiment_signature"], "source_signature": manifest["source_signature"],
              "source_signature_unchanged": True, "frozen_metadata_and_inner_ids_verified": True,
              "pending": [], "input_review": inputs, "case_head_review": heads,
              "head_aggregate": {"comparison_status": "fullcoverage", "decision_ready": True,
                                 "candidate_recommendation": {"status": "candidate" if accepted else "deferred"},
                                 "head_summaries": {head: {"completed_cases": 6, "missing_cases": 0, "numerical_failure_cases": 0,
                                                           "strict_accepted": {"numerator": 6 if accepted else 0, "denominator": 6}}
                                                    for head in ("H0", "H1", "H2", "H3")}}}
    write(report_dir / "summary.json", report)
    write(report_dir / "report.md", "# fixture report\n")
    write(root / "docs/TASK1_FORMAL_PROTOCOL.md", "# fixture formal protocol\n")
    return run_dir, report_dir, evidence_dir


class ControlledFreezeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="task1_controlled_freeze_", dir="/tmp")
        self.root = Path(self.temporary.name)
        self.run, self.report, self.evidence = fixture(self.root)

    def tearDown(self):
        # The tests own only this temporary tree. Restore permissions so it can
        # be removed without touching any project evidence.
        for path in [self.root, *self.root.rglob("*")]:
            if not path.is_symlink():
                path.chmod(stat.S_IMODE(path.stat().st_mode) | (0o700 if path.is_dir() else 0o600))
        self.temporary.cleanup()

    def freeze(self):
        return freeze.freeze_evidence(self.root, self.evidence, self.run, self.report,
                                      [Path("docs/TASK1_FORMAL_PROTOCOL.md")], "2026-10-12")

    def test_low_pass_counts_can_be_frozen_and_reverified(self):
        result = self.freeze()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["strict_pass_counts"], dict.fromkeys(("H0", "H1", "H2", "H3"), 0))
        manifest = freeze.read_json(self.evidence / "freeze_manifest.json")
        self.assertEqual(manifest["completed_jobs"], {"input": 7, "heads": 24})
        self.assertFalse(manifest["candidate_recipe"]["supported_by_all_six_cases"])
        self.assertIn("not a recognition-performance milestone", manifest["candidate_recipe"]["scope"])
        self.assertEqual(freeze.verify_evidence(self.root, self.evidence), result)
        self.assertEqual(self.freeze(), result)

    def test_freezes_originals_and_snapshot_without_copying_data(self):
        self.freeze()
        for directory in (self.run, self.report, self.evidence):
            self.assertEqual(directory.stat().st_mode & 0o222, 0)
        self.assertEqual((self.run / "heads/case_0/H0/failure_trace.png").stat().st_mode & 0o222, 0)
        self.assertNotEqual((self.root / "scripts/run_task1_controlled.py").stat().st_mode & 0o222, 0)
        self.assertTrue((self.evidence / "source_snapshot/docs/TASK1_FORMAL_PROTOCOL.md").is_file())
        self.assertFalse((self.evidence / "source_snapshot/data").exists())
        self.assertFalse((self.evidence / "source_snapshot/evidence").exists())

    def test_hash_tampering_is_detected(self):
        self.freeze()
        path = self.run / "heads/case_0/H0/final.pt"
        path.chmod(0o644)
        path.write_text("changed checkpoint\n")
        path.chmod(0o444)
        with self.assertRaisesRegex(ValueError, "Missing/size changed|Hash changed"):
            freeze.verify_evidence(self.root, self.evidence)

    def test_missing_case_is_rejected_before_creating_evidence(self):
        (self.run / "heads/case_5/H3/metrics.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing completed case"):
            self.freeze()
        self.assertFalse(self.evidence.exists())
        self.assertNotEqual(self.run.stat().st_mode & 0o222, 0)

    def test_incomplete_report_is_rejected(self):
        path = self.report / "summary.json"
        report = freeze.read_json(path)
        report["pending"] = [{"case": "case_5"}]
        write(path, report)
        with self.assertRaisesRegex(ValueError, "full-coverage"):
            self.freeze()
        self.assertFalse(self.evidence.exists())

    def test_report_signature_mismatch_is_rejected(self):
        path = self.report / "summary.json"
        report = freeze.read_json(path)
        report["source_signature"] = "changed"
        write(path, report)
        with self.assertRaisesRegex(ValueError, "report source/experiment signature"):
            self.freeze()

    def test_current_source_change_is_rejected(self):
        write(self.root / "scripts/run_task1_controlled.py", "# changed trained source\n")
        with self.assertRaisesRegex(ValueError, "controlled source differs"):
            self.freeze()

    def test_symbolic_link_artifact_is_rejected(self):
        (self.run / "linked_checkpoint.pt").symlink_to(self.run / "heads/case_0/H0/final.pt")
        with self.assertRaisesRegex(ValueError, "symbolic-link"):
            self.freeze()
        self.assertFalse(self.evidence.exists())

    def test_unindexed_file_is_detected(self):
        self.freeze()
        self.run.chmod(0o755)
        write(self.run / "unexpected.txt", "new artifact\n")
        self.run.chmod(0o555)
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            freeze.verify_evidence(self.root, self.evidence)

    def test_unindexed_file_beside_snapshot_is_detected(self):
        self.freeze()
        self.evidence.chmod(0o755)
        write(self.evidence / "unexpected.txt", "new artifact\n")
        self.evidence.chmod(0o555)
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            freeze.verify_evidence(self.root, self.evidence)

    def test_aggregate_disagreement_is_rejected(self):
        path = self.report / "summary.json"
        report = freeze.read_json(path)
        report["head_aggregate"]["head_summaries"]["H0"]["strict_accepted"]["numerator"] = 1
        write(path, report)
        with self.assertRaisesRegex(ValueError, "Aggregated acceptance"):
            self.freeze()

    def test_manifest_tampering_is_detected_by_marker(self):
        self.freeze()
        path = self.evidence / "freeze_manifest.json"
        content = freeze.read_json(path)
        content["freeze_date"] = "2026-10-13"
        path.chmod(0o644)
        write(path, content)
        path.chmod(0o444)
        with self.assertRaisesRegex(ValueError, "manifest differs from marker"):
            freeze.verify_evidence(self.root, self.evidence)


if __name__ == "__main__":
    unittest.main()
