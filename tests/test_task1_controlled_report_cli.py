"""Handwritten metadata/report fixtures only; no data copies or model fitting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import report_task1_controlled as wrapper
from scripts.task1_controlled_reporting import aggregate_heads


CASES = [f"case_{index}" for index in range(6)]
CONFIG = {
    "training": {"epochs": 4, "learning_rate": 0.001},
    "acceptance": {"window": 2, "CHID_F1_threshold": None},
    "input_stage": {"fold": 0, "seed": 42, "variants": [], "geometry_baselines": []},
    "head_stage": {"H3_learning_rate": 0.0001, "versions": ["H0", "H1", "H2", "H3"],
                   "cases": {case: {"backbone": "m2plus", "fold": 0, "seed": 42} for case in CASES}},
}


def diagnostic():
    return {"logit_between_structure_var": 1.0, "centered_logit_between_structure_var": 0.5,
            "logits": {"max_coordinate_range": 2.0}, "centered_logits": {"max_coordinate_range": 1.0},
            "probabilities": {"max_coordinate_range": 0.1},
            "prediction_counts": {"OPCID": 2, "CHIN": 6, "CHID": 2},
            "majority_prediction_fraction": 0.6,
            "head": {"available": True, "total_units": 4, "inactive_units": 0, "active_units": 4,
                     "effective_fraction": 0.5, "epsilon": 1e-6}}


def saved_record(case="case_0", head="H1"):
    history = []
    for epoch, loss in enumerate([1.2, 0.9, 0.7, 0.42, 0.4]):
        history.append({"epoch": epoch, "weighted_train_ce": loss,
                        "train_diagnostics": diagnostic(), "validation_diagnostics": diagnostic(),
                        "gradient_norms": None if epoch == 0 else {
                            group: {"mean": 1.0, "min": 0.5, "max": 2.0,
                                    "finite": True, "nonzero_fraction": 1.0}
                            for group in ("total", "encoder", "head")}})
    return {"case": case, "head": head, "status": "completed",
            "metrics": {"acceptance": {"accepted": True}}, "history": history}


class ControlledReportWrapperTests(unittest.TestCase):
    def _protocol_fixture(self, root):
        directory = root / "job"
        directory.mkdir()
        frozen = root / "frozen/folds"
        frozen.mkdir(parents=True)
        signal = {"signature": "fake-signal", "train_ids": ["train_0", "train_1"]}
        fold = {"fold": 1, "train_ids": ["train_0", "train_1"],
                "validation_ids": ["val_0"], "test_ids": ["outer_0"], "preprocessing": signal}
        (frozen / "fold_1.json").write_text(json.dumps(fold), encoding="utf-8")
        job = {"fold": 1, "name": "I3", "input_variant": "I3", "kind": "neural"}
        manifest = {"dataset_signature": "fake-dataset", "config": {
            "frozen_run_dir": "frozen", "training": {"epochs": 40}, "acceptance": {"window": 10}}}
        preprocessing = {
            "fold": 1, "dataset_signature": "fake-dataset",
            "frozen_fold_signature": wrapper.json_signature(fold),
            "signal_preprocessing": signal, "frozen_signal_preprocessing_signature": "fake-signal",
            "train_ids": ["train_0", "train_1"], "validation_ids": ["val_0"],
            "outer_images_used": False, "signal_statistics_refitted": False, "splits_refitted": False,
            "length_scaler": {"fitted_unique_structures": 2},
        }
        protocol = {"job": job, "training": manifest["config"]["training"],
                    "acceptance": manifest["config"]["acceptance"], "preprocessing": preprocessing}
        result = {"train_structures": 2, "validation_structures": 1, "outer_images_used": False}
        self._save_protocol(directory, protocol, result)
        return directory, job, result, manifest, protocol

    def _save_protocol(self, directory, protocol, result):
        pre = protocol["preprocessing"]
        pre["signature"] = wrapper.json_signature({key: value for key, value in pre.items() if key != "signature"})
        result["preprocessing_signature"] = pre["signature"]
        (directory / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")

    def test_protocol_rejects_resigned_outer_use_and_changed_inner_ids(self):
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="microcture-report-test-") as temporary:
            root = Path(temporary)
            directory, job, result, manifest, original = self._protocol_fixture(root)
            with patch.object(wrapper, "ROOT", root):
                verified = wrapper.verify_protocol(directory, job, result, manifest)
                self.assertEqual(verified["train_ids"], ["train_0", "train_1"])
                edits = [
                    ("outer_images_used", True, "Outer evaluation"),
                    ("signal_statistics_refitted", True, "Outer evaluation"),
                    ("validation_ids", ["outer_0"], "validation structure IDs"),
                    ("train_ids", ["train_0", "train_0"], "train structure IDs"),
                ]
                for key, value, error in edits:
                    protocol = copy.deepcopy(original)
                    metrics = copy.deepcopy(result)
                    protocol["preprocessing"][key] = value
                    self._save_protocol(directory, protocol, metrics)
                    with self.subTest(field=key), self.assertRaisesRegex(ValueError, error):
                        wrapper.verify_protocol(directory, job, metrics, manifest)
                self._save_protocol(directory, original, result)
                result["outer_images_used"] = True
                with self.assertRaisesRegex(ValueError, "Outer evaluation"):
                    wrapper.verify_protocol(directory, job, result, manifest)

    def test_protocol_rejects_changed_preprocessing_signature(self):
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="microcture-report-test-") as temporary:
            root = Path(temporary)
            directory, job, result, manifest, protocol = self._protocol_fixture(root)
            protocol["preprocessing"]["signature"] = "not-the-canonical-hash"
            result["preprocessing_signature"] = "not-the-canonical-hash"
            (directory / "protocol.json").write_text(json.dumps(protocol))
            with patch.object(wrapper, "ROOT", root), self.assertRaisesRegex(ValueError, "Preprocessing signature"):
                wrapper.verify_protocol(directory, job, result, manifest)

    def test_csv_uses_canonical_head_key_and_observed_denominators(self):
        failed = {"case": "case_1", "head": "H1", "status": "numerical_failure", "metrics": {}, "history": []}
        aggregate = aggregate_heads([saved_record(), failed], CASES, CONFIG)
        rows = wrapper.aggregate_csv_rows(aggregate)
        self.assertEqual([row["head"] for row in rows], ["H0", "H1", "H2", "H3"])
        row = next(row for row in rows if row["head"] == "H1")
        self.assertEqual((row["strict_pass_cases"], row["strict_observed_terminal_cases"]), (1, 2))
        self.assertEqual((row["completed_cases"], row["missing_cases"]), (1, 4))
        self.assertEqual((row["encoder_effective_gradient_numerator"], row["encoder_effective_gradient_denominator"]), (1, 1))
        self.assertEqual(row["validation_probabilities_constant_post_epoch_incidence_denominator"], 4)
        self.assertEqual(row["validation_probabilities_constant_final_epoch_case_incidence_denominator"], 1)
        self.assertEqual(next(row for row in rows if row["head"] == "H0")["strict_observed_terminal_cases"], 0)

    def test_markdown_uses_nonempty_canonical_candidate_recipe(self):
        records = [saved_record(case, head) for case in CASES for head in ("H0", "H1", "H2", "H3")]
        aggregate = aggregate_heads(records, CASES, CONFIG)
        report = {"input_review": [], "case_head_review": [], "head_aggregate": aggregate}
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="microcture-report-test-") as temporary:
            path = Path(temporary) / "report.md"
            wrapper.write_markdown(path, report)
            text = path.read_text()
        self.assertIn('"status": "candidate"', text)
        self.assertIn('"head_architecture": "H1"', text)
        self.assertNotIn("```json\n{}\n```", text)

    def test_build_report_writes_explicit_defer_decision_when_jobs_are_missing(self):
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="microcture-report-test-") as temporary:
            root = Path(temporary)
            run_dir, output_dir = root / "run", root / "report"
            run_dir.mkdir()
            config = copy.deepcopy(CONFIG)
            config["frozen_run_dir"] = "frozen"
            frozen = root / "frozen/folds"
            frozen.mkdir(parents=True)
            (frozen / "fold_0.json").write_text(json.dumps({
                "fold": 0, "train_ids": ["train_0"], "validation_ids": ["val_0"], "test_ids": ["outer_0"]}))
            metadata = {"structures": [
                {"structure_id": "train_0", "type": "CHIN"},
                {"structure_id": "val_0", "type": "CHID"},
                {"structure_id": "outer_0", "type": "CHID"}], "group_id": [0, 1, 2]}
            manifest = {"config": config, "source_signature": "fake-source", "dataset_signature": "fake-dataset",
                        "target_scope_preflight": {"historical_center_only_conflicts": []}}
            manifest["experiment_signature"] = wrapper.json_signature(manifest)
            (run_dir / "manifest.json").write_text(json.dumps(manifest))
            with patch.object(wrapper, "ROOT", root), \
                 patch.object(wrapper, "controlled_source_signature", return_value="fake-source"), \
                 patch.object(wrapper, "verify_frozen_metadata", return_value=metadata), \
                 patch.object(wrapper, "read_verified_record", return_value=None):
                report = wrapper.build_report(run_dir, output_dir)
            decision = json.loads((output_dir / "head_decision.json").read_text())
            self.assertEqual(decision, report["head_aggregate"]["candidate_recommendation"])
            self.assertEqual(decision["status"], "defer")
            self.assertIsNone(decision["recipe"])
            self.assertTrue(decision["reason"])
            self.assertTrue((output_dir / "head_aggregate.csv").is_file())
            self.assertIn('"status": "defer"', (output_dir / "report.md").read_text())
            self.assertEqual(report["input_scope"]["validation_structures"], 1)
            self.assertEqual(report["input_scope"]["CHID_independent_groups"], 1)


if __name__ == "__main__":
    unittest.main()
