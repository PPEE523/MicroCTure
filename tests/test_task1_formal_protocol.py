"""Synthetic provenance/planning checks; no training, images or raw-band reads."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from microcture.task1_formal import protocol


PROJECT = Path(__file__).resolve().parents[1]


def fixture_config() -> dict:
    return json.loads((PROJECT / "configs/task1_formal.json").read_text(encoding="utf-8"))


def put_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def create_fixture(root: Path) -> dict:
    """Fifteen structures, five isolated spatial blocks and tiny file stubs."""
    config = fixture_config()
    for package, entry in (("task1", "run_task1_comparison.py"),
                           ("task1_controlled", "run_task1_controlled.py"),
                           ("task1_formal", "run_task1_formal.py")):
        directory = root / "microcture" / package
        directory.mkdir(parents=True)
        (directory / "__init__.py").write_text(f'"""{package}."""\n')
        (root / "scripts").mkdir(exist_ok=True)
        (root / "scripts" / entry).write_text(f'"""{entry}."""\n')
    cache = root / config["cache_dir"]
    cache.mkdir(parents=True)
    labels = ["OPCID", "CHIN", "CHID"]
    structures = [{"structure_id": f"{label}_{block}", "type": label,
                   "start": str(1000 + block * 20000), "end": str(2000 + block * 20000)}
                  for block in range(5) for label in labels]
    ids = [row["structure_id"] for row in structures]
    folds, records = [], []
    for fold in range(5):
        test = ids[3 * fold:3 * fold + 3]
        validation_fold = (fold + 1) % 5
        validation = ids[3 * validation_fold:3 * validation_fold + 3]
        training = [item for item in ids if item not in test + validation]
        folds.append({"fold": fold, "train_ids": training + validation, "test_ids": test})
        records.append({"fold": fold, "train_ids": training, "validation_ids": validation,
                        "test_ids": test})
    source = root / "data/processed/structures.csv"
    source.parent.mkdir(parents=True)
    source.write_text("structure_id,type\nfixture,CHID\n")
    samples = []
    for replica in (1, 2):
        band = cache / f"rep{replica}.band.npy"
        band.write_bytes(b"band")
        raw = root / "micro-c数据" / f"rep{replica}.cool"
        raw.parent.mkdir(exist_ok=True)
        raw.write_bytes(b"raw")
        samples.append({"band_file": band.name, "band_size_bytes": band.stat().st_size,
                        "band_mtime_ns": band.stat().st_mtime_ns, "raw_path": str(raw),
                        "raw_size_bytes": raw.stat().st_size, "raw_mtime_ns": raw.stat().st_mtime_ns})
    metadata = {"project_root": str(root), "config": {"resolution_bp": 100, "window_bins": [64, 256],
                 "output_size": 64, "guard_bins": 10, "normalization": "oe", "train_augmentation": "reverse"},
                "structures": structures, "group_id": [block for block in range(5) for _ in labels],
                "label_order": labels, "samples": samples,
                "input_sha256": {"data/processed/structures.csv": protocol.sha256(source)}}
    metadata["signature"] = protocol.json_signature({"metadata": metadata, "folds": folds})
    put_json(cache / "metadata.json", metadata)
    put_json(cache / "folds.json", folds)
    for record in records:
        fit = {"dataset_signature": metadata["signature"], "train_ids": record["train_ids"],
               "fit_scope": "training structure footprints and training replicate images only",
               "expected": [{"expected": [0.0, 0.0, 1.0], "training_footprint_bins": 3} for _ in (1, 2)],
               "scalers": [{"mean": 0.1, "std": 0.5, "fitted_valid_pooled_pixels": 20} for _ in (1, 2)]}
        fit["signature"] = protocol.json_signature(fit)
        record["preprocessing"] = fit
        put_json(root / config["frozen_run_dir"] / "folds" / f"fold_{record['fold']}.json", record)
    original = {"config": {"models": config["models"]}, "dataset_signature": metadata["signature"]}
    original["run_signature"] = protocol.json_signature(original)
    put_json(root / config["frozen_run_dir"] / "manifest.json", original)
    failed = {"run_dir": config["frozen_run_dir"], "cache_dir": config["cache_dir"],
              "dataset_signature": metadata["signature"], "run_signature": original["run_signature"],
              "source_signature": protocol._package_signature(root, "task1", "run_task1_comparison.py"),
              "files": []}
    for path in sorted((root / config["frozen_run_dir"]).rglob("*.json")) + sorted(cache.glob("*.json")):
        failed["files"].append({"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size,
                                "sha256": protocol.sha256(path)})
    failed_path = root / config["failed_evidence_dir"] / "freeze_manifest.json"
    put_json(failed_path, failed)
    put_json(root / failed["run_dir"] / "FROZEN.json", {"evidence_manifest_sha256": protocol.sha256(failed_path)})
    cases = {f"case_{index}": {"backbone": "m2plus", "fold": index % 5, "seed": 42} for index in range(6)}
    control = {"config": {"acceptance": config["acceptance"], "head_stage": {"cases": cases}}}
    control["experiment_signature"] = protocol.json_signature(control)
    put_json(root / "results/task1_controlled_v1/manifest.json", control)
    for name, case in cases.items():
        record = {"status": "completed", "acceptance": {"accepted": True},
                  "experiment_signature": control["experiment_signature"],
                  "job": {**case, "head": "H3", "input_variant": "I3", "learning_rate": 1e-4}}
        put_json(root / "results/task1_controlled_v1/heads" / name / "H3/metrics.json", record)
    controlled = {"run_dir": "results/task1_controlled_v1", "dataset_signature": metadata["signature"],
                  "source_signature": protocol._package_signature(root, "task1_controlled", "run_task1_controlled.py"),
                  "experiment_signature": control["experiment_signature"],
                  "original_run_signature": original["run_signature"],
                  "strict_pass_counts": {"H3": 6}, "strict_pass_denominator_per_head": 6,
                  "old_evidence_reference": {"sha256": protocol.sha256(failed_path)}, "files": []}
    for path in sorted((root / controlled["run_dir"]).rglob("*.json")):
        controlled["files"].append({"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size,
                                    "sha256": protocol.sha256(path)})
    controlled_path = root / config["controlled_evidence_dir"] / "freeze_manifest.json"
    put_json(controlled_path, controlled)
    put_json(root / controlled["run_dir"] / "FROZEN.json", {"evidence_manifest_sha256": protocol.sha256(controlled_path)})
    return config


class PlanningTests(unittest.TestCase):
    def test_registered_counts_and_no_duplicate_directories(self):
        config = fixture_config()
        self.assertEqual({stage: len(protocol.planned_jobs(config, stage)) for stage in
                          ("main", "ablations", "baselines", "all")},
                         {"main": 45, "ablations": 90, "baselines": 20, "all": 155})
        jobs = protocol.planned_jobs(config)
        self.assertEqual(len({protocol.job_directory(Path("results/new"), job) for job in jobs}), 155)
        self.assertEqual(len(protocol.planned_jobs(config, model="pca_lr")), 5)
        self.assertTrue(all(job["seed"] == 42 for job in jobs if job["kind"] == "baseline"))

    def test_filters_keep_paired_neural_cells(self):
        jobs = protocol.planned_jobs(fixture_config(), model="m2plus", fold=2, seed=43)
        self.assertEqual([job["condition"] for job in jobs], ["I3", "I0", "G0"])
        self.assertTrue(all(job["head"] == "H3" and job["learning_rate"] == 1e-4 for job in jobs))
        with self.assertRaises(ValueError):
            protocol.planned_jobs(fixture_config(), fold=True)

    def test_registered_settings_cannot_silently_drift(self):
        changes = [("epochs", 100), ("learning_rate", 1e-3), ("checkpoint_selection", "best_f1"),
                   ("early_stopping", True), ("scheduler", True), ("refit", True), ("head", "H2")]
        for key, value in changes:
            with self.subTest(key=key):
                config = fixture_config()
                config["training"][key] = value
                with self.assertRaises(ValueError):
                    protocol.validate_protocol(config)
        config = fixture_config()
        config["acceptance"]["CHID_F1_threshold"] = 0.9
        with self.assertRaises(ValueError):
            protocol.validate_protocol(config)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="microcture-formal-protocol-", dir="/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = create_fixture(self.root)

    def prepare(self):
        with patch.object(protocol, "environment_record", return_value={"python": "fixture"}):
            return protocol.prepare_manifest(self.root, self.config)

    def test_manifest_checks_tiny_provenance_without_result_creation(self):
        before = {str(path) for path in self.root.rglob("*")}
        manifest, metadata = self.prepare()
        self.assertEqual({str(path) for path in self.root.rglob("*")}, before)
        self.assertFalse((self.root / self.config["run_dir"]).exists())
        self.assertEqual(manifest["dataset_signature"], metadata["signature"])
        self.assertEqual(manifest["job_counts"]["all"], 155)
        self.assertEqual(len(manifest["fold_ids"]), 5)
        self.assertEqual(sum(len(row["test_ids"]) for row in manifest["fold_ids"].values()), 15)
        self.assertEqual(protocol.json_signature({k: v for k, v in manifest.items() if k != "experiment_signature"}),
                         manifest["experiment_signature"])
        # A second preflight is deterministic; source hashes include new code.
        self.assertEqual(self.prepare()[0], manifest)

    def test_modified_original_source_is_rejected(self):
        (self.root / "microcture/task1/__init__.py").write_text("changed source\n")
        with self.assertRaisesRegex(ValueError, "Original comparison source changed"):
            self.prepare()

    def test_changed_controlled_source_is_rejected(self):
        (self.root / "scripts/run_task1_controlled.py").write_text("changed source\n")
        with self.assertRaisesRegex(ValueError, "Controlled comparison source changed"):
            self.prepare()

    def test_freeze_marker_hash_detects_rewritten_inventory(self):
        path = self.root / self.config["failed_evidence_dir"] / "freeze_manifest.json"
        freeze = json.loads(path.read_text())
        freeze["files"] = []
        put_json(path, freeze)
        with self.assertRaisesRegex(ValueError, "differs from marker"):
            self.prepare()

    def test_modified_small_evidence_is_rejected(self):
        path = self.root / "results/task1_controlled_v1/heads/case_0/H3/metrics.json"
        value = json.loads(path.read_text())
        value["acceptance"]["accepted"] = False
        put_json(path, value)
        with self.assertRaisesRegex(ValueError, "Frozen file|Frozen small-file"):
            self.prepare()

    def test_unsafe_destination_and_symlink_are_rejected(self):
        for value in ("data/formal", "evidence/formal", "results", "results/task1_comparison_v1/new",
                      "results/task1_controlled_v1", "reports/new", str(self.root.parent / "outside")):
            with self.subTest(path=value):
                config = copy.deepcopy(self.config)
                config["run_dir"] = value
                with self.assertRaises(ValueError):
                    protocol.validate_run_directory(self.root, config)
        alias = self.root / "results/link"
        alias.symlink_to(self.root / "results/task1_comparison_v1", target_is_directory=True)
        self.config["run_dir"] = "results/link/new"
        with self.assertRaises(ValueError):
            self.prepare()

    def test_existing_new_freeze_blocks_descendant_and_ancestor(self):
        put_json(self.root / "results/other_frozen/FROZEN.json", {})
        for value in ("results/other_frozen/new", "results/other_frozen"):
            self.config["run_dir"] = value
            with self.assertRaisesRegex(ValueError, "protected"):
                self.prepare()

    def test_fold_group_leakage_is_rejected_without_fitting(self):
        cache = self.root / self.config["cache_dir"]
        metadata = json.loads((cache / "metadata.json").read_text())
        metadata["group_id"] = [0] * 15
        cached = json.loads((cache / "folds.json").read_text())
        with self.assertRaisesRegex(ValueError, "Spatial groups"):
            protocol._validate_folds(self.root, self.config, metadata, cached)

    def test_signal_scaler_training_ownership_is_rejected_without_fitting(self):
        cache = self.root / self.config["cache_dir"]
        metadata = json.loads((cache / "metadata.json").read_text())
        cached = json.loads((cache / "folds.json").read_text())
        path = self.root / self.config["frozen_run_dir"] / "folds/fold_0.json"
        record = json.loads(path.read_text())
        fit = record["preprocessing"]
        fit["train_ids"] = record["train_ids"] + record["test_ids"]
        fit["signature"] = protocol.json_signature({k: v for k, v in fit.items() if k != "signature"})
        put_json(path, record)
        with self.assertRaisesRegex(ValueError, "inner-training ID"):
            protocol._validate_folds(self.root, self.config, metadata, cached)


if __name__ == "__main__":
    unittest.main()
