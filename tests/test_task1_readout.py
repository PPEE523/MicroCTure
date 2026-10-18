"""Handwritten forward tensors and mocked fits only; no actual training."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from microcture.task1.evaluation import save_prediction_csv
from microcture.task1_controlled.models import build_controlled_model
from microcture.task1_controlled.runner import state_fingerprint
from microcture.task1_readout.models import build_model, target_gate, model_spec
from microcture.task1_readout.protocol import planned_jobs, validate_protocol, json_signature
from microcture.task1_readout import runner
from microcture.task1_readout import evaluation
from microcture.task1.evaluation import metrics

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/task1_readout.json").read_text())


def samples():
    return {"x": np.ones((3, 2, 3, 64, 64), dtype=np.float32),
            "length": np.asarray([[-1], [0], [1]], dtype=np.float32),
            "y": np.arange(3, dtype=np.int64), "structure_id": np.asarray(["OPCID_0", "CHIN_0", "CHID_0"]),
            "group_id": np.arange(3), "replicate": np.full(3, "fixture")}


class ReadoutModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_seed_state_and_budget_match_both_variants_and_frozen_m2plus(self):
        states = []
        for factory in (lambda: build_controlled_model("m2plus", CONFIG["models"], "H3"),
                        lambda: build_model("O0", CONFIG["models"]), lambda: build_model("O1", CONFIG["models"])):
            torch.manual_seed(42)
            model = factory()
            self.assertEqual(sum(p.numel() for p in model.parameters()), 110437)
            states.append((state_fingerprint(model), set(model.state_dict()), torch.get_rng_state()))
        self.assertEqual(states[0][0], states[1][0])
        self.assertEqual(states[1][0], states[2][0])
        self.assertEqual(states[0][1], states[2][1])
        self.assertTrue(torch.equal(states[0][2], states[2][2]))
        self.assertEqual(model_spec("O1", CONFIG["models"])["gate_parameters"], 0)

    def test_original_O0_and_empty_full_target_O1_are_exactly_equal(self):
        torch.manual_seed(44)
        original = build_controlled_model("m2plus", CONFIG["models"], "H3").eval()
        o0, o1 = build_model("O0", CONFIG["models"]).eval(), build_model("O1", CONFIG["models"]).eval()
        o0.load_state_dict(original.state_dict()); o1.load_state_dict(original.state_dict())
        x, length = torch.randn(2, 2, 3, 64, 64), torch.tensor([[-1.], [1.]])
        with torch.inference_mode():
            self.assertTrue(torch.equal(original(x, length), o0(x, length)))
            for value in (0., 1.):
                x[:, :, 2] = value
                self.assertTrue(torch.equal(o0(x, length), o1(x, length)))

    def test_gate_weights_and_reversal_are_hand_calculated(self):
        mask = torch.zeros(2, 1, 64, 64)
        mask[0, 0, :16, :16] = 1
        mask[1] = 1
        gate = target_gate(mask)
        expected = torch.full((4, 4), .5); expected[0, 0] = 8.5
        self.assertTrue(torch.equal(gate[0, 0], expected))
        self.assertTrue(torch.equal(gate[1], torch.ones_like(gate[1])))
        self.assertTrue(torch.equal(gate.mean((-2, -1)), torch.ones(2, 1)))
        self.assertTrue(torch.equal(target_gate(mask.flip(-1, -2)), gate.flip(-1, -2)))

    def test_gate_precedes_projection_and_position_scale_codes_are_unchanged(self):
        torch.manual_seed(42)
        model = build_model("O1", CONFIG["models"]).eval()
        x = torch.randn(1, 2, 3, 64, 64)
        x[:, :, 2] = 0; x[:, :, 2, :32, :32] = 1
        captured = {}
        def pool_hook(module, inputs, out): captured["pool"] = out.detach().clone()
        def projection_pre(module, inputs): captured["projection_in"] = inputs[0].detach().clone()
        def projection_hook(module, inputs, out): captured["projection_out"] = out.detach().clone()
        def block_pre(module, inputs): captured["tokens"] = inputs[0].detach().clone()
        handles = [model.backbone.spatial_pool.register_forward_hook(pool_hook),
                   model.backbone.token_projection.register_forward_pre_hook(projection_pre),
                   model.backbone.token_projection.register_forward_hook(projection_hook),
                   model.backbone.blocks.register_forward_pre_hook(block_pre)]
        try:
            with torch.inference_mode(): model.spatial_readout(x)
        finally:
            for handle in handles: handle.remove()
        expected = (captured["pool"] * target_gate(x[:, :, 2:3].reshape(2, 1, 64, 64))).flatten(2).transpose(1, 2)
        self.assertTrue(torch.equal(captured["projection_in"], expected))
        projected = captured["projection_out"].reshape(1, 2, 16, 64)
        expected_tokens = (projected + model.backbone.position_encoding[None, None]
                           + model.backbone.scale_embedding.weight[None, :, None]).reshape(1, 32, 64)
        self.assertTrue(torch.equal(captured["tokens"], expected_tokens))

    def test_protocol_declares_exact_60_jobs_and_rejects_other_changes(self):
        jobs = planned_jobs(CONFIG)
        self.assertEqual(len(jobs), 60)
        self.assertEqual(len(planned_jobs(CONFIG, "O1", "I3")), 15)
        self.assertEqual(len(planned_jobs(CONFIG, "O0", fold=2, seed=43)), 2)
        for key, value in (("gate_beta", .8), ("conditions", ["I3"]), ("seeds", [42])):
            c = copy.deepcopy(CONFIG); c[key] = value
            with self.assertRaises(ValueError): validate_protocol(c)
        c = copy.deepcopy(CONFIG); c["training"]["learning_rate"] = .001
        with self.assertRaises(ValueError): validate_protocol(c)


class ReadoutRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="microcture-readout-test-", dir="/tmp")
        self.root = Path(self.temp.name)
        self.config = copy.deepcopy(CONFIG)
        self.job = planned_jobs(self.config, "O1", "G0", fold=0, seed=42)[0]
        self.sample, self.pre = samples(), {"signature": "fixture-pre"}
        self.fold = {"train_ids": list(self.sample["structure_id"]), "validation_ids": list(self.sample["structure_id"]),
                     "test_ids": list(self.sample["structure_id"])}
        self.manifest = {"config": self.config, "source_signature": "fixture-source", "dataset_signature": "fixture-data",
                         "experiment_signature": "fixture-exp", "evaluation_scope": "fixture"}
        self.output = self.root / "job"; self.output.mkdir()
        self.prob = np.eye(3, dtype=np.float32) * .7 + .1

    def tearDown(self): self.temp.cleanup()

    def fake_fit(self, job, train, validation, config, model_config, device, output):
        self.assertTrue(np.all(train["x"][:, :, :2] == 0))
        self.assertTrue(np.all(train["x"][:, :, 2] == 1))
        np.testing.assert_array_equal(train["length"], self.sample["length"])
        (output / "final.pt").write_bytes(b"handwritten sealed endpoint fixture")
        save_prediction_csv(output / "validation_predictions.csv", validation, self.prob)
        (output / "history.json").write_text(json.dumps({"epochs": [{"epoch": i, "weighted_train_ce": .4} for i in range(41)]}))
        return {"status": "completed", "epochs": 40, "final_train_ce": .4, "acceptance": {"accepted": False}}

    def test_mock_endpoint_is_sealed_before_outer_and_constant_acceptance_does_not_suppress_score(self):
        def outer(dataset, fold, pre):
            endpoint = json.loads((self.output / "endpoint.json").read_text())
            self.assertFalse(endpoint["outer_images_constructed"])
            self.assertEqual(endpoint["checkpoint_sha256"], runner.sha256(self.output / "final.pt"))
            return self.sample
        def predict(model, outer, config, device):
            self.assertTrue(np.all(outer["x"][:, :, :2] == 0))
            return self.prob, {"fixture": True}
        with patch.object(runner, "fit_diagnostic", side_effect=self.fake_fit), \
             patch.object(runner, "prepare_outer", side_effect=outer), \
             patch.object(runner, "restore_endpoint", return_value=object()), \
             patch.object(runner, "outer_neural_endpoint", side_effect=predict):
            result = runner.fit_one(self.job, self.sample, self.sample, self.pre, {}, self.fold,
                                    self.config, self.manifest, self.output, torch.device("cpu"))
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["acceptance"]["accepted"])
        self.assertEqual(runner.assert_resume(self.output, self.job, self.manifest), result)
        (self.output / "outer_predictions.csv").write_text("changed probabilities")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            runner.assert_resume(self.output, self.job, self.manifest)

    def test_failed_fit_never_constructs_outer_and_saved_failure_is_preserved(self):
        with patch.object(runner, "fit_diagnostic", side_effect=RuntimeError("fixture failure")), \
             patch.object(runner, "prepare_outer") as outer:
            result = runner.fit_one(self.job, self.sample, self.sample, self.pre, {}, self.fold,
                                    self.config, self.manifest, self.output, torch.device("cpu"))
        outer.assert_not_called()
        self.assertEqual(result["status"], "numerical_failure")
        self.assertEqual(runner.assert_resume(self.output, self.job, self.manifest)["status"], "numerical_failure")

    def test_restore_rejects_cross_variant_endpoint(self):
        checkpoint = {"job": self.job, "epoch": 40, "model_config": CONFIG["models"], "head_version": "H3",
                      "backbone": "m2plus", "readout_variant": "O0"}
        with patch.object(runner.torch, "load", return_value=checkpoint), self.assertRaisesRegex(ValueError, "readout final"):
            runner.restore_endpoint(self.job, self.config, self.output / "final.pt", torch.device("cpu"))

    def test_output_never_overlaps_old_formal_results(self):
        with self.assertRaisesRegex(ValueError, "original formal"):
            runner.guard_output(self.root, self.root / "results/task1_formal_v1/child")

    def test_keyboard_interrupt_publishes_partial_trajectory_and_stops_entire_run(self):
        run_dir = self.root / "results/new"
        config = {**self.config, "run_dir": "results/new"}
        manifest = {**self.manifest, "config": config}
        first_job = planned_jobs(config, fold=0, seed=42)[0]
        def interrupted(job, train, validation, config, model_config, device, output):
            (output / "history.json").write_text(json.dumps({"epochs": [{"epoch": 0}, {"epoch": 1}]}))
            (output / "initialization.json").write_text(json.dumps({"initial_state_sha256": "fixture-init",
                                                                     "common_initial_state_sha256": "fixture-common"}))
            raise KeyboardInterrupt()
        with patch.object(runner, "prepare_manifest", return_value=(manifest, {})), \
             patch.object(runner, "load_dataset", return_value={"signature": "fixture-data"}), \
             patch.object(runner, "frozen_fold", return_value=self.fold), \
             patch.object(runner, "prepare_inner", return_value=(self.sample, self.sample, self.pre)), \
             patch.object(runner, "fit_diagnostic", side_effect=interrupted) as fitter, \
             patch.object(runner, "prepare_outer") as outer:
            with self.assertRaisesRegex(runner.RunStopped, "KeyboardInterrupt"):
                runner.run_jobs(config, self.root, run_dir, "cpu", fold=0, seed=42)
        fitter.assert_called_once(); outer.assert_not_called()
        destination = runner.job_directory(run_dir, first_job)
        self.assertTrue((destination / "ARTIFACTS.json").is_file())
        saved = runner.assert_resume(destination, first_job, manifest)
        self.assertEqual(saved["status"], "execution_failure")
        self.assertEqual(saved["error_type"], "KeyboardInterrupt")
        self.assertEqual(json.loads((destination / "history.json").read_text())["epochs"], [{"epoch":0},{"epoch":1}])
        with patch.object(runner, "prepare_manifest", return_value=(manifest, {})), \
             patch.object(runner, "load_dataset") as loader, \
             patch.object(runner, "fit_diagnostic") as fitter, \
             patch.object(evaluation, "summarize_readout", return_value={"status":"pending"}):
            runner.run_jobs(config, self.root, run_dir, "cpu", variant=first_job["variant"],
                            condition=first_job["condition"], fold=0, seed=42)
        loader.assert_not_called(); fitter.assert_not_called()

    def test_io_failure_during_failure_write_moves_scientific_staging_and_blocks_retry(self):
        run_dir = self.root / "results/new"
        config = {**self.config, "run_dir": "results/new"}
        manifest = {**self.manifest, "config": config}
        first_job = planned_jobs(config, fold=0, seed=42)[0]
        original_write = runner.write_json
        def disk_error(job, train, validation, config, model_config, device, output):
            (output / "history.csv").write_text("epoch,loss\n0,1.0\n1,0.9\n")
            raise OSError("fixture model-I/O error")
        def failed_write(path, value):
            if path.name == "failure.json": raise OSError("fixture failure-index I/O error")
            return original_write(path, value)
        with patch.object(runner, "prepare_manifest", return_value=(manifest, {})), \
             patch.object(runner, "load_dataset", return_value={"signature":"fixture-data"}), \
             patch.object(runner, "frozen_fold", return_value=self.fold), \
             patch.object(runner, "prepare_inner", return_value=(self.sample, self.sample, self.pre)), \
             patch.object(runner, "fit_diagnostic", side_effect=disk_error) as fitter, \
             patch.object(runner, "write_json", side_effect=failed_write):
            with self.assertRaisesRegex(runner.RunStopped, "unsealed scientific files retained"):
                runner.run_jobs(config, self.root, run_dir, "cpu", fold=0, seed=42)
        fitter.assert_called_once()
        preserved = runner.failed_staging_directory(run_dir, first_job)
        self.assertEqual((preserved / "history.csv").read_text(), "epoch,loss\n0,1.0\n1,0.9\n")
        with patch.object(runner, "prepare_manifest", return_value=(manifest, {})), \
             patch.object(runner, "load_dataset") as loader, \
             patch.object(runner, "fit_diagnostic") as fitter:
            with self.assertRaisesRegex(ValueError, "no automatic retry"):
                runner.run_jobs(config, self.root, run_dir, "cpu", variant=first_job["variant"],
                                condition=first_job["condition"], fold=0, seed=42)
        loader.assert_not_called(); fitter.assert_not_called()

    def test_persistent_storage_failure_keeps_tmp_marker_and_same_run_cannot_retry(self):
        original_write, original_mkdir = runner.write_json, Path.mkdir
        for fault in ("move", "mkdir"):
            with self.subTest(fault=fault):
                run_dir = self.root / f"results/fault_{fault}"
                config = {**self.config, "run_dir": f"results/fault_{fault}"}
                manifest = {**self.manifest, "config": config}
                first_job = planned_jobs(config, fold=0, seed=42)[0]
                def disk_error(job, train, validation, config, model_config, device, output):
                    (output / "history.csv").write_text("epoch,loss\n0,1.0\n1,0.9\n")
                    raise OSError("fixture model-I/O error")
                def failed_write(path, value):
                    if path.name == "failure.json": raise OSError("fixture failure-index error")
                    return original_write(path, value)
                def failed_mkdir(path, *args, **kwargs):
                    if "failed_staging" in path.parts: raise OSError("fixture result-storage mkdir error")
                    return original_mkdir(path, *args, **kwargs)
                try:
                    with patch.object(runner, "prepare_manifest", return_value=(manifest, {})), \
                         patch.object(runner, "load_dataset", return_value={"signature":"fixture-data"}), \
                         patch.object(runner, "frozen_fold", return_value=self.fold), \
                         patch.object(runner, "prepare_inner", return_value=(self.sample, self.sample, self.pre)), \
                         patch.object(runner, "fit_diagnostic", side_effect=disk_error) as fitter, \
                         patch.object(runner, "write_json", side_effect=failed_write), \
                         patch.object(runner.shutil, "move", side_effect=OSError("fixture move error")) if fault=="move" else patch.object(Path, "mkdir", new=failed_mkdir):
                        with self.assertRaisesRegex(runner.RunStopped, "unsealed scientific files retained"):
                            runner.run_jobs(config, self.root, run_dir, "cpu", fold=0, seed=42)
                    fitter.assert_called_once()
                    found = runner.unpublished_tmp_staging(self.root, run_dir)
                    identity = (first_job["condition"],first_job["model"],first_job["seed"],first_job["fold"])
                    self.assertEqual(len(found[identity]), 1)
                    preserved = found[identity][0]
                    self.assertEqual((preserved / "history.csv").read_text(), "epoch,loss\n0,1.0\n1,0.9\n")
                    marker = json.loads((preserved / runner.STAGING_CONTEXT).read_text())
                    self.assertEqual(marker["project_root"], str(self.root.resolve()))
                    self.assertEqual(marker["run_dir"], str(run_dir.resolve()))
                    self.assertEqual(marker["job"], first_job)
                    with patch.object(runner, "prepare_manifest", return_value=(manifest, {})), \
                         patch.object(runner, "load_dataset") as loader, \
                         patch.object(runner, "fit_diagnostic") as fitter:
                        with self.assertRaisesRegex(ValueError, "Unpublished /tmp.*no automatic retry"):
                            runner.run_jobs(config, self.root, run_dir, "cpu", variant=first_job["variant"],
                                            condition=first_job["condition"], fold=0, seed=42)
                    loader.assert_not_called(); fitter.assert_not_called()
                finally:
                    for directories in runner.unpublished_tmp_staging(self.root, run_dir).values():
                        for directory in directories: shutil.rmtree(directory)


class ReadoutSummaryTests(unittest.TestCase):
    """Synthetic complete/pending OOF; validates pairing without any fits."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="microcture-readout-summary-test-", dir="/tmp")
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "results/new"; self.run_dir.mkdir(parents=True)
        self.config = copy.deepcopy(CONFIG); self.config["run_dir"] = "results/new"
        names = ("OPCID", "CHIN", "CHID")
        rows = [{"structure_id": f"{names[i%3]}_{i}", "type": names[i%3], "start": i*1000, "end": i*1000+100}
                for i in range(15)]
        self.metadata = {"structures": rows, "group_id": list(range(15)), "signature": "fixture-data"}
        self.fold_ids = {}
        for fold in range(5):
            test = list(range(3*fold, 3*fold+3)); validation = list(range(3*((fold+1)%5), 3*((fold+1)%5)+3))
            train = [i for i in range(15) if i not in test+validation]
            self.fold_ids[str(fold)] = {key: [rows[i]["structure_id"] for i in ix]
                                       for key, ix in (("train_ids",train),("validation_ids",validation),("test_ids",test))}
        self.manifest = {"config": self.config, "protocol": self.config["protocol"], "source_signature": "fixture-source",
                         "dataset_signature": "fixture-data", "fold_ids": self.fold_ids}
        self.manifest["experiment_signature"] = json_signature(self.manifest)
        (self.run_dir / "manifest.json").write_text(json.dumps(self.manifest))
        self.pending = None

    def tearDown(self): self.temp.cleanup()

    def fixture_record(self, run_dir, job, manifest, identity, folds):
        if self.pending == (job["model"], job["condition"], job["seed"], job["fold"]):
            return {"job": job, "status": "pending"}
        ids = folds[job["fold"]]["test_ids"]
        labels = np.array([identity[sid]["y"] for sid in ids])
        probabilities = np.eye(3)[labels]*.7+.1
        if job["model"] == "O0":
            probabilities[labels==2] = [.1,.8,.1]
        rows = [{"structure_id": sid, "y_true": int(labels[i]), "y_pred": int(probabilities[i].argmax()),
                 **{f"p_{name}": float(probabilities[i,j]) for j,name in enumerate(("OPCID","CHIN","CHID"))}}
                for i,sid in enumerate(ids)]
        score = metrics(labels, probabilities)
        value = {"initial_state_sha256": f"init-{job['seed']}", "common_initial_state_sha256": f"common-{job['seed']}",
                 "minibatch_order_sha256": f"order-{job['seed']}", "preprocessing_signature": "fixture-pre",
                 "class_weights": [1,1,1], "acceptance": {"accepted": True}, "final_train_ce": .4,
                 "runtime_seconds": 0}
        return {"job": job, "status": "completed", "metrics": value, "history": [],
                "predictions": {"structure_id": ids, "probabilities": probabilities, "rows": rows, "metrics": score}}

    def summarize(self):
        with patch.object(evaluation, "_load_job", side_effect=self.fixture_record), \
             patch.object(evaluation, "assert_resume", return_value=None), \
             patch.object(evaluation, "_neural_summary", return_value={"strict_learning_passes": 15}), \
             patch.object(evaluation, "_project_path", side_effect=lambda path: self.root / path):
            return evaluation.summarize_readout(self.run_dir, self.metadata)

    def test_complete_60_jobs_create_exact_oof_and_variant_pairs(self):
        summary = self.summarize()
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["completed_jobs"], 60)
        self.assertEqual(len(summary["comparison"]), 4)
        for name in ("I3:O1-O0", "G0:O1-O0"):
            comparison = summary["paired_comparisons"][name]
            self.assertEqual(comparison["completed_pairs"], 15)
            self.assertEqual(len(comparison["complete_seed_oof_deltas"]), 3)
            self.assertTrue(all(row["CHID_f1_delta"]==1 for row in comparison["complete_seed_oof_deltas"]))
        self.assertEqual(summary["models"]["I3/O1"]["seeds"]["42"]["oof"]["n_structures"], 15)

    def test_missing_one_fold_never_becomes_a_zero_or_partial_mean(self):
        self.pending = ("O1", "I3", 42, 0)
        summary = self.summarize()
        self.assertEqual(summary["pending_jobs"], 1)
        row = next(row for row in summary["comparison"] if (row["condition"],row["model"]) == ("I3","O1"))
        self.assertIsNone(row["macro_f1_mean"])
        self.assertEqual(row["complete_seeds"], 2)
        status = json.loads((self.run_dir / "oof/I3/O1/seed_42/status.json").read_text())
        self.assertFalse(status["oof_current"])
        self.assertEqual(status["pending_folds"], [0])

    def test_interrupted_partial_history_keeps_declared_failure_without_invented_diagnostics(self):
        job = planned_jobs(self.config, "O0", "I3", fold=0, seed=42)[0]
        directory = runner.job_directory(self.run_dir, job); directory.mkdir(parents=True)
        value = {"job":job, "status":"execution_failure", "preprocessing_signature":"fixture-pre",
                 **{key:self.manifest[key] for key in ("source_signature","experiment_signature","dataset_signature")}}
        (directory / "metrics.json").write_text(json.dumps(value))
        (directory / "history.json").write_text('{"epochs":[')
        identity = evaluation._identity(self.metadata)
        folds = evaluation._fold_ids(self.manifest, identity, self.run_dir)
        record = evaluation._load_job(self.run_dir, job, self.manifest, identity, folds)
        self.assertEqual(record["status"], "execution_failure")
        self.assertEqual(record["history"], [])
        self.assertIn("history_read_error", record)
        self.assertEqual((directory / "history.json").read_text(), '{"epochs":[')


if __name__ == "__main__": unittest.main()
