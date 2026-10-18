"""Portable final workflow checks; no model training or raw pixel scanning."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from microcture.tasks.task1 import prediction, protocol, comparison as runner

ROOT = Path(__file__).resolve().parents[1]


class CountingPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, length):
        self.calls += 1
        return torch.cat((length, -length, torch.zeros_like(length)), dim=1)


class SubmissionWorkflowChecks(unittest.TestCase):
    def setUp(self):
        self.config = protocol.load_protocol(ROOT / "configs/task1_local.json")

    def test_exact_model_and_experiment_counts(self):
        self.assertEqual({name: len(protocol.planned_jobs(self.config, name)) for name in ("main", "ablations", "baselines", "all")},
                         {"main": 50, "ablations": 90, "baselines": 15, "all": 155})
        jobs = protocol.planned_jobs(self.config, "main")
        self.assertEqual(sum(job["kind"] == "neural" for job in jobs), 45)
        self.assertEqual(sum(job["model"] == "pca_lr" for job in jobs), 5)
        self.assertEqual(len({str(protocol.job_directory(Path("new"), job)) for job in protocol.planned_jobs(self.config)}), 155)
        for name in ("m2", "m2plus", "cnn_transformer"):
            self.assertEqual(len(protocol.planned_jobs(self.config, "main", model=name)), 15)
        self.assertEqual(len(protocol.planned_jobs(self.config, "main", model="pca_lr")), 5)

    def test_protocol_does_not_allow_optimizer_or_checkpoint_drift(self):
        for field, value in (("learning_rate", .001), ("epochs", 100), ("head", "H0"), ("refit", True)):
            changed = copy.deepcopy(self.config)
            changed["training"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                protocol.validate_protocol(changed)

    def test_check_uses_only_packaged_small_metadata(self):
        with patch.object(runner, "load_dataset", side_effect=AssertionError("Count bands must not be opened")), \
                patch.object(runner, "prepare_inner", side_effect=AssertionError("No image construction")), \
                patch.object(runner, "fit_diagnostic", side_effect=AssertionError("No training")):
            checked = runner.check_protocol(ROOT, self.config, ROOT / "results/task1/model_comparison/check_only_fixture", "cpu")
        self.assertEqual(checked["n_structures"], 344)
        self.assertFalse(checked["training_launched"])
        self.assertFalse(checked["real_images_preprocessed"])
        self.assertEqual(checked["models"]["m2plus"]["input_shape"], [2, 3, 64, 64])
        self.assertFalse((ROOT / "results/task1/model_comparison/check_only_fixture").exists())

    def test_packaged_fold_wrapper_is_parsed_without_changing_signature(self):
        directory = ROOT / self.config["frozen_run_dir"]
        manifest, _ = protocol.prepare_manifest(ROOT, self.config, "cpu")
        fold, target = protocol.fold_record(directory, 0)
        self.assertEqual(runner.frozen_fold(ROOT, self.config, manifest, 0), fold)
        self.assertEqual(target["dataset_signature"], manifest["dataset_signature"])

    def test_merged_models_load_registered_checkpoint_and_reproduce_demo(self):
        """A refactor must preserve saved tensor keys and actual predictions."""
        import numpy as np
        checkpoint = ROOT / "submission/artifacts/task1/m2plus/seed_42/fold_0/final.pt"
        expected = json.loads((ROOT / "submission/demo/task1/expected.json").read_text())
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(4)
            model, saved = prediction.load_checkpoint(checkpoint, torch.device("cpu"))
            self.assertEqual(set(model.state_dict()), set(saved["state_dict"]))
            samples = prediction.load_input_npz(ROOT / "submission/demo/task1/input.npz")
            probabilities = prediction.predict_samples(model, samples, torch.device("cpu"))
            np.testing.assert_allclose(probabilities, expected["replicate_probabilities"], rtol=1e-5, atol=1e-6)
        finally:
            torch.set_num_threads(previous_threads)

    def test_batched_predict_averages_replicates_after_softmax(self):
        model = CountingPredictor()
        samples = {"x": np.zeros((4, 2, 3, 64, 64), np.float32),
                   "length": np.array([[1.], [-2.], [3.], [0.]], np.float32),
                   "structure_id": np.array(["A", "B", "A", "B"]), "replicate": np.array([1, 1, 2, 2])}
        probabilities = prediction.predict_samples(model, samples, torch.device("cpu"), 3)
        self.assertEqual(model.calls, 2)
        rows = prediction.prediction_rows(samples, probabilities)
        expected = probabilities[[0, 2]].astype(np.float64).mean(axis=0)
        np.testing.assert_allclose([rows[0]["p_OPCID"], rows[0]["p_CHIN"], rows[0]["p_CHID"]], expected)
        self.assertEqual(rows[0]["replicate_count"], 2)
        self.assertNotIn("y_true", rows[0])

    def test_candidate_parser_accepts_unlabelled_intervals_and_rejects_bad_bounds(self):
        with tempfile.TemporaryDirectory(prefix="microcture-submission-", dir="/tmp") as temporary:
            path = Path(temporary) / "candidates.csv"
            path.write_text("structure_id,start,end\nnew,100,500\n")
            rows = prediction.read_intervals(path, 10000)
            self.assertEqual(rows[0]["center"], 300)
            self.assertNotIn("type", rows[0])
            path.write_text("structure_id,start,end\nnew,500,100\n")
            with self.assertRaises(ValueError):
                prediction.read_intervals(path, 10000)

    def test_candidate_windows_follow_candidate_centers_without_fitting(self):
        from microcture.tasks.task1 import data as signal
        from microcture.tasks.task1 import data as geometry
        dataset = {"signature": "synthetic", "n_bins": 4000,
                   "config": {"window_bins": [64, 256], "output_size": 64, "resolution_bp": 100},
                   "structure_id": np.array(["old"]), "center_bin": np.array([1000]),
                   "window_footprints": np.arange(872, 1128)[None, :],
                   "metadata": {"chrom_length": 400000, "structures": []}}
        signal_fit = {"expected": [], "scalers": []}
        signal_fit["signature"] = protocol.json_signature(signal_fit)
        preprocessing = {"dataset_signature": "synthetic", "signal_preprocessing": signal_fit,
                         "length_scaler": {"mean": 3., "std": .5}}
        preprocessing["signature"] = protocol.json_signature(preprocessing)
        rows = [{"structure_id": "new", "start": 179000., "end": 181000., "center": 180000.}]
        raw = {"x": np.zeros((1, 2, 2, 64, 64), np.float32), "structure_id": np.array(["new"]), "y": np.array([0])}
        def extract(candidate, indices, expected):
            np.testing.assert_array_equal(candidate["center_bin"], [1800])
            np.testing.assert_array_equal(candidate["window_footprints"], np.arange(1672, 1928)[None, :])
            return raw
        with patch.object(signal, "_extract_inputs", side_effect=extract), \
                patch.object(signal, "_apply_scalers", return_value=raw), \
                patch.object(signal, "_fit_expected", side_effect=AssertionError("No signal fitting")), \
                patch.object(geometry, "_fit_length_scaler", side_effect=AssertionError("No length fitting")):
            result = prediction.prepare_candidates(dataset, rows, preprocessing)
        self.assertEqual(result["x"].shape, (1, 2, 3, 64, 64))
        self.assertAlmostEqual(float(result["length"][0, 0]), (np.log1p(20)-3)/.5, places=6)

    def test_fresh_prediction_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="microcture-submission-", dir="/tmp") as temporary:
            root = Path(temporary)
            path = root / "results/task1/inference/new_v1/predictions.csv"
            rows = [{"structure_id": "A", "predicted": "CHIN", "replicate_count": 2,
                     "p_OPCID": .2, "p_CHIN": .7, "p_CHID": .1}]
            prediction.save_predictions(path, rows, root=root)
            with self.assertRaises(FileExistsError):
                prediction.save_predictions(path, rows, root=root)

    def test_prediction_output_guard_protects_markerless_retained_trees_and_source(self):
        with tempfile.TemporaryDirectory(prefix="microcture-prediction-guard-", dir="/tmp") as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            retained = {"result": "results/task1/ablation/dropout/v1",
                        "report": "reports/task1/audit/v1", "cache": "data/task1/cache/saved_v1",
                        "raw": "data/raw/micro_c", "source": "microcture/tasks/task1"}
            (root / "configs/layout.json").write_text(json.dumps({"moves": [
                {"old": f"legacy/{kind}", "new": name, "kind": kind}
                for kind, name in retained.items()
            ]}))
            for kind, name in retained.items():
                with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "retained"):
                    prediction.guard_prediction_output(root, root / name / "new.csv")
            for name in ("microcture/untracked/new.csv", "results/task2/inference/v1/new.csv",
                         "results/task1/new.csv", "results/task1/inference/v1/output.json"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "fresh CSV"):
                    prediction.guard_prediction_output(root, root / name)
            fresh = root / "results/task1/inference/new_v1/predictions.csv"
            self.assertEqual(prediction.guard_prediction_output(root, fresh), fresh)
            self.assertFalse(fresh.parent.exists())

    def test_prediction_cli_rejects_markerless_retained_output_before_computation(self):
        from scripts.task1 import predict_task1_final as cli
        with tempfile.TemporaryDirectory(prefix="microcture-prediction-guard-", dir="/tmp") as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            retained = root / "results/task1/ablation/dropout/v1"
            retained.mkdir(parents=True)
            evidence = retained / "metrics.json"
            evidence.write_bytes(b'{"status":"complete"}\n')
            before = evidence.read_bytes()
            (root / "configs/layout.json").write_text(json.dumps({"moves": [
                {"old": "results/task1_dropout_v1", "new": str(retained.relative_to(root)), "kind": "result"}
            ]}))
            self.assertFalse((retained / "FROZEN.json").exists())
            self.assertFalse((retained / "ARTIFACTS.json").exists())
            for destination in (retained / "new.csv", root / "results/task1_dropout_v1/new.csv"):
                with patch.object(cli, "ROOT", root), \
                        patch.object(sys, "argv", ["predict_task1_final.py", "--output", str(destination)]), \
                        patch.object(prediction, "load_checkpoint", side_effect=AssertionError("No weights loaded")), \
                        patch.object(torch, "set_num_threads", side_effect=AssertionError("No computation settings")):
                    with self.assertRaisesRegex(ValueError, "retained"):
                        cli.main()
            self.assertEqual([path.name for path in retained.iterdir()], ["metrics.json"])
            self.assertEqual(evidence.read_bytes(), before)

    def test_output_guard_rejects_other_workspace_and_packaged_artifacts(self):
        with self.assertRaises(ValueError):
            runner.guard_output(ROOT, ROOT / "submission/new")
        with tempfile.TemporaryDirectory(prefix="microcture-submission-", dir="/tmp") as temporary:
            with self.assertRaises(ValueError):
                runner.guard_output(ROOT, Path(temporary) / "results/new")

    def test_output_guard_requires_task_experiment_and_version_but_allows_resume(self):
        with tempfile.TemporaryDirectory(prefix="microcture-result-guard-", dir="/tmp") as temporary:
            root = Path(temporary)
            for name in ("results/new", "results/task1", "results/task1/model_comparison",
                         "results/task2/model_comparison/new_v1"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "results/task1"):
                    runner.guard_output(root, root / name)
            fresh = root / "results/task1/model_comparison/new_v1"
            self.assertEqual(runner.guard_output(root, fresh), fresh)
            self.assertFalse(fresh.exists())
            fresh.mkdir(parents=True)
            self.assertEqual(runner.guard_output(root, fresh), fresh)
            deeper = root / "results/task1/ablation/input/new_v1"
            self.assertEqual(runner.guard_output(root, deeper), deeper)

    def test_output_guard_rejects_markerless_retained_results_and_old_aliases(self):
        with tempfile.TemporaryDirectory(prefix="microcture-result-guard-", dir="/tmp") as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            retained = root / "results/task1/ablation/roi/v1"
            retained.mkdir(parents=True)
            (retained / "metrics.json").write_text("{}")
            (root / "configs/layout.json").write_text(json.dumps({"moves": [
                {"old": "results/task1_roi_v1", "new": str(retained.relative_to(root)), "kind": "result"}
            ]}))
            self.assertFalse((retained / "FROZEN.json").exists())
            for path in (retained, retained / "extra", retained.parent, root / "results/task1_roi_v1"):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, "retained"):
                    runner.guard_output(root, path)
            self.assertEqual(runner.guard_output(root, retained.parent / "v2"), retained.parent / "v2")

    def test_output_guard_preserves_marker_and_symlink_rejections(self):
        with tempfile.TemporaryDirectory(prefix="microcture-result-guard-", dir="/tmp") as temporary:
            root = Path(temporary)
            protected = root / "results/task1/ablation/marker/v1"
            protected.mkdir(parents=True)
            (protected / "FROZEN.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "frozen"):
                runner.guard_output(root, protected / "extra")
            outside = root / "external"
            outside.mkdir()
            linked = root / "results/task1/model_comparison/linked_v1"
            linked.parent.mkdir(parents=True)
            linked.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Symbolic-link"):
                runner.guard_output(root, linked)
            existing = root / "results/task1/model_comparison/existing_v1"
            existing.mkdir()
            (existing / "linked_artifact").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "Symbolic-link artifacts"):
                runner.guard_output(root, existing)


if __name__ == "__main__":
    unittest.main()
