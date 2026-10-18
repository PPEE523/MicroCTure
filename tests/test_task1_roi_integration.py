"""Independent hand fixtures for controls and scientific endpoint boundaries.

No optimizer, model fit, raw-band scan or real experiment directory is used.
"""
from __future__ import annotations

import json
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from microcture.task1_roi import experiment as E


ROOT = Path(__file__).resolve().parents[1]


def samples():
    x = np.zeros((6, 2, 3, 64, 64), dtype=np.float32)
    x[:, :, 1:] = 1
    x[:, :, 0, 0, 1] = x[:, :, 0, 1, 0] = 2
    x[:, :, 0, 1, 2] = x[:, :, 0, 2, 1] = 8
    x[:, :, 1, 1, 2] = x[:, :, 1, 2, 1] = .5
    return {"x": x, "length": np.array([[.2], [.2], [.4], [.4], [.6], [.6]], dtype=np.float32),
            "y": np.array([0, 0, 1, 1, 2, 2]), "structure_id": np.array(["A", "A", "B", "B", "C", "C"]),
            "group_id": np.array([0, 0, 1, 1, 2, 2]), "replicate": np.array([1, 2, 1, 2, 1, 2]),
            "sample_id": np.array(["rep1", "rep2"] * 3), "roi_resampling": [{"native_bin_count": 10}] * 6}


class ControlMeaningTests(unittest.TestCase):
    def test_conditions_separate_geometry_validity_and_signal(self):
        source = samples()
        g = E.apply_condition(source, "G0")
        v = E.apply_condition(source, "V0")
        np.testing.assert_array_equal(g["x"][:, :, :2], 0)
        np.testing.assert_array_equal(g["x"][:, :, 2], source["x"][:, :, 2])
        np.testing.assert_array_equal(v["x"][:, :, 0], 0)
        self.assertEqual(v["x"][:, :, 1:].tobytes(), source["x"][:, :, 1:].tobytes())
        self.assertEqual(v["length"].tobytes(), source["length"].tobytes())
        self.assertTrue((source["x"][:, :, 0] != 0).any())

    def test_profile_control_keeps_distance_weighted_statistics(self):
        source = samples()
        result = E.apply_condition(source, "P0")
        offset = np.abs(np.arange(64)[:, None] - np.arange(64)[None, :])
        for distance in (0, 1, 2, 63):
            mask = offset == distance
            valid = source["x"][:, :, 1, mask].astype(np.float64)
            before = (source["x"][:, :, 0, mask].astype(np.float64) * valid).sum(-1)
            after = (result["x"][:, :, 0, mask].astype(np.float64) * valid).sum(-1)
            np.testing.assert_allclose(after, before, atol=1e-5, rtol=1e-6)
        expected = 2 * (2 + 8 * .5) / (2 * (1 + .5) + 2 * 61)
        self.assertAlmostEqual(float(result["x"][0, 0, 0, 0, 1]), expected, places=7)
        self.assertEqual(result["x"][:, :, 1:].tobytes(), source["x"][:, :, 1:].tobytes())
        np.testing.assert_allclose(result["x"], result["x"].transpose(0, 1, 2, 4, 3), atol=1e-7, rtol=0)

    def test_profile_control_invalid_pixels_are_zero(self):
        source = samples()
        source["x"][:, :, 1, 0, 1] = source["x"][:, :, 1, 1, 0] = 0
        source["x"][:, :, 0, 0, 1] = source["x"][:, :, 0, 1, 0] = 100000
        result = E.apply_condition(source, "P0")
        np.testing.assert_array_equal(result["x"][:, :, 0, 0, 1], 0)
        np.testing.assert_array_equal(result["x"][:, :, 0, 1, 0], 0)

    def test_registered_budget_is_75_unique_paths(self):
        config = json.loads((ROOT / "configs/task1_roi.json").read_text())
        jobs = E.planned_jobs(config)
        self.assertEqual(len(jobs), 75)
        self.assertEqual(len({str(E.job_directory(Path('/tmp/run'), j)) for j in jobs}), 75)
        self.assertEqual(sum(j["alignment"] == "R0" for j in jobs), 15)
        self.assertTrue(all(j["variant"] == "O0" and j["learning_rate"] == 1e-4 for j in jobs))


class ReplicateScientificChecks(unittest.TestCase):
    def row(self):
        records = [{"replicate": 1, "probabilities": [.8, .1, .1], "predicted": 0},
                   {"replicate": 2, "probabilities": [.1, .7, .2], "predicted": 1}]
        return {"replicate_count": "2", "replicate_probabilities_json": json.dumps(records),
                "replicate_disagree": "1", "p_OPCID": ".45", "p_CHIN": ".4", "p_CHID": ".15"}

    def test_two_wt_mean_with_real_disagreement(self):
        E.check_replicate_rows([self.row()])

    def test_duplicate_wt_and_wrong_mean_are_rejected(self):
        row = self.row()
        records = json.loads(row["replicate_probabilities_json"])
        records[1]["replicate"] = 1
        duplicate = {**row, "replicate_probabilities_json": json.dumps(records)}
        wrong_mean = {**row, "p_OPCID": ".46", "p_CHIN": ".39"}
        for invalid in (duplicate, wrong_mean):
            with self.subTest(row=invalid):
                with self.assertRaises(ValueError):
                    E.check_replicate_rows([invalid])

    def test_invalid_wt_argmax_probability_and_disagreement_rejected(self):
        row = self.row()
        wrong_prediction = copy.deepcopy(row)
        records = json.loads(row["replicate_probabilities_json"])
        records[0]["predicted"] = 2
        wrong_prediction["replicate_probabilities_json"] = json.dumps(records)
        invalid_probability = copy.deepcopy(row)
        records = json.loads(row["replicate_probabilities_json"])
        records[0]["probabilities"] = [.8, .1, .2]
        invalid_probability["replicate_probabilities_json"] = json.dumps(records)
        for invalid in (wrong_prediction, invalid_probability, {**row, "replicate_disagree": "0"}):
            with self.subTest(row=invalid):
                with self.assertRaises(ValueError):
                    E.check_replicate_rows([invalid])


class EndpointBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/task1_roi.json").read_text())
        self.job = E.planned_jobs(self.config, alignment="R1", condition="I3", fold=0, seed=42)[0]
        self.fold = {f"{part}_ids": ["A", "B", "C"] for part in ("train", "validation", "test")}
        self.manifest = {"selection_protocol": "fixed final40", "evaluation_scope": "fixture",
                         "experiment_signature": "experiment", "source_signature": "source", "dataset_signature": "data"}
        self.preprocessing = {"signature": "preprocess"}
        self.source = samples()
        self.probabilities = np.array([[.8, .1, .1], [.6, .3, .1], [.1, .8, .1], [.1, .6, .3], [.1, .1, .8], [.3, .1, .6]])

    def test_outer_waits_for_sealed_endpoint_then_two_wt_mean_csv(self):
        events = []
        with tempfile.TemporaryDirectory(prefix="microcture-roi-integration-", dir="/tmp") as temp:
            output = Path(temp)

            def fit(*args):
                events.append("inner_fit")
                (output / "final.pt").write_bytes(b"fixed fixture weights")
                return {"status": "completed"}

            def outer(*args):
                endpoint = json.loads((output / "endpoint.json").read_text())
                self.assertTrue(endpoint["training_and_selection_complete"])
                self.assertFalse(endpoint["outer_images_constructed"])
                self.assertEqual(endpoint["checkpoint_sha256"], E.sha256(output / "final.pt"))
                events.append("outer")
                return self.source

            with patch.object(E, "fit_diagnostic", side_effect=fit), \
                 patch.object(E, "prepare_roi_outer", side_effect=outer), \
                 patch.object(E, "restore_endpoint", return_value=object()), \
                 patch.object(E, "outer_neural_endpoint", return_value=(self.probabilities, {})):
                result = E.fit_one(self.job, self.source, self.source, self.preprocessing, {}, self.fold,
                                   self.config, self.manifest, output, torch.device("cpu"))
            self.assertEqual(result["status"], "completed")
            self.assertEqual(events, ["inner_fit", "outer"])
            rows = E._read_prediction_csv(output / "outer_predictions.csv")["rows"]
            self.assertEqual(len(rows), 3)
            self.assertEqual([int(row["replicate_count"]) for row in rows], [2, 2, 2])
            np.testing.assert_allclose([float(rows[0][f"p_{c}"]) for c in E.CLASS_NAMES], [.7, .2, .1])
            self.assertTrue((output / "ARTIFACTS.json").is_file())

    def test_interrupt_keeps_existing_trajectory_and_never_outer(self):
        with tempfile.TemporaryDirectory(prefix="microcture-roi-integration-", dir="/tmp") as temp:
            output = Path(temp)

            def interrupted(*args):
                E.write_json(output / "history.json", {"epochs": [{"epoch": 0}]})
                raise KeyboardInterrupt("fixture interruption")

            with patch.object(E, "fit_diagnostic", side_effect=interrupted), patch.object(E, "prepare_roi_outer") as outer:
                result = E.fit_one(self.job, self.source, self.source, self.preprocessing, {}, self.fold,
                                   self.config, self.manifest, output, torch.device("cpu"))
            outer.assert_not_called()
            self.assertEqual(result["status"], "execution_failure")
            self.assertTrue(result["run_stop_requested"])
            self.assertEqual(result["failure_stage"], "inner_fit")
            artifact = json.loads((output / "ARTIFACTS.json").read_text())
            self.assertIn("history.json", {f["name"] for f in artifact["files"]})

    def test_missing_checkpoint_is_failed_without_outer(self):
        with tempfile.TemporaryDirectory(prefix="microcture-roi-integration-", dir="/tmp") as temp:
            with patch.object(E, "fit_diagnostic", return_value={"status": "completed"}), \
                 patch.object(E, "prepare_roi_outer") as outer:
                result = E.fit_one(self.job, self.source, self.source, self.preprocessing, {}, self.fold,
                                   self.config, self.manifest, Path(temp), torch.device("cpu"))
            outer.assert_not_called()
            self.assertEqual(result["status"], "execution_failure")
            self.assertTrue(result["run_stop_requested"])


if __name__ == "__main__":
    unittest.main()
