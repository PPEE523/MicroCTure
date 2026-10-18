"""CLI policy/filter boundaries exercised with mocks; never fit or infer."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from scripts import run_task1_reevaluation as entry
from microcture.task1_reevaluation import protocol
from microcture.task1_reevaluation import runner as actual_runner


class ReevaluationCLIIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.config = {"run_dir": "results/task1_reevaluation_fixture_never_created"}
        self.runner = ModuleType("microcture.task1_reevaluation.runner")
        self.runner.guard_output = Mock(side_effect=lambda root, path: path)
        self.runner.check_protocol = Mock(return_value={"status": "checked", "no_training": True})
        self.runner.run_jobs = Mock(return_value={"status": "partial", "expected_fits": 155,
                                                   "completed_fits": 50, "expected_endpoints": 290,
                                                   "completed_endpoints": 95, "numerical_failure_fits": 0,
                                                   "execution_failure_fits": 0})

    def invoke(self, args):
        with patch.dict(sys.modules, {"microcture.task1_reevaluation.runner": self.runner}), \
             patch.object(protocol, "load_protocol", return_value=self.config), \
             patch.object(sys, "argv", ["run_task1_reevaluation.py", *args]), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return entry.main()

    def test_main_passes_subset_filters_without_new_policy_argument(self):
        self.assertEqual(self.invoke(["main", "--model", "m2plus", "--condition", "I3", "--fold", "2", "--seed", "44"]), 0)
        args, kwargs = self.runner.run_jobs.call_args
        self.assertEqual(args[0], self.config)
        self.assertEqual(kwargs, {"stage": "main", "model": "m2plus", "condition": "I3", "fold": 2, "seed": 44})
        self.assertFalse(self.runner.check_protocol.called)

    def test_metadata_check_cannot_start_run_jobs(self):
        self.assertEqual(self.invoke(["check"]), 0)
        self.runner.check_protocol.assert_called_once()
        self.runner.run_jobs.assert_not_called()

    def test_failed_fit_summary_returns_failure_code(self):
        self.runner.run_jobs.return_value = {"status": "partial", "numerical_failure_fits": 1,
                                              "execution_failure_fits": 0}
        self.assertEqual(self.invoke(["all"]), 1)

    def test_invalid_summary_returns_failure_code(self):
        self.runner.run_jobs.return_value = {"status": "invalid", "numerical_failure_fits": 0,
                                              "execution_failure_fits": 0}
        self.assertEqual(self.invoke(["main"]), 1)

    def test_unregistered_seed_is_rejected_before_output_guard(self):
        with self.assertRaises(SystemExit) as error:
            self.invoke(["main", "--seed", "45"])
        self.assertEqual(error.exception.code, 2)
        self.runner.run_jobs.assert_not_called()
        self.runner.guard_output.assert_not_called()

    def test_policy_override_is_not_a_cli_option(self):
        with self.assertRaises(SystemExit) as error:
            self.invoke(["main", "--policy", "outer_best"])
        self.assertEqual(error.exception.code, 2)
        self.runner.run_jobs.assert_not_called()

    def test_output_does_not_exist_after_mocked_cli_calls(self):
        self.invoke(["main"])
        self.assertFalse((entry.ROOT / self.config["run_dir"]).exists())

    def test_saved_own_manifest_is_accepted_for_resume(self):
        with tempfile.TemporaryDirectory(prefix="microcture-reevaluation-guard-test-", dir="/tmp") as temporary:
            root = Path(temporary)
            directory = root / "results/reevaluation"
            directory.mkdir(parents=True)
            marker = directory / "manifest.json"
            marker.write_text(json.dumps({"protocol": protocol.PROTOCOL}), encoding="utf-8")
            self.assertEqual(actual_runner.guard_output(root, directory), directory)
            marker.write_text(json.dumps({"protocol": "unregistered_or_previous_experiment"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                actual_runner.guard_output(root, directory)

    def test_empty_subset_stops_before_manifest_and_creates_no_output(self):
        with tempfile.TemporaryDirectory(prefix="microcture-reevaluation-filter-test-", dir="/tmp") as temporary:
            root = Path(temporary)
            directory = root / "results/reevaluation"
            config = {"formal_run_dir": "results/formal", "readout_run_dir": "results/readout"}
            with patch.object(actual_runner, "planned_jobs", return_value=[]), \
                 patch.object(actual_runner, "prepare_manifest") as prepare:
                with self.assertRaisesRegex(ValueError, "No registered fits"):
                    actual_runner.run_jobs(config, root, directory, stage="main", model="pca_lr", seed=43)
                prepare.assert_not_called()
            self.assertFalse(directory.exists())


if __name__ == "__main__":
    unittest.main()
