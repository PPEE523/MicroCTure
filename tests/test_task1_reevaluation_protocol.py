"""Preregistered job accounting and forbidden protocol changes; no fitting."""
import copy
import json
from pathlib import Path
import unittest

from microcture.task1_reevaluation.protocol import (endpoint_directory, endpoint_policies,
    planned_jobs, validate_protocol)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/task1_reevaluation.json").read_text())


class ReevaluationProtocolTests(unittest.TestCase):
    def test_registered_fit_endpoint_counts_and_main_contains_pca(self):
        expected = {"main": (50, 95), "ablations": (90, 180), "baselines": (20, 20), "all": (155, 290)}
        for stage, (fits, endpoints) in expected.items():
            jobs = planned_jobs(CONFIG, stage)
            self.assertEqual(len(jobs), fits)
            self.assertEqual(sum(len(endpoint_policies(job)) for job in jobs), endpoints)
        self.assertEqual(sum(job["model"] == "pca_lr" for job in planned_jobs(CONFIG, "main")), 5)
        self.assertEqual(len({(j["condition"], j["model"], j["seed"], j["fold"]) for j in planned_jobs(CONFIG)}), 155)

    def test_filters_preserve_task_identity_and_baseline_single_seed(self):
        jobs = planned_jobs(CONFIG, "main", "m2plus", "I3", 0, 43)
        self.assertEqual(len(jobs), 1)
        self.assertEqual((jobs[0]["stage"], jobs[0]["head"], jobs[0]["learning_rate"]), ("reevaluation", "H3", 1e-4))
        self.assertEqual(planned_jobs(CONFIG, "main", "pca_lr", seed=43), [])
        self.assertEqual(planned_jobs(CONFIG, "ablations", "pca_lr"), [])

    def test_only_checkpoint_policy_can_change(self):
        mutations = [("training", "learning_rate", 2e-4), ("training", "epochs", 20),
                     ("training", "early_stopping", True), ("selection", "tie_break", "latest"),
                     ("selection", "inner_ce_metric", "CHID_F1")]
        for section, field, value in mutations:
            changed = copy.deepcopy(CONFIG)
            changed[section][field] = value
            with self.assertRaises(ValueError):
                validate_protocol(changed)
        changed = copy.deepcopy(CONFIG)
        changed["unregistered_tuning"] = True
        with self.assertRaises(ValueError):
            validate_protocol(changed)

    def test_policy_paths_forbid_cross_kind_endpoint(self):
        neural = planned_jobs(CONFIG, "main", "m2", fold=0, seed=42)[0]
        baseline = planned_jobs(CONFIG, "main", "pca_lr", fold=0)[0]
        self.assertEqual(endpoint_directory(Path("/tmp/fixture"), neural, "inner_ce").name, "inner_ce")
        with self.assertRaises(ValueError):
            endpoint_directory(Path("/tmp/fixture"), baseline, "inner_ce")
        with self.assertRaises(ValueError):
            endpoint_directory(Path("/tmp/fixture"), neural, "../../escape")


if __name__ == "__main__":
    unittest.main()
