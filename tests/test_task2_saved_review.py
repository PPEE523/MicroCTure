"""Small, independent saved-table fixtures; no model training or raw data."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from microcture.tasks.task2 import audit_discovery


class SavedDiscoveryReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def js(self, name, value):
        (self.root/name).write_text(json.dumps(value), encoding="utf-8")

    def csv(self, name, rows, fields=None):
        with (self.root/name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def table(self, name):
        with (self.root/name).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def fixture(self, missing_candidate=False, missing_controls=False, trained=False, ambiguous_cluster=False):
        cfg = dict(resolution_bp=100, seeds=[7] if trained else [], full_candidate_quotas={"2": 2},
                   test_candidate_quotas={"2": 1}, random_references=1,
                   min_valid_fraction=.8, max_zero_fraction=.8,
                   controls_per_candidate=2, min_controls=2, control_quantile=.95,
                   replicate_min_correlation=.5, clustering_seeds=[42, 43], clusters=[2],
                   min_new_cluster_members=2 if ambiguous_cluster else 5, cluster_min_jaccard=.6,
                   primary_method="shape", primary_k=2, primary_cluster_seed=42,
                   ae_max_epochs=2, ae_max_train_windows=20)
        scan = [dict(window_id=f"window_{i}", start_bin=i*10, width_bins=2,
                     center_bin=i*10+1, split=[0, 0, 0, 1, 2][i], eligible=True,
                     valid_fraction=1, zero_fraction=.1, mean_oe=1,
                     known_ids="known_A" if i == 0 else "", recalled_ids="") for i in range(5)]
        self.csv("scan_rep1.csv", scan)
        selected = {"shape": [0, 1], "density": [1, 4]}
        if trained:
            selected.update({f"{family}{fusion}_7": [0, 1] for family in ("ae", "dae") for fusion in ("", "_fusion")})
        frozen = dict(selected=selected, test_selected={method: [4] for method in selected},
                      random_selected={"full": [[0, 4]], "test": [[4]]},
                      controls={str(i): [2, 3] for i in [0, 1, 4]},
                      clustering={f"{method}:2:{seed}": dict(labels=[0, 1], centers=[])
                                  for method in selected for seed in [42, 43]})
        if ambiguous_cluster:
            for seed in [42, 43]:
                frozen["clustering"][f"density:2:{seed}"]["labels"] = [0, 0]
        self.js("frozen_before_rep2_reload.json", frozen)
        corr = [.9, None if missing_candidate else .6, None if missing_controls else .2,
                None if missing_controls else .3, .8]
        self.csv("replicate_correlations.csv", [dict(window_index=i, correlation=x) for i, x in enumerate(corr)])
        validation = []
        for i in [0, 1, 4]:
            validation.append(dict(window_index=i, window_id=f"window_{i}", start_bp=i*1000,
                                   window_length_bp=200, wraps_origin=False,
                                   known_ids=scan[i]["known_ids"], split=scan[i]["split"],
                                   correlation=corr[i], matched_controls=0 if missing_controls else 2,
                                   cutoff=None if missing_controls else .5,
                                   replicate_supported=not missing_controls and corr[i] is not None,
                                   coordinate_status="provisional"))
        self.csv("candidate_validation.csv", validation)
        supported = {row["window_index"]: row["replicate_supported"] for row in validation}
        clusters = []
        for method, ids in selected.items():
            for seed in [42, 43]:
                labels = frozen["clustering"][f"{method}:2:{seed}"]["labels"]
                for label in set(labels):
                    members = [ids[j] for j, value in enumerate(labels) if value == label]
                    annotated = sum(i == 0 for i in members)
                    support_count = sum(supported[i] for i in members)
                    rule = annotated == 0 and support_count >= cfg["min_new_cluster_members"]
                    clusters.append(dict(method=method, k=2, seed=seed, cluster=label, members=len(members),
                                         annotated_members=annotated, supported_members=support_count,
                                         min_seed_jaccard=1, passes_original_rule=rule, stable_candidate_new_cluster=rule))
        self.csv("clusters.csv", clusters)
        self.csv("cluster_stability.csv", [dict(method=method, k=2, mean_pairwise_ari=1) for method in selected])
        methods = [dict(method=method, candidates=2, covered_bins=4,
                        unannotated=sum(i != 0 for i in ids),
                        replicate_supported=sum(supported[i] for i in ids),
                        unannotated_supported=sum(i != 0 and supported[i] for i in ids),
                        passing_cluster_configurations=0, stable_cluster_configurations=0)
                   for method, ids in selected.items()]
        self.csv("method_summary.csv", methods)
        union = [row for row in validation if row["window_index"] != 0 and row["replicate_supported"]]
        self.csv("unannotated_supported_union.csv", union, list(validation[0]))
        self.csv("unannotated_supported_disjoint.csv", sorted(union, key=lambda row: -row["correlation"]), list(validation[0]))
        training = []
        if trained:
            for family in ("ae", "dae"):
                model = f"{family}_7"
                self.js(f"{model}_history.json", [dict(epoch=1, train_loss=.4, validation_loss=.3),
                                                 dict(epoch=2, train_loss=.35, validation_loss=.3000005)])
                training.append(dict(model=model, family=family, seed=7, epochs=2, best_epoch=1,
                                     train_windows=3, validation_windows=1, validation_loss=.3, test_loss=.32))
        self.csv("training_summary.csv", training, ["model", "family", "seed", "epochs", "best_epoch", "train_windows", "validation_windows", "validation_loss", "test_loss"])
        self.js("run_manifest.json", dict(config=cfg))
        self.js("report.json", dict(config=cfg, scan_windows=5, eligible_windows=5,
                                    partition_windows=dict(train=3, validation=1, test=1), training=training, methods=methods,
                                    unique_selected_windows=3, unannotated_supported_union=len(union),
                                    unannotated_supported_disjoint=len(union), primary_candidate_new_clusters=0,
                                    all_passing_cluster_configurations=0, cluster_runs=len(selected)*2,
                                    frozen_sha256=hashlib.sha256((self.root/"frozen_before_rep2_reload.json").read_bytes()).hexdigest()))

    def reseal_fixture(self):
        report = json.loads((self.root/"report.json").read_text())
        report["frozen_sha256"] = hashlib.sha256((self.root/"frozen_before_rep2_reload.json").read_bytes()).hexdigest()
        self.js("report.json", report)

    def test_dynamic_fixture_reconstructs_decisions(self):
        self.fixture()
        result = audit_discovery(self.root, genome_bins=100)
        self.assertEqual(result["summary"]["methods"], 2)
        self.assertEqual(result["summary"]["unannotated_supported_union"], 2)
        self.assertEqual(result["summary"]["nonempty_cluster_records"], 8)
        self.assertEqual([r["window_index"] for r in result["disjoint_candidate_rows"]], [4, 1])
        self.assertTrue(all(row["status"] == "passed" for row in result["checks"]))
        self.assertTrue(all(Path(source["path"]).is_absolute() for source in result["sources"]))
        json.dumps(result, allow_nan=False)

    def test_threshold_tampering_is_rejected(self):
        self.fixture()
        rows = self.table("candidate_validation.csv")
        rows[0]["cutoff"] = .4
        self.csv("candidate_validation.csv", rows)
        with self.assertRaisesRegex(ValueError, "control threshold mismatch"):
            audit_discovery(self.root, genome_bins=100)

    def test_cluster_label_tampering_is_rejected_even_with_updated_hash(self):
        self.fixture()
        frozen = json.loads((self.root/"frozen_before_rep2_reload.json").read_text())
        frozen["clustering"]["shape:2:42"]["labels"] = [0, 0]
        self.js("frozen_before_rep2_reload.json", frozen)
        self.reseal_fixture()
        with self.assertRaisesRegex(ValueError, "nonempty cluster record"):
            audit_discovery(self.root, genome_bins=100)

    def test_fixed_budget_tampering_is_rejected(self):
        self.fixture()
        frozen = json.loads((self.root/"frozen_before_rep2_reload.json").read_text())
        frozen["selected"]["shape"] = [0]
        self.js("frozen_before_rep2_reload.json", frozen)
        self.reseal_fixture()
        with self.assertRaisesRegex(ValueError, "budget/quota mismatch"):
            audit_discovery(self.root, genome_bins=100)

    def test_missing_candidate_correlation_stays_na(self):
        self.fixture(missing_candidate=True)
        result = audit_discovery(self.root, genome_bins=100)
        row = next(r for r in result["candidate_rows"] if r["window_index"] == 1)
        self.assertIsNone(row["replicate_supported"])
        self.assertEqual(row["assessment_status"], "not_evaluable")
        self.assertEqual(row["assessment_reason"], "missing_replicate_correlation")
        self.assertEqual(result["summary"]["not_evaluable_candidates"], 1)
        self.assertFalse(row["in_unannotated_supported_union"])
        json.dumps(result, allow_nan=False)

    def test_missing_control_correlations_stay_na(self):
        self.fixture(missing_controls=True)
        result = audit_discovery(self.root, genome_bins=100)
        self.assertEqual(result["summary"]["not_evaluable_candidates"], 3)
        self.assertTrue(all(r["replicate_supported"] is None for r in result["candidate_rows"]))
        self.assertTrue(all(r["cutoff"] is None for r in result["candidate_rows"]))

    def test_circular_overlap_is_rejected(self):
        self.fixture()
        rows = self.table("scan_rep1.csv")
        rows[0]["start_bin"] = 99
        rows[1]["start_bin"] = 0
        self.csv("scan_rep1.csv", rows)
        with self.assertRaisesRegex(ValueError, "Overlapping circular windows"):
            audit_discovery(self.root, genome_bins=100)

    def test_duplicate_scan_id_is_rejected(self):
        self.fixture()
        rows = self.table("scan_rep1.csv")
        rows[1]["window_id"] = rows[0]["window_id"]
        self.csv("scan_rep1.csv", rows)
        with self.assertRaisesRegex(ValueError, "Duplicate scan window ID"):
            audit_discovery(self.root, genome_bins=100)

    def test_frozen_hash_mismatch_is_rejected(self):
        self.fixture()
        path = self.root/"frozen_before_rep2_reload.json"
        path.write_text(path.read_text()+"\n")
        with self.assertRaisesRegex(ValueError, "Frozen selection/labels hash mismatch"):
            audit_discovery(self.root, genome_bins=100)

    def test_histories_can_pass_while_checkpoint_reproduction_is_unavailable(self):
        self.fixture(trained=True)
        result = audit_discovery(self.root, genome_bins=100)
        reproduction = result["reproducibility"]
        self.assertEqual(reproduction["saved_table_audit"], "passed")
        self.assertEqual(reproduction["model_checkpoint_status"], "unavailable_missing_checkpoints")
        self.assertEqual(reproduction["missing_checkpoints"], ["ae_7.pt", "dae_7.pt"])
        self.assertFalse(reproduction["model_weights_loaded"])

    def test_tampered_best_epoch_does_not_match_saved_history(self):
        self.fixture(trained=True)
        rows = self.table("training_summary.csv")
        rows[0]["best_epoch"] = 2
        self.csv("training_summary.csv", rows)
        with self.assertRaisesRegex(ValueError, "best_epoch mismatch"):
            audit_discovery(self.root, genome_bins=100)

    def test_legacy_false_cluster_promotes_to_unknown_when_support_is_missing(self):
        self.fixture(missing_candidate=True, ambiguous_cluster=True)
        result = audit_discovery(self.root, genome_bins=100)
        self.assertIsNone(result["summary"]["all_passing_cluster_configurations"])
        self.assertEqual(result["summary"]["confirmed_passing_cluster_configurations"], 0)
        self.assertEqual(result["summary"]["not_evaluable_cluster_configurations"], 2)
        density = [row for row in result["cluster_rows"] if row["method"] == "density"]
        self.assertTrue(all(row["stable_candidate_new_cluster"] is None for row in density))

    def test_current_nullable_cluster_and_report_are_supported(self):
        self.fixture(missing_candidate=True, ambiguous_cluster=True)
        clusters = self.table("clusters.csv")
        for row in clusters:
            if row["method"] == "density":
                row["passes_original_rule"] = None
                row["stable_candidate_new_cluster"] = None
        self.csv("clusters.csv", clusters)
        report = json.loads((self.root/"report.json").read_text())
        report["all_passing_cluster_configurations"] = None
        report["confirmed_passing_cluster_configurations"] = 0
        report["not_evaluable_cluster_configurations"] = 2
        self.js("report.json", report)
        result = audit_discovery(self.root, genome_bins=100)
        self.assertIsNone(result["summary"]["all_passing_cluster_configurations"])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
