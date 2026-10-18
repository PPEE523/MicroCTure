"""Synthetic identity, missing-evidence and completion checks; no model training."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from microcture.layout import bootstrap
bootstrap()
import discovery_v2_core as core
import discover_task2_v2 as workflow
from discover_task2 import annotations


def config():
    return json.loads((ROOT / 'configs/task2/discovery/discovery_v2.json').read_text())


class SyntheticAE(torch.nn.Module):
    """One scalar for testing publication; the optimizer is replaced by a mock."""
    def __init__(self, latent):
        super().__init__()
        self.level = torch.nn.Parameter(torch.tensor(.1))
        self.latent = latent

    def forward(self, x):
        return self.level.expand(len(x), x.shape[-2], x.shape[-1]), torch.zeros(len(x), self.latent)


class ProtocolValidationTests(unittest.TestCase):
    def test_registered_configuration_is_valid(self):
        core.validate_config(config())

    def test_overlapping_missing_and_out_of_range_roles_are_rejected(self):
        for change in [dict(validation_block_remainders=[5, 6, 7]),
                       dict(test_block_remainders=[8]), dict(test_block_remainders=[8, 10])]:
            cfg = config()
            cfg.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                core.validate_config(cfg)
            with self.assertRaises(ValueError):
                core.bin_roles(100, cfg)

    def test_unsupported_and_ambiguous_protocols_fail_before_output(self):
        changes = [dict(resolution_bp=500), dict(seeds=[42, 42]), dict(clustering_seeds=[42]),
                   dict(clusters=[1, 4]), dict(primary_k=100), dict(primary_method='cherry_picked'),
                   dict(primary_cluster_seed=0), dict(min_controls=21), dict(mask_probability=float('nan')),
                   dict(full_candidate_quotas={'64': 0, '256': 49}),
                   dict(test_candidate_quotas={'64': 20}), dict(ae_batch_size=True)]
        for change in changes:
            cfg = config()
            cfg.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                core.validate_config(cfg)

    def test_invalid_run_never_enters_signature_or_creates_output(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(prefix='microcture-task2-integrity-') as folder:
            path = Path(folder)
            cfg = config()
            cfg['resolution_bp'] = 500
            (path / 'config.json').write_text(json.dumps(cfg))
            args = SimpleNamespace(config=path / 'config.json', output_dir=path / 'never')
            with mock.patch.object(workflow, 'signatures', side_effect=AssertionError('inputs read')):
                with self.assertRaisesRegex(ValueError, '100'):
                    workflow.run_experiment(args)
            self.assertFalse(args.output_dir.exists())


class ScanIdentityTests(unittest.TestCase):
    def setUp(self):
        self.n = 512
        self.cfg = dict(window_bins=[64], stride_bins=64)
        self.known = [dict(structure_id='S1', type='CHIN', start='12800', end='16000', center='14400')]
        self.rows = []
        for center in range(0, self.n, 64):
            start = center - 32
            hits, recalled = annotations(start, 64, self.n, self.known)
            self.rows.append(dict(window_id=f'w64_c{center}', start_bin=str(start), width_bins='64',
                                  center_bin=str(center), known_ids=';'.join(hits), recalled_ids=';'.join(recalled)))

    def test_complete_grid_and_current_annotation_are_accepted(self):
        actual = workflow.validate_scan_identity(self.rows, self.known, self.n, self.cfg)
        self.assertEqual(len(actual), len(self.rows))

    def test_duplicate_missing_extra_and_geometry_are_rejected(self):
        wrong_geometry = copy.deepcopy(self.rows)
        wrong_geometry[0]['start_bin'] = '99'
        extra = dict(self.rows[0], window_id='extra')
        for rows in [self.rows + [self.rows[0]], self.rows[:-1], self.rows + [extra], wrong_geometry]:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                workflow.validate_scan_identity(rows, self.known, self.n, self.cfg)

    def test_stale_and_duplicate_annotations_fail_without_relabeling(self):
        for key, value in [('known_ids', ''), ('recalled_ids', ''), ('known_ids', 'S1;S1')]:
            rows = copy.deepcopy(self.rows)
            target = next(row for row in rows if row['known_ids'] and row['recalled_ids'])
            target[key] = value
            before = copy.deepcopy(rows)
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'annotation mismatch'):
                workflow.validate_scan_identity(rows, self.known, self.n, self.cfg)
            self.assertEqual(rows, before)


class NullableReplicationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(min_controls=2, control_quantile=.95, replicate_min_correlation=.5)

    def test_missing_correlation_or_controls_are_not_negative(self):
        for corr, controls in [(None, [.2, .3]), (float('nan'), [.2, .3]), (.8, [None, .2]), (.8, [float('inf'), .2])]:
            result = workflow.assess_replicate(corr, controls, self.cfg)
            self.assertIsNone(result['replicate_supported'])
            self.assertEqual(result['replicate_status'], 'not_evaluable')
            self.assertTrue(result['replicate_reason'])
            json.dumps(result, allow_nan=False)

    def test_observed_positive_and_negative_rules_are_unchanged(self):
        positive = workflow.assess_replicate(.8, [.2, .3], self.cfg)
        negative = workflow.assess_replicate(.4, [.2, .3], self.cfg)
        equal = workflow.assess_replicate(.5, [.2, .3], self.cfg)
        self.assertIs(positive['replicate_supported'], True)
        self.assertIs(negative['replicate_supported'], False)
        self.assertIs(equal['replicate_supported'], False)

    def test_cluster_is_nullable_only_when_missing_members_can_change_decision(self):
        self.assertEqual(workflow.assess_cluster(0, 3, 2, 5, .8, .6), (None, None))
        self.assertEqual(workflow.assess_cluster(0, 5, 2, 5, .8, .6), (True, True))
        self.assertEqual(workflow.assess_cluster(0, 2, 2, 5, .8, .6), (False, False))
        self.assertEqual(workflow.assess_cluster(1, 3, 2, 5, .8, .6), (False, False))
        self.assertEqual(workflow.assess_cluster(0, 3, 2, 5, .5, .6), (None, False))

    def test_historical_972_observed_candidate_decisions_are_unchanged(self):
        path = ROOT / 'results/task2/discovery/v2/candidate_validation.csv'
        if not path.exists():
            self.skipTest('Local historical numeric evidence is not distributed')
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 972)
        for row in rows:
            self.assertTrue(row['correlation'])
            self.assertTrue(row['cutoff'])
            self.assertEqual(float(row['correlation']) > float(row['cutoff']), row['replicate_supported'] == 'True')


class ModelArtifactTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='microcture-task2-artifacts-')
        self.addCleanup(self.folder.cleanup)
        self.out = Path(self.folder.name)
        self.cfg = config()
        self.x = np.zeros((6, 2, 64, 64), dtype=np.float32)
        self.x[:, 1] = 1
        self.roles = np.array([0, 0, 1, 1, 2, 2])
        self.eligible = np.ones(6, bool)
        self.train = np.random.default_rng(42).choice([0, 1], 2, replace=False)
        self.validation = np.array([2, 3])
        self.identity = core.model_identity(self.x, self.roles, self.eligible, self.cfg, 'ae', 42, self.train, self.validation)

    def publish_fixture(self, change=None):
        torch.manual_seed(42)
        model = core.AE(self.cfg['latent_dim'])
        payload = dict(state_dict=model.state_dict(), family='ae', seed=42, epoch=1, validation_loss=1.,
                       identity_sha256=core._json_sha256(self.identity),
                       train_indices=self.train.tolist(), validation_indices=self.validation.tolist())
        if change:
            payload.update(change)
        torch.save(payload, self.out / 'ae_42.pt')
        (self.out / 'ae_42_history.json').write_text(json.dumps([dict(epoch=1, train_loss=1., validation_loss=1.)]))
        core.publish_model_artifacts(self.out, 'ae_42', self.identity)

    def test_complete_model_is_loaded_read_only_without_optimizer(self):
        self.publish_fixture()
        before = {p.name: p.read_bytes() for p in self.out.iterdir()}
        with mock.patch.object(core.torch.optim, 'Adam', side_effect=AssertionError('No training')):
            scores, features, summary = core.fit_model(self.x, self.roles, self.eligible, self.cfg, self.out, 'ae', 42)
        self.assertEqual(scores.shape, (6,))
        self.assertEqual(features.shape, (6, self.cfg['latent_dim']))
        self.assertEqual(summary['checkpoint_sha256'], core.file_sha256(self.out / 'ae_42.pt'))
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.out.iterdir()})

    def test_partial_model_is_rejected_without_any_overwrite(self):
        for suffix in ['.pt', '_history.json', '_artifacts.json']:
            with self.subTest(suffix=suffix):
                path = self.out / ('ae_42' + suffix)
                path.write_text('partial; must survive')
                with mock.patch.object(core, 'AE', side_effect=AssertionError('No model construction')):
                    # A patched model changes source inspection, so test the loader directly.
                    with self.assertRaisesRegex(ValueError, 'Incomplete'):
                        core.load_model_artifacts(self.out, 'ae_42', self.identity)
                self.assertEqual(path.read_text(), 'partial; must survive')
                path.unlink()

    def test_fresh_model_publication_and_read_only_replay_with_mock_optimizer(self):
        cfg = dict(self.cfg, ae_max_epochs=1)
        with mock.patch.object(core, 'AE', SyntheticAE), mock.patch.object(core.torch.optim, 'Adam') as optimizer:
            scores, features, summary = core.fit_model(self.x, self.roles, self.eligible, cfg, self.out, 'ae', 42)
            self.assertEqual(optimizer.return_value.step.call_count, 1)
            self.assertEqual(summary['epochs'], 1)
            self.assertEqual(len(list(self.out.glob('ae_42*'))), 3)
            before = {p.name: p.read_bytes() for p in self.out.iterdir()}
            optimizer.reset_mock()
            replay_scores, replay_features, replay_summary = core.fit_model(self.x, self.roles, self.eligible, cfg, self.out, 'ae', 42)
            optimizer.assert_not_called()
            np.testing.assert_array_equal(scores, replay_scores)
            np.testing.assert_array_equal(features, replay_features)
            self.assertEqual(summary, replay_summary)
            self.assertEqual(before, {p.name: p.read_bytes() for p in self.out.iterdir()})

    def test_changed_tensor_config_validation_roles_or_eligibility_rejected(self):
        self.publish_fixture()
        altered_x = self.x.copy(); altered_x[5, 0, 0, 0] = 1
        altered_roles = self.roles.copy(); altered_roles[3], altered_roles[4] = 2, 1
        altered_eligible = self.eligible.copy(); altered_eligible[4] = False
        altered_cfg = dict(self.cfg, ae_learning_rate=.002)
        for x, roles, eligible, cfg in [(altered_x, self.roles, self.eligible, self.cfg),
                                       (self.x, altered_roles, self.eligible, self.cfg),
                                       (self.x, self.roles, altered_eligible, self.cfg),
                                       (self.x, self.roles, self.eligible, altered_cfg)]:
            with mock.patch.object(core.torch.optim, 'Adam', side_effect=AssertionError('No training')):
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    core.fit_model(x, roles, eligible, cfg, self.out, 'ae', 42)

    def test_wrong_checkpoint_family_seed_or_validation_rejected(self):
        for change in [dict(family='dae'), dict(seed=43), dict(validation_indices=[2, 4]), dict(epoch=2)]:
            self.publish_fixture(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                core.load_model_artifacts(self.out, 'ae_42', self.identity)

    def test_tampered_checkpoint_or_history_is_rejected(self):
        for name in ['ae_42.pt', 'ae_42_history.json']:
            self.publish_fixture()
            with (self.out / name).open('ab') as stream:
                stream.write(b'changed')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'bytes mismatch'):
                core.load_model_artifacts(self.out, 'ae_42', self.identity)

    def test_run_completion_indexes_weights_and_is_published_last(self):
        with self.assertRaises(FileNotFoundError):
            workflow.complete_run(self.out, ['ae_42'])
        self.assertFalse((self.out / 'COMPLETED.json').exists())
        self.publish_fixture()
        workflow.complete_run(self.out, ['ae_42'])
        index = json.loads((self.out / 'ARTIFACTS.json').read_text())
        completion = json.loads((self.out / 'COMPLETED.json').read_text())
        self.assertEqual(index['checkpoint_count'], 1)
        self.assertIn('ae_42.pt', index['files'])
        self.assertEqual(completion['artifacts_sha256'], core.file_sha256(self.out / 'ARTIFACTS.json'))


class PreprocessingEvidenceTests(unittest.TestCase):
    def test_both_replicates_have_symmetric_training_fit_evidence(self):
        roles = np.repeat([0, 1, 2], 20)
        splits, eligible = np.array([0, 1, 2]), np.ones(3, bool)
        with tempfile.TemporaryDirectory(prefix='microcture-task2-preprocessing-') as folder:
            fields = []
            for replicate in [1, 2]:
                band = dict(band=np.ones((60, 6)) * replicate, valid=np.ones(60, bool))
                fitted = core.train_background(band, roles)
                path = Path(folder) / f'rep{replicate}.npz'
                workflow.save_preprocessing(path, fitted, roles, splits, eligible, f'rep{replicate}')
                with np.load(path, allow_pickle=False) as saved:
                    fields.append(set(saved.files))
                    np.testing.assert_array_equal(saved['training_fit_bins'], (roles == 0) & fitted['valid'])
                    self.assertIn('opportunities', saved.files)
                    self.assertIn('coverage', saved.files)
                    identity = json.loads(str(saved['fit_identity']))
                    self.assertEqual(str(saved['fit_identity_sha256']), core._json_sha256(identity))
            self.assertEqual(fields[0], fields[1])


if __name__ == '__main__':
    unittest.main()
