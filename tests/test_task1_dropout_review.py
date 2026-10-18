"""Hand-counted review fixtures; synthetic CPU arithmetic, no fit/score reads."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scripts import audit_task1_dropout as A
from microcture.task1_formal.protocol import json_signature, sha256


class DropoutReviewTests(unittest.TestCase):
    def test_confusion_hand_arithmetic(self):
        values, support, predicted = A.scores_from_confusion([[1, 0, 0], [0, 1, 0], [0, 1, 0]])
        self.assertAlmostEqual(float(values['macro_f1']), 5 / 9)
        self.assertEqual(float(values['CHID_f1']), 0.)
        np.testing.assert_array_equal(support, [1, 1, 1])
        np.testing.assert_array_equal(predicted, [1, 2, 0])

    def test_bootstrap_matches_pairs_and_retains_zero_support(self):
        identity = {str(i): {'y': i, 'group_id': i} for i in range(3)}
        oof = {}
        for variant, condition in A.ARMS:
            for seed in A.SEEDS:
                rs = []
                for i in range(3):
                    pred = 1 if variant == 'D5' and condition != 'I3' and i == 2 else i
                    rs.append({'structure_id': str(i), 'y_true': i, 'y_pred': pred, 'group_id': i})
                oof[variant, condition, seed] = {'rows': rs}
        result = A.group_bootstrap(oof, identity, repeats=120, rng_seed=11)
        self.assertEqual(result['group_count'], 3)
        self.assertGreater(result['zero_CHID_support_bootstraps'], 0)
        for metric in result['pairs']['D5/I3-D2/I3'].values():
            self.assertEqual(metric['point_mean'], 0.)
            self.assertEqual(metric['ci95'], [0., 0.])
        self.assertAlmostEqual(result['pairs']['D5/G0-D2/G0']['macro_f1']['point_mean'], -4 / 9)
        self.assertAlmostEqual(result['dropout_effect_beyond_control']['G0']['macro_f1']['point_mean'], 4 / 9)
        self.assertEqual(result, A.group_bootstrap(oof, identity, repeats=120, rng_seed=11))
        oof['D2', 'I3', 42]['rows'].append(oof['D2', 'I3', 42]['rows'][0])
        with self.assertRaisesRegex(ValueError, 'Matched full'):
            A.group_bootstrap(oof, identity, repeats=1)

    def test_completion_gate_precedes_any_fit_or_manifest_read(self):
        with tempfile.TemporaryDirectory(prefix='microcture-dropout-audit-fixture-', dir='/tmp') as tmp:
            run = Path(tmp);(run / 'summary.json').write_text(json.dumps({'status': 'pending', 'completed_jobs': 59, 'complete_oof': 11}))
            with patch.object(A.D, 'prepare_manifest') as prepare:
                with self.assertRaisesRegex(ValueError, 'all60'):
                    A.build_report(run, run / 'never_created')
                prepare.assert_not_called()
            self.assertFalse((run / 'never_created').exists())

    def test_recursive_inventory_refuses_unindexed_evidence(self):
        with tempfile.TemporaryDirectory(prefix='microcture-dropout-inventory-fixture-', dir='/tmp') as tmp:
            root = Path(tmp);directory = root / 'fit';directory.mkdir()
            (directory / 'x.json').write_text('{}')
            job = {'condition': 'I3'};manifest = {'experiment_signature': 'fixture'}
            files = [{'name': 'x.json', 'size_bytes': 2, 'sha256': sha256(directory / 'x.json')}]
            (directory / 'ARTIFACTS.json').write_text(json.dumps({'job': job, 'experiment_signature': 'fixture', 'files': files, 'signature': json_signature(files)}))
            with patch.object(A, 'ROOT', root):
                self.assertEqual(len(A.check_inventory(directory, job, manifest, 'D5')), 1)
                (directory / 'extra.txt').write_text('preserve me')
                with self.assertRaisesRegex(ValueError, 'Recursive'):
                    A.check_inventory(directory, job, manifest, 'D5')


if __name__ == '__main__':
    unittest.main()
