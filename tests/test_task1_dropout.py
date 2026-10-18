"""Dropout-only CPU forward/metadata fixtures; no optimization or real training."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from microcture.task1_dropout_v1 import experiment as E
from microcture.task1_readout.models import build_model
from microcture.task1_controlled.runner import state_fingerprint

ROOT=Path(__file__).resolve().parents[1]


class DropoutProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'configs/task1_dropout.json').read_text())
        self.job=E.planned_jobs(self.config,condition='I3',fold=0,seed=42)[0]

    def test_exact_60_only_D5_registered(self):
        jobs=E.planned_jobs(self.config)
        self.assertEqual(len(jobs),60)
        self.assertEqual({j['dropout_variant'] for j in jobs},{'D5'})
        self.assertEqual({j['alignment'] for j in jobs},{'R1'})
        self.assertEqual(len(E.planned_jobs(self.config,condition='I3')),15)
        self.assertEqual(len({(j['condition'],j['fold'],j['seed']) for j in jobs}),60)

    def test_no_sweep_or_other_training_changes(self):
        for field,value in [('learning_rate',2e-4),('epochs',20),('dropout',.4),('loss','plain_cross_entropy')]:
            cfg=copy.deepcopy(self.config);cfg['training'][field]=value
            with self.assertRaisesRegex(ValueError,'Only global'):
                E.validate_protocol(cfg)
        cfg=copy.deepcopy(self.config);cfg['models']['dropout']=.2
        with self.assertRaisesRegex(ValueError,'Only global'):E.validate_protocol(cfg)

    def test_all_previous_experiments_protected(self):
        for name in ('task1_roi_v1','task1_native_roi_v2','task1_formal_v1','task1_readout_v1'):
            with self.assertRaises(ValueError):E.guard_output(ROOT,ROOT/'results'/name)

    def test_paired_baseline_exact_and_mismatches_refused(self):
        ref=E.reference_manifest(ROOT,self.config);old_job=E._reference_job(ref['config'],self.job)
        directory=E.reference_roi.job_directory(ROOT/self.config['reference_roi_run_dir'],old_job)
        old=json.loads((directory/'metrics.json').read_text());pp=json.loads((directory/'protocol.json').read_text())['preprocessing']
        self.assertTrue(E.assert_paired_fit(self.config,self.job,old,pp)['initialization_minibatch_class_weights_preprocessing_exact'])
        for field in ('initial_state_sha256','common_initial_state_sha256','minibatch_order_sha256','class_weights'):
            bad=copy.deepcopy(old);bad[field]='corrupted'
            with self.assertRaisesRegex(ValueError,'paired'):E.assert_paired_fit(self.config,self.job,bad,pp)
        badpp=copy.deepcopy(pp);badpp['alignment']='R0'
        with self.assertRaisesRegex(ValueError,'input protocol'):E.assert_paired_fit(self.config,self.job,old,badpp)
        with self.assertRaisesRegex(ValueError,'before training'):E.assert_reference_preprocessing(ROOT,self.config,0,badpp)

    def test_same_weights_parameter_count_and_eval_outputs(self):
        torch.set_num_threads(1)
        old_config=copy.deepcopy(self.config['models']);old_config['dropout']=.2
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42);old=build_model('O0',old_config)
            torch.manual_seed(42);new=build_model('O0',self.config['models'])
            self.assertEqual(state_fingerprint(old),state_fingerprint(new))
            self.assertEqual(sum(p.numel() for p in new.parameters()),110437)
            self.assertEqual(E.dropout_modules(old),{'head.dropout':.2,'backbone.blocks.0.ffn.2':.2})
            self.assertEqual(E.dropout_modules(new),{'head.dropout':.5,'backbone.blocks.0.ffn.2':.5})
            x=torch.randn(2,2,3,64,64);length=torch.randn(2,1)
            old.eval();new.eval()
            with torch.inference_mode():self.assertTrue(torch.equal(old(x,length),new(x,length)))
            old.train();new.train()
            with torch.inference_mode():
                torch.manual_seed(99);a=old(x,length)
                torch.manual_seed(99);b=new(x,length)
            self.assertFalse(torch.equal(a,b))

    def test_partial_failure_keeps_scientific_trace_and_stops(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            out=Path(tmp)
            def interrupted(*args,**kwargs):
                (out/'history.json').write_text('{"epochs":[{"epoch":0}]}')
                raise KeyboardInterrupt('fixture interruption')
            manifest={key:'fixture' for key in ('experiment_signature','source_signature','dataset_signature','selection_protocol','evaluation_scope')}
            fold={'train_ids':['a'],'validation_ids':['b'],'test_ids':['c']}
            pp={'signature':'fixed','alignment':'R1'}
            with patch.object(E,'apply_condition',side_effect=lambda samples,condition:samples),patch.object(E,'fit_diagnostic',side_effect=interrupted),patch.object(E,'write_artifact_index') as index:
                result=E.fit_one(self.job,{}, {},pp,{},fold,self.config,manifest,out,torch.device('cpu'))
            self.assertEqual(result['status'],'execution_failure')
            self.assertTrue(result['run_stop_requested'])
            self.assertEqual(json.loads((out/'history.json').read_text())['epochs'][0]['epoch'],0)
            self.assertTrue((out/'failure.json').exists());index.assert_called_once()


if __name__=='__main__':unittest.main()
