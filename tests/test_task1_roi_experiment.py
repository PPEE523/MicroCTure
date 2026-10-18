"""Small hand-computed controls and publication mocks; no optimizer or real images."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from microcture.task1_roi import experiment as e
from microcture.task1_readout import runner as frozen_runner
from microcture.task1_formal.runner import verify_artifact_index

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/task1_roi.json").read_text())


def samples():
    x=np.zeros((2,2,3,64,64),dtype=np.float32)
    x[:,:,0]=5;x[:,:,1]=1;x[:,:,2]=.5
    return {"x":x,"length":np.array([[.2],[.2]],dtype=np.float32),"y":np.array([2,2]),
        "structure_id":np.array(["test","test"]),"group_id":np.array([1,1]),"replicate":np.array([1,2])}


class ProtocolTests(unittest.TestCase):
    def test_exact_jobs_and_filters(self):
        c=config();jobs=e.planned_jobs(c)
        self.assertEqual(len(jobs),75);self.assertEqual(sum(j['alignment']=='R0' for j in jobs),15)
        self.assertEqual(len(e.planned_jobs(c,alignment='R1',condition='V0',fold=4,seed=43)),1)
        self.assertEqual(e.planned_jobs(c,alignment='R0',condition='G0'),[])
        self.assertTrue(all(j['variant']=='O0' and j['model']=='m2plus' for j in jobs))
        self.assertEqual(len({str(e.job_directory(Path('/tmp/run'),j)) for j in jobs}),75)

    def test_gate_and_seed_mutation_rejected(self):
        for mutate in [lambda c:c['development_gates'].update(macro_f1=.8),lambda c:c.update(seeds=[42])]:
            c=config();mutate(c)
            with self.assertRaises(ValueError):e.validate_protocol(c)
        with self.assertRaises(ValueError):e.planned_jobs(config(),seed=True)

    def test_fit_loop_is_frozen_import(self):
        self.assertIs(e.fit_diagnostic,frozen_runner.fit_diagnostic)
        self.assertIs(e.restore_endpoint,frozen_runner.restore_endpoint)

    def test_output_protection_and_protocol_resume(self):
        for name in ['task1_formal_v1','task1_readout_v1','task1_reevaluation_v1','task1_morphology_probe_v1']:
            with self.assertRaises(ValueError):e.guard_output(ROOT,ROOT/'results'/name)
        with tempfile.TemporaryDirectory(prefix='microcture-roi-test-') as d:
            root=Path(d);run=root/'results/new';run.mkdir(parents=True)
            (run/'manifest.json').write_text(json.dumps({'protocol':e.PROTOCOL}))
            self.assertEqual(e.guard_output(root,run),run)
            (run/'manifest.json').write_text('{}')
            with self.assertRaises(ValueError):e.guard_output(root,run)


class ConditionTests(unittest.TestCase):
    def test_i3_copy_and_g0_scope(self):
        s=samples();a=e.apply_condition(s,'I3');self.assertTrue(np.array_equal(a['x'],s['x']));self.assertIsNot(a['x'],s['x'])
        g=e.apply_condition(s,'G0');self.assertEqual(float(g['x'][:,:,:2].sum()),0)
        self.assertTrue(np.array_equal(g['x'][:,:,2],s['x'][:,:,2]));self.assertTrue(np.array_equal(g['length'],s['length']))

    def test_v0_preserves_quality_and_geometry(self):
        s=samples();s['x'][0,0,1,3,4]=.2
        v=e.apply_condition(s,'V0');self.assertEqual(float(v['x'][:,:,0].sum()),0)
        self.assertTrue(np.array_equal(v['x'][:,:,1:],s['x'][:,:,1:]));self.assertEqual(float(s['x'][0,0,0,0,0]),5)

    def test_p0_offset_weighted_mean_and_invalid_pixels(self):
        s=samples();s['x'][:,:,1]=0;s['x'][:,:,0]=0
        # Same grid-distance layer: signal10 validity1, signal2 validity0.5.
        for i,j,signal,valid in [(0,2,10,1),(2,0,10,1),(1,3,2,.5),(3,1,2,.5)]:
            s['x'][0,0,0,i,j]=signal;s['x'][0,0,1,i,j]=valid
        p=e.apply_condition(s,'P0');expected=22/3
        self.assertAlmostEqual(float(p['x'][0,0,0,0,2]),expected,places=5)
        self.assertAlmostEqual(float(p['x'][0,0,0,1,3]),expected,places=5)
        self.assertEqual(float(p['x'][0,0,0,4,6]),0)
        self.assertTrue(np.array_equal(p['x'][:,:,1:],s['x'][:,:,1:]));self.assertTrue(np.array_equal(p['x'][0,0,0],p['x'][0,0,0].T))


class ScientificPublicationTests(unittest.TestCase):
    def test_fixed_endpoint_precedes_outer_and_keeps_alignment(self):
        c=config();j=e.planned_jobs(c,alignment='R1',condition='I3',fold=0,seed=42)[0]
        s=samples();pp={'signature':'p'};fold={'train_ids':['test'],'validation_ids':['test'],'test_ids':['test']}
        manifest={'experiment_signature':'exp','source_signature':'src','dataset_signature':'data','selection_protocol':'final40','evaluation_scope':'fixture'}
        with tempfile.TemporaryDirectory(prefix='microcture-roi-test-') as d:
            output=Path(d);events=[]
            def fit(*args):
                events.append('fit');self.assertEqual(args[0]['variant'],'O0');self.assertEqual(args[0]['alignment'],'R1')
                (output/'final.pt').write_bytes(b'fixed weights fixture');return {'status':'completed'}
            def outer(*args):
                events.append('outer');record=json.loads((output/'endpoint.json').read_text())
                self.assertTrue(record['training_and_selection_complete']);self.assertFalse(record['outer_images_constructed'])
                self.assertEqual(record['checkpoint_sha256'],e.sha256(output/'final.pt'));return s
            p=np.array([[.1,.2,.7],[.2,.1,.7]],dtype=np.float32)
            with patch.object(e,'fit_diagnostic',side_effect=fit),patch.object(e,'prepare_roi_outer',side_effect=outer),patch.object(e,'restore_endpoint',return_value=object()),patch.object(e,'outer_neural_endpoint',return_value=(p,{})):
                result=e.fit_one(j,s,s,pp,{},fold,c,manifest,output,torch.device('cpu'))
            self.assertEqual(events,['fit','outer']);self.assertEqual(result['status'],'completed')
            verify_artifact_index(output,j,manifest);self.assertEqual(json.loads((output/'metrics.json').read_text())['job']['alignment'],'R1')

    def test_keyboard_interrupt_preserves_partial_history_and_seal(self):
        c=config();j=e.planned_jobs(c,alignment='R0',fold=0,seed=42)[0];s=samples();pp={'signature':'p'}
        fold={key:['test'] for key in ['train_ids','validation_ids','test_ids']}
        manifest={'experiment_signature':'exp','source_signature':'src','dataset_signature':'data','selection_protocol':'final40','evaluation_scope':'fixture'}
        with tempfile.TemporaryDirectory(prefix='microcture-roi-test-') as d:
            output=Path(d)
            def interrupted(*args):
                (output/'history.json').write_text('{"epochs":[{"epoch":0}]}');raise KeyboardInterrupt()
            with patch.object(e,'fit_diagnostic',side_effect=interrupted),patch.object(e,'prepare_roi_outer') as outer:
                result=e.fit_one(j,s,s,pp,{},fold,c,manifest,output,torch.device('cpu'))
            self.assertEqual(result['status'],'execution_failure');self.assertTrue(result['run_stop_requested']);outer.assert_not_called()
            self.assertTrue((output/'history.json').exists());verify_artifact_index(output,j,manifest)

    def test_fallback_keeps_unsealed_files_without_cleaning(self):
        j=e.planned_jobs(config(),alignment='R0',fold=0,seed=42)[0]
        with tempfile.TemporaryDirectory(prefix='microcture-roi-test-') as d:
            root=Path(d);run=root/'results/test';output=root/'staging';output.mkdir();(output/'history.json').write_text('{}')
            e.staging_context(output,root,run,j)
            with patch.object(e.shutil,'move',side_effect=OSError('mock double storage fault')):
                retained=e.retain_staging(output,run,j,OSError('mock original fault'),root)
            self.assertEqual(retained,output);self.assertTrue((output/'history.json').exists())


class OOFTests(unittest.TestCase):
    def fixture(self,root,failed=False):
        c=config();run=root/'results/fixture';run.mkdir(parents=True);c['run_dir']='results/fixture'
        structures=[{'structure_id':f'g{g}_{cls}','type':cls,'start':g*10000+i*100,'end':g*10000+i*100+100}
            for g in range(5) for i,cls in enumerate(e.CLASS_NAMES)]
        metadata={'structures':structures,'group_id':[g for g in range(5) for _ in range(3)]}
        identity=e._identity(metadata);allids=list(identity)
        folds={str(g):{'test_ids':[sid for sid in allids if identity[sid]['group_id']==g],
            'validation_ids':[sid for sid in allids if identity[sid]['group_id']==(g+1)%5],
            'train_ids':[sid for sid in allids if identity[sid]['group_id'] not in (g,(g+1)%5)]} for g in range(5)}
        manifest={'config':c,'protocol':e.PROTOCOL,'requested_device':'cpu','fold_ids':folds,'source_signature':'fixture',
            'experiment_signature':'fixture','dataset_signature':'fixture','evaluation_scope':'tiny saved-CSV fixture'}
        (run/'manifest.json').write_text(json.dumps(manifest))
        def load(*args):
            j=args[1]
            if failed and j['alignment']=='R1' and j['condition']=='I3' and j['seed']==43 and j['fold']==2:
                return {'job':j,'status':'numerical_failure','metrics':{},'history':[]}
            perfect=j['condition']=='I3';rows=[]
            for sid in folds[str(j['fold'])]['test_ids']:
                true=identity[sid]['y'];p=np.eye(3)[true if perfect else 1]
                rows.append({'structure_id':sid,'y_true':str(true),'y_pred':str(int(p.argmax())),
                    'group_id':str(j['fold']),**{f'p_{cls}':str(float(p[i])) for i,cls in enumerate(e.CLASS_NAMES)}})
            probs=np.asarray([[float(r[f'p_{cls}']) for cls in e.CLASS_NAMES] for r in rows]);labels=np.array([int(r['y_true']) for r in rows])
            return {'job':j,'status':'completed','history':[],'metrics':{'initial_state_sha256':'same','common_initial_state_sha256':'same',
                'minibatch_order_sha256':'same','class_weights':[1,1,1],'runtime_seconds':0.,'final_train_ce':.1,'acceptance':{'accepted':True}},
                'predictions':{'rows':rows,'structure_id':np.array([r['structure_id'] for r in rows]),'probabilities':probs,'metrics':e.metrics(labels,probs)}}
        return run,manifest,metadata,load

    @staticmethod
    def learning(items,c):
        return {'strict_learning_passes':sum(r['status']=='completed' for r in items)}

    def test_all_complete_oof_preserves_constant_control_and_pairs(self):
        with tempfile.TemporaryDirectory(prefix='microcture-roi-test-') as d:
            root=Path(d);run,manifest,metadata,load=self.fixture(root)
            with patch.object(e,'prepare_manifest',return_value=(manifest,metadata)),patch.object(e,'_load_saved',side_effect=load),patch.object(e,'_neural_summary',side_effect=self.learning):
                result=e.summarize(run,root)
            self.assertEqual(result['completed_jobs'],75);self.assertEqual(result['complete_oof'],15);self.assertEqual(result['status'],'complete')
            self.assertEqual(len(result['paired_deltas']),12)
            constant=[r for r in result['comparison'] if r['condition']=='V0'][0]
            self.assertAlmostEqual(constant['macro_f1_mean'],1/6);self.assertFalse(constant['development_gate_pass'])
            good=[r for r in result['comparison'] if r['alignment']=='R1' and r['condition']=='I3'][0]
            self.assertEqual(good['macro_f1_mean'],1);self.assertTrue(good['development_gate_pass'])
            self.assertEqual(result['models']['R1/V0']['stable_wrong_ids'],sorted(sid for sid in e._identity(metadata) if not sid.endswith('CHIN')))

    def test_failed_seed_never_scored_or_filled_with_zero(self):
        with tempfile.TemporaryDirectory(prefix='microcture-roi-test-') as d:
            root=Path(d);run,manifest,metadata,load=self.fixture(root,failed=True)
            with patch.object(e,'prepare_manifest',return_value=(manifest,metadata)),patch.object(e,'_load_saved',side_effect=load),patch.object(e,'_neural_summary',side_effect=self.learning):
                result=e.summarize(run,root)
            self.assertEqual(result['status'],'failed');self.assertEqual(result['numerical_failure_jobs'],1)
            self.assertEqual(result['complete_oof'],14)
            row=[r for r in result['comparison'] if r['alignment']=='R1' and r['condition']=='I3'][0]
            self.assertIsNone(row['macro_f1_mean']);self.assertEqual(row['complete_seeds'],2);self.assertFalse(row['development_gate_pass'])
            self.assertFalse((run/'oof/R1/I3/seed_43/metrics.json').exists())
            self.assertEqual(result['models']['R1/I3']['seeds']['43']['fit_status_counts']['numerical_failure'],1)


if __name__=='__main__':unittest.main()
