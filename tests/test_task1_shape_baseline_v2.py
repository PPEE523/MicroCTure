"""Handcrafted descriptors and tiny synthetic fits, without project count matrices."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from microcture.task1_shape_baseline_v2.features import features,_weighted
from microcture.task1_shape_baseline_v2 import experiment as e

ROOT=Path(__file__).resolve().parents[1]


def config():return json.loads((ROOT/'configs/task1_shape_baseline_v2.json').read_text())


def sample(n=3):
    x=np.zeros((n,2,3,64,64));x[:,:,0]=3;x[:,:,1:]=1
    return {'x':x,'length':np.arange(n).reshape(-1,1)*.1,'y':np.arange(n)%3,
        'structure_id':np.array([f's{i}' for i in range(n)]),'group_id':np.arange(n),'replicate':np.ones(n,dtype=int)}


class ShapeTests(unittest.TestCase):
    def test_455_names_and_reversal_invariance(self):
        s=sample();rng=np.random.default_rng(4);s['x'][:,:,0]=rng.normal(size=(3,2,64,64));s['x'][:,:,1]=rng.uniform(size=(3,2,64,64))
        f,names=features(s);g,other=features({**s,'x':s['x'][...,::-1,::-1]})
        self.assertEqual(f.shape,(3,455));self.assertEqual(len(set(names)),455);self.assertEqual(names,other)
        np.testing.assert_array_equal(f,g)

    def test_missing_zero_not_treated_as_measured_zero(self):
        mean,std=_weighted(np.array([[10.,2.,999.]]),np.array([[1.,.5,0.]]))
        self.assertAlmostEqual(mean[0],22/3);self.assertAlmostEqual(std[0],np.sqrt((100+2)/1.5-(22/3)**2))
        mean,std=_weighted(np.array([[999.]]),np.array([[0.]]));self.assertEqual(mean[0],0);self.assertEqual(std[0],0)

    def test_uniform_signal_has_zero_distance_residuals(self):
        f,names=features(sample())
        self.assertTrue(np.all(np.abs(f[:,[i for i,n in enumerate(names) if '_residual_' in n]])<1e-10))
        self.assertTrue(np.all(f[:,[i for i,n in enumerate(names) if '_signal_grid' in n]]==3))

    def test_P0_theoretical_zero_residuals_remain_exactly_zero(self):
        from microcture.task1_roi.experiment import apply_condition
        from sklearn.preprocessing import StandardScaler
        s=sample(8);rng=np.random.default_rng(17)
        signal=rng.normal(2,1,size=(8,2,64,64));valid=rng.uniform(size=(8,2,64,64))
        s['x'][:,:,0]=((signal+signal.swapaxes(-1,-2))/2).astype(np.float32)
        s['x'][:,:,1]=((valid+valid.swapaxes(-1,-2))/2).astype(np.float32)
        f,names=features(apply_condition(s,'P0'));res=f[:,[i for i,n in enumerate(names) if '_residual_' in n]]
        self.assertEqual(res.shape,(8,128));np.testing.assert_array_equal(res,np.zeros_like(res))
        np.testing.assert_array_equal(StandardScaler().fit_transform(res),np.zeros_like(res))

    def test_constant_weighted_field_has_exactly_zero_std(self):
        rng=np.random.default_rng(18);values=np.full((5,127),3.141592653589793);weights=rng.uniform(size=values.shape)
        _,std=_weighted(values,weights);np.testing.assert_array_equal(std,np.zeros(5))

    def test_no_labels_or_ids_enter_features(self):
        a=sample();b={**a,'y':np.array([2,2,2]),'structure_id':np.array(['fake']*3),'group_id':np.array([999]*3)}
        np.testing.assert_array_equal(features(a)[0],features(b)[0])

    def test_fixed40_no_C_gamma_search(self):
        c=config();jobs=e.planned_jobs(c);self.assertEqual(len(jobs),40)
        self.assertEqual(len({str(e.directory(Path('/tmp/fake'),j)) for j in jobs}),40)
        c['models']['svc_shape']['C']=10
        with self.assertRaises(ValueError):e.validate(c)

    def test_tiny_fits_and_svc_scores(self):
        from sklearn.preprocessing import StandardScaler
        values=np.array([[-3.,0.],[-2.,0.],[0.,3.],[0.,2.],[3.,0.],[2.,0.]])
        y=np.array([0,0,1,1,2,2]);scaler=StandardScaler().fit(values)
        for name in e.MODELS:
            model=e.build_classifier(name,{0:1.,1:1.,2:1.});model.fit(scaler.transform(values),y)
            payload={'scaler':scaler,'model':model,'model_name':name};p=e.predict_scores(payload,values)
            np.testing.assert_allclose(p.sum(1),1);self.assertEqual(p.shape,(6,3));np.testing.assert_array_equal(p.argmax(1),y)
            if name=='svc_shape':
                self.assertFalse(model.probability);self.assertTrue(model.break_ties)
                decision=model.decision_function(scaler.transform(values));soft=np.exp(decision-decision.max(1,keepdims=True));soft/=soft.sum(1,keepdims=True)
                np.testing.assert_allclose(p,soft)

    def test_fixed_checkpoint_before_outer(self):
        c=config();j=e.planned_jobs(c,model='lr_shape',condition='I3',fold=0)[0];s=sample(6)
        # Real tiny inner fit; the outer boundary remains a mock and has no project data.
        pp={'signature':'fixture'};fold={k:list(s['structure_id']) for k in ('train_ids','validation_ids','test_ids')}
        manifest={'selection':'fixed','experiment_signature':'exp','source_signature':'src','dataset_signature':'data'}
        with tempfile.TemporaryDirectory(prefix='microcture-shape-test-') as d:
            output=Path(d)
            def outer(*args):
                record=json.loads((output/'endpoint.json').read_text());self.assertFalse(record['outer_images_constructed']);self.assertTrue(record['training_and_selection_complete'])
                self.assertEqual(record['checkpoint_sha256'],e.sha256(output/'model.joblib'));return s
            with patch.object(e,'prepare_roi_outer',side_effect=outer):r=e.fit_one(j,s,s,pp,fold,{},c,manifest,output)
            self.assertEqual(r['status'],'completed');self.assertFalse(r['outer_fit_or_selection'])


if __name__=='__main__':unittest.main()
