"""Independent dense arithmetic and mocked scientific data-flow fixtures."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from microcture.task1_native_roi import data as native
from microcture.task1_roi.data import roi_area_weights


class FakeCooler:
    def __init__(self, matrix):
        self.array=np.asarray(matrix,dtype=float)
        self.calls=[]

    def matrix(self,**kwargs):
        if kwargs != {"balance":False,"sparse":True}:
            raise AssertionError("Raw unbalanced sparse native read required")
        return self

    def __getitem__(self,index):
        self.calls.append(index)
        return sparse.csr_matrix(self.array[index])


class SparseDataTests(unittest.TestCase):
    def test_fft_matches_direct_circular_and_zero_opportunities(self):
        mask=np.zeros(200,dtype=bool);mask[20:80]=True;mask[120:150]=True
        observed=native.fft_pair_opportunities(mask,60)
        expected=np.array([np.count_nonzero(mask&np.roll(mask,-d)) for d in range(61)])
        self.assertTrue(np.array_equal(observed,expected))

    def test_owned_counts_unique_both_training_endpoints(self):
        n=240;i,j=np.indices((n,n));matrix=(1+(i+j)%3).astype(float)
        mask=np.zeros(n,bool);mask[30:95]=True;mask[130:165]=True
        # Very strong outside contacts must never enter training-only sums.
        matrix[~(mask[:,None]&mask[None,:])]=10000
        clr=FakeCooler(matrix)
        observed,facts=native.owned_native_counts(clr,mask,np.array([[30,95],[130,165]]),max_distance=45,owner_rows=13)
        expected=np.zeros(46)
        for a in range(n):
            for b in range(a+1,n):
                if mask[a] and mask[b] and 20<=b-a<=45:
                    expected[b-a]+=matrix[a,b]
        self.assertTrue(np.array_equal(observed,expected))
        self.assertEqual(facts['eligible_counts_sum'],expected.sum())
        self.assertTrue(all(mask[np.arange(row.start,row.stop)].all() for row,col in clr.calls))

    def test_circular_guard_prevents_silent_remote_pair_omission(self):
        mask=np.zeros(240,bool);mask[:20]=True;mask[-20:]=True
        self.assertTrue(native.circular_wrap_pair_guard(mask,45))
        with self.assertRaisesRegex(ValueError,'circular'):
            native.owned_native_counts(FakeCooler(np.ones((240,240))),mask,np.array([[0,20],[220,240]]),max_distance=45)

    def test_both_projection_branches_equal_independent_dense_formula(self):
        n=80;i,j=np.indices((n,n));counts=(1+(i+j)%9).astype(float);counts[(i+j)%4==0]=0
        good=np.ones(n);good[30:40]=0;expected=1+np.arange(n)/80;starts=np.arange(n)*10;interval=(33,773)
        result,facts=native.resample_matched_native(sparse.csr_matrix(counts),starts,interval,good,expected)
        a=roi_area_weights(*interval,starts,resolution_bp=10,output_size=64)
        v=np.outer(good,good)*(np.abs(i-j)>=20)
        field=np.where(v>0,np.log1p(counts/expected[np.abs(i-j)]),0)
        den=a@v@a.T;fine=np.divide(a@(field*v)@a.T,den,out=np.zeros_like(den),where=den>0)
        e=np.maximum(0,np.minimum(starts+10,interval[1])-np.maximum(starts,interval[0]))/10
        b=np.zeros((n,8));b[np.arange(n),np.arange(n)//10]=1
        pn=b.T@(field*v*np.outer(e,e))@b;pd=b.T@(v*np.outer(e,e))@b
        p=np.divide(pn,pd,out=np.zeros_like(pn),where=pd>0);broadcast=b@p@b.T
        coarse=np.divide(a@(broadcast*v)@a.T,den,out=np.zeros_like(den),where=den>0)
        self.assertTrue(np.allclose(result['C0'][0],coarse,atol=1e-6,rtol=0))
        self.assertTrue(np.allclose(result['C1'][0],fine,atol=1e-6,rtol=0))
        self.assertTrue(np.allclose(result['C0'][1],den,atol=1e-7,rtol=0))
        self.assertTrue(np.array_equal(result['C0'][1:],result['C1'][1:]))
        self.assertFalse(facts['native_dense_matrix_allocated'])

    def test_zero_contacts_do_not_change_frozen_coverage(self):
        result,facts=native.resample_matched_native(sparse.csr_matrix((80,80)),np.arange(80)*10,(0,800),np.ones(80),np.ones(80))
        self.assertGreater(facts['target_validity_fraction'],0)
        self.assertTrue((result['C0'][0]==0).all() and (result['C1'][0]==0).all())
        self.assertTrue(np.array_equal(result['C0'][1:],result['C1'][1:]))

    def test_normalizer_cache_reuse_and_corruption_refusal(self):
        native._EXPECTED_CACHE.clear()
        mask=np.zeros(300,bool);mask[60:240]=True;components=np.array([[60,240]])
        sources=[{"path":f"fake{r}","replicate":r,"sample_id":str(r)} for r in (1,2)]
        matrices={f"fake{r}":FakeCooler(np.full((300,300),r)) for r in (1,2)}
        dataset={"signature":"fixture"};fold={"fold":0,"train_ids":["a"],"validation_ids":["b"],"test_ids":["c"]}
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary, patch.object(native,'MAX_DISTANCE_BINS',40), \
             patch.object(native,'native_source_records',return_value=sources), \
             patch.object(native,'training_native_mask',return_value=(mask,components)), \
             patch.object(native.cooler,'Cooler',side_effect=lambda path:matrices[Path(path).name]):
            # The production audit spot-checks long distances; use same direct
            # exposure oracle for the small fixture's out-of-range distances.
            original_opps=native.fft_pair_opportunities
            with patch.object(native,'fft_pair_opportunities',side_effect=lambda m,d:original_opps(m,d)):
                record=native.prepare_shared_expected(dataset,fold,temporary)
                self.assertTrue(np.array_equal(np.array(record['records'][0]['expected'])[20:],np.ones(21)))
                self.assertTrue(np.array_equal(np.array(record['records'][1]['expected'])[20:],np.full(21,2.)))
                calls=sum(len(c.calls) for c in matrices.values())
                native._EXPECTED_CACHE.clear()
                second=native.prepare_shared_expected(dataset,fold,temporary)
                self.assertEqual(record,second)
                self.assertEqual(calls,sum(len(c.calls) for c in matrices.values()))
                cached=next(Path(temporary).rglob('fold_0.json'))
                value=json.loads(cached.read_text());value['records'][0]['expected'][20]=2.
                cached.write_text(json.dumps(value));native._EXPECTED_CACHE.clear()
                with self.assertRaisesRegex(ValueError,'signature'):
                    native.prepare_shared_expected(dataset,fold,temporary)

    def test_failed_native_normalizer_is_retained_and_never_retried(self):
        native._EXPECTED_CACHE.clear()
        mask=np.zeros(300,bool);mask[60:240]=True;sources=[{"path":'zero',"replicate":1,"sample_id":'1'},{"path":'zero2',"replicate":2,"sample_id":'2'}]
        dataset={"signature":"failure"};fold={"fold":1,"train_ids":["a"],"validation_ids":["b"],"test_ids":["c"]}
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary,patch.object(native,'MAX_DISTANCE_BINS',40), \
             patch.object(native,'native_source_records',return_value=sources), \
             patch.object(native,'training_native_mask',return_value=(mask,np.array([[60,240]]))), \
             patch.object(native.cooler,'Cooler',return_value=FakeCooler(np.zeros((300,300)))):
            actual=native.fft_pair_opportunities
            with patch.object(native,'fft_pair_opportunities',side_effect=lambda m,d:actual(m,d)):
                with self.assertRaisesRegex(ValueError,'zero distance'):
                    native.prepare_shared_expected(dataset,fold,temporary)
                failure=next(Path(temporary).rglob('fold_1.failure.json'))
                value=json.loads(failure.read_text());self.assertEqual(value['status'],'failure');self.assertEqual(len(value['records']),1)
                with self.assertRaisesRegex(ValueError,'failure forbids'):
                    native.prepare_shared_expected(dataset,fold,temporary)


if __name__=='__main__':
    unittest.main()
