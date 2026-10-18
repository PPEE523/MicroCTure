"""Sparse native ROI algebra checked against a small independent dense fixture."""
import unittest
import numpy as np
from scipy import sparse

from scripts.probe_native_roi_feasibility import common_native_roi
from microcture.task1_roi.data import roi_area_weights


class NativeROIFixtureTests(unittest.TestCase):
    def test_sparse_both_branches_equal_dense_formula(self):
        n=80
        i,j=np.indices((n,n))
        counts=((i*3+j*5)%7).astype(float)
        counts=(counts+counts.T)/2
        counts[(i+j)%5==0]=0
        good=np.ones(n);good[30:40]=0
        expected=1+np.arange(n)/50
        starts=np.arange(n)*10
        interval=(33,773)
        c0,c1,facts=common_native_roi(sparse.csr_matrix(counts),starts,interval,good,expected)
        a=roi_area_weights(*interval,starts,resolution_bp=10,output_size=64)
        v=np.outer(good,good)*(abs(i-j)>=20)
        field=np.log1p(counts/expected[abs(i-j)])
        field=np.where(v>0,field,0)
        den=a@v@a.T
        fine=np.divide(a@(field*v)@a.T,den,out=np.zeros_like(den),where=den>0)
        e=np.maximum(0,np.minimum(starts+10,interval[1])-np.maximum(starts,interval[0]))/10
        b=np.zeros((n,8));b[np.arange(n),np.arange(n)//10]=1
        coarse_den=b.T@(v*np.outer(e,e))@b
        coarse_num=b.T@(field*v*np.outer(e,e))@b
        proj=np.divide(coarse_num,coarse_den,out=np.zeros_like(coarse_num),where=coarse_den>0)
        broadcast=b@proj@b.T
        coarse=np.divide(a@(broadcast*v)@a.T,den,out=np.zeros_like(den),where=den>0)
        self.assertTrue(np.allclose(c0[0],coarse,atol=1e-6,rtol=0))
        self.assertTrue(np.allclose(c1[0],fine,atol=1e-6,rtol=0))
        self.assertTrue(np.allclose(c0[1],den,atol=1e-7,rtol=0))
        self.assertTrue(np.array_equal(c0[1:],c1[1:]))
        self.assertFalse(facts['native_dense_n_by_n_allocated'])

    def test_zero_native_counts_keep_frozen_coverage(self):
        n=80
        c0,c1,facts=common_native_roi(sparse.csr_matrix((n,n)),np.arange(n)*10,(0,800),np.ones(n),np.ones(n))
        self.assertGreater(facts['common_validity_mean'],0)
        self.assertTrue((c0[0]==0).all() and (c1[0]==0).all())
        self.assertTrue(np.array_equal(c0[1],c1[1]))

    def test_same_constant_field_survives_projection(self):
        n=80
        i,j=np.indices((n,n))
        counts=np.where(abs(i-j)>=20,2.,0.)
        c0,c1,_=common_native_roi(sparse.csr_matrix(counts),np.arange(n)*10,(0,800),np.ones(n),np.ones(n))
        self.assertTrue(np.allclose(c0[0],c1[0],atol=1e-6,rtol=0))

    def test_asymmetric_storage_and_invalid_expectation_rejected(self):
        n=80
        bad=sparse.csr_matrix(([1.],([0],[30])),shape=(n,n))
        with self.assertRaisesRegex(ValueError,'Symmetric'):
            common_native_roi(bad,np.arange(n)*10,(0,800),np.ones(n),np.ones(n))
        expected=np.ones(n);expected[25]=0
        with self.assertRaisesRegex(ValueError,'expectations'):
            common_native_roi(sparse.csr_matrix((n,n)),np.arange(n)*10,(0,800),np.ones(n),expected)


if __name__=='__main__':
    unittest.main()
