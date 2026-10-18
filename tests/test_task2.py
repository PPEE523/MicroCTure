import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
import cooler
import torch
from discover_task2 import make_band,extract,annotations,disjoint_select,correlation,AE,losses


class DiscoveryTest(unittest.TestCase):
    def test_circular_band_counts_and_zero_opportunities(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tiny.cool';starts=np.arange(0,1052,10)
            bins=pd.DataFrame({'chrom':'chr','start':starts,'end':np.minimum(starts+10,1052)})
            pixels=pd.DataFrame(sorted([(i,i,1) for i in range(0,106,10)]+[(0,20,4),(0,90,6)]),columns=['bin1_id','bin2_id','count'])
            cooler.create_cooler(str(path),bins,pixels)
            b=make_band(path,11,4,chunk=4)
            self.assertAlmostEqual(b['expected'][2],10/9)
            raw,oe,mask=extract(b,9,3)
            self.assertEqual(raw[0,2],6);self.assertFalse(mask[1].any())
            np.testing.assert_equal(raw,raw.T)

    def test_deduplication_handles_origin(self):
        rows=[dict(start_bin=-3,width_bins=6),dict(start_bin=98,width_bins=6),dict(start_bin=40,width_bins=6)]
        self.assertEqual(disjoint_select([0,1,2],rows,100),[0,2])

    def test_overlap_not_equal_recall(self):
        known=[dict(start=1000,end=2000,center=1500,structure_id='k')]
        hit,recall=annotations(9,3,100,known)
        self.assertEqual(hit,['k']);self.assertEqual(recall,[])
        hit,recall=annotations(10,10,100,known)
        self.assertEqual(recall,['k'])

    def test_correlation_excludes_invalid_pixels(self):
        a=np.arange(400,dtype=float).reshape(20,20);b=a.copy();mask=np.ones((20,20),bool)
        mask[0]=False;b[0]=100000
        self.assertAlmostEqual(correlation(a,b,mask),1)
        self.assertIsNone(correlation(a,np.zeros_like(a),mask))

    def test_ae_masked_loss_and_features(self):
        torch.set_num_threads(1);x=torch.zeros(2,2,64,64);x[:,1]=1
        model=AE();pred,z=model(x)
        self.assertEqual(tuple(z.shape),(2,16));losses(pred,x).mean().backward()
        self.assertTrue(torch.isfinite(losses(pred,x)).all())
        x[:,1]=0;self.assertEqual(float(losses(pred,x).sum().detach()),0.)

if __name__=='__main__':unittest.main()
