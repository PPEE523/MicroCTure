import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import tempfile
import unittest
import numpy as np
import pandas as pd
import cooler
from prepare_dataset import group_windows,assert_disjoint,fit_expected,pool_masked,fit_scale,apply_scale


class DatasetTest(unittest.TestCase):
    def test_transitive_and_origin_grouping(self):
        rows=[{'center':c*100} for c in (2,12,22,98,60)]
        groups,ids=group_windows(rows,100,6,1)
        self.assertEqual(len(set(groups[:4])),1)
        self.assertNotEqual(groups[0],groups[4])
        assert_disjoint(ids,np.array([0,0,0,0,1]),100,1)
        with self.assertRaises(ValueError):assert_disjoint(ids,np.array([0,1,0,0,1]),100,1)

    def test_heldout_counts_cannot_change_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            bins=pd.DataFrame({'chrom':['chr']*10,'start':np.arange(10)*100,'end':np.arange(1,11)*100})
            train=np.array([True]*5+[False]*5)
            results=[]
            for index,value in enumerate((1,1000000)):
                pixels=pd.DataFrame([(0,2,8),(0,3,4),(1,3,6),(2,4,4),(5,7,value),(0,7,value)],columns=['bin1_id','bin2_id','count']).sort_values(['bin1_id','bin2_id'])
                path=Path(temp)/f'{index}.cool'
                cooler.create_cooler(str(path),bins,pixels)
                results.append(fit_expected(path,train,1,3,2))
            np.testing.assert_equal(results[0][0],results[1][0])
            self.assertEqual(results[0][1],results[1][1])
            self.assertEqual(results[0][2][2],3)

    def test_mask_pooling(self):
        a=np.array([[2.,999.],[4.,6.]])
        mask=np.array([[True,False],[True,True]])
        pooled=pool_masked(a,mask,1)
        self.assertEqual(pooled[0,0,0],4)
        self.assertEqual(pooled[1,0,0],.75)

    def test_train_scaler_and_missing_values(self):
        train=np.ones((2,1,2,2,2),dtype=np.float32)
        train[0,0,0]=2;train[1,0,0]=4
        params=fit_scale(train)
        heldout=train.copy();heldout[:,0,0]=100
        heldout[0,0,1,0,0]=0
        transformed=apply_scale(heldout,params)
        self.assertEqual(transformed[0,0,0,0,0],0)
        self.assertEqual(params,[[3.,1.]])
        self.assertEqual(transformed[1,0,0,0,0],97)


if __name__=='__main__':unittest.main()
