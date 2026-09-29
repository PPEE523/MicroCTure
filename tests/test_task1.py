import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import unittest
import numpy as np
import torch
from train_task1 import CNN,grouped,inputs,SPECS

class Task1Test(unittest.TestCase):
    def test_replicates_average_before_classification(self):
        d={'structure_id':np.array(['a','b','a','b']),'y':np.array([0,1,0,1])}
        p=np.array([[.9,.1,0],[.2,.8,0],[.3,.7,0],[.4,.6,0]])
        ids,y,out=grouped(d,p)
        np.testing.assert_allclose(out,[[.6,.4,0],[.3,.7,0]])
        self.assertEqual(list(y),[0,1])
    def test_variants_have_valid_gradients(self):
        torch.set_num_threads(1)
        d={k:np.ones((2,2,2,64,64),dtype='float32') for k in ['X_oe','X_depth']}
        for spec in SPECS.values():
            model=CNN(len(spec[1]),spec[2]);x=torch.from_numpy(inputs(d,spec)).requires_grad_(True)
            output=model(x)
            self.assertEqual(tuple(output.shape),(2,3))
            output.sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())

if __name__=='__main__':unittest.main()
