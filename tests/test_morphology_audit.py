import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_morphology import remove_row_effects


class MorphologyAuditTest(unittest.TestCase):
    def test_additive_row_pattern_is_removed_without_modifying_input(self):
        row=np.linspace(-1,1,20)
        matrix=row[:,None]+row[None,:]+2
        original=matrix.copy();mask=np.ones_like(matrix,bool)
        residual,effect=remove_row_effects(matrix,mask,40)
        np.testing.assert_allclose(residual,0,atol=1e-10)
        np.testing.assert_allclose(residual+effect,matrix)
        np.testing.assert_array_equal(matrix,original)

    def test_masked_values_do_not_affect_fit(self):
        matrix=np.arange(100,dtype=float).reshape(10,10)
        matrix=(matrix+matrix.T)/2
        mask=np.ones((10,10),bool);mask[0]=False;mask[:,0]=False
        altered=matrix.copy();altered[~mask]=1000000
        a,_=remove_row_effects(matrix,mask,10)
        b,_=remove_row_effects(altered,mask,10)
        np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
