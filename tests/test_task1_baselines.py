import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import unittest
import numpy as np
from task1_baselines import LogisticRegression,StandardScaler,PCA,confusion_matrix,f1_score

class BaselinesTest(unittest.TestCase):
    def test_metrics_hand_calculated(self):
        y=np.array([0,0,1,1,2,2]);p=np.array([0,1,1,1,0,2])
        np.testing.assert_equal(confusion_matrix(y,p),[[1,1,0],[0,2,0],[1,0,1]])
        self.assertAlmostEqual(f1_score(y,p),(.5+.8+2/3)/3)
    def test_softmax_fits_three_separated_classes(self):
        x=np.array([[-3,0],[-2,0],[0,3],[0,2],[3,0],[2,0]],dtype=float)
        y=np.array([0,0,1,1,2,2]);model=LogisticRegression().fit(x,y);p=model.predict_proba(x)
        np.testing.assert_equal(p.argmax(1),y);np.testing.assert_allclose(p.sum(1),1)
    def test_transforms_do_not_refit_on_prediction(self):
        x=np.array([[0.,1],[1,2],[2,3]])
        scaler=StandardScaler().fit(x);before=scaler.mean.copy();scaler.transform(x+100)
        np.testing.assert_equal(scaler.mean,before)
        pca=PCA(1).fit(x);np.testing.assert_allclose(pca.transform(x).mean(0),0,atol=1e-6)

if __name__=='__main__':unittest.main()
