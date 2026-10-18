import json,unittest
import numpy as np
from scripts.evaluate_task1_fixed_fusion import blend_rows


def row(sid,probs):
    mean=np.asarray(probs).mean(0)
    return {'structure_id':sid,'y_true':'2','group_id':'7',**{f'p_{c}':str(mean[i]) for i,c in enumerate(('OPCID','CHIN','CHID'))},'replicate_probabilities_json':json.dumps([{'replicate':str(i+1),'probabilities':p} for i,p in enumerate(probs)])}


class FusionTests(unittest.TestCase):
    def test_fixed_half_mixture_commutes_with_WT_mean(self):
        lr=[[.1,.2,.7],[.2,.4,.4]];cnn=[[.4,.2,.4],[.3,.1,.6]]
        data,p=blend_rows([row('x',lr)],[[row('x',cnn)]])
        np.testing.assert_allclose(p,.5*np.asarray(lr)+.5*np.asarray(cnn))
        np.testing.assert_allclose(p.mean(0),.5*np.mean(lr,axis=0)+.5*np.mean(cnn,axis=0))
        self.assertEqual(data['replicate'].tolist(),[1,2])

    def test_pooled_mean_is_diagnostic_arithmetic(self):
        lr=[[1.,0.,0.],[0.,1.,0.]];cnn1=[[0.,1.,0.],[0.,0.,1.]];cnn2=[[0.,0.,1.],[1.,0.,0.]]
        _,p=blend_rows([row('x',lr)],[[row('x',cnn1)],[row('x',cnn2)]])
        np.testing.assert_allclose(p,.5*np.asarray(lr)+.25*(np.asarray(cnn1)+np.asarray(cnn2)))

    def test_ID_and_truth_mismatch_fail(self):
        r=row('x',[[.2,.3,.5]]*2)
        with self.assertRaises(ValueError):blend_rows([r],[[row('other',[[.2,.3,.5]]*2)]])
        bad={**r,'y_true':'1'}
        with self.assertRaises(ValueError):blend_rows([r],[[bad]])


if __name__=='__main__':unittest.main()
