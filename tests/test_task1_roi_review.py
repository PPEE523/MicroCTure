"""Hand-computed metric and paired-cluster fixtures; no model fitting or images."""
from __future__ import annotations
import unittest
import numpy as np
from scripts.audit_task1_roi_review import scores_from_confusion, group_bootstrap, ARMS, SEEDS, _without_time


class ReviewTests(unittest.TestCase):
    def test_fixed_three_class_confusion_arithmetic(self):
        cm=np.array([[2,0,1],[1,3,0],[0,2,1]])
        m,support,predicted=scores_from_confusion(cm)
        self.assertEqual(support.tolist(),[3,4,3]);self.assertEqual(predicted.tolist(),[3,5,2])
        self.assertAlmostEqual(m['CHID_precision'],.5);self.assertAlmostEqual(m['CHID_recall'],1/3)
        self.assertAlmostEqual(m['CHID_f1'],.4);self.assertAlmostEqual(m['macro_f1'],(2/3+2/3+.4)/3)

    def test_identical_paired_predictions_have_zero_difference(self):
        identity={f'{g}_{y}':{'group_id':g,'y':y} for g in range(4) for y in range(3)};oof={}
        rows=[{'structure_id':sid,'group_id':r['group_id'],'y_true':r['y'],'y_pred':r['y']} for sid,r in identity.items()]
        for arm in ARMS:
            for seed in SEEDS:oof[(*arm,seed)]={'rows':rows}
        x=group_bootstrap(oof,identity,repeats=50,rng_seed=7)
        self.assertEqual(x['arms']['R1/I3']['macro_f1']['ci95'],[1.,1.])
        self.assertEqual(x['pairs']['R1/I3-R0/I3']['CHID_f1']['ci95'],[0.,0.])
        self.assertEqual(x['zero_CHID_support_bootstraps'],0)

    def test_replay_ignores_only_measurement_time(self):
        a={'epochs':[{'epoch':0,'epoch_seconds':2.,'loss':1.}]};b={'epochs':[{'epoch':0,'epoch_seconds':9.,'loss':1.}]}
        self.assertEqual(_without_time(a),_without_time(b))
        b['epochs'][0]['loss']=.9;self.assertNotEqual(_without_time(a),_without_time(b))


if __name__=='__main__':unittest.main()
