import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from analyze_conditions import compare,local_signal

class ConditionsTest(unittest.TestCase):
    def test_all_cross_repeat_contrasts_required(self):
        self.assertEqual(compare([1,1.1],[3,3.2],.01,1.5)['direction'],'increased')
        self.assertEqual(compare([3,3.2],[1,1.1],.01,1.5)['direction'],'decreased')
        self.assertEqual(compare([1,3],[2,4],.01,1.5)['direction'],'inconsistent_or_small')
    def test_repeat_order_does_not_change_effect(self):
        a=compare([1,2],[4,5],.01,1.5);b=compare([2,1],[5,4],.01,1.5)
        self.assertEqual(a['log2_fold_change'],b['log2_fold_change'])
        self.assertEqual(a['direction'],b['direction'])
    def test_signal_counts_both_endpoints_and_opportunities(self):
        band=np.zeros((8,4));band[0,2]=4
        b={'band':band,'total':8}
        common=np.ones(8,bool)
        s=local_signal(b,common,2,2)
        self.assertEqual(s[0],250000);self.assertEqual(s[2],250000);self.assertEqual(s[1],0)
        common[2]=False;s=local_signal(b,common,2,2)
        self.assertTrue(np.isnan(s[2]));self.assertEqual(s[0],0)
    def test_pseudocount_finite_for_zero(self):
        r=compare([0,0],[0,0],.01,1.5)
        self.assertEqual(r['log2_fold_change'],0)

if __name__=='__main__':unittest.main()
