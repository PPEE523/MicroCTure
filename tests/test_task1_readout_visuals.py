"""Hand-computed statistics fixtures; no model execution or training."""
import unittest
from scripts.plot_task1_readout import confusion_metrics, aggregate_complete_seed_scores, paired_seed_effects, KEYS, SCORE_NAMES


class ReadoutStatisticsTests(unittest.TestCase):
    def test_asymmetric_three_class_counts(self):
        # CM = [[2,0,0],[0,1,1],[0,1,0]]. CHID TP=0, FP=1,FN=1.
        score = confusion_metrics([0, 0, 1, 1, 2], [0, 0, 1, 2, 1])
        self.assertEqual(score["confusion_matrix"], [[2,0,0],[0,1,1],[0,1,0]])
        self.assertAlmostEqual(score["macro_f1"], .5)
        self.assertEqual(score["CHID_f1"], 0)
        self.assertEqual(score["predicted_counts"], [2,2,1])

    def test_fixed_absent_class_is_zero(self):
        self.assertAlmostEqual(confusion_metrics([1,1],[1,1])["macro_f1"], 1/3)

    def test_chid_precision_recall_denominators(self):
        score = confusion_metrics([0,1,1,2,2,2], [0,2,1,2,2,1])
        self.assertAlmostEqual(score["CHID_precision"], 2/3)
        self.assertAlmostEqual(score["CHID_recall"], 2/3)
        self.assertAlmostEqual(score["CHID_f1"], 2/3)

    def test_seed_sd_is_sample_sd_and_no_missing_seed_mean(self):
        scores = {seed: {name: value for name in SCORE_NAMES} for seed, value in zip([42,43,44],[.2,.4,.6])}
        aggregate = aggregate_complete_seed_scores(scores, [42,43,44])
        self.assertAlmostEqual(aggregate["macro_f1"]["mean"], .4)
        self.assertAlmostEqual(aggregate["macro_f1"]["seed_sample_sd"], .2)
        with self.assertRaises(ValueError):
            aggregate_complete_seed_scores({42:scores[42],43:scores[43]}, [42,43,44])

    def test_paired_seed_identity(self):
        scores = {key: {seed: {name: index/10 + seed/1000 for name in SCORE_NAMES}
                       for seed in [42,43,44]} for index, key in enumerate(KEYS)}
        effects = paired_seed_effects(scores,[42,43,44])
        self.assertAlmostEqual(effects["I3: O1 - O0"]["macro_f1"]["42"], .1)
        self.assertAlmostEqual(effects["O0: I3 - G0"]["CHID_f1"]["44"], -.2)
        del scores["G0/O1"][43]
        with self.assertRaises(ValueError):
            paired_seed_effects(scores,[42,43,44])


if __name__ == "__main__":
    unittest.main()
