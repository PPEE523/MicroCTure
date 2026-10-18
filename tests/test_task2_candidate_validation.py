"""Scientific invariants for frozen candidates, controls and retrieval diagnostics."""
import unittest
import numpy as np

from microcture.tasks.task2.validation import (
    formal_capacity, known_query_summaries, match_controls, maximum_disjoint_intervals,
    rank_query, retrieval, six_cells, summarize_retrieval, support_decision,
)


def entity(name,lo,hi,kind='known',label='CHIN'):
    return dict(entity_id=name,envelope_start_bin=lo,envelope_end_bin=hi,kind=kind,type=label)


def grid():
    return [dict(variant_id=f'{a}_{w}_{j}',anchor=a,width_bins=w)
            for a in ('original','boundary') for w in (64,128,256) for j in range(9)]


class SupportThresholdTests(unittest.TestCase):
    def test_original_floor_and_strict_equality(self):
        row=support_decision(.5,[.1]*10)
        self.assertEqual(row['cutoff'],.5)
        self.assertFalse(row['replicate_supported'])

    def test_control_upper_tail_is_not_replaced_by_candidate_outcome(self):
        controls=[.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]
        low=support_decision(.6,controls);high=support_decision(.99,controls)
        self.assertAlmostEqual(low['cutoff'],.955)
        self.assertEqual(low['cutoff'],high['cutoff'])
        self.assertFalse(low['replicate_supported']);self.assertTrue(high['replicate_supported'])

    def test_invalid_or_nine_controls_stay_na(self):
        row=support_decision(.99,[.1]*9+[None,float('nan'),float('inf')])
        self.assertEqual(row['matched_controls'],9)
        self.assertIsNone(row['replicate_supported']);self.assertIsNone(row['cutoff'])

    def test_nonfinite_candidate_is_not_a_negative(self):
        row=support_decision(float('nan'),[.1]*20)
        self.assertEqual(row['assessment_reason'],'nonfinite_candidate_score')
        self.assertIsNone(row['replicate_supported'])


class SpatialControlTests(unittest.TestCase):
    def test_exact_capacity_is_not_coordinate_order_greedy(self):
        rows=[entity('wide',0,100),entity('a',1,10),entity('b',10,20),entity('c',20,30)]
        self.assertEqual([r['entity_id'] for r in maximum_disjoint_intervals(rows)],['a','b','c'])

    def test_primary_capacity_keeps_boundary_quality_and_role_filters(self):
        rows=[]
        for i in range(12):
            r=entity(str(i),i*10,i*10+10)
            r.update(role=1,boundary_reproducible=i!=0,all_variants_quality_pass=i!=1,
                     reference_geometry_eligible=i!=2,calibration_member=i>2)
            rows.append(r)
        result,pool=formal_capacity(rows,10)
        self.assertEqual(len(pool),9);self.assertEqual(result['maximum_nonoverlapping_windows'],9)
        self.assertFalse(result['feasible'])

    def test_matching_ignores_scores_and_obeys_same_width_and_disjointness(self):
        focal=dict(width_bins=64,mean_oe=1,valid_fraction=.9,correlation=.1)
        pool=[]
        for i,(lo,w) in enumerate([(0,64),(5,64),(20,256),(20,64)]):
            r=entity(str(i),lo,lo+10)
            r.update(window_index=i,width_bins=w,mean_oe=1,valid_fraction=.9,correlation=.999)
            pool.append(r)
        chosen=[r['entity_id'] for r,_ in match_controls(focal,pool,3)]
        self.assertEqual(chosen,['0','3'])
        focal['correlation']=.99
        for r in pool:r['correlation']=-1
        self.assertEqual([r['entity_id'] for r,_ in match_controls(focal,pool,3)],chosen)


class CandidateGridTests(unittest.TestCase):
    def test_missing_variant_is_an_error_not_a_smaller_denominator(self):
        settings=grid();decisions={v['variant_id']:dict(replicate_supported=True,margin=.1) for v in settings[:-1]}
        with self.assertRaisesRegex(ValueError,'Missing or extra'):six_cells(settings,decisions)

    def test_unassessed_variant_prevents_complete_cell_fraction(self):
        settings=grid();decisions={v['variant_id']:dict(replicate_supported=True,margin=.1) for v in settings}
        decisions[settings[0]['variant_id']]=dict(replicate_supported=None,margin=None)
        rows=six_cells(settings,decisions)
        self.assertEqual(rows[0]['assessed_variants'],8)
        self.assertEqual(rows[0]['above_threshold'],8)
        self.assertEqual(rows[0]['not_evaluable'],1)
        self.assertIsNone(rows[0]['full_cell_above_fraction'])


class KnownRetrievalTests(unittest.TestCase):
    def test_class_retrieval_excludes_self_and_overlapping_context(self):
        q=entity('query',10,20)
        gallery=[q,entity('overlap',15,25),entity('separate',30,40)]
        x=np.array([[1.,0],[1.,.001],[0,1.]])
        result=rank_query(x[0],x,q,gallery,'euclidean')
        self.assertEqual([r['entity_id'] for r,_ in result],['separate'])

    def test_identity_retrieval_keeps_only_true_self_as_overlapping_positive(self):
        q=entity('query',10,20)
        gallery=[q,entity('overlap',15,25),entity('separate',30,40)]
        x=np.array([[1.,0],[1.,.001],[0,1.]])
        result=rank_query(x[0],x,q,gallery,'euclidean',identity=True)
        self.assertEqual([r['entity_id'] for r,_ in result],['query','separate'])

    def test_absent_chid_queries_report_na_and_complete_grid_not_best_variant(self):
        settings=grid()
        entities=[entity('ref_a',0,10,label='CHIN'),entity('ref_b',20,30,label='OPCID'),entity('query',100,110,label='OPCID')]
        x=np.tile(np.array([[1.,0],[0,1.],[.1,1.]])[:,None,None,:],(1,54,2,1))
        arrays=dict(pca=x,shape=x,quality=np.ones((3,54,2),bool),training=np.array([0,1]),calibration=np.array([2]))
        rows=retrieval('test',entities,arrays,settings)
        self.assertEqual(len(rows),54*2*3)
        summary=summarize_retrieval(rows)
        for r in summary:
            if r['type']=='CHID':self.assertIsNone(r['top1']);self.assertEqual(r['query_ids'],0)
        queries=known_query_summaries(rows)
        self.assertTrue(all(r['variants']==54 for r in queries))
        self.assertTrue(all(r['top1_count']==54 for r in queries))


if __name__=='__main__':unittest.main()
