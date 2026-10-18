"""Scientific invariants of paired saved-prediction diagnostics; no training."""
import copy
import unittest
import numpy as np
from microcture.tasks.task5.analysis import (MODELS, METRICS, paired_rows, collapse_replicates,
    ssim_values, pixel_metrics, contact_profile, boundary_error, select_example, score_stats)


class Task5AnalysisTests(unittest.TestCase):
    def pair_fixture(self):
        meta=[dict(structure_id='CHIN_1',replicate=r,type='CHIN',group_id='1',start=100,end=200,start_bin=0) for r in [1,2]]
        records=[]
        for m in MODELS:
            for r in meta:
                value=([10.,20.,30.][int(m[-2:])-42] if m[-2:].isdigit() else 5.)
                records.append(dict(model=m,**r,**{k:value for k in METRICS}))
        return meta,records

    def test_scores_average_seeds_before_pairing(self):
        meta,records=self.pair_fixture();rows,seeds=paired_rows(records,meta)
        self.assertEqual(len(rows),4);self.assertEqual(len(seeds),12)
        self.assertTrue(all(r['cnn_psnr']==20 and r['delta_psnr']==15 for r in rows))

    def test_missing_seed_is_rejected(self):
        meta,r=self.pair_fixture()
        with self.assertRaises(ValueError):paired_rows(r[:-1],meta)

    def test_duplicate_pair_is_rejected(self):
        meta,r=self.pair_fixture()
        with self.assertRaises(ValueError):paired_rows(r+[r[0]],meta)

    def test_repeat_pairing_does_not_duplicate_positions(self):
        meta,r=self.pair_fixture();rows,_=paired_rows(r,meta)
        positions=collapse_replicates(rows,['family','structure_id'],[f'{p}_{k}' for k in METRICS for p in ['cnn','bicubic','delta']])
        self.assertEqual(len(positions),2);self.assertNotIn('replicate',positions[0])

    def test_one_repeat_is_not_a_complete_position(self):
        with self.assertRaises(ValueError):collapse_replicates([dict(structure_id='a',replicate=1,psnr=3)],['structure_id'],['psnr'])

    def test_distance_partition_is_disjoint_and_exhaustive(self):
        sep=np.abs(np.arange(128)[:,None]-np.arange(128)[None,:])*200;upper=np.triu(np.ones((128,128),bool),2)
        bands=[upper&(sep>=lo)&(sep<hi) for lo,hi in [(400,1600),(1600,6400),(6400,25600)]]
        np.testing.assert_array_equal(np.sum(bands,axis=0),upper.astype(int))

    def test_identical_images_have_unit_ssim_and_zero_mse(self):
        x=np.arange(32*32,dtype='float32').reshape(32,32)/1024;m=np.ones_like(x,bool);s,inter=ssim_values(x,x,m)
        r=pixel_metrics(x,x,m,s,inter);self.assertEqual(r['mse'],0);self.assertAlmostEqual(r['ssim'],1,6)

    def test_ssim_neighborhood_does_not_fill_missing_pixels(self):
        x=np.ones((32,32),dtype='float32');m=np.ones_like(x,bool);m[16,16]=False;s,inter=ssim_values(x,x,m)
        self.assertFalse(inter[16,16]);self.assertFalse(inter[15,15]);self.assertTrue(inter[8,8])

    def test_far_boundary_proxy_is_na_not_zero(self):
        x=np.ones((32,32),dtype='float32');m=np.ones_like(x,bool);p=contact_profile(x,m,band=(6400,25600))
        self.assertTrue(np.isnan(p).all());error,n=boundary_error(p,p,dict(start=3000,end=4000,start_bin=0));self.assertIsNone(error);self.assertEqual(n,0)

    def test_near_ssim_without_full_valid_neighborhood_is_na(self):
        x=np.ones((32,32),dtype='float32');d=np.abs(np.arange(32)[:,None]-np.arange(32)[None,:]);m=d>=2
        s,inter=ssim_values(x,x,m);near=m&(d>=2)&(d<8)&np.triu(np.ones_like(m),2)
        result=pixel_metrics(x,x,near,s,inter);self.assertIsNone(result['ssim']);self.assertEqual(result['ssim_pixels'],0)

    def test_boundary_shift_uses_fixed_resolution(self):
        x=np.full(40,np.nan);y=x.copy();x[10:17]=[2,1,0,1,2,3,4];y[10:17]=[3,2,1,0,1,2,3]
        value,n=boundary_error(x,y,dict(start=2600,end=2600,start_bin=0),radius=3)
        self.assertEqual(value,200);self.assertEqual(n,2)

    def test_example_is_selected_by_genomic_start(self):
        metadata=[dict(type='CHIN',replicate=1,start=s,structure_id=f'id{s}',psnr=100-s) for s in [9,1,7,3]]
        _,row=select_example(metadata,dict(type='CHIN',replicate=1));self.assertEqual(row['start'],3)

    def test_ties_and_na_have_separate_counts(self):
        s=score_stats([None,-1,0,1,1e-12]);self.assertEqual(s['n_evaluable'],4);self.assertEqual((s['positive'],s['negative'],s['tied']),(1,1,2))


if __name__=='__main__':unittest.main()
