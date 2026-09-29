"""Recompute both calibration experiments, nearest choices and robust decisions."""
import json
import hashlib
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image
from discover_task2_v2 import ROOT, read_csv


def main():
    cfg=json.loads((ROOT/'configs/centering_sensitivity.json').read_text())
    extra=json.loads((ROOT/'configs/centering_supplement.json').read_text())
    settings=read_csv(f"{cfg['output_dir']}/variants.csv")
    variants={r['variant_id']:i for i,r in enumerate(settings)}
    assert len(variants)==54
    # The supplemental cohort changes calibration only, never candidate crops or features.
    with np.load(ROOT/cfg['output_dir']/'features.npz') as primary:
        primary_candidates=primary['candidates'].copy()
        primary_features=primary['shape'][primary_candidates].copy()
    results=[]
    for directory in [cfg['output_dir'],extra['output_dir']]:
        out=ROOT/directory
        manifest=json.loads((out/'run_manifest.json').read_text())
        for name,digest in manifest['sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        entities=read_csv(f'{directory}/entities.csv')
        with np.load(out/'features.npz') as z:
            xs={name:z[name] for name in ['pca','shape']};quality=z['quality'];training=z['training'];calibration=z['calibration'];candidates=z['candidates']
        np.testing.assert_array_equal(xs['shape'][candidates],primary_features)
        candidate_ranges=[(int(entities[i]['envelope_start_bin']),int(entities[i]['envelope_end_bin'])) for i in candidates]
        def overlap(a,b):return a[0]<b[1] and b[0]<a[1]
        cal_ranges=[]
        for i in list(training)+list(calibration):
            e=entities[i];interval=(int(e['envelope_start_bin']),int(e['envelope_end_bin']))
            assert not any(overlap(interval,r) for r in candidate_ranges)
            assert e['boundary_reproducible']=='True' and quality[i].all()
            assert int(e['role'])==0 if i in training else int(e['role']) in ([1] if directory==cfg['output_dir'] else [1,2])
            if i in calibration:
                assert not any(overlap(interval,r) for r in cal_ranges);cal_ranges.append(interval)
        for e in entities:
            for v in settings:
                center=int(e['original_center_bin'] if v['anchor']=='original' else e['boundary_center_bin'])
                start=center+int(v['shift_bins'])-int(v['width_bins'])//2
                assert start>=int(e['envelope_start_bin']) and start+int(v['width_bins'])<=int(e['envelope_end_bin'])
        thresholds={}
        saved=read_csv(f'{directory}/calibration_distances.csv')
        for metric,distance in [('pca','euclidean'),('shape','cosine')]:
            x=xs[metric]
            for vi,v in enumerate(settings):
                vals=[r for r in saved if r['metric']==metric and r['variant_id']==v['variant_id']]
                assert len(vals)==len(calibration)
                for r in vals:
                    i,j=int(r['entity_index']),int(r['reference_index'])
                    assert i in calibration and j in training
                    expected=training[cdist(x[i:i+1,vi,0],x[training,vi,0],metric=distance).argmin()]
                    assert j==expected
                    ds=[cdist(x[i:i+1,vi,rep],x[j:j+1,vi,rep],metric=distance)[0,0] for rep in [0,1]]
                    np.testing.assert_allclose(ds,[float(r['rep1_distance']),float(r['rep2_distance'])])
                thresholds[vi,metric]=np.quantile([[float(r['rep1_distance']),float(r['rep2_distance'])] for r in vals],cfg['distance_quantile'],axis=0) if len(calibration)>=cfg['min_calibration_windows'] else np.array([np.nan,np.nan])
        rows=read_csv(f'{directory}/candidate_variants.csv');assert len(rows)==2268
        flags={}
        for r in rows:
            i,vi,metric=int(r['entity_index']),variants[r['variant_id']],r['metric'];x=xs[metric];distance='euclidean' if metric=='pca' else 'cosine'
            j=training[cdist(x[i:i+1,vi,0],x[training,vi,0],metric=distance).argmin()]
            assert int(r['reference_index'])==j
            ds=[cdist(x[i:i+1,vi,rep],x[j:j+1,vi,rep],metric=distance)[0,0] for rep in [0,1]]
            np.testing.assert_allclose(ds,[float(r['rep1_distance']),float(r['rep2_distance'])])
            cutoff=thresholds[vi,metric]
            np.testing.assert_allclose(cutoff,[float(r['rep1_cutoff']),float(r['rep2_cutoff'])],equal_nan=True)
            passed=bool(quality[i,vi].all() and np.isfinite(cutoff).all() and np.all(ds>cutoff))
            assert r['outside_both']==str(passed);flags[i,vi,metric]=passed
        summaries=read_csv(f'{directory}/candidate_summary.csv')
        by_name={e['entity_id']:i for i,e in enumerate(entities)}
        for r in summaries:
            i=by_name[r['entity_id']];fs=[]
            for anchor in cfg['anchors']:
                ids=[vi for vi,v in enumerate(settings) if v['anchor']==anchor]
                assert int(r[anchor+'_outside_variants'])==sum(flags[i,vi,'pca'] and flags[i,vi,'shape'] for vi in ids)
            for width in cfg['width_bins']:
                ids=[vi for vi,v in enumerate(settings) if v['anchor']=='boundary' and int(v['width_bins'])==width]
                fraction=float(np.mean([flags[i,vi,'pca'] and flags[i,vi,'shape'] for vi in ids]));fs.append(fraction)
                np.testing.assert_allclose(fraction,float(r[f'boundary_w{width}_outside_fraction']))
            robust=bool(len(calibration)>=cfg['min_calibration_windows'] and quality[i].all() and entities[i]['boundary_reproducible']=='True' and min(fs)>=cfg['robust_fraction'])
            assert r['robust_recentered_anomaly']==str(robust)
        factors=read_csv(f'{directory}/isolated_factor_sensitivity.csv');assert len(factors)==5670
        cached={}
        for r in factors:
            i=by_name[r['entity_id']];a,b=variants[r['baseline_variant']],variants[r['target_variant']];metric=r['metric'];x=xs[metric];distance='euclidean' if metric=='pca' else 'cosine'
            va,vb=settings[a],settings[b]
            field={'recenter':'anchor','translation':'shift_bins','scale':'width_bins','occlusion':'occlusion'}[r['factor']]
            assert [key for key in ['anchor','width_bins','shift_bins','occlusion'] if va[key]!=vb[key]]==[field]
            key=(a,b,metric)
            if key not in cached:
                vals=[[cdist(x[j:j+1,a,rep],x[j:j+1,b,rep],metric=distance)[0,0] for rep in [0,1]] for j in calibration]
                cached[key]=np.quantile(vals,cfg['distance_quantile'],axis=0) if len(calibration)>=cfg['min_calibration_windows'] else np.array([np.nan,np.nan])
            ds=[cdist(x[i:i+1,a,rep],x[i:i+1,b,rep],metric=distance)[0,0] for rep in [0,1]];cutoff=cached[key]
            np.testing.assert_allclose(ds,[float(r['rep1_change']),float(r['rep2_change'])])
            np.testing.assert_allclose(cutoff,[float(r['known_rep1_q95']),float(r['known_rep2_q95'])],equal_nan=True)
            assert r['unusually_sensitive_both']==str(bool(quality[i,a].all() and quality[i,b].all() and np.isfinite(cutoff).all() and np.all(ds>cutoff)))
        results.append(dict(directory=directory,calibration_windows=len(calibration),candidate_comparisons=len(rows),isolated_factor_comparisons=len(factors),status='passed'))
    for directory in [cfg['output_dir'],extra['output_dir']]:
        with Image.open(ROOT/directory/'all_variants.png') as image:image.verify()
    report=dict(status='passed',experiments=results,identical_candidate_features=True,reference_contexts_disjoint_from_candidates=True,only_one_factor_changes_in_paired_contrasts=True,source_hashes_match=True,independent_confirmation=False)
    (ROOT/'reports/centering_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
