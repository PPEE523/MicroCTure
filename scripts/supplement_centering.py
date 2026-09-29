"""Separate prespecified role-1+2 calibration supplement; preserve primary insufficiency."""
import json
import hashlib
import numpy as np
from scipy.spatial.distance import cdist
from discover_task2_v2 import ROOT, read_csv
from discover_task2 import load_band, extract, write_csv
from discovery_v2_core import bin_roles, train_background
from review_novelty import shape_feature
from centering_sensitivity import occlusion_mask


def main():
    extra=json.loads((ROOT/'configs/centering_supplement.json').read_text())
    cfg=json.loads((ROOT/extra['base_config']).read_text())
    source=json.loads((ROOT/cfg['source_config']).read_text())
    data=json.loads((ROOT/'configs/data.json').read_text())
    base=ROOT/cfg['output_dir'];out=ROOT/extra['output_dir'];out.mkdir(parents=True,exist_ok=True)
    sources=['configs/centering_supplement.json',extra['base_config'],'scripts/supplement_centering.py','scripts/centering_sensitivity.py',
             f"{cfg['output_dir']}/features.npz",f"{cfg['output_dir']}/entities.csv",f"{cfg['output_dir']}/all_entity_geometry.csv"]
    manifest=dict(config=extra,sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources})
    path=out/'run_manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Changed inputs')
    path.write_text(json.dumps(manifest,indent=2))
    settings=read_csv(f"{cfg['output_dir']}/variants.csv")
    entities=read_csv(f"{cfg['output_dir']}/entities.csv")
    geometry=read_csv(f"{cfg['output_dir']}/all_entity_geometry.csv")
    with np.load(base/'features.npz') as z:
        features=z['shape'];pca=z['pca'];quality=z['quality'];training=z['training'];candidates=z['candidates']
    n=(data['chrom_length']+99)//100;roles=bin_roles(n,source)
    bands=[train_background(load_band(s,n,256),roles) for s in data['samples']]
    occupied=np.zeros(n,bool)
    for i in candidates:
        e=entities[i];occupied[np.arange(int(e['envelope_start_bin']),int(e['envelope_end_bin']))%n]=True
    with np.load(ROOT/'results/task2_v2/shape_representation.npz') as z:mean,components=z['pca_mean'],z['components']
    newfeatures=[];newquality=[]
    for e in geometry:
        lo,hi=int(e['envelope_start_bin']),int(e['envelope_end_bin'])
        if e['kind']!='known' or int(e['role'])!=2 or e['boundary_reproducible']!='True' or occupied[lo:hi].any():continue
        fs=[];qs=[];cache={}
        for v in settings:
            w,s=int(v['width_bins']),int(v['shift_bins'])
            center=int(e['original_center_bin'] if v['anchor']=='original' else e['boundary_center_bin'])
            start=center+s-w//2;occlusion=occlusion_mask(w,float(v['fraction']),int(v['seed']))
            pair=[];good=[]
            for rep,b in enumerate(bands):
                key=(start,w,rep)
                if key not in cache:cache[key]=extract(b,start,w)
                raw,oe,mask=cache[key];upper=np.triu(mask,2)
                good.append(bool(mask.sum()/(w*w-3*w+2)>=cfg['min_valid_fraction'] and upper.any() and (raw[upper]==0).mean()<=cfg['max_zero_fraction']))
                pair.append(shape_feature(oe,mask&occlusion))
            fs.append(pair);qs.append(good)
        e=dict(e);e['all_variants_quality_pass']=str(bool(np.all(qs)));e['reference_member']='False';e['calibration_member']='False'
        entities.append(e);newfeatures.append(fs);newquality.append(qs)
    if newfeatures:
        newfeatures=np.asarray(newfeatures,dtype=np.float32)
        features=np.concatenate([features,newfeatures]);pca=np.concatenate([pca,(newfeatures-mean)@components.T]);quality=np.concatenate([quality,newquality])
    pool=[i for i,e in enumerate(entities) if e['kind']=='known' and int(e['role']) in extra['calibration_roles'] and e['boundary_reproducible']=='True' and e['all_variants_quality_pass']=='True']
    calibration=[];used=np.zeros(n,bool)
    for i in sorted(pool,key=lambda i:(int(entities[i]['envelope_start_bin']),entities[i]['entity_id'])):
        lo,hi=int(entities[i]['envelope_start_bin']),int(entities[i]['envelope_end_bin'])
        if not used[lo:hi].any():calibration.append(i);used[lo:hi]=True
    for i,e in enumerate(entities):e['calibration_member']=str(i in calibration)
    write_csv(out/'entities.csv',entities)
    np.savez_compressed(out/'features.npz',shape=features,pca=pca,quality=quality,training=training,calibration=calibration,candidates=candidates)
    (out/'cohort_frozen.json').write_text(json.dumps(dict(training=training.tolist(),calibration=calibration,candidates=candidates.tolist(),reason=extra['reason']),indent=2))
    rows=[];thresholds=[];calrows=[];flags={};factors=[]
    lookup={(v['anchor'],int(v['width_bins']),int(v['shift_bins']),v['occlusion']):i for i,v in enumerate(settings)}
    for metric,x,distance in [('pca',pca,'euclidean'),('shape',features,'cosine')]:
        for vi,v in enumerate(settings):
            matches=training[cdist(x[calibration,vi,0],x[training,vi,0],metric=distance).argmin(1)]
            values=[]
            for i,j in zip(calibration,matches):
                ds=[float(cdist(x[i:i+1,vi,rep],x[j:j+1,vi,rep],metric=distance)[0,0]) for rep in [0,1]]
                values.append(ds);calrows.append(dict(variant_id=v['variant_id'],metric=metric,entity_index=i,reference_index=int(j),rep1_distance=ds[0],rep2_distance=ds[1]))
            limits=np.quantile(values,cfg['distance_quantile'],axis=0) if len(calibration)>=cfg['min_calibration_windows'] else np.array([np.nan,np.nan])
            thresholds.append(dict(variant_id=v['variant_id'],metric=metric,calibration_windows=len(calibration),rep1_cutoff=float(limits[0]),rep2_cutoff=float(limits[1])))
            matches=training[cdist(x[candidates,vi,0],x[training,vi,0],metric=distance).argmin(1)]
            for i,j in zip(candidates,matches):
                ds=[float(cdist(x[i:i+1,vi,rep],x[j:j+1,vi,rep],metric=distance)[0,0]) for rep in [0,1]]
                flag=bool(quality[i,vi].all() and np.isfinite(limits).all() and np.all(np.array(ds)>limits));flags[i,vi,metric]=flag
                rows.append(dict(entity_id=entities[i]['entity_id'],entity_index=int(i),variant_id=v['variant_id'],metric=metric,reference_index=int(j),nearest_reference=entities[j]['entity_id'],rep1_distance=ds[0],rep2_distance=ds[1],rep1_cutoff=float(limits[0]),rep2_cutoff=float(limits[1]),quality_pass=bool(quality[i,vi].all()),outside_both=flag))
            a,w,s,o=v['anchor'],int(v['width_bins']),int(v['shift_bins']),v['occlusion'];contrasts=[]
            if a=='boundary':contrasts.append(('recenter',lookup['original',w,s,o]))
            if s:contrasts.append(('translation',lookup[a,w,0,o]))
            if w!=64:contrasts.append(('scale',lookup[a,64,s,o]))
            if o!='none':contrasts.append(('occlusion',lookup[a,w,s,'none']))
            for factor,bvi in contrasts:
                vals=np.array([[cdist(x[i:i+1,bvi,rep],x[i:i+1,vi,rep],metric=distance)[0,0] for rep in [0,1]] for i in calibration])
                cutoff=np.quantile(vals,cfg['distance_quantile'],axis=0) if len(calibration)>=cfg['min_calibration_windows'] else [np.nan,np.nan]
                for i in candidates:
                    ds=[float(cdist(x[i:i+1,bvi,rep],x[i:i+1,vi,rep],metric=distance)[0,0]) for rep in [0,1]]
                    factors.append(dict(entity_id=entities[i]['entity_id'],factor=factor,metric=metric,baseline_variant=settings[bvi]['variant_id'],target_variant=v['variant_id'],rep1_change=ds[0],rep2_change=ds[1],known_rep1_q95=float(cutoff[0]),known_rep2_q95=float(cutoff[1]),quality_pass=bool(quality[i,bvi].all() and quality[i,vi].all()),unusually_sensitive_both=bool(quality[i,bvi].all() and quality[i,vi].all() and np.isfinite(cutoff).all() and np.all(np.array(ds)>cutoff))))
    for name,values in [('candidate_variants.csv',rows),('thresholds.csv',thresholds),('calibration_distances.csv',calrows),('isolated_factor_sensitivity.csv',factors)]:write_csv(out/name,values)
    summaries=[]
    for i in candidates:
        row=dict(entity_id=entities[i]['entity_id'],boundary_reproducible=entities[i]['boundary_reproducible']=='True',calibration_sufficient=len(calibration)>=cfg['min_calibration_windows'],all_variants_quality_pass=bool(quality[i].all()))
        for anchor in cfg['anchors']:
            ids=[vi for vi,v in enumerate(settings) if v['anchor']==anchor]
            row[anchor+'_outside_variants']=sum(flags[i,vi,'pca'] and flags[i,vi,'shape'] for vi in ids)
        fractions=[]
        for w in cfg['width_bins']:
            ids=[vi for vi,v in enumerate(settings) if v['anchor']=='boundary' and int(v['width_bins'])==w]
            f=float(np.mean([flags[i,vi,'pca'] and flags[i,vi,'shape'] for vi in ids]));fractions.append(f);row[f'boundary_w{w}_outside_fraction']=f
        row['robust_recentered_anomaly']=bool(row['boundary_reproducible'] and row['calibration_sufficient'] and row['all_variants_quality_pass'] and min(fractions)>=cfg['robust_fraction'])
        summaries.append(row)
    write_csv(out/'candidate_summary.csv',summaries)
    report=dict(reason=extra['reason'],reference_windows=len(training),calibration_windows=len(calibration),candidates=21,variants=54,
                candidates_with_any_two_metric_anomaly=sum(r['original_outside_variants']+r['boundary_outside_variants']>0 for r in summaries),
                candidates_with_recentered_two_metric_anomaly=sum(r['boundary_outside_variants']>0 for r in summaries),robust_recentered_anomalies=sum(r['robust_recentered_anomaly'] for r in summaries),
                candidate_metric_comparisons=len(rows),isolated_factor_metric_comparisons=len(factors))
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
