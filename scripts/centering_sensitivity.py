"""Matched recentering, scale, shift and symmetric occlusion sensitivity experiment."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/microcture-mpl')
import hashlib
import itertools
import json
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from discover_task2_v2 import ROOT, read_csv
from discover_task2 import load_band, extract, write_csv
from discovery_v2_core import bin_roles, train_background
from review_novelty import shape_feature, boundary_profile


def occlusion_mask(width, fraction, seed):
    rng = np.random.default_rng(seed)
    remove = np.triu(rng.random((width, width)) < fraction, 2)
    return ~(remove | remove.T)


def anchor_position(band, center, cfg):
    width = cfg['anchor_search_bins']
    start = center - width//2
    _, oe, mask = extract(band, start, width)
    profile = boundary_profile(oe, mask, cfg['boundary_flank_bins'], cfg['boundary_min_pairs'])
    if not np.isfinite(profile).any():
        return center, float('nan'), False
    index = int(np.nanargmin(profile))
    return start+index, float(profile[index]), True


def variants(cfg):
    return [dict(anchor=a, width_bins=w, shift_bins=s, occlusion=o['name'], fraction=o['fraction'], seed=o['seed'],
                 variant_id=f'{a}_w{w}_s{s}_{o["name"]}')
            for a, w, s, o in itertools.product(cfg['anchors'], cfg['width_bins'], cfg['shift_bins'], cfg['occlusions'])]


def main():
    cfg = json.loads((ROOT/'configs/centering_sensitivity.json').read_text())
    source = json.loads((ROOT/cfg['source_config']).read_text())
    data = json.loads((ROOT/'configs/data.json').read_text())
    out = ROOT/cfg['output_dir']
    out.mkdir(parents=True, exist_ok=True)
    files = ['configs/centering_sensitivity.json', cfg['source_config'], cfg['candidate_source'], 'configs/data.json',
             'scripts/centering_sensitivity.py', 'scripts/review_novelty.py', 'scripts/discovery_v2_core.py',
             'scripts/discover_task2.py', 'scripts/prepare_dataset.py', 'data/processed/structures.csv',
             'results/task2_v2/shape_representation.npz']
    manifest = dict(config=cfg, sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
                    samples={s['sample_id']:dict(size=(ROOT/s['path']).stat().st_size, mtime_ns=(ROOT/s['path']).stat().st_mtime_ns) for s in data['samples']})
    path = out/'run_manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Inputs changed; use a new versioned directory')
    path.write_text(json.dumps(manifest, indent=2))
    settings = variants(cfg)
    write_csv(out/'variants.csv', settings)
    n = (data['chrom_length']+99)//100
    roles = bin_roles(n, source)
    bands = [train_background(load_band(s, n, 256), roles) for s in data['samples']]
    candidates = read_csv(cfg['candidate_source'])
    known = read_csv('data/processed/structures.csv')
    entities = []
    # Same fixed 6.4 kb search around the original center for every entity.
    for kind, collection in [('candidate', candidates), ('known', known)]:
        for row in collection:
            center = (int(row['start_bp'])+int(row['window_length_bp'])//2)//100 if kind=='candidate' else int(float(row['center'])//100)
            a1, score1, ok1 = anchor_position(bands[0], center, cfg)
            a2, score2, ok2 = anchor_position(bands[1], center, cfg)
            radius = max(cfg['width_bins'])//2 + max(abs(s) for s in cfg['shift_bins'])
            lo, hi = min(center, a1)-radius, max(center, a1)+radius
            role = int(roles[lo]) if 0<=lo<hi<=n and (roles[lo:hi]==roles[lo]).all() else -1
            entities.append(dict(entity_id=row['window_id'] if kind=='candidate' else row['structure_id'], kind=kind,
                                 type='' if kind=='candidate' else row['type'], original_center_bin=center,
                                 boundary_center_bin=a1, rep2_boundary_bin=a2, rep1_boundary_score=score1, rep2_boundary_score=score2,
                                 boundary_reproducible=bool(ok1 and ok2 and score1<0 and score2<0 and abs(a1-a2)<=cfg['boundary_max_shift_bins']),
                                 envelope_start_bin=lo, envelope_end_bin=hi, role=role))
    candidate_bins = np.zeros(n, bool)
    for entity in entities:
        if entity['kind']=='candidate':
            candidate_bins[np.arange(entity['envelope_start_bin'],entity['envelope_end_bin'])%n] = True
    # Known windows must exclude the entire perturbation footprint, not only the source window.
    for entity in entities:
        lo, hi = entity['envelope_start_bin'], entity['envelope_end_bin']
        entity['reference_geometry_eligible'] = bool(entity['kind']=='known' and entity['role'] in [0,1]
                                                     and not candidate_bins[np.arange(lo,hi)%n].any())
    with np.load(ROOT/'results/task2_v2/shape_representation.npz') as z:
        pca_mean, components = z['pca_mean'], z['components']
    retained = [e for e in entities if e['kind']=='candidate' or e['reference_geometry_eligible']]
    features = np.zeros((len(retained), len(settings), 2, 256), np.float32)
    quality = np.ones((len(retained),len(settings),2), bool)
    masks = {v['variant_id']:occlusion_mask(v['width_bins'],v['fraction'],v['seed']) for v in settings}
    for i, entity in enumerate(retained):
        # Cache native matrices for three mask realizations of the same crop.
        cache = {}
        for vi, setting in enumerate(settings):
            center = entity['original_center_bin'] if setting['anchor']=='original' else entity['boundary_center_bin']
            width = setting['width_bins']
            start = center+setting['shift_bins']-width//2
            for rep, band in enumerate(bands):
                key = (start,width,rep)
                if key not in cache:
                    cache[key] = extract(band,start,width)
                raw, oe, mask = cache[key]
                upper = np.triu(mask,2)
                quality[i,vi,rep] = mask.sum()/(width*width-3*width+2)>=cfg['min_valid_fraction'] and upper.any() and (raw[upper]==0).mean()<=cfg['max_zero_fraction']
                features[i,vi,rep] = shape_feature(oe,mask & masks[setting['variant_id']])
        entity['all_variants_quality_pass'] = bool(quality[i].all())
        if (i+1)%50==0:
            print('Processed',i+1,'entities',flush=True)
    # Fixed reference/calibration cohort across ALL variants prevents composition confounding.
    training = [i for i,e in enumerate(retained) if e['kind']=='known' and e['role']==0 and e['boundary_reproducible'] and e['all_variants_quality_pass']]
    pool = [i for i,e in enumerate(retained) if e['kind']=='known' and e['role']==1 and e['boundary_reproducible'] and e['all_variants_quality_pass']]
    calibration, occupied = [], np.zeros(n,bool)
    for i in sorted(pool,key=lambda i:(retained[i]['envelope_start_bin'],retained[i]['entity_id'])):
        lo,hi = retained[i]['envelope_start_bin'],retained[i]['envelope_end_bin']
        if not occupied[lo:hi].any():
            occupied[lo:hi]=True
            calibration.append(i)
    if not training or not calibration:
        raise ValueError('No usable matched reference or calibration cohort')
    candidate_indices = [i for i,e in enumerate(retained) if e['kind']=='candidate']
    for i,e in enumerate(retained):
        e['reference_member']=i in training
        e['calibration_member']=i in calibration
    write_csv(out/'entities.csv',retained)
    write_csv(out/'all_entity_geometry.csv',entities)
    pca = (features-pca_mean)@components.T
    np.savez_compressed(out/'features.npz',shape=features,pca=pca,quality=quality,training=training,calibration=calibration,candidates=candidate_indices)
    print('Frozen cohort:',len(training),'reference,',len(calibration),'calibration,',len(candidate_indices),'candidates;',len(settings),'variants',flush=True)
    comparisons, cal_rows, threshold_rows = [], [], []
    metric_results = {}
    for metric, x, distance in [('pca',pca,'euclidean'),('shape',features,'cosine')]:
        for vi,v in enumerate(settings):
            distances = cdist(x[calibration,vi,0],x[training,vi,0],metric=distance)
            matches = np.array(training)[distances.argmin(1)]
            cal_values=[]
            for i,j in zip(calibration,matches):
                ds=[float(cdist(x[i:i+1,vi,rep],x[j:j+1,vi,rep],metric=distance)[0,0]) for rep in [0,1]]
                cal_values.append(ds)
                cal_rows.append(dict(variant_id=v['variant_id'],metric=metric,entity_index=i,reference_index=int(j),rep1_distance=ds[0],rep2_distance=ds[1]))
            cutoff = np.quantile(cal_values,cfg['distance_quantile'],axis=0) if len(calibration)>=cfg['min_calibration_windows'] else [np.nan,np.nan]
            threshold_rows.append(dict(variant_id=v['variant_id'],metric=metric,calibration_windows=len(calibration),rep1_cutoff=float(cutoff[0]),rep2_cutoff=float(cutoff[1])))
            d = cdist(x[candidate_indices,vi,0],x[training,vi,0],metric=distance)
            nearest = np.array(training)[d.argmin(1)]
            for i,j in zip(candidate_indices,nearest):
                ds=[float(cdist(x[i:i+1,vi,rep],x[j:j+1,vi,rep],metric=distance)[0,0]) for rep in [0,1]]
                outside=bool(quality[i,vi].all() and np.isfinite(cutoff).all() and np.all(np.array(ds)>cutoff))
                metric_results[(i,vi,metric)]=outside
                comparisons.append(dict(entity_id=retained[i]['entity_id'],entity_index=i,variant_id=v['variant_id'],anchor=v['anchor'],width_bins=v['width_bins'],shift_bins=v['shift_bins'],occlusion=v['occlusion'],metric=metric,
                                        nearest_reference=retained[j]['entity_id'],reference_index=int(j),rep1_distance=ds[0],rep2_distance=ds[1],
                                        rep1_cutoff=float(cutoff[0]),rep2_cutoff=float(cutoff[1]),quality_pass=bool(quality[i,vi].all()),outside_both=outside))
    write_csv(out/'candidate_variants.csv',comparisons)
    write_csv(out/'calibration_distances.csv',cal_rows)
    write_csv(out/'thresholds.csv',threshold_rows)
    # Paired sensitivity relative to the same entity's original, unshifted, unmasked 6.4 kb crop.
    baseline = next(i for i,v in enumerate(settings) if v['anchor']=='original' and v['width_bins']==64 and v['shift_bins']==0 and v['occlusion']=='none')
    sensitivity=[]
    for metric,x,distance in [('pca',pca,'euclidean'),('shape',features,'cosine')]:
        for vi,v in enumerate(settings):
            cal_change=np.array([[cdist(x[i:i+1,baseline,rep],x[i:i+1,vi,rep],metric=distance)[0,0] for rep in [0,1]] for i in calibration])
            limits=np.quantile(cal_change,cfg['distance_quantile'],axis=0) if len(calibration)>=cfg['min_calibration_windows'] else [np.nan,np.nan]
            for i in candidate_indices:
                ds=[float(cdist(x[i:i+1,baseline,rep],x[i:i+1,vi,rep],metric=distance)[0,0]) for rep in [0,1]]
                sensitivity.append(dict(entity_id=retained[i]['entity_id'],variant_id=v['variant_id'],metric=metric,rep1_change=ds[0],rep2_change=ds[1],
                                        known_rep1_change_q95=float(limits[0]),known_rep2_change_q95=float(limits[1])))
    write_csv(out/'paired_sensitivity.csv',sensitivity)
    summaries=[]
    for i in candidate_indices:
        row=dict(entity_id=retained[i]['entity_id'],boundary_reproducible=retained[i]['boundary_reproducible'],
                 boundary_shift_bp=abs(retained[i]['boundary_center_bin']-retained[i]['rep2_boundary_bin'])*100,
                 all_variants_quality_pass=retained[i]['all_variants_quality_pass'],calibration_sufficient=len(calibration)>=cfg['min_calibration_windows'])
        for anchor in cfg['anchors']:
            ids=[vi for vi,v in enumerate(settings) if v['anchor']==anchor]
            flags=[metric_results[i,vi,'pca'] and metric_results[i,vi,'shape'] for vi in ids]
            row[anchor+'_outside_variants']=sum(flags)
            row[anchor+'_outside_fraction']=float(np.mean(flags))
        # Robust candidate must persist at every scale, across >=80% of perturbations per scale.
        fractions=[]
        for width in cfg['width_bins']:
            ids=[vi for vi,v in enumerate(settings) if v['anchor']=='boundary' and v['width_bins']==width]
            fraction=float(np.mean([metric_results[i,vi,'pca'] and metric_results[i,vi,'shape'] for vi in ids]))
            row[f'boundary_w{width}_outside_fraction']=fraction
            fractions.append(fraction)
        row['robust_recentered_anomaly']=bool(row['calibration_sufficient'] and row['all_variants_quality_pass'] and row['boundary_reproducible'] and min(fractions)>=cfg['robust_fraction'])
        summaries.append(row)
    write_csv(out/'candidate_summary.csv',summaries)
    report=dict(config=cfg,candidates=len(candidate_indices),variants_per_candidate=len(settings),reference_windows=len(training),calibration_windows=len(calibration),
                reproducible_candidate_boundaries=sum(r['boundary_reproducible'] for r in summaries),
                candidates_with_any_two_metric_anomaly=sum(r['original_outside_variants']+r['boundary_outside_variants']>0 for r in summaries),
                candidates_with_recentered_two_metric_anomaly=sum(r['boundary_outside_variants']>0 for r in summaries),
                robust_recentered_anomalies=sum(r['robust_recentered_anomaly'] for r in summaries),
                candidate_metric_comparisons=len(comparisons),calibration_metric_comparisons=len(cal_rows),interpretation=cfg['interpretation'])
    (out/'report.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(16,9),layout='constrained',sharey=True)
    for ax,anchor in zip(axes,cfg['anchors']):
        ids=[vi for vi,v in enumerate(settings) if v['anchor']==anchor]
        matrix=np.array([[int(metric_results[i,vi,'pca'])+int(metric_results[i,vi,'shape']) for vi in ids] for i in candidate_indices])
        im=ax.imshow(matrix,vmin=0,vmax=2,cmap='viridis',aspect='auto',interpolation='nearest')
        ax.set(title=anchor,yticks=range(len(candidate_indices)),yticklabels=[retained[i]['entity_id'] for i in candidate_indices],xticks=range(len(ids)),
               xticklabels=[f"{settings[vi]['width_bins']*.1:g}k/{settings[vi]['shift_bins']*100:+d}/{settings[vi]['occlusion']}" for vi in ids])
        ax.tick_params(axis='x',labelrotation=90,labelsize=7)
    fig.colorbar(im,ax=axes,ticks=[0,1,2],label='Metrics outside calibrated range in BOTH repeats (0/1/2)',shrink=.5)
    fig.suptitle('Matched recentering / scale / shift / occlusion; full grid, no best-variant selection')
    fig.savefig(out/'all_variants.png',dpi=130)
    plt.close(fig)
    (out/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>重新居中与扰动敏感性</title><style>body{{max-width:1500px;margin:30px auto;font:16px/1.6 sans-serif;padding:20px}}img{{width:100%}}</style>
<h1>候选与已知参考的同规则重新居中和扰动实验</h1><p>21 个候选，每个 54 个变体；参考 {len(training)} 个、互不重叠校准 {len(calibration)} 个。稳定重新居中异常：{report['robust_recentered_anomalies']}。探索性形态异常不等于新类型或新簇。</p>
<p><a href="candidate_summary.csv">候选汇总</a> · <a href="candidate_variants.csv">全部变体距离与判定</a> · <a href="paired_sensitivity.csv">相对自身基线的扰动敏感性及已知范围</a> · <a href="thresholds.csv">每变体校准阈值</a> · <a href="entities.csv">中心位置与固定参考队列</a></p>
<p>两组中心：原中心、rep1 在固定 6.4 kb 搜索区的局部隔离最低点；rep2 使用同一 rep1 中心。尺度 6.4/12.8/25.6 kb，平移 −800/0/+800 bp，无遮挡及两次固定种子的 10% 对称随机像素遮挡。所有实体和重复共享对应遮挡掩码。每个变体独立用相同已知校准队列的 95% 分位数作参照。参考排除候选全部扰动足迹。</p>
<p>热图颜色为两个重复均超范围的指标数。质量不合格或校准不足会计为未通过；请结合 CSV 标记，不能把未通过一律解释为已知形态。稳定异常还要求边界重复偏差 ≤500 bp、全变体质量合格，且三个尺度各至少 80% 的重新居中变体同时通过两指标。</p><img src="all_variants.png" alt="全部变体结果">''')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
