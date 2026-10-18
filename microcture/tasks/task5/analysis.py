"""Paired and distance-specific diagnostics of frozen task-5 reconstructions.

All scores average per-seed metrics, never metrics of averaged predictions.
Original global boundary errors remain distinct from the exploratory band proxy.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[3]
SEEDS = (42, 43, 44)
FAMILIES = ('bilinear', 'bicubic', 'plain_cnn', 'residual_cnn')
MODELS = FAMILIES[:2] + tuple(f'{f}_{s}' for f in FAMILIES[2:] for s in SEEDS)
METRICS = ('psnr', 'ssim', 'boundary_error_bp')
COLORS = dict(zip(FAMILIES, ('#8c9aa8', '#2563a6', '#dd8d32', '#b64060')))
LABELS = dict(zip(FAMILIES, ('Bilinear', 'Bicubic', 'Plain CNN', 'Residual CNN')))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def write_json(path, value):
    with Path(path).open('x') as f: json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def family(name):
    return name.rsplit('_', 1)[0] if name.rsplit('_', 1)[-1].isdigit() else name


def finite_mean(values):
    values = list(values)
    if all(v is None for v in values): return None
    if any(v is None for v in values): raise ValueError('Inconsistent evaluability across paired measurements')
    return float(np.mean(values))


def ssim_values(prediction, target, mask, window=7):
    """The frozen SSIM arithmetic, including full-neighborhood validity."""
    ux, uy = uniform_filter(prediction, window), uniform_filter(target, window)
    vx = np.maximum(uniform_filter(prediction**2, window) - ux**2, 0)
    vy = np.maximum(uniform_filter(target**2, window) - uy**2, 0)
    cov = uniform_filter(prediction * target, window) - ux * uy
    values = ((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux**2+uy**2+.01**2)*(vx+vy+.03**2))
    interior = uniform_filter(mask.astype(float), window, mode='constant') > 1-1e-8
    return values, interior


def pixel_metrics(prediction, target, keep, ssim, interior):
    use = keep & interior
    mse = float(np.mean((prediction[keep]-target[keep])**2)) if keep.any() else None
    return {'mse': mse, 'psnr': -10*math.log10(max(mse, 1e-12)) if mse is not None else None,
            'ssim': float(ssim[use].mean()) if use.any() else None,
            'evaluated_pixels': int(keep.sum()), 'ssim_pixels': int(use.sum())}


def contact_profile(matrix, mask, flank=8, resolution=200, band=None):
    """Distance bands restrict pairs inside the unchanged cross-cut geometry."""
    result = np.full(len(matrix), np.nan)
    sep = (np.arange(flank)[None, :] + flank - np.arange(flank)[:, None])*resolution
    possible = np.ones((flank, flank), bool) if band is None else (sep >= band[0]) & (sep < band[1])
    opportunities = int(possible.sum())
    if not opportunities: return result
    minimum = (flank*flank)//2 if band is None else math.ceil(opportunities/2)
    for cut in range(flank, len(matrix)-flank):
        keep = mask[cut-flank:cut, cut:cut+flank] & possible
        if keep.sum() >= minimum:
            result[cut] = matrix[cut-flank:cut, cut:cut+flank][keep].mean()
    return result


def boundary_error(predicted, actual, meta, resolution=200, radius=5):
    shifts = []
    for endpoint in ('start', 'end'):
        center = round((int(meta[endpoint])-int(meta['start_bin'])*100)/resolution)
        ids = np.arange(max(0, center-radius), min(len(predicted), center+radius+1))
        ids = ids[np.isfinite(predicted[ids]) & np.isfinite(actual[ids])]
        if len(ids):
            shifts.append(abs(int(ids[np.argmin(predicted[ids])])-int(ids[np.argmin(actual[ids])]))*resolution)
    return (float(np.mean(shifts)) if shifts else None), len(shifts)


def corr(a, b):
    keep = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[keep], b[keep])[0, 1]) if keep.sum() >= 3 and np.std(a[keep]) and np.std(b[keep]) else None


def paired_rows(records, metadata):
    """Strict paired identity and seed coverage; average seed scores first."""
    lookup = {}
    for r in records:
        key = (r['model'], r['structure_id'], int(r['replicate']))
        if key in lookup: raise ValueError('Duplicate model-window metric')
        lookup[key] = r
    expected = {(m, r['structure_id'], int(r['replicate'])) for m in MODELS for r in metadata}
    if set(lookup) != expected: raise ValueError('Missing or extra paired model/seed/window')
    output, seeds = [], []
    for f in FAMILIES[2:]:
        for meta in metadata:
            sid, rep = meta['structure_id'], int(meta['replicate']);base = lookup['bicubic', sid, rep]
            values = {k: [] for k in METRICS}
            for seed in SEEDS:
                v = lookup[f'{f}_{seed}', sid, rep]
                record = dict(family=f, seed=seed, **meta)
                for k in METRICS:
                    record[f'delta_{k}'] = float(v[k])-float(base[k]); values[k].append(float(v[k]))
                seeds.append(record)
            record = dict(family=f, **meta)
            for k in METRICS:
                record[f'bicubic_{k}'] = float(base[k]);record[f'cnn_{k}'] = float(np.mean(values[k]));record[f'delta_{k}'] = record[f'cnn_{k}']-record[f'bicubic_{k}']
            output.append(record)
    return output, seeds


def collapse_replicates(rows, keys, numeric):
    groups = {}
    for r in rows: groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    output = []
    for key, vs in groups.items():
        if len(vs) != 2 or {int(r['replicate']) for r in vs} != {1, 2}: raise ValueError('Expected both WT repeats')
        record = {k: v for k, v in vs[0].items() if k != 'replicate' and k not in numeric}
        for k in numeric: record[k] = finite_mean(v[k] for v in vs)
        output.append(record)
    return output


def score_stats(values, tolerance=1e-10):
    x = np.asarray([v for v in values if v is not None], float)
    return dict(n_evaluable=len(x), mean=float(x.mean()) if len(x) else None,
                median=float(np.median(x)) if len(x) else None,
                minimum=float(x.min()) if len(x) else None, maximum=float(x.max()) if len(x) else None,
                positive=int((x > tolerance).sum()), negative=int((x < -tolerance).sum()), tied=int((np.abs(x) <= tolerance).sum()))


def pair_summary(rows, positions, tolerance):
    summary = []
    for f in FAMILIES[2:]:
        for scope, source in [('rep1', [r for r in rows if r['replicate'] == 1]), ('rep2', [r for r in rows if r['replicate'] == 2]), ('positions', positions)]:
            for stratifier in ['all', 'type', 'group_id']:
                for val in ['all'] if stratifier == 'all' else sorted({r[stratifier] for r in source}):
                    selected = [r for r in source if r['family'] == f and (stratifier == 'all' or r[stratifier] == val)]
                    for k in METRICS:
                        summary.append(dict(family=f, scope=scope, stratifier=stratifier, stratum=val, metric=k, **score_stats([r[f'delta_{k}'] for r in selected], tolerance)))
    return summary


def select_example(metadata, cfg):
    candidates = [(i, r) for i, r in enumerate(metadata) if r['replicate'] == cfg['replicate'] and r['type'] == cfg['type']]
    candidates.sort(key=lambda v: (v[1]['start'], v[1]['structure_id']))
    return candidates[(len(candidates)-1)//2]


def recover_input(root, directory, manifest, metadata, reference, predictions, selection):
    """Recover one actual low-res input from frozen counts and saved QC masks.

    Pooling clipped normalized target values would NOT recover the original input.
    Both stored interpolations must match before calling this the actual input.
    """
    from microcture.layout import resolve
    import torch
    from torch.nn.functional import interpolate
    i, meta = selection
    logical = f'data/cache/discovery/WT_37C_rep{meta["replicate"]}.npz'
    path = resolve(root/logical, root=root)
    if digest(path) != manifest['input_sha256'][logical]: raise ValueError('Frozen contact-band hash differs')
    with np.load(path, allow_pickle=False) as z: band, total = z['band'], float(z['total'])
    ids = (meta['start_bin']+np.arange(256)) % len(band)
    a, b = ids[:, None], ids[None, :];delta = np.abs(a-b);sep = np.minimum(delta, len(band)-delta)
    anchor = np.where(delta <= len(band)//2, np.minimum(a,b), np.maximum(a,b));raw = band[anchor,sep]
    target = raw.reshape(128,2,128,2).sum((1,3))/4
    target *= 1e6/total
    mask = reference['mask'][i];scale = manifest['normalization']['log_scale']
    normalized = np.clip(np.log1p(target)/scale,0,1)
    np.testing.assert_allclose(normalized[mask], reference['target'][i][mask], atol=1e-7,rtol=1e-6)
    count = mask.reshape(32,4,32,4).sum((1,3));summed = np.where(mask,target,0).reshape(32,4,32,4).sum((1,3))
    low = np.divide(summed,count,out=np.zeros_like(summed),where=count>0)
    low = np.clip(np.log1p(low)/scale,0,1).astype('float32')
    errors = {}
    for mode in ('bilinear','bicubic'):
        rec = interpolate(torch.from_numpy(low[None,None]),scale_factor=4,mode=mode,align_corners=False).clamp(0,1).numpy()[0,0]
        error = float(np.max(np.abs(rec-predictions[mode][i])));errors[mode] = error
        if error > 1e-6: raise ValueError('Recovered input does not reproduce stored interpolation')
    return low, count > 0, {'selection':meta,'source_cache':str(path.relative_to(root)), 'source_sha256':digest(path),'interpolation_max_errors':errors,'source':'frozen 100 bp contact counts plus saved valid target mask; no raw scan'}


def summarize_methods(rows, distances=False):
    output = []
    for f in FAMILIES:
        for scope in ('rep1','rep2','positions'):
            subset = [r for r in rows if r['family'] == f]
            if scope == 'positions':
                keys = ['structure_id'] + (['band'] if distances else [])
                numeric = list(METRICS)
                subset = collapse_replicates(subset,keys,numeric)
            else: subset = [r for r in subset if r['replicate'] == int(scope[-1])]
            bands = sorted({r['band'] for r in subset}) if distances else ['all']
            for band in bands:
                selected = [r for r in subset if not distances or r['band'] == band]
                record = dict(family=f, scope=scope, band=band, windows=len(selected))
                for k in METRICS:
                    stat = score_stats([r[k] for r in selected]);record[k] = stat['mean'];record[f'{k}_n'] = stat['n_evaluable']
                output.append(record)
    return output


def compute(root, config):
    directory = root/config['input_dir'];manifest = json.loads((directory/'manifest.json').read_text());cfg = manifest['config']
    assert tuple(cfg['seeds']) == SEEDS and cfg['ssim_window'] == 7 and cfg['profile_flank_bins'] == 8 and cfg['boundary_radius_bins'] == 5
    windows = read_csv(directory/'windows.csv');test_indices = [i for i,r in enumerate(windows) if r['split']=='test']
    metadata = [{k:(int(windows[i][k]) if k in ('replicate','start_bin','start','end') else windows[i][k]) for k in ('structure_id','type','group_id','replicate','start_bin','start','end')} for i in test_indices]
    assert len(metadata)==106 and len({r['structure_id'] for r in metadata})==53
    saved = read_csv(directory/'metrics.csv');lookup = {(r['model'],r['structure_id'],int(r['replicate'])):r for r in saved}
    assert len(lookup)==len(saved)==848
    predictions = {};reference = None;records=[];distance_rows=[];errors={};sources=[]
    # Bind every saved numeric input to the previously published delivery signature.
    from microcture.layout import resolve
    prior=json.loads((root/config['frozen_delivery']).read_text())
    registered={str(resolve(root/r['path'],root=root)):r['sha256'] for r in prior['source_files']}
    for name in ('manifest.json','windows.csv','metrics.csv','summary.csv',*(m+'_reconstruction.npz' for m in MODELS)):
        p=directory/name;h=digest(p)
        if registered.get(str(p))!=h:raise ValueError(f'Prior delivery signature missing or changed: {p}')
        sources.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=h))
    for name in MODELS:
        with np.load(directory/(name+'_reconstruction.npz'),allow_pickle=False) as z:
            pred,tar,mask,indices=z['prediction'],z['target'],z['mask'],z['window_indices']
            assert indices.tolist()==test_indices and int(z['resolution_bp'])==200 and pred.shape==(106,128,128) and mask.dtype==bool
            assert np.all(np.isfinite(pred)) and pred.min()>=0 and pred.max()<=1
            np.testing.assert_allclose(pred,pred.transpose(0,2,1),atol=1e-6)
            if reference is None:reference={'target':tar,'mask':mask}
            else:
                np.testing.assert_array_equal(tar,reference['target']);np.testing.assert_array_equal(mask,reference['mask'])
            predictions[name]=pred
        sep=np.abs(np.arange(128)[:,None]-np.arange(128)[None,:])*200;upper=np.triu(np.ones((128,128),bool),2)
        for i,meta in enumerate(metadata):
            x,y,m=pred[i],tar[i],mask[i];ss,interior=ssim_values(x,y,m);keep=m&upper
            pix=pixel_metrics(x,y,keep,ss,interior);pp=contact_profile(x,m);tp=contact_profile(y,m);be,nb=boundary_error(pp,tp,meta)
            row=dict(model=name,**meta,psnr=pix['psnr'],ssim=pix['ssim'],boundary_error_bp=be)
            original=lookup[name,meta['structure_id'],meta['replicate']]
            checked={**pix,'boundary_shift_bp':be,'evaluated_boundaries':nb,'profile_correlation':corr(pp,tp),'shape_correlation':corr(x[keep],y[keep])}
            for k,v in checked.items():
                if v is None:assert original[k] in ('',None)
                else:
                    err=abs(float(original[k])-v);errors[k]=max(errors.get(k,0),err)
                    if err>1e-6:raise ValueError(f'Frozen metric mismatch: {name} {meta["structure_id"]} {k} {err}')
            records.append(row)
            for lo,hi in config['distance_bands_bp']:
                key=f'{lo}_{hi}';bandkeep=keep&(sep>=lo)&(sep<hi);v=pixel_metrics(x,y,bandkeep,ss,interior)
                bp=contact_profile(x,m,band=(lo,hi));bt=contact_profile(y,m,band=(lo,hi));db,dn=boundary_error(bp,bt,meta)
                reason='evaluable' if db is not None else ('not_evaluable_no_pairs_in_profile_geometry' if lo>=(2*8-1)*200 else 'not_evaluable_quality')
                distance_rows.append(dict(model=name,**meta,band=key,distance_min_bp=lo,distance_max_bp=hi,psnr=v['psnr'],ssim=v['ssim'],boundary_error_bp=db,evaluated_pixels=v['evaluated_pixels'],ssim_pixels=v['ssim_pixels'],evaluated_boundaries=dn,boundary_status=reason))
    pairs,seed_pairs=paired_rows(records,metadata)
    numeric=[f'{prefix}_{k}' for k in METRICS for prefix in ('bicubic','cnn','delta')]
    positions=collapse_replicates(pairs,['family','structure_id'],numeric)
    # Family means use the SAME metrics per seed and per window (no image ensemble).
    def reduce_seeds(source,dist):
        out=[]
        for f in FAMILIES:
            for meta in metadata:
                for band in [f'{lo}_{hi}' for lo,hi in config['distance_bands_bp']] if dist else ['all']:
                    selected=[r for r in source if family(r['model'])==f and r['structure_id']==meta['structure_id'] and r['replicate']==meta['replicate'] and (not dist or r['band']==band)]
                    assert len(selected)==(3 if f.endswith('cnn') else 1)
                    row=dict(family=f,**meta,band=band)
                    for k in METRICS:row[k]=finite_mean(r[k] for r in selected)
                    out.append(row)
        return out
    families=reduce_seeds(records,False);distance_families=reduce_seeds(distance_rows,True)
    example=select_example(metadata,config['example']);low,low_valid,recovery=recover_input(root,directory,manifest,metadata,reference,predictions,example)
    sources.append({'path':recovery['source_cache'],'bytes':(root/recovery['source_cache']).stat().st_size,'sha256':recovery['source_sha256']})
    # Retain numerical agreement with every existing distance-PSNR diagnostic.
    old=read_csv(directory/'distance_diagnostics.csv');legacy_error=0.0
    for r in old:
        if r['model'] not in MODELS:continue
        subrows=[v for v in distance_rows if v['model']==r['model'] and v['replicate']==int(r['replicate']) and v['distance_min_bp']==int(r['distance_min_bp'])]
        err=abs(float(np.mean([v['psnr'] for v in subrows]))-float(r['psnr']));legacy_error=max(legacy_error,err);assert err<1e-10
    seed_signs=[]
    for f in FAMILIES[2:]:
        for rep in (1,2):
            for k in METRICS:
                groups=[[r[f'delta_{k}'] for r in seed_pairs if r['family']==f and r['replicate']==rep and r['structure_id']==m['structure_id']] for m in metadata if m['replicate']==rep]
                seed_signs.append(dict(family=f,replicate=rep,metric=k,positions=len(groups),all_seeds_positive=sum(min(g)>1e-10 for g in groups),all_seeds_negative=sum(max(g)<-1e-10 for g in groups)))
    for name in ('distance_diagnostics.csv','diagnostic_protocol.json'):
        p=directory/name;sources.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p)))
    summary={'schema_version':1,'status':'saved_prediction_analysis','positions':53,'replicates':2,'spatial_groups':sorted({r['group_id'] for r in metadata},key=int),'families':list(FAMILIES),'seeds':list(SEEDS),'paired_unit':'mean of per-seed metrics within each window and WT repeat; then optional mean of the two repeats at each position','boundary_unit':'global original proxy and exploratory distance-restricted proxy are distinct; neither is expert-validated biological boundary error','distance_bands_bp':config['distance_bands_bp'],'ssim_distance_rule':config['ssim_distance_rule'],'band_boundary_rule':config['band_boundary_rule'],'frozen_metric_max_errors':errors,'historical_distance_psnr_max_error':legacy_error,'seed_sign_consistency':seed_signs,'pair_summary':pair_summary(pairs,positions,config['tie_tolerance']),'method_summary':summarize_methods(families),'distance_summary':summarize_methods(distance_families,True),'example':recovery,'training_performed':False,'raw_contact_scan_performed':False,'weights_restored':False,'limitations':['Exploratory reanalysis of repeatedly viewed WT data, not a new external validation.','53 positions in 7 spatial groups; repeated measurements/seeds are not independent biological samples.','Far-distance boundary proxy is NA because the fixed 8x8 profile has no such pairs.','Near-band SSIM is NA because no original full valid 7x7 neighborhood lies at these center distances.','SSIM bands assign map centers; the unchanged 7x7 neighborhood may cross band boundaries.','Synthetic degradation and noisy measured targets; no claim about true low-depth Hi-C to Micro-C recovery.']}
    return summary,dict(paired_windows=pairs,paired_positions=positions,paired_seeds=seed_pairs,distance_windows=distance_rows,distance_families=distance_families,method_windows=families),dict(predictions=predictions,reference=reference,metadata=metadata,example_index=example[0],low=low,low_valid=low_valid),sources


def plot_all(directory,summary,tables,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.savefig(directory/name,dpi=140,facecolor='white',metadata={'Software':'MicroCTure saved-prediction diagnostics'});plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,12),layout='constrained')
    for col,k in enumerate(('psnr','ssim')):
        for f in FAMILIES[2:]:
            rs=[r for r in tables['paired_windows'] if r['family']==f];ps=[r for r in tables['paired_positions'] if r['family']==f]
            axes[0,col].hist([r[f'delta_{k}'] for r in rs],bins=18,histtype='step',lw=2,color=COLORS[f],label=LABELS[f])
            for rep,marker in [(1,'o'),(2,'x')]:
                subr=[r for r in rs if r['replicate']==rep];axes[1,col].scatter([r[f'bicubic_{k}'] for r in subr],[r[f'cnn_{k}'] for r in subr],s=18,marker=marker,alpha=.65,c=COLORS[f])
            axes[2,col].scatter([(r['start']+r['end'])/2/1e6 for r in ps],[r[f'delta_{k}'] for r in ps],s=24,alpha=.75,c=COLORS[f],label=LABELS[f])
        axes[0,col].axvline(0,c='#777',ls='--');axes[0,col].set(xlabel=f'CNN minus Bicubic: {k.upper()}',ylabel='Window-repeat pairs',title=f'Paired Δ{k.upper()} | 106 pairs per family')
        lo,hi=axes[1,col].get_xlim();a,b=axes[1,col].get_ylim();lo=min(lo,a);hi=max(hi,b);axes[1,col].plot([lo,hi],[lo,hi],'--',c='#777');axes[1,col].set(xlabel=f'Bicubic {k.upper()}',ylabel=f'CNN {k.upper()}',title='Same-window comparison; circles=rep1, crosses=rep2')
        axes[2,col].axhline(0,c='#777',ls='--');axes[2,col].set(xlabel='Author interval center (Mb)',ylabel=f'Paired Δ{k.upper()}',title='All 53 positions; means over seeds then WT repeats')
        axes[0,col].legend(frameon=False)
    fig.suptitle('Task 5 | Does improvement extend across individual test windows?',fontsize=15,fontweight='bold')
    save(fig,'paired_improvement.png')
    fig,axes=plt.subplots(1,3,figsize=(14,5.0),layout='constrained');pooled=[r for r in summary['distance_summary'] if r['scope']=='positions'];bands=[f'{a}_{b}' for a,b in summary['distance_bands_bp']]
    for ax,k,title in zip(axes,METRICS,('PSNR (dB) ↑','SSIM at valid map centers ↑','Exploratory band boundary proxy (bp) ↓')):
        for j,f in enumerate(FAMILIES):
            vals=[next(r[k] for r in pooled if r['family']==f and r['band']==band) for band in bands]
            for i,v in enumerate(vals):
                if v is not None:ax.bar(i+(j-1.5)*.19,v,.18,color=COLORS[f],label=LABELS[f] if i==0 else None)
        ax.set(xticks=range(3),xticklabels=['0.4–1.6 kb','1.6–6.4 kb','6.4–25.6 kb'],title=title,xlim=(-.6,2.6));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        if k=='boundary_error_bp':ax.text(2,.12,'NA\nNo eligible pairs in\nfixed profile geometry',ha='center',transform=ax.get_xaxis_transform(),fontsize=9,color='#5d6470')
        if k=='ssim':ax.text(0,.12,'NA\nNo full valid 7×7\nneighborhood centers',ha='center',transform=ax.get_xaxis_transform(),fontsize=9,color='#5d6470')
    axes[0].legend(frameon=False,fontsize=8);fig.suptitle('Task 5 | Performance depends on contact distance',fontsize=15,fontweight='bold')
    fig.supxlabel('Fixed historical distance bands; global boundary metric remains a separate endpoint.\nSSIM uses the unchanged 7×7 map; band boundary proxy uses the fixed 8×8 profile. Far NA is not zero.',fontsize=9)
    save(fig,'distance_stratification.png')
    fig=plt.figure(figsize=(15,9));outer=fig.add_gridspec(1,3,width_ratios=[1.55,1,1],left=.05,right=.98,bottom=.13,top=.79,wspace=.3)
    ga=outer[0].subgridspec(3,2,hspace=.25,wspace=.1);i=data['example_index'];mask=data['reference']['mask'][i];target=data['reference']['target'][i];vmax=max(float(np.quantile(target[mask],.99)),1e-6)
    arrays=[target,data['low'],data['predictions']['bicubic'][i],data['predictions']['plain_cnn_42'][i],data['predictions']['residual_cnn_42'][i]]
    for j,(matrix,label) in enumerate(zip(arrays,['Measured reference\n200 bp','Retained input\n800 bp','Bicubic','Plain CNN | seed 42','Residual CNN | seed 42'])):
        ax=fig.add_subplot(ga[j//2,j%2]);ax.imshow(np.where(mask if matrix.shape==mask.shape else data['low_valid'],matrix,np.nan),origin='lower',vmin=0,vmax=vmax,cmap='magma',interpolation='nearest');ax.set_xticks([]);ax.set_yticks([]);ax.set_title(label,fontsize=9)
    ax=fig.add_subplot(ga[2,1]);ax.axis('off');ax.text(.5,.5,f'{data["metadata"][i]["structure_id"]}\nWT rep1 · 25.6 kb\nFixed median-start CHIN\nShared reference-derived scale',ha='center',va='center',fontsize=9)
    gb=outer[1].subgridspec(2,1,hspace=.4)
    method_positions=collapse_replicates(tables['method_windows'],['family','structure_id'],list(METRICS))
    for row,k in enumerate(('psnr','ssim')):
        ax=fig.add_subplot(gb[row]);values=[[r[k] for r in method_positions if r['family']==f] for f in FAMILIES]
        bp=ax.boxplot(values,patch_artist=True,tick_labels=[LABELS[f] for f in FAMILIES],showfliers=False)
        for patch,f in zip(bp['boxes'],FAMILIES):patch.set_facecolor(COLORS[f]);patch.set_alpha(.65)
        ax.tick_params(axis='x',rotation=24,labelsize=8);ax.set_ylabel(k.upper());ax.grid(axis='y',alpha=.2)
    ax=fig.add_subplot(outer[2]);values=[[r['boundary_error_bp'] for r in method_positions if r['family']==f] for f in FAMILIES]
    bp=ax.boxplot(values,patch_artist=True,tick_labels=[LABELS[f] for f in FAMILIES],showfliers=False)
    for j,(patch,f,vs) in enumerate(zip(bp['boxes'],FAMILIES,values)):
        patch.set_facecolor(COLORS[f]);patch.set_alpha(.65);ax.scatter(j+1,np.mean(vs),marker='D',c='#222',s=20);ax.text(j+1,np.mean(vs)+22,f'mean\n{np.mean(vs):.1f}',ha='center',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.75,pad=1))
    ax.tick_params(axis='x',rotation=24,labelsize=8);ax.set_ylabel('Original boundary proxy error (bp) ↓');ax.grid(axis='y',alpha=.2)
    for x,label in [(.17,'A | Same-window reconstruction'),(.57,'B | Pixel-level fidelity'),(.84,'C | Structure-boundary proxy')]:fig.text(x,.87,label,ha='center',fontsize=11,fontweight='bold')
    fig.suptitle('Pixel-level reconstruction improves, whereas\nstructure-boundary recovery remains unresolved.',fontsize=18,fontweight='bold',y=.98)
    fig.text(.5,.04,'B/C: 53 paired positions, average per-seed metrics first, then both WT repeats. No independent-sample significance test.\nBoundary reference = minima near author endpoints, not expert-certified physical boundaries. A: fixed rep1/seed42 example, not chosen by score.',ha='center',fontsize=9,color='#4d5969')
    save(fig,'final_summary.png')


def verify(directory,root=ROOT,require_local_inputs=False):
    directory=Path(directory);manifest=json.loads((directory/'ARTIFACTS.json').read_text());expected={r['path'] for r in manifest['files']}|{'ARTIFACTS.json','COMPLETED.json'}
    if {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()} != expected:raise ValueError('Incomplete or extra public analysis files')
    for r in manifest['files']:
        p=directory/r['path']
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:raise ValueError('Changed analysis artifact: '+r['path'])
    marker=json.loads((directory/'COMPLETED.json').read_text());assert marker['artifacts_sha256']==digest(directory/'ARTIFACTS.json')
    assert len(read_csv(directory/'paired_windows.csv'))==212 and len(read_csv(directory/'paired_positions.csv'))==106
    s=json.loads((directory/'summary.json').read_text());assert s['positions']==53 and s['replicates']==2
    if require_local_inputs:
        for r in json.loads((directory/'sources.json').read_text())['files']:
            p=root/r['path'];assert p.is_file() and digest(p)==r['sha256'],str(p)
    return {'status':'verified','paired_window_rows':212,'paired_position_rows':106,'artifact_files':len(manifest['files']),'training_performed':False}


def build(root,config_path):
    started=time.monotonic();config=json.loads(config_path.read_text());out=root/config['output_dir'];delivery=root/config['delivery_dir']
    for p in (out,delivery):
        if not p.resolve().is_relative_to(root.resolve()) or p.exists():raise ValueError('Use fresh analysis and delivery directories: '+str(p))
    summary,tables,data,sources=compute(root,config)
    for p in (out,delivery):p.mkdir(parents=True,exist_ok=False)
    for name,rows in tables.items():write_csv(out/(name+'.csv'),rows)
    for name in ('paired_windows','paired_positions'):write_csv(delivery/(name+'.csv'),tables[name])
    for name in ('method_summary','distance_summary','pair_summary'):write_csv(delivery/(name+'.csv'),summary[name])
    plot_all(delivery,summary,tables,data)
    summary['execution_seconds']=round(time.monotonic()-started,3);write_json(delivery/'summary.json',summary)
    for p in (config_path,Path(__file__),root/'scripts/task5/analyze_saved.py'):
        sources.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':digest(p)})
    write_json(delivery/'sources.json',{'files':sources})
    (delivery/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Task 5 paired analysis</title><style>body{max-width:1150px;margin:40px auto;font:17px/1.7 sans-serif;color:#25354a}img{width:100%}a{color:#2563a6}</style><h1>任务五：逐窗口改善与像素—边界取舍</h1><p>53位置 × 2WT；CNN先平均每个seed的指标，再作逐窗口配对。距离边界为探索性受限剖面；远距离NA不是零误差。</p>'+''.join(f'<h2>{title}</h2><img src="{name}.png" alt="{title}">' for name,title in [('paired_improvement','逐窗口分布、配对散点与基因组位置'),('distance_stratification','固定距离层的像素与边界代理评价'),('final_summary','最终结论：像素改善，边界恢复尚未解决')])+'<p>'+ ' · '.join(f'<a href="{name}.csv">{name}</a>' for name in ['paired_windows','paired_positions','distance_summary','pair_summary','method_summary'])+' · <a href="summary.json">完整汇总与解释边界</a></p>')
    files=[{'path':str(p.relative_to(delivery)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(delivery.iterdir()) if p.is_file()]
    write_json(delivery/'ARTIFACTS.json',{'files':files});write_json(delivery/'COMPLETED.json',{'artifacts_sha256':digest(delivery/'ARTIFACTS.json')})
    for r in sources:assert digest(root/r['path'])==r['sha256']
    return verify(delivery,root,True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['build','verify']);p.add_argument('--config',type=Path,default=ROOT/'configs/task5/paired_analysis/paired_v1.json');p.add_argument('--require-local-inputs',action='store_true');a=p.parse_args(argv);config=json.loads(a.config.read_text())
    result=build(ROOT,a.config) if a.action=='build' else verify(ROOT/config['delivery_dir'],ROOT,a.require_local_inputs)
    print(json.dumps(result,ensure_ascii=False))
