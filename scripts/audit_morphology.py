"""Audit replicated candidate morphology against multiple disjoint known contexts."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/microcture-mpl')
import hashlib
import json
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from discover_task2_v2 import ROOT,read_csv
from discover_task2 import load_band,extract,write_csv,correlation
from discovery_v2_core import bin_roles,train_background
from review_novelty import shape_feature


def remove_row_effects(matrix,mask,iterations):
    residual=np.where(mask,matrix,0).astype(float)
    for _ in range(iterations):
        count=mask.sum(1)
        means=np.divide(residual.sum(1),count,out=np.zeros(len(matrix)),where=count>0)
        residual=np.where(mask,residual-.5*(means[:,None]+means[None,:]),0)
    return residual,np.where(mask,matrix-residual,0)


def matrix_metrics(raw,oe,mask):
    upper=np.triu(mask,2);count=mask.sum(1)
    means=np.divide(np.where(mask,oe,0).sum(1),count,out=np.full(len(oe),np.nan),where=count>0)
    usable=np.flatnonzero(np.isfinite(means))
    top=usable[np.argsort(-means[usable],kind='stable')[:max(1,int(np.ceil(len(usable)*.1)))]]
    selected=np.zeros(len(oe),bool);selected[top]=True
    contacts=upper&(selected[:,None]|selected[None,:])
    mass=float(oe[contacts].sum()/oe[upper].sum()) if oe[upper].sum()>0 else float('nan')
    return dict(valid_fraction=float(mask.sum()/(len(oe)**2-3*len(oe)+2)),zero_fraction=float((raw[upper]==0).mean()),
                row_mean_cv=float(np.nanstd(means)/max(np.nanmean(means),1e-12)),top10_rows_contact_mass_fraction=mass,
                mean_oe=float(oe[upper].mean()))


def main():
    cfg=json.loads((ROOT/'configs/morphology_audit.json').read_text());out=ROOT/cfg['output_dir'];out.mkdir(parents=True,exist_ok=True)
    for name in ['comparisons','diagnostics']:(out/name).mkdir(exist_ok=True)
    paths=['configs/morphology_audit.json','scripts/audit_morphology.py',cfg['candidate_source'],cfg['references_source'],
           'scripts/review_novelty.py','scripts/discover_task2.py','scripts/discovery_v2_core.py','configs/discovery_v2.json',
           'results/task2_centering_supplement/candidate_summary.csv','results/task2_novelty_review/candidate_review.csv']
    manifest=dict(config=cfg,sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
    path=out/'run_manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('Inputs changed')
    path.write_text(json.dumps(manifest,indent=2))
    data=json.loads((ROOT/'configs/data.json').read_text());source=json.loads((ROOT/'configs/discovery_v2.json').read_text())
    n=(data['chrom_length']+99)//100;roles=bin_roles(n,source)
    bands=[train_background(load_band(s,n,256),roles) for s in data['samples']]
    candidates=read_csv(cfg['candidate_source']);references=[r for r in read_csv(cfg['references_source']) if r['role']=='0']
    prior={r['window_id']:r for r in read_csv('results/task2_novelty_review/candidate_review.csv')}
    variants={r['entity_id']:r for r in read_csv('results/task2_centering_supplement/candidate_summary.csv')}
    ref_features=[];ref_matrices=[]
    for r in references:
        matrices=[extract(b,int(r['start_bin']),int(r['width_bins'])) for b in bands]
        ref_matrices.append(matrices);ref_features.append(shape_feature(*matrices[0][1:]))
    ref_features=np.array(ref_features)
    results=[];metrics=[];matches=[];montage=[]
    for c in candidates:
        name=c['window_id'];start=int(c['start_bp'])//100;width=int(c['window_length_bp'])//100
        matrices=[extract(b,start,width) for b in bands]
        feature=shape_feature(*matrices[0][1:])
        available=[j for j,r in enumerate(references) if int(r['width_bins'])==width]
        distances=cdist(feature[None],ref_features[available],metric='cosine')[0]
        order=np.array(available)[np.argsort(distances,kind='stable')]
        chosen=[];occupied=np.zeros(n,bool)
        for j in order:
            ids=np.arange(int(references[j]['start_bin']),int(references[j]['start_bin'])+width)
            if not occupied[ids].any():chosen.append(int(j));occupied[ids]=True
            if len(chosen)==cfg['top_references']:break
        assert len(chosen)==cfg['top_references']
        logs=[np.log1p(m[1]) for m in matrices]
        residuals=[];effects=[]
        for rep,(raw,oe,mask) in enumerate(matrices):
            residual,effect=remove_row_effects(logs[rep],mask,cfg['row_effect_iterations']);residuals.append(residual);effects.append(effect)
            metrics.append(dict(entity_id=name,kind='candidate',replicate=rep+1,**matrix_metrics(raw,oe,mask)))
        common=matrices[0][2]&matrices[1][2]
        log_r=correlation(logs[0],logs[1],common);resid_r=correlation(residuals[0],residuals[1],common)
        row=dict(window_id=name,start_bp=start*100,end_bp=(start+width)*100,width_bp=width*100,
                 oe_replication=float(c['correlation']),log_oe_replication=log_r,residual_replication=resid_r,
                 annotation_gap_bp=int(prior[name]['annotation_gap_bp']),nearest_annotation=prior[name]['nearest_annotation_by_position'],
                 boundary_shift_bp=int(prior[name]['boundary_shift_bp']),supplement_original_anomaly_variants=int(variants[name]['original_outside_variants']),
                 supplement_boundary_anomaly_variants=int(variants[name]['boundary_outside_variants']),stable_anomaly=variants[name]['robust_recentered_anomaly'],
                 reference_ids=';'.join(references[j]['structure_id'] for j in chosen),coordinate_status='provisional')
        for rank,j in enumerate(chosen,1):
            rr=references[j];rm=ref_matrices[j]
            ds=[float(cdist(shape_feature(*matrices[rep][1:])[None],shape_feature(*rm[rep][1:])[None],metric='cosine')[0,0]) for rep in [0,1]]
            ref_res=[remove_row_effects(np.log1p(m[1]),m[2],cfg['row_effect_iterations'])[0] for m in rm]
            refcorr=correlation(ref_res[0],ref_res[1],rm[0][2]&rm[1][2])
            matches.append(dict(window_id=name,rank=rank,reference_id=rr['structure_id'],reference_type=rr['type'],reference_start_bin=int(rr['start_bin']),width_bins=width,
                                rep1_cosine_distance=ds[0],rep2_same_reference_cosine_distance=ds[1],reference_residual_replication=refcorr))
            for rep,m in enumerate(rm):metrics.append(dict(entity_id=f'{name}:ref{rank}:{rr["structure_id"]}',kind='reference',replicate=rep+1,**matrix_metrics(*m)))
        results.append(row)
        panels=[matrices]+[ref_matrices[j] for j in chosen]
        limit=float(np.quantile(np.concatenate([np.log1p(m[0][1])[m[0][2]] for m in panels]),.99))
        fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
        for col,ms in enumerate(panels):
            label=name if col==0 else references[chosen[col-1]]['structure_id']
            for rep,(raw,oe,mask) in enumerate(ms):
                im=axes[rep,col].imshow(np.where(mask,np.log1p(oe),np.nan),origin='lower',vmin=0,vmax=limit,cmap='magma')
                axes[rep,col].set(title=f'{label} / rep{rep+1}',xlabel='Local bin',ylabel='Local bin')
        fig.colorbar(im,ax=axes,label='log1p(O/E)',shrink=.5)
        fig.suptitle(f'{name} | same-scale, spatially disjoint top-3 known contexts selected by rep1\nReference label is not a candidate classification; known contexts include surrounding contacts',fontsize=11)
        fig.savefig(out/'comparisons'/f'{name}.png',dpi=115);plt.close(fig)
        montage.append((name,np.where(common,.5*(logs[0]+logs[1]),np.nan)))
        if name in cfg['focus_windows']:
            fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
            lim=float(np.nanquantile(np.abs(np.concatenate([residuals[r][matrices[r][2]] for r in [0,1]])),.99))
            for rep in [0,1]:
                mask=matrices[rep][2]
                for col,(m,title) in enumerate([(logs[rep],'log1p(O/E)'),(effects[rep],'Additive row/column component'),(residuals[rep],'Residual after row/column removal')]):
                    axes[rep,col].imshow(np.where(mask,m,np.nan),origin='lower',cmap='coolwarm' if col==2 else 'magma',vmin=-lim if col==2 else 0,vmax=lim if col==2 else limit)
                    axes[rep,col].set(title=f'{title} / rep{rep+1}')
            fig.suptitle(f'{name} | log-O/E r={log_r:.3f}; residual r={resid_r:.3f}\nRemoval can suppress real biological stripes too; this is not a technical-artifact test',fontsize=11)
            fig.savefig(out/'diagnostics'/f'{name}.png',dpi=115);plt.close(fig)
    write_csv(out/'evidence.csv',results);write_csv(out/'quality_and_shape_metrics.csv',metrics);write_csv(out/'top3_reference_matches.csv',matches)
    fig,axes=plt.subplots(7,3,figsize=(12,24),layout='constrained')
    for ax,(name,matrix) in zip(axes.flat,montage):
        ax.imshow(matrix,origin='lower',cmap='magma',vmin=0,vmax=float(np.nanquantile(matrix,.99)))
        ax.set_title(name,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('21 candidate mean log-O/E shapes; each panel independently scaled; no type labels assigned')
    fig.savefig(out/'candidate_montage.png',dpi=140);plt.close(fig)
    report=dict(candidates=len(results),top3_reference_comparisons=len(matches),figures=25,
                marginal_residual_correlation_min=min(r['residual_replication'] for r in results),
                marginal_residual_correlation_max=max(r['residual_replication'] for r in results),
                stable_distinct_morphologies_confirmed=0,interpretation=cfg['interpretation'])
    (out/'report.json').write_text(json.dumps(report,indent=2))
    page=['<!doctype html><meta charset="utf-8"><title>候选形态证据核查</title><style>body{max-width:1500px;margin:30px auto;padding:20px;font:16px/1.6 sans-serif}img{max-width:100%}</style><h1>21 个候选的多参考形态核查</h1><p>每个候选与三个空间不重叠的同尺度已知中心窗口比较；参考标签不是候选分类。没有新类型确认。</p><p><a href="evidence.csv">证据表</a> · <a href="quality_and_shape_metrics.csv">质量与接触集中度</a> · <a href="top3_reference_matches.csv">三个参考的距离及重复表现</a> · <a href="candidate_montage.png">21 个窗口形态总览</a></p>']
    for r in results:
        name=r['window_id'];page.append(f'<h2>{name}</h2><p>最近位置注释 {r["nearest_annotation"]}，间隔 {r["annotation_gap_bp"]} bp；log-O/E 重复相关 {r["log_oe_replication"]:.3f}，去行列分量后 {r["residual_replication"]:.3f}。</p><a href="comparisons/{name}.png"><img loading="lazy" src="comparisons/{name}.png"></a>')
        if name in cfg['focus_windows']:page.append(f'<p><a href="diagnostics/{name}.png">行列效应敏感性诊断（不能据此确认技术伪影）</a></p>')
    (out/'index.html').write_text('\n'.join(page));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
