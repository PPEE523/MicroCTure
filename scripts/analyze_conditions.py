"""Task 3 genome tracks and task 4 descriptive replicate-aware differential analysis."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/microcture-mpl')
import csv,json,hashlib
from pathlib import Path
import numpy as np
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator,StrMethodFormatter
from discover_task2 import load_band,extract
ROOT=Path(__file__).resolve().parents[1]
COLORS=['#1764ab','#dd8530','#b43858']


def write_csv(path,rows,fields=None):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)


def local_signal(b,common,lo,hi):
    signal=np.zeros(len(common));denom=np.zeros(len(common))
    for d in range(lo,hi+1):
        valid=common&np.roll(common,-d)
        counts=np.where(valid,b['band'][:,d],0)
        signal+=counts+np.roll(counts,d)
        denom+=valid+np.roll(valid,d).astype(int)
    return np.divide(signal*1e6/float(b['total']),denom,out=np.full_like(signal,np.nan),where=denom>0)


def compare(wt,treated,pseudo,fold):
    wt=np.asarray(wt);treated=np.asarray(treated)
    pairs=np.log2((treated[:,None]+pseudo)/(wt[None,:]+pseudo))
    effect=float(np.log2((treated.mean()+pseudo)/(wt.mean()+pseudo)))
    threshold=np.log2(fold)
    direction='increased' if np.min(pairs)>threshold else ('decreased' if np.max(pairs)<-threshold else 'inconsistent_or_small')
    return dict(log2_fold_change=effect,min_cross_repeat_log2_fc=float(pairs.min()),max_cross_repeat_log2_fc=float(pairs.max()),
                direction=direction,wt_repeat_log2_ratio=float(np.log2((wt[1]+pseudo)/(wt[0]+pseudo))),
                treatment_repeat_log2_ratio=float(np.log2((treated[1]+pseudo)/(treated[0]+pseudo))))


def main():
    c=json.loads((ROOT/'configs/conditions.json').read_text());audit=json.loads((ROOT/'reports/conditions_audit.json').read_text())
    assert audit['bin_coordinates_identical']
    out3=ROOT/'results/task3';out4=ROOT/'results/task4'
    for folder in [out3/'tracks',out4/'heatmaps']:folder.mkdir(parents=True,exist_ok=True)
    n=(c['chrom_length']+99)//100
    bands=[load_band(s,n,256) for s in c['samples']]
    common=np.logical_and.reduce([b['valid'] for b in bands])
    samples=c['samples'];groups=[[i for i,s in enumerate(samples) if s['condition']==cond] for cond in c['conditions']]
    assert all(len(g)==2 for g in groups)
    signal=np.array([local_signal(b,common,c['min_distance_bins'],c['max_distance_bins']) for b in bands])
    means=np.array([signal[g].mean(0) for g in groups])
    np.savez_compressed(out3/'signals.npz',signals=signal,condition_means=means,common_valid=common,positions=np.arange(n)*100)
    tracks=[]
    for i in range(n):
        row=dict(chrom=c['chrom'],start=i*100,end=min((i+1)*100,c['chrom_length']),valid=bool(common[i]))
        for j,s in enumerate(samples):row[s['sample_id']]=float(signal[j,i]) if np.isfinite(signal[j,i]) else ''
        tracks.append(row)
    write_csv(out3/'signals.csv',tracks)
    workbook=openpyxl.load_workbook(ROOT/c['annotation_path'],read_only=True,data_only=True)
    sheet=workbook['Supplementary Table 1'];genes=[]
    for row in sheet.iter_rows(min_row=2,values_only=True):
        if not row[0]:continue
        if isinstance(row[2],(int,float)) and isinstance(row[3],(int,float)) and 0<=row[2]<row[3]<=c['chrom_length']:
            genes.append(dict(gene=row[0],chrom=c['chrom'],start=int(row[2]),end=int(row[3]),strand=row[4],coordinate_status='provisional'))
    workbook.close();write_csv(ROOT/'data/processed/genes.csv',genes)
    known=list(csv.DictReader((ROOT/'data/processed/structures.csv').open()))
    for r in known:r['start']=int(r['start']);r['end']=int(r['end'])
    print('Loaded six matrices and',len(genes),'gene annotations; generating 465 tracks',flush=True)
    pages=[]
    for start in range(0,c['chrom_length'],c['track_window_bp']):
        end=min(start+c['track_window_bp'],c['chrom_length']);sl=slice(start//100,(end+99)//100);pos=np.arange(n)[sl]*100
        fig,axes=plt.subplots(4,1,figsize=(13,8),sharex=True,gridspec_kw={'height_ratios':[3,1.5,1.2,1.8]},layout='constrained')
        axes[0].fill_between(pos,0,np.sqrt(means[0,sl]),color='gray',alpha=.2,label='WT mean baseline')
        for j,(cond,g) in enumerate(zip(c['conditions'],groups)):
            for index in g:axes[0].plot(pos,np.sqrt(signal[index,sl]),color=COLORS[j],alpha=.3,lw=.6,ls=':')
            axes[0].plot(pos,np.sqrt(means[j,sl]),color=COLORS[j],label=cond,lw=1.1)
        axes[0].set_ylabel('sqrt(contacts per million / pair)');axes[0].legend(fontsize=8,ncol=2)
        axes[0].set_title(f'{c["chrom"]}:{start:,}-{end:,} | 100 bp bins | 200 bp-10 kb contact band')
        axes[1].fill_between(pos,0,np.mean(means[:,sl],axis=0),color='#768798',alpha=.65);axes[1].set_ylabel('Mean signal\n3 conditions')
        for r in genes:
            if r['start']<end and r['end']>start:
                lane=0 if r['strand']=='+' else 1
                left=max(start,r['start']);right=min(end,r['end'])
                axes[2].add_patch(Rectangle((left,lane-.25),right-left,.5,color='#3181bd'))
                axes[2].text((left+right)/2,lane+.3,r['gene'],fontsize=5,ha='center',rotation=20,clip_on=True)
        axes[2].set(ylim=(-.5,1.9),yticks=[0,1],yticklabels=['+','-'],ylabel='Genes')
        for r in known:
            if r['start']<end and r['end']>start:
                lane=['OPCID','CHIN','CHID'].index(r['type']);left=max(start,r['start']);right=min(end,r['end'])
                axes[3].add_patch(Rectangle((left,lane-.22),right-left,.44,color='#ee923c',alpha=.6))
                axes[3].text((left+right)/2,lane+.25,r['structure_id'],fontsize=5,ha='center',rotation=15,clip_on=True)
        axes[3].set(ylim=(-.5,2.9),yticks=[0,1,2],yticklabels=['OPCID','CHIN','CHID'],xlabel='Genomic position (bp)',xlim=(start,end))
        axes[3].xaxis.set_major_locator(MaxNLocator(6));axes[3].xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        name=f'{start:07d}_{end:07d}.png';fig.savefig(out3/'tracks'/name,dpi=110);plt.close(fig)
        pages.append(dict(start=start,end=end,file='tracks/'+name))
        if len(pages)%100==0:print('Track pages:',len(pages),flush=True)
    write_csv(out3/'pages.csv',pages)
    html=['<!doctype html><meta charset="utf-8"><title>Three-condition tracks</title><h1>Three-condition genome tracks</h1><p>WT, delta stpA, delta hns/stpA; dotted lines: repeats. Coordinates provisional.</p>']
    html += [f'<p><a href="{r["file"]}">{r["start"]:,}–{r["end"]:,} bp</a></p>' for r in pages]
    (out3/'index.html').write_text('\n'.join(html))
    # Use exactly the same pair mask across all samples for each structure.
    region_values=[];differences=[]
    for r in known:
        start=r['start']//100;width=(r['end']+99)//100-start
        if width>256:raise ValueError('Structure exceeds cached band')
        ids=(start+np.arange(width))%n;d=np.abs(np.arange(width)[:,None]-np.arange(width)[None,:])
        possible=(d>=c['min_distance_bins'])&(d<=c['max_distance_bins'])&np.triu(np.ones((width,width),bool),1)
        mask=possible&common[ids,None]&common[ids][None,:]
        fraction=float(mask.sum()/max(possible.sum(),1));ok=mask.sum()>=c['structure_min_pairs'] and fraction>=c['structure_min_valid_fraction']
        depth=[];oes=[]
        for s,b in zip(samples,bands):
            raw,oe,_=extract(b,start,width)
            value=float(raw[mask].mean()*1e6/float(b['total'])) if mask.any() else float('nan')
            o=float(oe[mask].mean()) if mask.any() else float('nan');depth.append(value);oes.append(o)
            region_values.append(dict(structure_id=r['structure_id'],type=r['type'],sample=s['sample_id'],condition=s['condition'],replicate=s['replicate'],signal=value,mean_oe=o,valid_pairs=int(mask.sum()),valid_fraction=fraction))
        for cond,g in zip(c['conditions'][1:],groups[1:]):
            wt=np.array(depth)[groups[0]];treated=np.array(depth)[g]
            effect=compare(wt,treated,c['pseudocount_per_million'],c['fold_change_threshold'])
            oe_effect=compare(np.array(oes)[groups[0]],np.array(oes)[g],.01,c['fold_change_threshold'])
            supported=ok and effect['direction']!='inconsistent_or_small'
            differences.append(dict(structure_id=r['structure_id'],type=r['type'],chrom=c['chrom'],start=r['start'],end=r['end'],condition=cond,
                                    **effect,oe_log2_fold_change=oe_effect['log2_fold_change'],oe_direction=oe_effect['direction'],
                                    wt_rep1=wt[0],wt_rep2=wt[1],treated_rep1=treated[0],treated_rep2=treated[1],
                                    valid_pairs=int(mask.sum()),valid_fraction=fraction,quality_pass=ok,descriptive_change=supported,
                                    oe_direction_agrees=bool(np.sign(effect['log2_fold_change'])==np.sign(oe_effect['log2_fold_change']))))
    write_csv(out4/'structure_signals.csv',region_values);write_csv(out4/'differential_structures.csv',differences)
    chosen=[]
    for cond in c['conditions'][1:]:
        ranked=sorted([r for r in differences if r['condition']==cond and r['quality_pass']],key=lambda r:abs(r['log2_fold_change']),reverse=True)
        chosen+=ranked[:c['top_heatmaps_per_condition']]
    for r in chosen:
        center=(r['start']+r['end'])//200;w=min(256,max(64,int(np.ceil((r['end']-r['start'])/100*1.4))));start=center-w//2
        matrices=[extract(b,start,w) for b in bands];limit=np.quantile(np.concatenate([np.log1p(oe[mask]) for raw,oe,mask in matrices]),.99)
        fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
        for col,(cond,g) in enumerate(zip(c['conditions'],groups)):
            for row,index in enumerate(g):
                raw,oe,mask=matrices[index];im=axes[row,col].imshow(np.where(mask,np.log1p(oe),np.nan),origin='lower',vmin=0,vmax=limit,cmap='magma',extent=(start*.1,(start+w)*.1,start*.1,(start+w)*.1))
                axes[row,col].set(title=f'{cond} / rep{row+1}',xlabel='Position (kb)',ylabel='Position (kb)')
        fig.colorbar(im,ax=axes,label='log1p(O/E)',shrink=.6)
        fig.suptitle(f"{r['structure_id']} | {r['condition']} depth log2FC={r['log2_fold_change']:.2f}\nDescriptive change={r['descriptive_change']}; O/E panels show shape, not depth-normalized magnitude")
        fig.savefig(out4/'heatmaps'/f"{r['condition']}_{r['structure_id']}.png",dpi=130);plt.close(fig)
    summary=[]
    for cond in c['conditions'][1:]:
        for kind in ['OPCID','CHIN','CHID']:
            subset=[r for r in differences if r['condition']==cond and r['type']==kind]
            good=[r for r in subset if r['quality_pass']]
            summary.append(dict(condition=cond,type=kind,total=len(subset),quality_pass=len(good),
                                increased=sum(r['descriptive_change'] and r['direction']=='increased' for r in subset),
                                decreased=sum(r['descriptive_change'] and r['direction']=='decreased' for r in subset),
                                median_log2_fc=float(np.median([r['log2_fold_change'] for r in good])) if good else None))
    write_csv(out4/'summary.csv',summary)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,cond in zip(axes,c['conditions'][1:]):
        for i,kind in enumerate(['OPCID','CHIN','CHID']):
            values=[r['log2_fold_change'] for r in differences if r['condition']==cond and r['type']==kind and r['quality_pass']]
            ax.scatter(np.full(len(values),i),values,alpha=.35,s=12,color=COLORS[i])
        ax.axhline(0,color='gray');ax.set(xticks=[0,1,2],xticklabels=['OPCID','CHIN','CHID'],ylabel='Depth-normalized log2 fold change',title=cond)
    fig.savefig(out4/'effects.png',dpi=150);plt.close(fig)
    report=dict(conditions=c['conditions'],samples=6,track_pages=len(pages),genes=len(genes),common_valid_bins=int(common.sum()),total_bins=n,
                comparisons=len(differences),summary=summary,heatmaps=len(chosen),coordinate_status='provisional',
                interpretation='Descriptive fold changes and all four cross-repeat contrasts; no p-values or claims of significance/causality.',
                limitations=['Library-size normalization gives relative, not absolute contacts.', 'Common low-coverage masking can omit condition-specific losses.',
                             'Gene coordinates from supplied Table 1, not verified as complete reference annotation.', 'Each condition has only two biological repeats; no pixel-level pseudoreplication.'],
                input_sha256={'config':hashlib.sha256((ROOT/'configs/conditions.json').read_bytes()).hexdigest(),'samples_audit':hashlib.sha256((ROOT/'reports/conditions_audit.json').read_bytes()).hexdigest()})
    (ROOT/'reports/task34_results.json').write_text(json.dumps(report,indent=2));print('COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
