"""Rep1-only local discovery, then frozen-coordinate rep2 validation."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/microcture-mpl')
import csv,json,hashlib,random,time
from pathlib import Path
import numpy as np
import h5py
import torch
from torch import nn
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_examples import distance
from prepare_dataset import pool_masked
from task1_baselines import PCA,StandardScaler,LogisticRegression

ROOT=Path(__file__).resolve().parents[1]


def write_csv(path,rows,fields=None):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)


def make_band(path,n,width,chunk=2_000_000):
    """Coarsen upper-triangle counts into a circular local band, not a dense genome."""
    band=np.zeros((n,width),np.float64);coverage=np.zeros(n);total=0
    with h5py.File(path,'r') as f:
        p=f['pixels']
        for off in range(0,len(p['count']),chunk):
            sl=slice(off,off+chunk);a=p['bin1_id'][sl]//10;b=p['bin2_id'][sl]//10;v=p['count'][sl]
            if (v<0).any():raise ValueError('Negative counts')
            total+=int(v.sum());coverage+=np.bincount(a,weights=v,minlength=n)+np.bincount(b,weights=v,minlength=n)
            linear=b-a;d=np.minimum(linear,n-linear);anchor=np.where(linear<=n//2,a,b)
            use=(d>=2)&(d<width)
            np.add.at(band,(anchor[use],d[use]),v[use])
        assert total==int(f.attrs['sum'])
        length=int(f['chroms/length'][0])
    threshold=float(np.median(coverage[coverage>0])*.1);valid=coverage>=threshold
    if length%100:valid[-1]=False
    expected=np.zeros(width);opportunities=np.zeros(width,int)
    for d in range(2,width):
        good=valid & np.roll(valid,-d);opportunities[d]=good.sum()
        expected[d]=band[good,d].sum()/max(good.sum(),1)
    return dict(band=band.astype('float32'),valid=valid,expected=expected,coverage=coverage,
                threshold=threshold,total=total,opportunities=opportunities)


def load_band(sample,n,width):
    path=ROOT/sample['path'];cache=ROOT/'data/cache/discovery';cache.mkdir(parents=True,exist_ok=True)
    file=cache/(sample['sample_id']+'.npz')
    signature=dict(version=1,size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,width=width)
    if file.exists():
        with np.load(file) as z:
            if json.loads(str(z['signature']))==signature:return {k:z[k].copy() for k in z.files if k!='signature'}
    print('Streaming circular band:',sample['sample_id'],flush=True)
    b=make_band(path,n,width);np.savez_compressed(file,**b,signature=json.dumps(signature));return b


def extract(b,start,width):
    n=len(b['valid']);ids=(start+np.arange(width))%n
    a=ids[:,None];c=ids[None,:];delta=np.abs(a-c);d=np.minimum(delta,n-delta)
    anchor=np.where(delta<=n//2,np.minimum(a,c),np.maximum(a,c))
    raw=b['band'][anchor,d]
    valid=b['valid'][ids];mask=valid[:,None]&valid[None,:]&(d>=2)
    oe=np.divide(raw,b['expected'][d],out=np.zeros_like(raw),where=b['expected'][d]>0)
    return raw,oe,mask


def tensor(oe,mask):
    pooled=pool_masked(np.log1p(oe),mask,64)
    use=pooled[1]>0;signal=pooled[0];mean=float(signal[use].mean());sd=max(float(signal[use].std()),.1)
    pooled[0]=np.where(use,np.clip((signal-mean)/sd,-4,4)/4,0)
    return pooled


def annotations(start,width,n,known):
    bins=(start+np.arange(width))%n
    hits=[];recalled=[]
    for r in known:
        s=int(r['start'])//100;e=(int(r['end'])+99)//100
        count=np.count_nonzero((bins>=s)&(bins<e))
        if count:hits.append(r['structure_id'])
        center=int(float(r['center'])//100)
        if count>=.5*(e-s) and center in bins:recalled.append(r['structure_id'])
    return hits,recalled


class AE(nn.Module):
    def __init__(self,latent=16):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(2,8,3,2,1),nn.ReLU(),nn.Conv2d(8,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU())
        self.encode=nn.Linear(32*8*8,latent);self.expand=nn.Linear(latent,32*8*8)
        self.decode=nn.Sequential(nn.ConvTranspose2d(32,16,4,2,1),nn.ReLU(),nn.ConvTranspose2d(16,8,4,2,1),nn.ReLU(),nn.ConvTranspose2d(8,1,4,2,1),nn.Tanh())
    def forward(self,x):
        z=self.encode(self.conv(x).flatten(1));return self.decode(self.expand(z).reshape(-1,32,8,8))[:,0],z


def losses(output,x):
    weight=x[:,1];return ((output-x[:,0]).square()*weight).sum((1,2))/weight.sum((1,2)).clamp(min=1)


def fit_ae(x,rows,cfg,out):
    train=[];val=[]
    for i,r in enumerate(rows):
        lo=r['start_bin'];hi=lo+r['width_bins']-1
        if lo<0 or hi>=46417 or lo//cfg['spatial_block_bins']!=hi//cfg['spatial_block_bins'] or not r['eligible']:continue
        (val if lo//cfg['spatial_block_bins']%5==4 else train).append(i)
    rng=np.random.default_rng(cfg['seed']);train=rng.choice(train,min(len(train),cfg['ae_max_train_windows']),replace=False)
    model=AE(cfg['latent_dim']);opt=torch.optim.Adam(model.parameters(),lr=cfg['ae_learning_rate'])
    xt=torch.from_numpy(x);best=float('inf');pat=0;history=[]
    for epoch in range(cfg['ae_max_epochs']):
        model.train();running=[]
        for ids in np.array_split(rng.permutation(train),max(1,int(np.ceil(len(train)/cfg['ae_batch_size'])))):
            opt.zero_grad();pred,_=model(xt[ids]);loss=losses(pred,xt[ids]).mean();loss.backward();opt.step();running.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation=np.concatenate([losses(model(xt[ids])[0],xt[ids]).numpy() for ids in np.array_split(val,max(1,int(np.ceil(len(val)/128))))]).mean()
        history.append(dict(epoch=epoch+1,train_loss=float(np.mean(running)),validation_loss=float(validation)))
        if validation<best-1e-6:
            best=validation;pat=0;torch.save(dict(state_dict=model.state_dict(),latent=cfg['latent_dim'],epoch=epoch+1),out/'autoencoder.pt')
        else:pat+=1
        if (epoch+1)%5==0:print('AE epoch',epoch+1,'validation loss',round(float(validation),5),flush=True)
        if pat>=cfg['ae_patience']:break
    model.load_state_dict(torch.load(out/'autoencoder.pt',weights_only=False)['state_dict']);model.eval()
    score=[];features=[]
    with torch.no_grad():
        for offset in range(0,len(x),128):
            batch=xt[offset:offset+128];recon,z=model(batch);score.extend(losses(recon,batch).tolist());features.extend(z.numpy())
    (out/'ae_history.json').write_text(json.dumps(history,indent=2))
    (out/'ae_spatial_split.json').write_text(json.dumps(dict(train_indices=train.tolist(),validation_indices=val,rule='whole windows inside 200kb blocks; every fifth block validates'),indent=2))
    return np.array(score),np.array(features)


def disjoint_select(indices,rows,n):
    occupied=np.zeros(n,bool);selected=[]
    for i in indices:
        ids=(rows[i]['start_bin']+np.arange(rows[i]['width_bins']))%n
        if not occupied[ids].any():selected.append(int(i));occupied[ids]=True
    return selected


def correlation(a,b,mask):
    use=mask&np.triu(np.ones_like(mask,bool),2);x=a[use];y=b[use]
    if len(x)<100 or np.std(x)==0 or np.std(y)==0:return None
    return float(np.corrcoef(x,y)[0,1])


def main():
    cfg=json.loads((ROOT/'configs/discovery.json').read_text());data=json.loads((ROOT/'configs/data.json').read_text());known=list(csv.DictReader((ROOT/'data/processed/structures.csv').open()))
    out=ROOT/cfg['output_dir'];out.mkdir(parents=True,exist_ok=True)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    n=(data['chrom_length']+99)//100;width=max(cfg['window_bins'])
    manifest=dict(config=cfg,input_signatures={s['sample_id']:dict(path=s['path'],size=(ROOT/s['path']).stat().st_size,mtime_ns=(ROOT/s['path']).stat().st_mtime_ns) for s in data['samples']},
                  sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['configs/discovery.json','data/processed/structures.csv','scripts/discover_task2.py']})
    if (out/'run_manifest.json').exists() and json.loads((out/'run_manifest.json').read_text())!=manifest:raise ValueError('Discovery inputs changed; use a new output directory')
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    b1=load_band(data['samples'][0],n,width)
    rows=[];xs=[]
    for w in cfg['window_bins']:
        for center in range(0,n,cfg['stride_bins']):
            start=center-w//2;raw,oe,mask=extract(b1,start,w);hits,recall=annotations(start,w,n,known)
            upper=np.triu(mask,2);quality=float(mask.sum()/(w*w-3*w+2));zero=float(np.mean(raw[upper]==0)) if upper.any() else 1.
            rows.append(dict(window_id=f'w{w}_c{center}',start_bin=start,width_bins=w,center_bin=center,
                             valid_fraction=quality,zero_fraction=zero,mean_oe=float(oe[upper].mean()) if upper.any() else 0.,
                             eligible=quality>=cfg['min_valid_fraction'] and zero<=cfg['max_zero_fraction'],
                             known_ids=';'.join(hits),recalled_ids=';'.join(recall)))
            xs.append(tensor(oe,mask))
    x=np.stack(xs);print('Scanned',len(rows),'windows; eligible',sum(r['eligible'] for r in rows),flush=True)
    ae,z=fit_ae(x,rows,cfg,out)
    smooth=gaussian_filter(x[:,0]*x[:,1],sigma=(0,2,2))/np.maximum(gaussian_filter(x[:,1],sigma=(0,2,2)),.1)
    shape=(smooth*smooth*x[:,1]).sum((1,2))/x[:,1].sum((1,2))
    eligible=np.array([r['eligible'] for r in rows]);fusion=np.zeros(len(rows))
    # Rank within physical scale; residuals are window-standardized to suppress density shortcuts.
    for w in cfg['window_bins']:
        ids=np.where(eligible & (np.array([r['width_bins'] for r in rows])==w))[0]
        fusion[ids]=.5*rankdata(ae[ids])/len(ids)+.5*rankdata(shape[ids])/len(ids)
    for i,r in enumerate(rows):r.update(ae_score=float(ae[i]),shape_score=float(shape[i]),score=float(fusion[i]))
    ranking=np.where(eligible)[0][np.argsort(-fusion[eligible],kind='stable')]
    take=int(np.ceil(len(ranking)*cfg['candidate_fraction']));selected=disjoint_select(ranking[:take],rows,n)
    recall_rows=[]
    for method,values in [('density',np.array([r['mean_oe'] for r in rows])),('shape',shape),('autoencoder',ae),('fusion',fusion)]:
        # Comparable scale-specific percentiles for all baseline curves.
        ranks=np.zeros(len(rows))
        for w in cfg['window_bins']:
            ids=np.where(eligible & (np.array([r['width_bins'] for r in rows])==w))[0];ranks[ids]=rankdata(values[ids])/len(ids)
        order=np.where(eligible)[0][np.argsort(-ranks[eligible],kind='stable')]
        for fraction in cfg['recall_fractions']:
            subset=order[:int(np.ceil(len(order)*fraction))];recalled=set();covered=np.zeros(n,bool)
            for i in subset:
                recalled.update(filter(None,rows[i]['recalled_ids'].split(';')));covered[(rows[i]['start_bin']+np.arange(rows[i]['width_bins']))%n]=True
            for kind in ('OPCID','CHIN','CHID'):
                ids={r['structure_id'] for r in known if r['type']==kind}
                recall_rows.append(dict(method=method,top_fraction=fraction,type=kind,recalled=len(ids&recalled),total=len(ids),recall=len(ids&recalled)/len(ids),genome_coverage=float(covered.mean()),windows=len(subset)))
    write_csv(out/'recall_curves.csv',recall_rows)
    # Fixed 8-dimensional PCA of rep1 latent features, then deterministic multi-start k-means.
    scaler=StandardScaler().fit(z[eligible]);zs=scaler.transform(z);pca=PCA(8).fit(zs[eligible]);embedding=pca.transform(zs)
    centers,labels=None,None;best=float('inf')
    for seed in range(10):
        c,l=kmeans2(embedding[selected],cfg['clusters'],minit='++',seed=seed,iter=100)
        loss=np.square(embedding[selected]-c[l]).sum()
        if loss<best:best=loss;centers=c;labels=l
    np.savez_compressed(out/'representation.npz',latent=z,pca=embedding,pca_mean=pca.mean,pca_components=pca.components,feature_mean=scaler.mean,feature_std=scaler.std,cluster_centers=centers,selected_indices=selected,cluster_labels=labels)
    # Known-vs-unannotated linear probe, spatial train/holdout blocks; diagnostic only.
    binary=np.array([bool(r['known_ids']) for r in rows],int)
    blocks=np.array([r['center_bin']//cfg['spatial_block_bins'] for r in rows]);contained=np.array([r['start_bin']>=0 and (r['start_bin']+r['width_bins']-1)//cfg['spatial_block_bins']==r['start_bin']//cfg['spatial_block_bins'] for r in rows])
    # Use a nearest-centroid probe to avoid treating incomplete labels as a supervised truth set.
    train=eligible&contained&(blocks%5!=4);hold=eligible&contained&(blocks%5==4)
    centroids=np.array([z[train&(binary==label)].mean(0) for label in (0,1)])
    predicted=cdist(z[hold],centroids).argmin(1);probe_recall=[float(np.mean(predicted[binary[hold]==label]==label)) for label in (0,1)]
    probe=dict(balanced_accuracy=float(np.mean(probe_recall)),class_recall=probe_recall,chance_balanced_accuracy=.5,n_holdout=int(hold.sum()),note='Annotation-overlap diagnostic only; AE saw these validation blocks for early stopping. Not an independent performance estimate.')
    # Freeze candidates and density/quality-matched, mutually nonoverlapping background controls before rep2 access.
    candidate_bins=np.zeros(n,bool)
    for i in selected:candidate_bins[(rows[i]['start_bin']+np.arange(rows[i]['width_bins']))%n]=True
    background=[]
    for i,r in enumerate(rows):
        if r['eligible'] and not r['known_ids'] and not candidate_bins[(r['start_bin']+np.arange(r['width_bins']))%n].any():background.append(i)
    controls={}
    for i in selected:
        matches=[j for j in background if rows[j]['width_bins']==rows[i]['width_bins']]
        matches.sort(key=lambda j:abs(np.log1p(rows[j]['mean_oe'])-np.log1p(rows[i]['mean_oe']))+abs(rows[j]['valid_fraction']-rows[i]['valid_fraction']))
        controls[str(i)]=disjoint_select(matches,rows,n)[:cfg['controls_per_candidate']]
    frozen=dict(selected_indices=selected,cluster_labels=labels.tolist(),controls=controls,probe=probe,replicate_min_correlation=cfg['replicate_min_correlation'],control_quantile=cfg['control_quantile'])
    (out/'frozen_before_rep2.json').write_text(json.dumps(frozen,indent=2));write_csv(out/'scan_rep1.csv',rows)
    print('Frozen',len(selected),'independent candidate windows; now reading rep2',flush=True)
    b2=load_band(data['samples'][1],n,width)
    used=sorted(set(selected)|{j for value in controls.values() for j in value});correlations={}
    for i in used:
        r=rows[i];_,o1,m1=extract(b1,r['start_bin'],r['width_bins']);_,o2,m2=extract(b2,r['start_bin'],r['width_bins']);correlations[i]=correlation(o1,o2,m1&m2)
    candidates=[];known_features=embedding[np.where(eligible&(binary==1))[0]]
    for index,i in enumerate(selected):
        r=rows[i];values=[correlations[j] for j in controls[str(i)] if correlations[j] is not None]
        cutoff=max(cfg['replicate_min_correlation'],float(np.quantile(values,cfg['control_quantile']))) if len(values)>=cfg['min_controls'] else None
        corr=correlations[i];passes=cutoff is not None and corr is not None and corr>cutoff
        candidates.append(dict(candidate_id=f'C{index+1:04d}',window_index=i,chrom=data['chrom'],start_bp=(r['start_bin']%n)*100,
                               end_bp=min(((r['start_bin']+r['width_bins'])%n)*100,data['chrom_length']),center_bp=r['center_bin']*100,
                               window_length_bp=r['width_bins']*100,wraps_origin=r['start_bin']<0 or r['start_bin']+r['width_bins']>n,
                               cluster=int(labels[index]),score=r['score'],known_ids=r['known_ids'],replicate_correlation=corr,
                               matched_controls=len(values),control_threshold=cutoff,replicate_supported=passes,
                               nearest_annotated_embedding_distance=float(cdist(embedding[i:i+1],known_features).min()),
                               distinction='overlaps_known_annotation' if r['known_ids'] else 'unannotated_window_not_proven_new_type'))
    cluster_summary=[]
    for k in range(cfg['clusters']):
        members=[r for r in candidates if r['cluster']==k];annotated=sum(bool(r['known_ids']) for r in members);supported=sum(r['replicate_supported'] for r in members)
        novel=annotated==0 and supported>=cfg['min_new_cluster_members']
        cluster_summary.append(dict(cluster=k,members=len(members),annotated_members=annotated,replicate_supported_members=supported,passes_operational_new_cluster_rule=novel))
    novel_clusters={r['cluster'] for r in cluster_summary if r['passes_operational_new_cluster_rule']}
    for r in candidates:r['candidate_new_cluster']=r['cluster'] in novel_clusters and r['replicate_supported']
    write_csv(out/'candidates.csv',candidates);write_csv(out/'clusters.csv',cluster_summary)
    write_csv(out/'candidate_new_structures.csv',[r for r in candidates if r['candidate_new_cluster']],list(candidates[0]))
    write_csv(out/'unannotated_replicated_windows.csv',[r for r in candidates if not r['known_ids'] and r['replicate_supported']],list(candidates[0]))
    write_csv(out/'controls.csv',[dict(window_index=i,correlation=correlations[i],**rows[i]) for i in used if i not in selected])
    # Unified O/E color scale: chosen on rep1 only, fixed for every candidate and repeat.
    limit=float(np.quantile(np.concatenate([np.log1p(extract(b1,rows[i]['start_bin'],rows[i]['width_bins'])[1])[extract(b1,rows[i]['start_bin'],rows[i]['width_bins'])[2]] for i in selected]),.99))
    (out/'heatmaps').mkdir(exist_ok=True)
    for candidate,i in zip(candidates,selected):
        r=rows[i];fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
        for ax,b,rep in zip(axes,[b1,b2],[1,2]):
            _,oe,mask=extract(b,r['start_bin'],r['width_bins']);lo=r['start_bin']*.1;hi=lo+r['width_bins']*.1
            cm=plt.get_cmap('magma').copy();cm.set_bad('#bdc5cc');im=ax.imshow(np.where(mask,np.log1p(oe),np.nan),origin='lower',extent=(lo,hi,lo,hi),vmin=0,vmax=limit,cmap=cm)
            ax.set(title=f'rep{rep}',xlabel='Unwrapped binned position (kb)',ylabel='Position (kb)')
        fig.colorbar(im,ax=axes,label='log1p(O/E)',shrink=.8)
        fig.suptitle(f"{candidate['candidate_id']} | cluster {candidate['cluster']} | r={candidate['replicate_correlation']}\n{candidate['distinction']}",fontsize=10)
        fig.savefig(out/'heatmaps'/f"{candidate['candidate_id']}.png",dpi=110);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    scatter=axes[0].scatter(embedding[selected,0],embedding[selected,1],c=labels,cmap='tab10',s=30)
    for row,k in zip(candidates,selected):
        if row['known_ids']:axes[0].scatter(*embedding[k,:2],marker='x',c='black',s=15)
    axes[0].set(title='Rep1 AE latent PCA; x = known overlap',xlabel='PC1',ylabel='PC2');fig.colorbar(scatter,ax=axes[0],label='Cluster')
    for kind in ('OPCID','CHIN','CHID'):
        items=[r for r in recall_rows if r['method']=='fusion' and r['type']==kind]
        axes[1].plot([r['genome_coverage'] for r in items],[r['recall'] for r in items],marker='o',label=kind)
    axes[1].set(title='Known recovery vs covered genome',xlabel='Genome fraction covered',ylabel='Recall',ylim=(0,1.05));axes[1].legend();fig.savefig(out/'overview.png',dpi=150);plt.close(fig)
    recalled=set()
    for i in selected:recalled.update(filter(None,rows[i]['recalled_ids'].split(';')))
    dedup_recall={kind:sum(r['structure_id'] in recalled for r in known if r['type']==kind)/sum(r['type']==kind for r in known) for kind in ('OPCID','CHIN','CHID')}
    report=dict(config=cfg,scan_windows=len(rows),eligible_windows=int(eligible.sum()),independent_candidates=len(candidates),
                known_overlap_candidates=sum(bool(r['known_ids']) for r in candidates),unannotated_candidates=sum(not r['known_ids'] for r in candidates),
                replicate_supported=sum(r['replicate_supported'] for r in candidates),unannotated_replicate_supported=sum(r['replicate_supported'] and not r['known_ids'] for r in candidates),
                new_cluster_candidates=sum(r['candidate_new_cluster'] for r in candidates),clusters=cluster_summary,representation_probe=probe,
                deduplicated_known_recall=dedup_recall,candidate_genome_coverage=float(candidate_bins.mean()),rep1_color_limit_log1p=limit,
                limitations=['Local diagonal windows only, up to 25.6kb; not all distant 2D contacts.',
                'Coordinates provisional; circular bins approximate up to 48bp at origin.',
                'Annotations incomplete; unannotated is not synonymous with a new type.',
                'AE residual can reflect noise; shape ranks, quality filters and replication are complementary, not proof.',
                'Matched controls use rep1 density and quality; too few controls => insufficient validation, not passed.',
                'K=6 and member>=5 are operational conventions; no mechanism claims.'])
    (out/'report.json').write_text(json.dumps(report,indent=2));(ROOT/'reports/task2_results.json').write_text(json.dumps(report,indent=2));print('COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
