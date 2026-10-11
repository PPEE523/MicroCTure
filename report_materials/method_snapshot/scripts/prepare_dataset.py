"""Grouped, spatially disjoint supervised data; all fitted transforms use train only."""
import csv
import json
from pathlib import Path
import hashlib

import cooler
import h5py
import numpy as np

from plot_examples import aggregate_local, distance

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ('train', 'validation', 'test')


def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def group_windows(rows, n, half, guard):
    """Union overlapping/nearby circular footprints, including the origin."""
    parent = list(range(len(rows)))
    ids = [(np.arange(-half,half)+int(float(r['center'])//100)) % n for r in rows]
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(rows)):
        for j in range(i):
            ci, cj = ids[i][half], ids[j][half]
            if distance(ci,cj,n) < 2*half+guard:
                parent[find(i)] = find(j)
    roots = sorted(set(find(i) for i in range(len(rows))))
    groups = np.array([roots.index(find(i)) for i in range(len(rows))])
    return groups, ids


def split_groups(rows, groups, footprints, explored, config, n):
    forced = set()
    explored_bins = np.zeros(n, bool)
    for row in explored:
        explored_bins[(np.arange(int(row['width_bins']))+int(row['start_bin'])) % n] = True
    for group, ids in zip(groups, footprints):
        if explored_bins[ids].any(): forced.add(int(group))
    k = int(groups.max())+1
    labels = config['label_order']
    counts = np.zeros((k,3),int)
    for row,g in zip(rows,groups): counts[g,labels.index(row['type'])] += 1
    target = np.array(config['split_fractions'])[:,None]*counts.sum(axis=0)
    rng = np.random.default_rng(config['seed'])
    best, score = None, float('inf')
    for _ in range(config['split_search_trials']):
        assignment = rng.choice(3,k,p=config['split_fractions'])
        assignment[list(forced)] = 0
        totals = np.array([counts[assignment==s].sum(axis=0) for s in range(3)])
        if (totals < np.array([1,1,3])[None,:]).any(): continue
        current = np.square((totals-target)/np.maximum(target,1)).sum()
        if current < score: best,score=assignment.copy(),float(current)
    if best is None: raise ValueError('Cannot form splits with >=3 CHID per split')
    return best[groups], sorted(forced)


def assert_disjoint(footprints, split, n, guard):
    occupancy = [np.zeros(n,bool) for _ in SPLITS]
    for ids,s in zip(footprints,split): occupancy[s][ids]=True
    for i in range(3):
        for j in range(i):
            for shift in range(-guard,guard+1):
                if (occupancy[i] & np.roll(occupancy[j],shift)).any():
                    raise ValueError('Cross-split footprint/guard overlap')
    return occupancy


def fit_expected(path, train_bins, factor, max_d, chunk=2_000_000):
    """No counts with a validation/test endpoint enter these fitted statistics."""
    n=len(train_bins)
    sums=np.zeros(max_d+1)
    total=0
    with h5py.File(path,'r') as f:
        p=f['pixels']
        for offset in range(0,len(p['count']),chunk):
            sl=slice(offset,offset+chunk)
            a=p['bin1_id'][sl]//factor; b=p['bin2_id'][sl]//factor
            v=p['count'][sl]
            keep=train_bins[a]&train_bins[b]
            total+=int(v[keep].sum())
            d=distance(a,b,n)
            keep &= (d>=2)&(d<=max_d)
            sums+=np.bincount(d[keep],weights=v[keep],minlength=max_d+1)
    opportunities=np.array([np.count_nonzero(train_bins & np.roll(train_bins,-d)) for d in range(max_d+1)])
    expected=np.divide(sums,opportunities,out=np.zeros_like(sums),where=opportunities>0)
    if total<=0 or (expected[2:]<=0).any(): raise ValueError('Insufficient train background')
    return expected,total,opportunities


def pool_masked(array, valid, size):
    factor=len(array)//size
    shape=(size,factor,size,factor)
    count=valid.reshape(shape).sum(axis=(1,3))
    summed=np.where(valid,array,0).reshape(shape).sum(axis=(1,3))
    value=np.divide(summed,count,out=np.zeros_like(summed),where=count>0)
    return np.stack([value,count/(factor*factor)]).astype('float32')


def fit_scale(x):
    result=[]
    for scale in range(x.shape[1]):
        v=x[:,scale,0][x[:,scale,1]>0]
        result.append([float(v.mean()),max(float(v.std()),1e-6)])
    return result


def apply_scale(x, parameters):
    x=x.copy()
    for scale,(mean,std) in enumerate(parameters):
        x[:,scale,0]=np.where(x[:,scale,1]>0,(x[:,scale,0]-mean)/std,0)
    return x


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    c=json.loads((ROOT/'configs/dataset.json').read_text())
    data=json.loads((ROOT/'configs/data.json').read_text())
    assert c['resolution_bp']==100, 'Current pipeline validated at 100bp only'
    rows=list(csv.DictReader((ROOT/'data/processed/structures.csv').open()))
    explored=list(csv.DictReader((ROOT/c['explored_regions']).open()))
    n=(data['chrom_length']+99)//100
    half=max(c['window_bins'])//2
    groups,footprints=group_windows(rows,n,half,c['guard_bins'])
    split,forced=split_groups(rows,groups,footprints,explored,c,n)
    occupancy=assert_disjoint(footprints,split,n,c['guard_bins'])
    train_bins=occupancy[0].copy()
    if data['chrom_length']%100:train_bins[-1]=False
    out=ROOT/c['output_dir'];out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for row,g,s,ids in zip(rows,groups,split,footprints):
        manifest.append(dict(structure_id=row['structure_id'],type=row['type'],label=c['label_order'].index(row['type']),
                             group_id=int(g),split=SPLITS[s],center=row['center'],start=row['start'],end=row['end'],
                             max_window_start_bin=int(ids[0]),max_window_end_bin_exclusive=int((ids[-1]+1)%n),
                             length_bp=row['length_bp'],coordinate_status=row['coordinate_status']))
    write_csv(ROOT/'data/processed/classification_split.csv',manifest)
    totals={s:{t:sum(r['split']==s and r['type']==t for r in manifest) for t in c['label_order']} for s in SPLITS}
    print('Split counts (unique structures):',totals,flush=True)
    print('Groups:',len(set(groups)),'forced training groups:',len(forced),flush=True)
    arrays={key:[] for key in ('oe','depth')};metadata=[];normalization={}
    for sample in data['samples']:
        print('Fitting train-only expected:',sample['sample_id'],flush=True)
        clr=cooler.Cooler(str(ROOT/sample['path']))
        expected,total,opportunities=fit_expected(ROOT/sample['path'],train_bins,10,2*half)
        normalization[sample['sample_id']]={'expected':expected.tolist(),'train_pixel_sum':total,'opportunities':opportunities.tolist()}
        for i,(row,ids) in enumerate(zip(rows,footprints)):
            matrix=aggregate_local(clr,ids,10)
            d=distance(ids[:,None],ids[None,:],n)
            # Per-example missing-data rule, not a statistic fitted on held-out samples.
            good=matrix.sum(axis=0)>0
            good &= ids != n-1 if data['chrom_length']%100 else True
            valid=good[:,None]&good[None,:]&(d>=2)
            oe=np.divide(matrix,expected[d],out=np.zeros_like(matrix),where=expected[d]>0)
            views={key:[] for key in arrays}
            for width in c['window_bins']:
                sl=slice(half-width//2,half+width//2)
                mask=valid[sl,sl]
                views['oe'].append(pool_masked(np.log1p(oe[sl,sl]),mask,c['output_size']))
                views['depth'].append(pool_masked(np.log1p(matrix[sl,sl]*1e6/total),mask,c['output_size']))
            for key in arrays:arrays[key].append(np.stack(views[key]))
            metadata.append(dict(**manifest[i],sample_id=sample['sample_id'],replicate=sample['replicate'],
                                 local_mean_count=float(matrix[valid].mean()),valid_fraction=float(valid.mean())))
        print('Extracted',len(rows),'paired-scale examples for',sample['sample_id'],flush=True)
    arrays={key:np.stack(values) for key,values in arrays.items()}
    training=np.array([m['split']=='train' for m in metadata])
    scalers={key:fit_scale(x[training]) for key,x in arrays.items()}
    arrays={key:apply_scale(x,scalers[key]) for key,x in arrays.items()}
    for s in SPLITS:
        selected=np.array([m['split']==s for m in metadata])
        np.savez_compressed(out/f'{s}.npz',X_oe=arrays['oe'][selected],X_depth=arrays['depth'][selected],
                            y=np.array([m['label'] for m in metadata])[selected],
                            structure_id=np.array([m['structure_id'] for m in metadata])[selected],
                            group_id=np.array([m['group_id'] for m in metadata])[selected],
                            replicate=np.array([m['replicate'] for m in metadata])[selected])
    write_csv(ROOT/'data/processed/classification_examples.csv',metadata)
    report=dict(config=c,counts=totals,groups=int(groups.max()+1),forced_training_groups=forced,
                cross_split_shared_bins=0,guard_bp=c['guard_bins']*100,
                train_bin_count=int(train_bins.sum()),tensor_layout='N, scale(6.4kb/25.6kb), channel(signal/valid_fraction), 64, 64',
                scaler_fit_split='train',scalers=scalers,normalization=normalization,
                train_class_weights=(len(np.where(split==0)[0])/(3*np.bincount([r['label'] for r in manifest if r['split']=='train'],minlength=3))).tolist(),
                coordinate_status='provisional',notes=['Explored windows forced to training; earlier global QC was exploratory exposure, not a pristine external test.',
                'No augmentation has been generated; future augmentation is train-only and must stay inside fixed footprints.',
                '3-class annotation-centered recognition; not a genome-wide detector or unknown-class rejection model.',
                'Circular coarse-bin approximation retained; terminal partial bin masked.',
                'No low-coverage fitted threshold from step 2 is reused. Only local zero-coverage bins and d<2 are masked.'])
    for name in ('configs/dataset.json','configs/data.json','data/processed/structures.csv',c['explored_regions']):
        report.setdefault('input_sha256',{})[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    (ROOT/'reports/classification_dataset.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Dataset ready:',out,flush=True)


if __name__=='__main__':main()
