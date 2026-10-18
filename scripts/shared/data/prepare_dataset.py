"""Grouped, spatially disjoint supervised data; all fitted transforms use train only."""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()
import csv
import json
from pathlib import Path
import hashlib

import cooler
import h5py
import numpy as np

from plot_examples import aggregate_local, distance

ROOT = _layout_root
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
    """Retired production entry; retain the pure numerical helpers for review."""
    raise RuntimeError(
        "Legacy dataset generation is retired: retained classification caches and task5 "
        "split/example tables are read-only. For a fresh task1 cache, use "
        "python -B scripts/task1/prepare_task1.py --cache data/task1/cache/next_v2. "
        "Keep the historical task5 splits unchanged.")


if __name__ == '__main__':
    import sys
    try:
        main()
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
