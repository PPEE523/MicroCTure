"""Step 2: circular distance background, paired local matrices and figures."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/microcture-matplotlib')
import argparse
import csv
import json
from pathlib import Path

import cooler
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]


def distance(a, b, n):
    delta = np.abs(a-b)
    return np.minimum(delta, n-delta)


def aggregate_local(clr, ids, factor):
    """Fetch contiguous pieces, including all cross-origin rectangles, then sum."""
    breaks = np.r_[0, np.where(np.diff(ids) != 1)[0]+1, len(ids)]
    result = np.zeros((len(ids), len(ids)), dtype=float)
    for l, r in zip(breaks[:-1], breaks[1:]):
        for u, v in zip(breaks[:-1], breaks[1:]):
            i0, i1 = int(ids[l]*factor), min(int((ids[r-1]+1)*factor), clr.info['nbins'])
            j0, j1 = int(ids[u]*factor), min(int((ids[v-1]+1)*factor), clr.info['nbins'])
            block = clr.matrix(balance=False, sparse=True)[i0:i1, j0:j1].tocoo()
            np.add.at(result, (l+block.row//factor, u+block.col//factor), block.data)
    return result


def expected_background(path, resolution, max_distance, chunk_size):
    """Two streaming passes: endpoint coverage, then valid-pair distance sums."""
    clr = cooler.Cooler(str(path))
    factor = resolution // clr.binsize
    length = int(clr.chromsizes.iloc[0])
    n = (length+resolution-1)//resolution
    coverage = np.zeros(n)
    nnz, total = clr.info['nnz'], 0
    with h5py.File(path, 'r') as f:
        pixels = f['pixels']
        for offset in range(0, nnz, chunk_size):
            sl = slice(offset, offset+chunk_size)
            a = pixels['bin1_id'][sl]//factor
            b = pixels['bin2_id'][sl]//factor
            counts = pixels['count'][sl].astype(float)
            if not np.isfinite(counts).all() or (counts < 0).any():
                raise ValueError('Invalid contact counts')
            total += int(counts.sum())
            coverage += np.bincount(a, weights=counts, minlength=n)
            coverage += np.bincount(b, weights=counts, minlength=n)
        positive = coverage[coverage > 0]
        threshold = float(np.median(positive)*0.1)
        valid = coverage >= threshold
        if length % resolution:
            valid[-1] = False
        sums = np.zeros(max_distance+1)
        for offset in range(0, nnz, chunk_size):
            sl = slice(offset, offset+chunk_size)
            a = pixels['bin1_id'][sl]//factor
            b = pixels['bin2_id'][sl]//factor
            counts = pixels['count'][sl]
            d = distance(a, b, n)
            keep = valid[a] & valid[b] & (d >= 2) & (d <= max_distance)
            sums += np.bincount(d[keep], weights=counts[keep], minlength=max_distance+1)
    # Every possible valid pair counts, including zero-contact pairs.
    opportunities = np.array([np.count_nonzero(valid & np.roll(valid, -d))
                              for d in range(max_distance+1)])
    expected = np.divide(sums, opportunities, out=np.full_like(sums, np.nan), where=opportunities>0)
    expected[:2] = np.nan
    if total != clr.info['sum']:
        raise ValueError('Full pixel sum differs from metadata')
    return dict(expected=expected, opportunities=opportunities, coverage=coverage,
                valid=valid, threshold=threshold, total=total, n=n)


def select_regions(rows, n, resolution, seed):
    rng = np.random.default_rng(seed)
    regions = []
    occupied = np.zeros(n, dtype=bool)
    for row in rows:
        occupied[int(row['start'])//resolution:(int(row['end'])+resolution-1)//resolution] = True
    for kind in ('OPCID', 'CHIN', 'CHID'):
        group = sorted([r for r in rows if r['type']==kind], key=lambda r: int(r['length_bp']))
        chosen = [group[round((len(group)-1)*q)] for q in (.25, .5, .75)]
        for row in chosen:
            width = int(np.ceil(max(6000, int(row['length_bp'])*1.8)/resolution))
            center = int(float(row['center'])//resolution)
            begin = center-width//2
            regions.append(dict(region_id=row['structure_id'], type=kind, start_bin=begin,
                                width_bins=width, structure_start=int(row['start']),
                                structure_end=int(row['end']), matched_to=''))
            for candidate in rng.permutation(n):
                ids = (np.arange(width)+int(candidate)) % n
                if not occupied[ids].any() and n-1 not in ids:
                    regions.append(dict(region_id='BG_'+row['structure_id'], type='background',
                                        start_bin=int(candidate), width_bins=width,
                                        structure_start=None, structure_end=None,
                                        matched_to=row['structure_id']))
                    occupied[ids] = True
                    break
            else:
                raise ValueError('No unannotated background window of matching length')
    # A real cross-origin example exercises circular stitching; not a negative control.
    regions.append(dict(region_id='origin_wrap', type='origin_check', start_bin=-30,
                        width_bins=60, structure_start=None, structure_end=None, matched_to=''))
    return regions


def normalize(matrix, ids, background):
    d = distance(ids[:, None], ids[None, :], background['n'])
    valid = background['valid'][ids]
    mask = valid[:, None] & valid[None, :] & (d >= 2)
    raw = np.where(mask, matrix, np.nan)
    depth = raw*1e8/background['total']
    oe = np.divide(raw, background['expected'][d], out=np.full_like(raw, np.nan),
                   where=np.isfinite(background['expected'][d]) & (background['expected'][d]>0))
    return raw, depth, oe


def draw(ax, array, region, resolution, limit, mode):
    lo = region['start_bin']*resolution/1000
    hi = lo+region['width_bins']*resolution/1000
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('#c8cdd2')
    shown = np.log1p(array)
    image = ax.imshow(shown, origin='lower', extent=(lo, hi, lo, hi), cmap=cmap,
                      vmin=0, vmax=limit, interpolation='nearest', aspect='equal')
    if region['structure_start'] is not None:
        start = region['structure_start']/1000
        width = (region['structure_end']-region['structure_start'])/1000
        ax.add_patch(Rectangle((start, start), width, width, fill=False, edgecolor='#00e5cc', lw=1.2))
    ax.set_xlabel('Position (kb; unwrapped at origin)')
    ax.set_ylabel('Position (kb)')
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolution', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--chunk-size', type=int, default=2_000_000)
    parser.add_argument('--refresh-background', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT/'configs/data.json').read_text())
    if args.resolution < 10 or args.resolution % 10 or args.chunk_size <= 0:
        parser.error('resolution must be a positive multiple of 10; chunk size must be positive')
    out = ROOT/'results/local_matrices'/f'{args.resolution}bp'
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT/'data/cache/local_matrices'
    cache.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((ROOT/'data/processed/structures.csv').open()))
    n = (config['chrom_length']+args.resolution-1)//args.resolution
    regions = select_regions(rows, n, args.resolution, args.seed)
    max_distance = max(r['width_bins'] for r in regions)
    if max_distance >= n//2:
        parser.error('Windows must be smaller than half the circular genome')
    backgrounds, matrices, summaries = [], {}, []
    for sample in config['samples']:
        path = ROOT/sample['path']
        key = dict(version=1, resolution=args.resolution, max_distance=max_distance,
                   path=str(path), size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
        file = cache/f'{sample["sample_id"]}_{args.resolution}.npz'
        background = None
        if file.exists() and not args.refresh_background:
            with np.load(file) as saved:
                if json.loads(str(saved['key'])) == key:
                    background = {k: saved[k].copy() for k in saved.files if k != 'key'}
        if background is None:
            print(f'{sample["sample_id"]}: streaming coverage and circular expected contacts', flush=True)
            background = expected_background(path, args.resolution, max_distance, args.chunk_size)
            np.savez_compressed(file, **background, key=json.dumps(key))
        background['n'] = int(background['n'])
        backgrounds.append(background)
        print(f'{sample["sample_id"]}: plotting input windows; masked bins={np.count_nonzero(~background["valid"])}', flush=True)
        clr = cooler.Cooler(str(path))
        for region in regions:
            ids = (np.arange(region['width_bins'])+region['start_bin']) % n
            raw = aggregate_local(clr, ids, args.resolution//10)
            if not np.array_equal(raw, raw.T):
                raise ValueError('Asymmetric local matrix')
            values = normalize(raw, ids, background)
            matrices[region['region_id'], sample['sample_id']] = values
            np.savez_compressed(out/f'{region["region_id"]}_{sample["sample_id"]}.npz',
                                bin_ids=ids, raw=values[0], per_100m=values[1], oe=values[2])
    limits = [float(np.quantile(np.concatenate([np.log1p(v[i][np.isfinite(v[i])])
                                               for v in matrices.values()]), .99)) for i in range(3)]
    titles = ['log1p(raw counts)', 'log1p(contacts per 100M)', 'log1p(observed / expected)']
    for region in regions:
        fig, axes = plt.subplots(2, 3, figsize=(13, 8.5), layout='constrained')
        pair = [matrices[region['region_id'], s['sample_id']] for s in config['samples']]
        upper = np.triu(np.ones(pair[0][2].shape, bool), 2)
        use = upper & np.isfinite(pair[0][2]) & np.isfinite(pair[1][2])
        a, b = pair[0][2][use], pair[1][2][use]
        corr = float(np.corrcoef(a,b)[0,1]) if len(a)>2 and a.std()>0 and b.std()>0 else None
        summaries.append(dict(**region, chrom=config['chrom'],
                              window_start_bp_unwrapped=region['start_bin']*args.resolution,
                              window_end_bp_unwrapped=(region['start_bin']+region['width_bins'])*args.resolution,
                              oe_pearson=corr, compared_pixels=int(use.sum()),
                              zero_fraction_rep1=float(np.mean(pair[0][0][np.isfinite(pair[0][0])]==0)),
                              zero_fraction_rep2=float(np.mean(pair[1][0][np.isfinite(pair[1][0])]==0))))
        for j in range(3):
            for i in range(2):
                image = draw(axes[i,j], pair[i][j], region, args.resolution, limits[j], j)
                axes[i,j].set_title(f'rep{i+1} | {titles[j]}')
            fig.colorbar(image, ax=axes[:,j], shrink=.65, label=titles[j])
        corr_label = f'{corr:.3f}' if corr is not None else 'undefined'
        fig.suptitle(f'{region["region_id"]} | {args.resolution} bp | O/E replicate r={corr_label}\n'
                     'Cyan: annotation (provisional coordinates); gray: masked; global color scales', fontsize=13)
        fig.savefig(out/f'{region["region_id"]}.png', dpi=150)
        plt.close(fig)
    for is_background, filename in [(False, 'known_overview'), (True, 'background_overview')]:
        selected = [r for r in regions if (r['type']=='background') == is_background and r['type']!='origin_check']
        fig, axes = plt.subplots(3, 6, figsize=(22, 12), layout='constrained')
        for k, region in enumerate(selected):
            for i, sample in enumerate(config['samples']):
                ax = axes[k//3, (k%3)*2+i]
                image = draw(ax, matrices[region['region_id'],sample['sample_id']][2], region, args.resolution, limits[2], 2)
                ax.set_title(f'{region["region_id"]} / rep{i+1}', fontsize=10)
                ax.tick_params(labelsize=7)
        fig.colorbar(image, ax=axes, shrink=.6, label=titles[2])
        fig.suptitle('Unannotated matched-width controls' if is_background else 'Known structures: paired replicates (O/E)', fontsize=18)
        fig.savefig(out/f'{filename}.png', dpi=140)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for sample, background in zip(config['samples'], backgrounds):
        axes[0].loglog(np.arange(2,max_distance+1)*args.resolution, background['expected'][2:], label=sample['sample_id'])
        axes[1].hist(np.log10(background['coverage'][background['coverage']>0]), bins=80, histtype='step', label=sample['sample_id'])
        axes[1].axvline(np.log10(background['threshold']), ls='--', lw=1)
    axes[0].set(xlabel='Circular distance (bp, binned)', ylabel='Expected raw contacts / valid pair')
    axes[1].set(xlabel='log10(endpoint coverage)', ylabel='Number of bins')
    for ax in axes: ax.legend(fontsize=8)
    fig.savefig(out/'distance_and_coverage.png', dpi=160)
    plt.close(fig)
    with (out/'regions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    report = dict(resolution=args.resolution, seed=args.seed, color_limits_log1p=limits,
                  regions=summaries, masked_bins=[int(np.count_nonzero(~b['valid'])) for b in backgrounds],
                  coverage_thresholds=[float(b['threshold']) for b in backgrounds],
                  total_counts=[int(b['total']) for b in backgrounds],
                  coordinate_status='provisional', circular_distance='min(abs(i-j), n-abs(i-j)); partial terminal bin masked',
                  partial_bin_bp=config['chrom_length']%args.resolution,
                  notes=['O/E is not ICE balancing.', 'Mask d<2 and endpoint coverage below 10% of positive-bin median.',
                         'Expected denominator includes zero-contact valid pairs.',
                         'Backgrounds are annotation-free and width-matched, not verified biological negatives.',
                         'Correlations are descriptive, not a held-out discovery validation.'])
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Wrote {len(regions)} paired panels, 2 galleries, QC plot and numeric matrices to {out}', flush=True)


if __name__ == '__main__':
    main()
