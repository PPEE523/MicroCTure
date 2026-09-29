"""Render a portfolio cover from measured results, without retraining models."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/microcture-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-data', action='store_true', help='Export fixed measured example from local experiment outputs')
    args = parser.parse_args()
    assets = ROOT/'site/assets'; assets.mkdir(parents=True, exist_ok=True)
    path = assets/'matrix_preview.json'
    if args.refresh_data:
        source = ROOT/'results/task5/bicubic_reconstruction.npz'
        with np.load(source) as base, np.load(ROOT/'results/task5/residual_cnn_42_reconstruction.npz') as cnn:
            with (ROOT/'results/task5/windows.csv').open() as handle: rows = list(csv.DictReader(handle))
            i = next(i for i, ix in enumerate(base['window_indices']) if rows[ix]['type'] == 'CHID' and rows[ix]['replicate'] == '1')
            row = rows[base['window_indices'][i]]
            payload = dict(structure_id=row['structure_id'], replicate=1, resolution_bp=200, window_bp=25600,
                           model='residual_cnn_42', selection='First CHID test example; not selected by reconstruction quality.',
                           normalization='Clipped log1p contact intensity; common training-fitted [0,1] scale.',
                           source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                          [source, ROOT/'results/task5/residual_cnn_42_reconstruction.npz']},
                           target=base['target'][i].round(6).tolist(), bicubic=base['prediction'][i].round(6).tolist(),
                           cnn=cnn['prediction'][i].round(6).tolist(), mask=base['mask'][i].astype(int).tolist())
        path.write_text(json.dumps(payload, separators=(',', ':'))+'\n')
    data = json.loads(path.read_text())
    with (ROOT/'showcase/results/task6/graph_42_coordinates.csv').open() as handle:
        points = np.array([[float(r[k]) for k in ['x', 'y', 'z']] for r in csv.DictReader(handle)])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': '#edf2fa', 'axes.facecolor': '#101c30'})
    fig = plt.figure(figsize=(15, 7), facecolor='#0a1220')
    fig.text(.045, .9, 'MicroCTure', fontsize=43, weight='bold', color='#f5f8ff')
    fig.text(.047, .835, 'MACHINE LEARNING FOR GENOME CONTACT MAPS', fontsize=11, color='#76e3d2', fontfamily='monospace')
    fig.text(.047, .76, 'Sparse data. Reproducible models. Spatial insight.', fontsize=17, color='#adbcd1')
    for x, title, detail in [(.047, '507M', 'WT sparse records'), (.265, '36', 'training / fitting runs'),
                              (.49, '4×', 'synthetic super-resolution'), (.745, '929', 'whole-genome 3D nodes')]:
        fig.text(x, .625, title, fontsize=31, weight='bold', color='#f5f8ff')
        fig.text(x, .582, detail, fontsize=10, color='#8b9fbb')
    mask = np.array(data['mask'], bool)
    for pos, name, label in [(.045, 'target', '01 / MEASURED CONTACTS'), (.363, 'cnn', '02 / CNN RECONSTRUCTION')]:
        ax = fig.add_axes([pos, .1, .27, .405])
        ax.imshow(np.where(mask, data[name], np.nan), cmap='magma', vmin=0, vmax=1, origin='lower')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        fig.text(pos, .067, label, color='#afc0d7', fontsize=9, fontfamily='monospace')
    ax = fig.add_axes([.68, .1, .275, .405], projection='3d', facecolor='#0a1220')
    ax.scatter(*points.T, c=np.arange(len(points)), cmap='viridis', s=3, alpha=.85)
    ax.set_axis_off(); ax.view_init(elev=24, azim=45)
    fig.text(.68, .067, '03 / GRAPH-BASED 3D EMBEDDING', color='#afc0d7', fontsize=9, fontfamily='monospace')
    fig.savefig(assets/'overview.png', dpi=145, facecolor=fig.get_facecolor()); plt.close(fig)
    print('Portfolio cover built from measured matrices and saved coordinates')


if __name__ == '__main__':
    main()
