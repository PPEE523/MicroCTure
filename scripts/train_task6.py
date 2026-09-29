"""Transductive 3D graph decoding with training-edge-only MDS baselines."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/microcture-mpl')
import copy
import json
import numpy as np
import torch
from torch import nn
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optional_common import ROOT, read_csv, write_csv, write_json, sha256, correlation


def pair_partition(n, seed, train_fraction=.7, validation_fraction=.15):
    a, b = np.triu_indices(n, 1)
    separation = np.minimum(b-a, n-(b-a))
    a, b = a[separation >= 2], b[separation >= 2]
    random = np.random.default_rng(seed).random(len(a))
    roles = np.where(random < train_fraction, 0, np.where(random < train_fraction+validation_fraction, 1, 2))
    return a, b, roles


def classical_mds(distances):
    squared = distances**2
    gram = -.5*(squared-squared.mean(0)[None, :]-squared.mean(1)[:, None]+squared.mean())
    values, vectors = np.linalg.eigh(gram)
    return vectors[:, -3:]*np.sqrt(np.maximum(values[-3:], 0))[None, :]


def fit_geometry_inputs(counts, a, b, roles, cfg):
    n = len(counts)
    train = roles == 0
    separation = np.minimum(b-a, n-(b-a))
    contact = counts[a, b]
    raw_distance = (contact+cfg['pseudocount'])**(-cfg['contact_exponent'])
    scale = float(np.median(raw_distance[train]))
    target = raw_distance/scale
    background = np.zeros(n//2+1)
    log_background = np.zeros_like(background)
    for d in range(2, len(background)):
        use = train & (separation == d)
        background[d] = np.median(target[use])
        log_background[d] = np.mean(np.log1p(contact[use]))
    background[1] = background[2]*.8
    log_background[1] = log_background[2]
    delta = np.abs(np.arange(n)[:, None]-np.arange(n)[None, :])
    delta = np.minimum(delta, n-delta)
    filled = background[delta]
    filled[a[train], b[train]] = target[train]
    filled[b[train], a[train]] = target[train]
    np.fill_diagonal(filled, 0)
    adjacency = np.eye(n)
    weights = np.log1p(contact[train])
    adjacency[a[train], b[train]] = weights
    adjacency[b[train], a[train]] = weights
    degree = adjacency.sum(1)**-.5
    adjacency *= degree[:, None]*degree[None, :]
    return target, filled, adjacency, dict(distance_scale=scale, bond_target=float(background[1]),
                                         log_contact_background=log_background.tolist())


class GraphDecoder(nn.Module):
    def __init__(self, initial, hidden=32):
        super().__init__()
        n = len(initial)
        angle = torch.arange(n)*2*np.pi/n
        features = torch.cat([initial, torch.stack([torch.sin(angle), torch.cos(angle),
                              torch.sin(2*angle), torch.cos(2*angle)], 1)], 1)
        self.register_buffer('features', features)
        self.register_buffer('initial', initial)
        self.embedding = nn.Parameter(torch.randn(n, hidden)*.01)
        self.input_layer = nn.Linear(7, hidden)
        self.graph_layer = nn.Linear(hidden, hidden)
        self.output_layer = nn.Linear(hidden, 3)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, adjacency):
        h = torch.tanh(self.input_layer(self.features)+self.embedding)
        h = torch.tanh(self.graph_layer(adjacency@h)+h)
        xyz = self.initial+self.output_layer(h)
        return xyz-xyz.mean(0)


def distance_loss(xyz, a, b, target):
    distance = torch.linalg.vector_norm(xyz[a]-xyz[b], dim=1).clamp_min(1e-6)
    return ((torch.log(distance)-torch.log(target))**2).mean()


def evaluate(xyz, a, b, use, counts, target, background, cfg, fitted):
    distances = np.linalg.norm(xyz[a]-xyz[b], axis=1)
    predicted = np.maximum((np.maximum(distances, 1e-6)*fitted['distance_scale'])**(-1/cfg['contact_exponent'])-cfg['pseudocount'], 0)
    actual = counts[a, b]
    separation = np.minimum(b-a, len(xyz)-(b-a))
    residual_actual = np.log1p(actual)-background[separation]
    residual_predicted = np.log1p(predicted)-background[separation]
    stratified, weights = [], []
    for d in np.unique(separation[use]):
        selected = use & (separation == d)
        value = correlation(np.log1p(predicted[selected]), np.log1p(actual[selected]))
        if value is not None:
            stratified.append(value); weights.append(selected.sum())
    return dict(pairs=int(use.sum()),
                log_distance_rmse=float(np.sqrt(np.mean((np.log(np.maximum(distances[use], 1e-6))-np.log(target[use]))**2))),
                contact_log_pearson=correlation(np.log1p(predicted[use]), np.log1p(actual[use])),
                contact_spearman=float(spearmanr(predicted[use], actual[use]).statistic),
                distance_adjusted_contact_pearson=correlation(residual_predicted[use], residual_actual[use]),
                within_distance_pearson=float(np.average(stratified, weights=weights)) if weights else None)


def plot_and_html(out, geometries, metadata, matrices, cfg):
    schematic = json.loads((ROOT/'configs/macrodomain_schematic.json').read_text())
    n, length = len(matrices[0]), metadata['chromosome_length']
    positions = np.minimum((np.arange(n)+.5)*cfg['resolution_bp'], length-1)
    minutes = positions/length*100
    labels = np.empty(n, object)
    colors = np.empty(n, object)
    for domain in schematic['domains']:
        start, end = domain['start_min'], domain['end_min']
        use = (minutes >= start) & (minutes < end) if start < end else (minutes >= start) | (minutes < end)
        labels[use], colors[use] = domain['name'], domain['color']
    genes = {r['gene']: r for r in read_csv(ROOT/'data/processed/genes.csv')}
    markers = []
    for marker in schematic['markers']:
        position = ((int(genes[marker['gene']]['start'])+int(genes[marker['gene']]['end']))/2
                    if 'gene' in marker else marker['fraction']*length)
        markers.append(dict(name=marker['name'], bin=int(position//cfg['resolution_bp'])))
    primary = ['mds', 'geometry_42', 'graph_42']
    fig = plt.figure(figsize=(15, 5))
    for i, name in enumerate(primary):
        xyz = geometries[name]
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ring = np.vstack([xyz, xyz[0]])
        ax.plot(*ring.T, lw=.5, color='gray', alpha=.5)
        ax.scatter(*xyz.T, c=colors, s=4)
        for marker in markers:
            ax.text(*xyz[marker['bin']], marker['name'], fontsize=7)
        limits = np.max(np.ptp(xyz, axis=0))/2
        midpoint = (xyz.max(0)+xyz.min(0))/2
        ax.set(xlim=(midpoint[0]-limits, midpoint[0]+limits), ylim=(midpoint[1]-limits, midpoint[1]+limits),
               zlim=(midpoint[2]-limits, midpoint[2]+limits), title=name)
        ax.set_box_aspect([1, 1, 1])
    fig.suptitle('5 kb ensemble-contact embedding; schematic domains; arbitrary distance units')
    fig.tight_layout(); fig.savefig(out/'structures.png', dpi=150); plt.close(fig)
    payload = dict(models={name: xyz.round(6).tolist() for name, xyz in geometries.items()},
                   colors=colors.tolist(), labels=labels.tolist(), positions=positions.astype(int).tolist(), markers=markers)
    write_json(out/'viewer_data.json', payload)
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MicroCTure — 三维接触嵌入</title>
<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;background:#fafafa;color:#17233b}canvas{background:white;width:100%;border:1px solid #ddd}select,button{padding:8px}p{line-height:1.6}</style>
<h1>全基因组 5 kb 三维接触嵌入</h1><p>拖动旋转，滚轮缩放。颜色是文献启发的宏观结构域示意；环状连接是建模先验。坐标为相对单位，不代表唯一的单细胞构象。</p>
<label>模型 <select id="model"></select></label> <button id="reset">重置视角</button><span id="legend"></span>
<canvas id="scene" width="1100" height="720"></canvas><p>绿色 Ori；红色 Right；青色 Ter；蓝色 Left；灰色非结构区。mioC 仅标记 oriC 邻域；Ter 中点是示意定位。</p>
<script>const data=PAYLOAD;const canvas=document.querySelector('#scene'),ctx=canvas.getContext('2d'),select=document.querySelector('#model');
Object.keys(data.models).forEach(k=>select.add(new Option(k,k)));select.value='graph_42';let rx=.4,ry=.7,zoom=230,drag=false,last;
function draw(){const pts=data.models[select.value],scale=Math.max(...pts.map(p=>Math.hypot(...p)));const view=pts.map(p=>{let[x,y,z]=p.map(v=>v/scale);let a=x*Math.cos(ry)+z*Math.sin(ry),b=-x*Math.sin(ry)+z*Math.cos(ry);return [550+zoom*a,360+zoom*(y*Math.cos(rx)-b*Math.sin(rx)),y*Math.sin(rx)+b*Math.cos(rx)]});ctx.clearRect(0,0,1100,720);ctx.strokeStyle='#cbd5e1';ctx.beginPath();view.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.stroke();view.map((p,i)=>({p,i})).sort((a,b)=>a.p[2]-b.p[2]).forEach(({p,i})=>{ctx.fillStyle=data.colors[i];ctx.beginPath();ctx.arc(p[0],p[1],2.6,0,Math.PI*2);ctx.fill()});ctx.font='15px system-ui';ctx.fillStyle='#111827';data.markers.forEach(m=>{const p=view[m.bin];ctx.fillText(m.name,p[0]+6,p[1]-6)})}
canvas.onpointerdown=e=>{drag=true;last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointerup=()=>drag=false;canvas.onpointermove=e=>{if(drag){ry+=(e.clientX-last[0])*.008;rx+=(e.clientY-last[1])*.008;last=[e.clientX,e.clientY];draw()}};canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(70,Math.min(900,zoom*Math.exp(-e.deltaY*.001)));draw()};select.onchange=draw;document.querySelector('#reset').onclick=()=>{rx=.4;ry=.7;zoom=230;draw()};draw();</script></html>'''
    (out/'index.html').write_text(html.replace('PAYLOAD', json.dumps(payload)), encoding='utf-8')
    # Annotation is not used to fit coordinates. Domain compaction is exploratory.
    rows = []
    for name, xyz in geometries.items():
        for domain in schematic['domains']:
            selected = labels == domain['name']
            points = xyz[selected]
            rg = float(np.sqrt(np.mean(np.sum((points-points.mean(0))**2, axis=1))))
            null = []
            for shift in range(1, n):
                rotated = xyz[np.roll(selected, shift)]
                null.append(float(np.sqrt(np.mean(np.sum((rotated-rotated.mean(0))**2, axis=1)))))
            rows.append(dict(model=name, domain=domain['name'], bins=len(points),
                             radius_of_gyration=rg, matched_arc_median_rg=float(np.median(null)),
                             rg_ratio_to_matched_arc=float(rg/np.median(null)),
                             compactness_rank_fraction=float(np.mean(np.array(null) <= rg))))
    write_csv(out/'domain_geometry.csv', rows)


def main():
    cfg = json.loads((ROOT/'configs/optional_tasks.json').read_text())['task6']
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    out = ROOT/cfg['output_dir']; out.mkdir(parents=True, exist_ok=True)
    signatures = {name: sha256(ROOT/name) for name in ['configs/optional_tasks.json', 'configs/macrodomain_schematic.json',
                  'scripts/train_task6.py', 'scripts/prepare_task6.py', 'scripts/optional_common.py',
                  'data/cache/task6/WT_37C_rep1.npz', 'data/cache/task6/WT_37C_rep2.npz', 'data/processed/genes.csv']}
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text())['input_sha256'] != signatures:
        raise ValueError('Inputs changed: use a new output_dir, preserving previous fits')
    matrices, totals = [], []
    for replicate in [1, 2]:
        with np.load(ROOT/f'data/cache/task6/WT_37C_rep{replicate}.npz') as z:
            matrix, length, total = z['counts'].copy(), int(z['chromosome_length']), int(z['total'])
        sizes = np.minimum(cfg['resolution_bp'], length-np.arange(len(matrix))*cfg['resolution_bp'])/cfg['resolution_bp']
        matrix /= sizes[:, None]*sizes[None, :]
        matrices.append(matrix); totals.append(total)
    matrices[1] *= totals[0]/totals[1]
    n = len(matrices[0])
    a, b, roles = pair_partition(n, cfg['pair_split_seed'], cfg['train_fraction'], cfg['validation_fraction'])
    target, filled, adjacency, fitted = fit_geometry_inputs(matrices[0], a, b, roles, cfg)
    initial = classical_mds(filled).astype('float32')
    # Training-only scalar correction gives MDS its best log-distance scale.
    d0 = np.linalg.norm(initial[a]-initial[b], axis=1)
    initial *= np.exp(np.mean(np.log(target[roles == 0])-np.log(np.maximum(d0[roles == 0], 1e-6))))
    geometries = dict(mds=initial)
    write_json(out/'manifest.json', dict(config=cfg, input_sha256=signatures, fitted=fitted, bins=n,
               chromosome_length=length, pair_counts={name: int((roles == i).sum()) for i, name in enumerate(['train', 'validation', 'test'])}))
    np.savez_compressed(out/'evaluation_inputs.npz', counts_rep1=matrices[0], counts_rep2=matrices[1],
                        a=a, b=b, roles=roles, target=target)
    at, bt = torch.from_numpy(a), torch.from_numpy(b)
    tt = torch.from_numpy(target.astype('float32'))
    adj = torch.from_numpy(adjacency.astype('float32'))
    train, validation = torch.from_numpy(roles == 0), torch.from_numpy(roles == 1)
    runs = []
    for family in ['geometry', 'graph']:
        for seed in cfg['seeds']:
            name = f'{family}_{seed}'; torch.manual_seed(seed)
            if family == 'geometry':
                model = nn.Embedding(n, 3)
                with torch.no_grad(): model.weight.copy_(torch.from_numpy(initial)+torch.randn(n, 3)*.001)
                coordinates = lambda: model.weight-model.weight.mean(0)
            else:
                model = GraphDecoder(torch.from_numpy(initial), cfg['hidden_size'])
                coordinates = lambda: model(adj)
            checkpoint = out/f'{name}.pt'
            if not checkpoint.exists():
                optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
                best, stale, history, best_epoch, best_state = float('inf'), 0, [], None, None
                for epoch in range(1, cfg['epochs']+1):
                    optimizer.zero_grad(); xyz = coordinates()
                    fit = distance_loss(xyz, at[train], bt[train], tt[train])
                    bonds = torch.linalg.vector_norm(xyz-torch.roll(xyz, 1, 0), dim=1)
                    bond = ((bonds-fitted['bond_target'])**2).mean()
                    loss = fit+cfg['bond_weight']*bond
                    loss.backward(); optimizer.step()
                    with torch.no_grad(): val = float(distance_loss(coordinates(), at[validation], bt[validation], tt[validation]))
                    history.append(dict(epoch=epoch, train_loss=float(fit.detach()), bond_loss=float(bond.detach()), validation_loss=val))
                    if val < best-1e-7:
                        best, stale, best_epoch, best_state = val, 0, epoch, copy.deepcopy(model.state_dict())
                    else: stale += 1
                    if epoch % 100 == 0: print(name, epoch, 'validation', round(val, 6), flush=True)
                    if stale >= cfg['patience']: break
                torch.save(dict(state_dict=best_state, family=family, seed=seed, best_epoch=best_epoch,
                                validation_loss=best, config=cfg), checkpoint)
                write_json(out/f'{name}_history.json', history)
            saved = torch.load(checkpoint, weights_only=False); model.load_state_dict(saved['state_dict'])
            with torch.no_grad(): geometries[name] = coordinates().numpy().copy()
            runs.append(dict(model=name, best_epoch=saved['best_epoch'], validation_loss=saved['validation_loss']))
    metrics = []
    for name, xyz in geometries.items():
        write_csv(out/f'{name}_coordinates.csv', [dict(bin=i, chrom='NC_000913.3', start=i*cfg['resolution_bp'],
                  end=min((i+1)*cfg['resolution_bp'], length), x=float(x), y=float(y), z=float(z))
                  for i, (x, y, z) in enumerate(xyz)])
        distances = squareform(pdist(xyz))
        predicted = np.maximum((np.maximum(distances, 1e-6)*fitted['distance_scale'])**(-1/cfg['contact_exponent'])-cfg['pseudocount'], 0)
        np.fill_diagonal(predicted, 0)
        np.savez_compressed(out/f'{name}_maps.npz', distances=distances, predicted_contacts=predicted)
        for rep in [1, 2]:
            actual = matrices[rep-1]
            rep_target = (actual[a, b]+cfg['pseudocount'])**(-cfg['contact_exponent'])/fitted['distance_scale']
            for role, label in enumerate(['train', 'validation', 'test']):
                metrics.append(dict(model=name, replicate=rep, partition=label,
                       **evaluate(xyz, a, b, roles == role, actual, rep_target,
                                  np.array(fitted['log_contact_background']), cfg, fitted)))
    write_csv(out/'metrics.csv', metrics); write_csv(out/'training_summary.csv', runs)
    # Distance agreement is invariant to rotations, translations and reflections.
    concordance = []
    for family in ['geometry', 'graph']:
        for j, seed in enumerate(cfg['seeds']):
            for other in cfg['seeds'][:j]:
                concordance.append(dict(family=family, seed_a=other, seed_b=seed,
                     distance_correlation=correlation(pdist(geometries[f'{family}_{other}']), pdist(geometries[f'{family}_{seed}']))))
    write_csv(out/'seed_stability.csv', concordance)
    plot_and_html(out, geometries, dict(chromosome_length=length), matrices, cfg)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    vmax = np.quantile(np.log1p(matrices[0][a, b]), .99)
    for ax, name in zip(axes, ['observed rep1', 'mds', 'graph_42']):
        matrix = matrices[0] if name == 'observed rep1' else np.load(out/f'{name}_maps.npz')['predicted_contacts']
        ax.imshow(np.log1p(matrix), vmin=0, vmax=vmax, cmap='magma', origin='lower')
        ax.set_title(name); ax.set_xlabel('5 kb genomic bin')
    fig.tight_layout(); fig.savefig(out/'contact_maps.png', dpi=140); plt.close(fig)
    print('Task 6 complete:', n, 'coordinates per model', flush=True)


if __name__ == '__main__':
    main()
