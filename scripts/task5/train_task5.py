"""Spatially separated 800-to-200 bp synthetic super-resolution experiment."""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()
import copy
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import uniform_filter
from optional_common import ROOT, read_csv, write_csv, write_json, sha256, correlation
from optional_common import experiment_arguments, input_path, signatures as file_signatures, runtime_paths, temporary_plotting
from contact_band import extract
from spatial_tools import assert_disjoint


def pool(values, mask, factor):
    n = len(values)//factor
    shape = (n, factor, n, factor)
    count = mask.reshape(shape).sum((1, 3))
    sums = np.where(mask, values, 0).reshape(shape).sum((1, 3))
    return np.divide(sums, count, out=np.zeros_like(sums), where=count > 0), count/(factor*factor)


def prepare(cfg, *, data_root=ROOT, split_file=None, band_cache_dir=None):
    rows = read_csv(split_file or input_path(data_root,'data/processed/classification_split.csv'))
    n, width = 46417, cfg['window_source_bins']
    starts = [int(float(r['center'])//100)-width//2 for r in rows]
    footprints = [(s+np.arange(width)) % n for s in starts]
    roles = np.full(n, -1)
    assignment = np.array([['train', 'validation', 'test'].index(r['split']) for r in rows])
    assert_disjoint(footprints, assignment, n, 10)
    for ids, role in zip(footprints, assignment):
        roles[ids] = role
    raw_targets, masks, low_values, low_masks, metadata = [], [], [], [], []
    for replicate in [1, 2]:
        cache_file = band_cache_dir/f'WT_37C_rep{replicate}.npz' if band_cache_dir is not None else input_path(data_root,f'data/cache/discovery/WT_37C_rep{replicate}.npz')
        with np.load(cache_file, allow_pickle=False) as z:
            band = {key: z[key].copy() for key in z.files if key != 'signature'}
        # Partition-internal coverage; only train endpoints fit the QC threshold.
        coverage = np.zeros(n)
        for d in range(2, band['band'].shape[1]):
            counts = np.where(roles == np.roll(roles, -d), band['band'][:, d], 0)
            coverage += counts+np.roll(counts, d)
        coverage *= 1e6/float(band['total'])
        if replicate == 1:
            threshold = float(np.median(coverage[(roles == 0) & (coverage > 0)])*.1)
        band['valid'] = coverage >= threshold
        band['valid'][-1] = False
        for r, start in zip(rows, starts):
            counts, _, mask = extract(band, start, width)
            target, coverage = pool(counts, mask, 2)
            valid = coverage == 1
            target *= 1e6/float(band['total'])
            low, low_valid = pool(target, valid, cfg['scale_factor'])
            raw_targets.append(target)
            masks.append(valid)
            low_values.append(low)
            low_masks.append(low_valid)
            metadata.append(dict(structure_id=r['structure_id'], type=r['type'], group_id=r['group_id'],
                                 split=r['split'], replicate=replicate, start_bin=start,
                                 start=int(r['start']), end=int(r['end']), coordinate_status='provisional'))
    raw_targets, masks = np.array(raw_targets), np.array(masks)
    train = np.array([r['split'] == 'train' and r['replicate'] == 1 for r in metadata])
    scale = float(np.quantile(np.log1p(raw_targets[train])[masks[train]], cfg['normalization_quantile']))
    targets = np.clip(np.log1p(raw_targets)/scale, 0, 1).astype('float32')
    targets[~masks] = 0
    low = np.clip(np.log1p(np.array(low_values))/scale, 0, 1)
    inputs = np.stack([low, np.array(low_masks)], axis=1).astype('float32')
    clipping = {f'{split}_rep{rep}': float(np.mean(np.log1p(raw_targets[ix])[masks[ix]] > scale))
                for split in ['train', 'validation', 'test'] for rep in [1, 2]
                if (ix := np.array([r['split'] == split and r['replicate'] == rep for r in metadata])).any()}
    return inputs, targets[:, None], masks[:, None], metadata, dict(log_scale=scale, clipping_fraction=clipping)


class SuperResolutionCNN(nn.Module):
    def __init__(self, channels=16, residual=True, factor=4):
        super().__init__()
        self.residual, self.factor = residual, factor
        self.net = nn.Sequential(nn.Conv2d(2, channels, 5, padding=2), nn.ReLU(),
                                 nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(channels, 1, 3, padding=1))
        if residual:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        up = F.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=False)
        y = self.net(up)
        if self.residual:
            y = y+up[:, :1]
        return (y+y.transpose(-1, -2))/2


def masked_loss(prediction, target, mask):
    # Every window has equal weight, regardless of valid-pixel count.
    return (((prediction-target)**2*mask).sum((1, 2, 3))/mask.sum((1, 2, 3)).clamp_min(1)).mean()


def image_metrics(prediction, target, mask, window=7):
    valid = mask & np.triu(np.ones_like(mask, bool), 2)
    mse = float(np.mean((prediction[valid]-target[valid])**2))
    ux, uy = uniform_filter(prediction, window), uniform_filter(target, window)
    vx = np.maximum(uniform_filter(prediction**2, window)-ux**2, 0)
    vy = np.maximum(uniform_filter(target**2, window)-uy**2, 0)
    cov = uniform_filter(prediction*target, window)-ux*uy
    ssim = ((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux**2+uy**2+.01**2)*(vx+vy+.03**2))
    interior = uniform_filter(mask.astype(float), window, mode='constant') > 1-1e-8
    use = valid & interior
    return dict(mse=mse, psnr=float(-10*np.log10(max(mse, 1e-12))),
                ssim=float(ssim[use].mean()) if use.any() else None,
                shape_correlation=correlation(prediction[valid], target[valid]),
                evaluated_pixels=int(valid.sum()), ssim_pixels=int(use.sum()))


def profile(matrix, mask, flank=8):
    result = np.full(len(matrix), np.nan)
    for cut in range(flank, len(matrix)-flank):
        values = matrix[cut-flank:cut, cut:cut+flank]
        keep = mask[cut-flank:cut, cut:cut+flank]
        if keep.sum() >= flank*flank//2:
            result[cut] = values[keep].mean()
    return result


def structure_metrics(prediction, target, mask, row, cfg):
    p, t = [profile(x, mask, cfg['profile_flank_bins']) for x in [prediction, target]]
    shifts = []
    for boundary in [row['start'], row['end']]:
        center = round((boundary-row['start_bin']*100)/cfg['target_resolution_bp'])
        ids = np.arange(max(0, center-cfg['boundary_radius_bins']),
                        min(len(p), center+cfg['boundary_radius_bins']+1))
        ids = ids[np.isfinite(p[ids]) & np.isfinite(t[ids])]
        if len(ids):
            shifts.append(abs(int(ids[np.argmin(p[ids])])-int(ids[np.argmin(t[ids])]))*cfg['target_resolution_bp'])
    return dict(profile_correlation=correlation(p, t), boundary_shift_bp=float(np.mean(shifts)) if shifts else None,
                evaluated_boundaries=len(shifts))


def run_experiment(args):
    cfg = json.loads(args.config.read_text())['task5']
    source_names = ['scripts/train_task5.py', 'scripts/optional_common.py', 'scripts/contact_band.py', 'scripts/spatial_tools.py']
    cache_paths = [args.band_cache_dir.resolve()/f'WT_37C_rep{rep}.npz' if args.band_cache_dir else input_path(args.data_root,f'data/cache/discovery/WT_37C_rep{rep}.npz') for rep in [1,2]]
    split_file = args.split_file.resolve() if args.split_file else input_path(args.data_root,'data/processed/classification_split.csv')
    signature = file_signatures([args.config]+[_layout_resolve(ROOT/name) for name in source_names], [split_file]+cache_paths)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    inputs, targets, masks, metadata, normalization = prepare(cfg, data_root=args.data_root, split_file=split_file,
                                                          band_cache_dir=args.band_cache_dir.resolve() if args.band_cache_dir else None)
    write_csv(out/'windows.csv', metadata)
    train = np.array([i for i, r in enumerate(metadata) if r['split'] == 'train' and r['replicate'] == 1])
    validation = np.array([i for i, r in enumerate(metadata) if r['split'] == 'validation' and r['replicate'] == 1])
    test = np.array([i for i, r in enumerate(metadata) if r['split'] == 'test'])
    write_json(out/'manifest.json', dict(config=cfg, runtime=runtime_paths(args), input_sha256=signature, normalization=normalization,
               train_windows=len(train), validation_windows=len(validation), test_windows=len(test),
               interpretation='Synthetic bin aggregation; targets are noisy measurements, not latent truth.'))
    x, y, m = map(torch.from_numpy, [inputs, targets, masks.astype('float32')])
    predictions, runs = {}, []
    for mode in ['bilinear', 'bicubic']:
        predictions[mode] = F.interpolate(x[test, :1], scale_factor=cfg['scale_factor'], mode=mode,
                                         align_corners=False).clamp(0, 1).numpy()[:, 0]
    for family in cfg['models']:
        for seed in cfg['seeds']:
            name = f'{family}_{seed}'
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = SuperResolutionCNN(cfg['channels'], family == 'residual_cnn', cfg['scale_factor'])
            checkpoint = out/f'{name}.pt'
            if not checkpoint.exists():
                optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
                history, best, stale, best_state, best_epoch = [], float('inf'), 0, None, None
                for epoch in range(1, cfg['epochs']+1):
                    losses = []
                    model.train()
                    for batch in np.array_split(rng.permutation(train), int(np.ceil(len(train)/cfg['batch_size']))):
                        optimizer.zero_grad()
                        loss = masked_loss(model(x[batch]), y[batch], m[batch])
                        loss.backward()
                        optimizer.step()
                        losses.append(float(loss.detach()))
                    model.eval()
                    with torch.no_grad():
                        val = float(masked_loss(model(x[validation]), y[validation], m[validation]))
                    history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), validation_loss=val))
                    if val < best-1e-7:
                        best, stale, best_state, best_epoch = val, 0, copy.deepcopy(model.state_dict()), epoch
                    else:
                        stale += 1
                    if epoch % 5 == 0:
                        print(name, epoch, 'validation', round(val, 6), flush=True)
                    if stale >= cfg['patience']:
                        break
                torch.save(dict(state_dict=best_state, best_epoch=best_epoch, validation_loss=best,
                                config=cfg, family=family, seed=seed), checkpoint)
                write_json(out/f'{name}_history.json', history)
            saved = torch.load(checkpoint, weights_only=False)
            model.load_state_dict(saved['state_dict'])
            model.eval()
            with torch.no_grad():
                pred = np.concatenate([model(x[batch]).clamp(0, 1).numpy()[:, 0]
                                       for batch in np.array_split(test, 8)])
            predictions[name] = pred
            runs.append(dict(model=name, best_epoch=saved['best_epoch'], validation_loss=saved['validation_loss']))
    metrics = []
    for name, pred in predictions.items():
        for j, i in enumerate(test):
            row = dict(model=name, **metadata[i], **image_metrics(pred[j], targets[i, 0], masks[i, 0], cfg['ssim_window']))
            row.update(structure_metrics(pred[j], targets[i, 0], masks[i, 0], metadata[i], cfg))
            metrics.append(row)
        np.savez_compressed(out/f'{name}_reconstruction.npz', prediction=pred,
                            contact_per_million=np.expm1(pred*normalization['log_scale']),
                            target=targets[test, 0], mask=masks[test, 0], window_indices=test,
                            resolution_bp=cfg['target_resolution_bp'])
    write_csv(out/'metrics.csv', metrics)
    write_csv(out/'training_summary.csv', runs)
    summary = []
    for name in predictions:
        for replicate in [1, 2]:
            for kind in ['all', 'OPCID', 'CHIN', 'CHID']:
                rows = [r for r in metrics if r['model'] == name and r['replicate'] == replicate and (kind == 'all' or r['type'] == kind)]
                summary.append(dict(model=name, replicate=replicate, type=kind, n=len(rows), **{
                    metric: float(np.mean([r[metric] for r in rows if r[metric] is not None]))
                    for metric in ['psnr', 'ssim', 'shape_correlation', 'profile_correlation', 'boundary_shift_bp']}))
    write_csv(out/'summary.csv', summary)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, metric in zip(axes, ['psnr', 'ssim']):
        for rep, shift in [(1, -.18), (2, .18)]:
            vals = [r[metric] for r in summary if r['replicate'] == rep and r['type'] == 'all']
            ax.bar(np.arange(len(vals))+shift, vals, .36, label=f'rep{rep}')
        ax.set_xticks(range(len(predictions)), predictions, rotation=35, ha='right')
        ax.set_ylabel(metric.upper())
        ax.legend()
    fig.tight_layout(); fig.savefig(out/'comparison.png', dpi=140); plt.close(fig)
    # First fixed test example per class, no cherry-picking by model performance.
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for line, kind in enumerate(['OPCID', 'CHIN', 'CHID']):
        j = next(j for j, i in enumerate(test) if metadata[i]['type'] == kind and metadata[i]['replicate'] == 1)
        for ax, name in zip(axes[line], ['target', 'bilinear', 'residual_cnn_42', 'plain_cnn_42']):
            matrix = targets[test[j], 0] if name == 'target' else predictions[name][j]
            ax.imshow(np.where(masks[test[j], 0], matrix, np.nan), vmin=0, vmax=1, cmap='magma', origin='lower')
            ax.set_title(f'{metadata[test[j]]["structure_id"]}: {name}', fontsize=9)
    fig.suptitle('Fixed test examples; 25.6 kb window, 200 bp pixels; shared [0,1] scale')
    fig.tight_layout(); fig.savefig(out/'examples.png', dpi=140); plt.close(fig)
    print('Task 5 complete:', len(metrics), 'window-method evaluations', flush=True)


def main(argv=None):
    args = experiment_arguments(__doc__, 'configs/optional_tasks.json', argv,
                                extra_paths=[('--split-file', 'Read-only frozen split CSV; never rebuilds a classification split'),
                                             ('--band-cache-dir','Read-only directory of existing WT_37C_rep1/2.npz bands')])
    if args is None:
        return
    with temporary_plotting() as pyplot:
        global plt
        plt = pyplot
        run_experiment(args)


if __name__ == '__main__':
    main()
