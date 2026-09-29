"""Post-training diagnostics; no test-driven refitting or model selection."""
import json
import numpy as np
from optional_common import ROOT, write_csv, write_json, correlation
from train_task5 import prepare, image_metrics


def main():
    cfg = json.loads((ROOT/'configs/optional_tasks.json').read_text())['task5']
    out = ROOT/cfg['output_dir']
    _, target, masks, metadata, _ = prepare(cfg)
    train = np.array([r['split'] == 'train' and r['replicate'] == 1 for r in metadata])
    test = np.array([i for i, r in enumerate(metadata) if r['split'] == 'test'])
    size = target.shape[-1]
    distance = np.abs(np.arange(size)[:, None]-np.arange(size)[None, :])
    background = np.zeros(size)
    for d in range(size):
        use = masks[train, 0] & (distance == d)
        if use.any(): background[d] = float(target[train, 0][use].mean())
    decay = background[distance]
    prediction = np.repeat(decay[None], len(test), axis=0)
    np.savez_compressed(out/'distance_only_diagnostic.npz', prediction=prediction, background=background)
    models = ['bilinear', 'bicubic']+[f'{family}_{seed}' for family in cfg['models'] for seed in cfg['seeds']]
    residual_rows, rows = [], []
    for model in models+['distance_only_diagnostic']:
        if model != 'distance_only_diagnostic':
            with np.load(out/f'{model}_reconstruction.npz') as z: prediction = z['prediction'].copy()
        else: prediction = np.repeat(decay[None], len(test), axis=0)
        for j, i in enumerate(test):
            for lower, upper in [(2, 8), (8, 32), (32, size)]:
                mask = masks[i, 0] & (distance >= lower) & (distance < upper)
                use = mask & np.triu(np.ones_like(mask, bool), 2)
                metrics = image_metrics(prediction[j], target[i, 0], mask, cfg['ssim_window'])
                residual_rows.append(dict(model=model, structure_id=metadata[i]['structure_id'],
                    replicate=metadata[i]['replicate'], distance_min_bp=lower*cfg['target_resolution_bp'],
                    distance_max_bp=upper*cfg['target_resolution_bp'], mse=metrics['mse'], psnr=metrics['psnr'],
                    residual_correlation=correlation((prediction[j]-decay)[use], (target[i, 0]-decay)[use])))
    for model in models+['distance_only_diagnostic']:
        for rep in [1, 2]:
            for distance_min in [400, 1600, 6400]:
                selected = [r for r in residual_rows if r['model'] == model and r['replicate'] == rep and r['distance_min_bp'] == distance_min]
                correlations = [r['residual_correlation'] for r in selected if r['residual_correlation'] is not None]
                rows.append(dict(model=model, replicate=rep, distance_min_bp=distance_min,
                    distance_max_bp=selected[0]['distance_max_bp'], windows=len(selected),
                    psnr=float(np.mean([r['psnr'] for r in selected])),
                    residual_correlation=float(np.mean(correlations)) if correlations else None))
    write_csv(out/'distance_diagnostics.csv', rows)
    write_json(out/'diagnostic_protocol.json', dict(status='post_training_diagnostic',
        purpose='Check whether PSNR gains mainly reflect genomic-distance decay; not used for selecting models.',
        background_fit='train split, rep1 only; mean normalized target per separation',
        no_new_training=True))
    print('Task 5 distance diagnostics:', len(rows), 'summaries')


if __name__ == '__main__':
    main()
