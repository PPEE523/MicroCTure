"""Recompute optional-task metrics from saved matrices and coordinates."""
import json
import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import uniform_filter
from optional_common import ROOT, read_csv, write_json, sha256
from prepare_dataset import assert_disjoint


def check_hashes(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    for path, digest in manifest['input_sha256'].items():
        assert sha256(ROOT/path) == digest, path
    return manifest


def check5():
    cfg = json.loads((ROOT/'configs/optional_tasks.json').read_text())['task5']
    out = ROOT/cfg['output_dir']; manifest = check_hashes(out)
    windows = read_csv(out/'windows.csv')
    rows = read_csv(out/'metrics.csv')
    split = read_csv(ROOT/'data/processed/classification_split.csv')
    footprints = [(int(float(r['center'])//100)-128+np.arange(256)) % 46417 for r in split]
    roles = np.array([['train', 'validation', 'test'].index(r['split']) for r in split])
    assert_disjoint(footprints, roles, 46417, 10)
    assert manifest['train_windows'] == 240 and manifest['validation_windows'] == 51 and manifest['test_windows'] == 106
    assert len(rows) == 848
    models = sorted({r['model'] for r in rows})
    assert len(models) == 8
    for model in models:
        with np.load(out/f'{model}_reconstruction.npz') as z:
            prediction, target, mask, indices = z['prediction'], z['target'], z['mask'], z['window_indices']
            assert prediction.shape == target.shape == mask.shape == (106, 128, 128)
            assert np.isfinite(prediction).all() and prediction.min() >= 0 and prediction.max() <= 1
            np.testing.assert_allclose(prediction, prediction.transpose(0, 2, 1), atol=1e-6)
            np.testing.assert_allclose(z['contact_per_million'], np.expm1(prediction*manifest['normalization']['log_scale']))
            selected = [r for r in rows if r['model'] == model]
            for i, (record, index) in enumerate(zip(selected, indices)):
                assert windows[index]['split'] == 'test' and windows[index]['structure_id'] == record['structure_id']
                valid = mask[i] & np.triu(np.ones((128, 128), bool), 2)
                mse = np.mean((prediction[i][valid]-target[i][valid])**2)
                np.testing.assert_allclose(float(record['mse']), mse)
                np.testing.assert_allclose(float(record['psnr']), -10*np.log10(max(mse, 1e-12)))
                x, y = prediction[i], target[i]
                ux, uy = uniform_filter(x, 7), uniform_filter(y, 7)
                vx, vy = np.maximum(uniform_filter(x*x, 7)-ux*ux, 0), np.maximum(uniform_filter(y*y, 7)-uy*uy, 0)
                cov = uniform_filter(x*y, 7)-ux*uy
                ssim = ((2*ux*uy+.0001)*(2*cov+.0009))/((ux*ux+uy*uy+.0001)*(vx+vy+.0009))
                interior = valid & (uniform_filter(mask[i].astype(float), 7, mode='constant') > 1-1e-8)
                np.testing.assert_allclose(float(record['ssim']), ssim[interior].mean(), atol=1e-6)
    for summary in read_csv(out/'summary.csv'):
        selected = [r for r in rows if r['model'] == summary['model'] and r['replicate'] == summary['replicate']
                    and (summary['type'] == 'all' or r['type'] == summary['type'])]
        assert len(selected) == int(summary['n'])
        for metric in ['psnr', 'ssim', 'shape_correlation', 'profile_correlation', 'boundary_shift_bp']:
            np.testing.assert_allclose(float(summary[metric]), np.mean([float(r[metric]) for r in selected if r[metric] != '']))
    # Restore saved weights and reproduce predictions from the original paired inputs.
    from train_task5 import prepare, SuperResolutionCNN
    inputs, _, _, metadata, _ = prepare(cfg)
    indices = [i for i, row in enumerate(metadata) if row['split'] == 'test']
    for family in cfg['models']:
        for seed in cfg['seeds']:
            name = f'{family}_{seed}'
            saved = torch.load(out/f'{name}.pt', weights_only=False)
            model = SuperResolutionCNN(cfg['channels'], family == 'residual_cnn', cfg['scale_factor'])
            model.load_state_dict(saved['state_dict']); model.eval()
            history = json.loads((out/f'{name}_history.json').read_text())
            assert saved['validation_loss'] <= min(r['validation_loss'] for r in history)+1e-7
            with torch.no_grad():
                actual = np.concatenate([model(torch.from_numpy(inputs[batch])).clamp(0, 1).numpy()[:, 0]
                                         for batch in np.array_split(indices, 8)])
            with np.load(out/f'{name}_reconstruction.npz') as z:
                np.testing.assert_allclose(actual, z['prediction'], atol=2e-6)
    return dict(status='passed', model_variants=8, trained_checkpoints=6, test_window_evaluations=len(rows),
                spatial_guard_bp=1000, source_hashes_match=True, independent_psnr_ssim_recomputed=True,
                checkpoint_predictions_reproduced=True)


def check6():
    cfg = json.loads((ROOT/'configs/optional_tasks.json').read_text())['task6']
    out = ROOT/cfg['output_dir']; manifest = check_hashes(out)
    rows = read_csv(out/'metrics.csv')
    assert len(rows) == 42
    with np.load(out/'evaluation_inputs.npz') as z:
        a, b, roles = z['a'], z['b'], z['roles']
        counts = [z['counts_rep1'], z['counts_rep2']]
        from train_task6 import fit_geometry_inputs, GraphDecoder
        _, _, adjacency, _ = fit_geometry_inputs(counts[0], a, b, roles, cfg)
        assert len(set(zip(a, b))) == len(a)
        assert (a < b).all() and set(roles) == {0, 1, 2}
        assert len(a) == 929*928//2-929
        for model in sorted({r['model'] for r in rows}):
            coordinates = read_csv(out/f'{model}_coordinates.csv')
            assert len(coordinates) == 929
            assert int(coordinates[0]['start']) == 0 and int(coordinates[-1]['end']) == manifest['chromosome_length']
            assert all(int(left['end']) == int(right['start']) for left, right in zip(coordinates, coordinates[1:]))
            xyz = np.array([[float(r[k]) for k in ['x', 'y', 'z']] for r in coordinates])
            assert np.isfinite(xyz).all()
            if model != 'mds':
                saved = torch.load(out/f'{model}.pt', weights_only=False)
                history = json.loads((out/f'{model}_history.json').read_text())
                assert saved['validation_loss'] <= min(r['validation_loss'] for r in history)+1e-7
                if model.startswith('graph'):
                    decoder = GraphDecoder(torch.zeros(929, 3), cfg['hidden_size'])
                    decoder.load_state_dict(saved['state_dict']); decoder.eval()
                    with torch.no_grad(): recovered = decoder(torch.from_numpy(adjacency.astype('float32'))).numpy()
                else:
                    recovered = saved['state_dict']['weight'].numpy()
                    recovered = recovered-recovered.mean(0)
                np.testing.assert_allclose(xyz, recovered, atol=2e-6)
            distances = squareform(pdist(xyz))
            with np.load(out/f'{model}_maps.npz') as stored:
                np.testing.assert_allclose(distances, stored['distances'])
                pred = np.maximum((np.maximum(distances, 1e-6)*manifest['fitted']['distance_scale'])**(-1/cfg['contact_exponent'])-cfg['pseudocount'], 0)
                np.fill_diagonal(pred, 0)
                np.testing.assert_allclose(pred, stored['predicted_contacts'])
            for record in [r for r in rows if r['model'] == model]:
                selection = roles == ['train', 'validation', 'test'].index(record['partition'])
                assert selection.sum() == int(record['pairs'])
                contact = counts[int(record['replicate'])-1][a, b]
                target = (contact+cfg['pseudocount'])**(-cfg['contact_exponent'])/manifest['fitted']['distance_scale']
                rmse = np.sqrt(np.mean((np.log(np.maximum(distances[a, b][selection], 1e-6))-np.log(target[selection]))**2))
                np.testing.assert_allclose(float(record['log_distance_rmse']), rmse, rtol=1e-5)
                corr = np.corrcoef(np.log1p(pred[a, b][selection]), np.log1p(contact[selection]))[0, 1]
                np.testing.assert_allclose(float(record['contact_log_pearson']), corr, atol=1e-6)
    for rep in [1, 2]:
        with np.load(ROOT/f'data/cache/task6/WT_37C_rep{rep}.npz') as z:
            np.testing.assert_array_equal(z['counts'], z['counts'].T)
            assert int(np.triu(z['counts']).sum()) == int(z['total'])
    return dict(status='passed', models=7, trained_checkpoints=6, bins_per_model=929,
                metric_rows=42, count_conservation=True, source_hashes_match=True,
                independent_distance_and_contact_metrics_recomputed=True, checkpoint_coordinates_reproduced=True)


def main():
    results = dict(task5=check5(), task6=check6())
    for name in ['task5/comparison.png', 'task5/examples.png', 'task6/structures.png', 'task6/contact_maps.png']:
        with Image.open(ROOT/'results'/name) as img: img.verify()
    write_json(ROOT/'reports/task56_verification.json', results)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
