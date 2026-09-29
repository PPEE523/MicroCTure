"""Verify v2 split isolation, matched budgets, checkpoints and cluster decisions."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from discover_task2_v2 import ROOT, read_csv


def main():
    cfg = json.loads((ROOT / 'configs/discovery_v2.json').read_text())
    out = ROOT / cfg['output_dir']
    manifest = json.loads((out / 'run_manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    report = json.loads((out / 'report.json').read_text())
    frozen = json.loads((out / 'frozen_before_rep2_reload.json').read_text())
    assert hashlib.sha256((out / 'frozen_before_rep2_reload.json').read_bytes()).hexdigest() == report['frozen_sha256']
    rows = read_csv(f"{cfg['output_dir']}/scan_rep1.csv")
    with np.load(out / 'spatial_preprocessing.npz') as z:
        roles, splits, eligible = z['bin_roles'], z['window_roles'], z['eligible']
    n = len(roles)
    def bins(i):
        return (int(rows[i]['start_bin']) + np.arange(int(rows[i]['width_bins']))) % n
    for i, row in enumerate(rows):
        if splits[i] >= 0:
            assert (roles[bins(i)] == splits[i]).all()
            start, width = int(row['start_bin']), int(row['width_bins'])
            assert 0 <= start < start + width <= n
            assert start // cfg['spatial_block_bins'] == (start + width - 1) // cfg['spatial_block_bins']
    for seed in cfg['seeds']:
        checkpoints = []
        for family in ['ae', 'dae']:
            name = f'{family}_{seed}'
            checkpoint = torch.load(out / f'{name}.pt', weights_only=False)
            checkpoints.append(checkpoint)
            train = np.array(checkpoint['train_indices'])
            val = np.array(checkpoint['validation_indices'])
            assert eligible[train].all() and (splits[train] == 0).all()
            assert eligible[val].all() and (splits[val] == 1).all()
            history = json.loads((out / f'{name}_history.json').read_text())
            assert len(history) <= cfg['ae_max_epochs']
            assert abs(min(h['validation_loss'] for h in history) - checkpoint['validation_loss']) <= 1e-6
            assert history[checkpoint['epoch'] - 1]['validation_loss'] == checkpoint['validation_loss']
        assert checkpoints[0]['train_indices'] == checkpoints[1]['train_indices']
        assert checkpoints[0]['validation_indices'] == checkpoints[1]['validation_indices']
    for domain, methods in [('full', frozen['selected']), ('test', frozen['test_selected'])]:
        selections = list(methods.values()) + frozen['random_selected'][domain]
        quota = cfg[f'{domain}_candidate_quotas']
        for selected in selections:
            occupied = np.zeros(n, bool)
            assert len(selected) == sum(quota.values())
            for i in selected:
                assert eligible[i]
                assert not occupied[bins(i)].any()
                occupied[bins(i)] = True
                if domain == 'test':
                    assert splits[i] == 2
            for width, count in quota.items():
                assert sum(int(rows[i]['width_bins']) == int(width) for i in selected) == count
            assert occupied.sum() == sum(int(width) * count for width, count in quota.items())
    correlations = {int(r['window_index']): float(r['correlation']) if r['correlation'] else None
                    for r in read_csv(f"{cfg['output_dir']}/replicate_correlations.csv")}
    validation = read_csv(f"{cfg['output_dir']}/candidate_validation.csv")
    supported = {}
    for row in validation:
        i = int(row['window_index'])
        occupied = np.zeros(n, bool)
        occupied[bins(i)] = True
        controls = frozen['controls'][str(i)]
        for j in controls:
            assert rows[j]['width_bins'] == rows[i]['width_bins'] and eligible[j] and not rows[j]['known_ids']
            assert not occupied[bins(j)].any()
            occupied[bins(j)] = True
        values = [correlations[j] for j in controls if correlations[j] is not None]
        assert len(values) == int(row['matched_controls'])
        cutoff = max(cfg['replicate_min_correlation'], float(np.quantile(values, cfg['control_quantile']))) if len(values) >= cfg['min_controls'] else None
        if cutoff is not None:
            np.testing.assert_allclose(cutoff, float(row['cutoff']))
        decision = cutoff is not None and correlations[i] is not None and correlations[i] > cutoff
        assert row['replicate_supported'] == str(decision)
        supported[i] = decision
    cluster_rows = read_csv(f"{cfg['output_dir']}/clusters.csv")
    for row in cluster_rows:
        method, k, seed = row['method'], row['k'], row['seed']
        labels = np.array(frozen['clustering'][f'{method}:{k}:{seed}']['labels'])
        positions = set(np.flatnonzero(labels == int(row['cluster'])).tolist())
        ids = [frozen['selected'][method][p] for p in positions]
        annotated = sum(bool(rows[i]['known_ids']) for i in ids)
        passes = sum(supported[i] for i in ids)
        assert len(ids) == int(row['members']) and annotated == int(row['annotated_members']) and passes == int(row['supported_members'])
        sims = []
        for other in cfg['clustering_seeds']:
            if other != int(seed):
                other_labels = np.array(frozen['clustering'][f'{method}:{k}:{other}']['labels'])
                groups = [set(np.flatnonzero(other_labels == j).tolist()) for j in np.unique(other_labels)]
                sims.append(max(len(positions & g) / len(positions | g) for g in groups))
        stability = min(sims)
        np.testing.assert_allclose(stability, float(row['min_seed_jaccard']))
        operational = annotated == 0 and passes >= cfg['min_new_cluster_members']
        assert row['passes_original_rule'] == str(operational)
        assert row['stable_candidate_new_cluster'] == str(operational and stability >= cfg['cluster_min_jaccard'])
    for row in report['methods']:
        ids = frozen['selected'][row['method']]
        assert row['replicate_supported'] == sum(supported[i] for i in ids)
        assert row['unannotated_supported'] == sum(supported[i] and not rows[i]['known_ids'] for i in ids)
    known = read_csv('data/processed/structures.csv')
    recall_rows = read_csv(f"{cfg['output_dir']}/known_recall.csv")
    for row in recall_rows:
        domain, method = row['domain'], row['method']
        if method.startswith('random_'):
            ids = frozen['random_selected'][domain][int(method.split('_')[1])]
        else:
            ids = frozen['selected' if domain == 'full' else 'test_selected'][method]
        truth = [r for r in known if r['type'] == row['type'] and (domain == 'full'
                 or (roles[int(r['start']) // 100:(int(r['end']) + 99) // 100] == 2).all())]
        recalled = 0
        for structure in truth:
            lo, hi = int(structure['start']) // 100, (int(structure['end']) + 99) // 100
            center = int(float(structure['center']) // 100)
            # Reconstruct the coordinate rule, not the saved recalled_ids field.
            recalled += any(center in bins(i) and np.count_nonzero((bins(i) >= lo) & (bins(i) < hi)) >= .5 * (hi - lo) for i in ids)
        assert len(truth) == int(row['total']) and recalled == int(row['recalled'])
        if truth:
            np.testing.assert_allclose(float(row['recall']), recalled / len(truth))
    disjoint = read_csv(f"{cfg['output_dir']}/unannotated_supported_disjoint.csv")
    occupied = np.zeros(n, bool)
    for row in disjoint:
        i = int(row['window_index'])
        assert supported[i] and not rows[i]['known_ids'] and not occupied[bins(i)].any()
        occupied[bins(i)] = True
    images = list((out / 'heatmaps').glob('*.png'))
    assert len(images) == report['heatmaps']
    for path in images:
        with Image.open(path) as image:
            image.verify()
    result = dict(status='passed', training_runs=6, paired_training_windows_identical=True,
                  spatial_partitions_disjoint=True, methods=len(frozen['selected']),
                  fixed_full_candidates=189, fixed_test_candidates=27, candidate_validations=len(validation),
                  cluster_runs=len(frozen['clustering']), checked_cluster_rows=len(cluster_rows),
                  pngs_verified=len(images), independently_checked_recall_rows=len(recall_rows),
                  source_hashes_match=True, independent_biological_confirmation=False)
    (ROOT / 'reports/task2_v2_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
