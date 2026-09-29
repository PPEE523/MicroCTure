"""Versioned AE/DAE rescanning with matched budgets and exploratory replication."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/microcture-mpl')
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from scipy.cluster.vq import kmeans2
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from discover_task2 import load_band, extract, tensor, correlation, disjoint_select, write_csv
from discovery_v2_core import bin_roles, window_role, train_background, fit_model, training_percentiles, quota_select
from task1_baselines import PCA, StandardScaler
from diagnose_discovery_v2 import adjusted_rand

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with (ROOT / path).open() as handle:
        return list(csv.DictReader(handle))


def fit_embedding(values, train, out, name, standardize=True):
    if standardize:
        scaler = StandardScaler().fit(values[train])
        normalized = scaler.transform(values)
        mean, std = scaler.mean, scaler.std
    else:
        normalized, mean, std = values, np.zeros(values.shape[1]), np.ones(values.shape[1])
    pca = PCA(8).fit(normalized[train])
    embedding = pca.transform(normalized)
    np.savez_compressed(out / f'{name}_representation.npz', latent=values, embedding=embedding,
                        feature_mean=mean, feature_std=std, pca_mean=pca.mean, components=pca.components)
    return embedding


def known_recall(selected, rows, known, allowed=None):
    recalled = {name for i in selected for name in rows[i]['recalled_ids'].split(';') if name}
    result = {}
    for kind in ['OPCID', 'CHIN', 'CHID']:
        ids = {r['structure_id'] for r in known if r['type'] == kind and (allowed is None or r['structure_id'] in allowed)}
        result[kind] = dict(recalled=len(ids & recalled), total=len(ids), recall=len(ids & recalled) / len(ids) if ids else None)
    return result


def main():
    cfg = json.loads((ROOT / 'configs/discovery_v2.json').read_text())
    data = json.loads((ROOT / 'configs/data.json').read_text())
    out = ROOT / cfg['output_dir']
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    sources = ['configs/discovery_v2.json', 'configs/data.json', 'data/processed/structures.csv',
               'scripts/discover_task2_v2.py', 'scripts/discovery_v2_core.py', 'scripts/discover_task2.py',
               'scripts/task1_baselines.py', 'scripts/prepare_dataset.py', 'scripts/diagnose_discovery_v2.py',
               'results/task2/scan_rep1.csv']
    manifest = dict(config=cfg, sha256={s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources},
                    samples={s['sample_id']: dict(path=s['path'], size=(ROOT / s['path']).stat().st_size,
                                                mtime_ns=(ROOT / s['path']).stat().st_mtime_ns) for s in data['samples']})
    manifest_path = out / 'run_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Configuration/source inputs changed; use a new versioned directory')
    manifest_path.write_text(json.dumps(manifest, indent=2))
    n = (data['chrom_length'] + 99) // 100
    roles = bin_roles(n, cfg)
    b1 = train_background(load_band(data['samples'][0], n, max(cfg['window_bins'])), roles)
    original = {r['window_id']: r for r in read_csv('results/task2/scan_rep1.csv')}
    known = read_csv('data/processed/structures.csv')
    rows, arrays = [], []
    for width in cfg['window_bins']:
        for center in range(0, n, cfg['stride_bins']):
            start = center - width // 2
            raw, oe, mask = extract(b1, start, width)
            upper = np.triu(mask, 2)
            valid = float(mask.sum() / (width * width - 3 * width + 2))
            zero = float((raw[upper] == 0).mean()) if upper.any() else 1.0
            old = original[f'w{width}_c{center}']
            rows.append(dict(window_id=old['window_id'], start_bin=start, width_bins=width, center_bin=center,
                             split=window_role(start, width, roles, cfg['spatial_block_bins']),
                             valid_fraction=valid, zero_fraction=zero, mean_oe=float(oe[upper].mean()) if upper.any() else 0,
                             eligible=valid >= cfg['min_valid_fraction'] and zero <= cfg['max_zero_fraction'],
                             known_ids=old['known_ids'], recalled_ids=old['recalled_ids']))
            arrays.append(tensor(oe, mask))
    x = np.stack(arrays).astype(np.float32)
    del arrays
    eligible = np.array([r['eligible'] for r in rows])
    splits = np.array([r['split'] for r in rows])
    widths = np.array([r['width_bins'] for r in rows])
    train = (splits == 0) & eligible
    holdout = (splits == 2) & eligible
    write_csv(out / 'scan_rep1.csv', rows)
    np.savez_compressed(out / 'spatial_preprocessing.npz', bin_roles=roles, window_roles=splits, eligible=eligible,
                        expected_rep1=b1['expected'], opportunities_rep1=b1['opportunities'], valid_rep1=b1['valid'], threshold_rep1=b1['threshold'])
    print('Scan prepared:', len(rows), 'eligible', int(eligible.sum()), 'train/val/test', [int(((splits == i) & eligible).sum()) for i in range(3)], flush=True)
    smooth = gaussian_filter(x[:, 0] * x[:, 1], sigma=(0, 2, 2)) / np.maximum(gaussian_filter(x[:, 1], sigma=(0, 2, 2)), .1)
    shape = (smooth * smooth * x[:, 1]).sum((1, 2)) / np.maximum(x[:, 1].sum((1, 2)), .1)
    del smooth
    shape_weights = x[:, 1].reshape(len(x), 16, 4, 16, 4).sum((2, 4))
    shape_values = (x[:, 0] * x[:, 1]).reshape(len(x), 16, 4, 16, 4).sum((2, 4))
    shape_features = np.divide(shape_values, shape_weights, out=np.zeros_like(shape_values), where=shape_weights > 0).reshape(len(x), -1)
    embeddings = {'shape': fit_embedding(shape_features, train, out, 'shape', standardize=False)}
    rank = lambda values: training_percentiles(values, widths, splits == 0, eligible)
    shape_rank = rank(shape)
    methods = {'shape': shape_rank, 'density': rank(np.array([r['mean_oe'] for r in rows]))}
    feature_for_method = {'shape': 'shape', 'density': 'shape'}
    training = []
    quality_correlations = []
    for family in ['ae', 'dae']:
        for seed in cfg['seeds']:
            scores, features, summary = fit_model(x, splits, eligible, cfg, out, family, seed)
            training.append(summary)
            name = f'{family}_{seed}'
            embedding = fit_embedding(features, train, out, name)
            embeddings[name] = embedding
            np.savez_compressed(out / f'{name}_scores.npz', residual=scores)
            methods[name] = rank(scores)
            fusion = f'{family}_fusion_{seed}'
            methods[fusion] = .5 * (methods[name] + shape_rank)
            feature_for_method[name] = feature_for_method[fusion] = name
            for metric in ['mean_oe', 'valid_fraction', 'zero_fraction']:
                values = np.array([r[metric] for r in rows])
                quality_correlations.append(dict(model=name, quality_metric=metric,
                    test_spearman=float(spearmanr(scores[holdout], values[holdout]).statistic)))
            print('Finished training', name, json.dumps(summary), flush=True)
    write_csv(out / 'training_summary.csv', training)
    write_csv(out / 'quality_correlations.csv', quality_correlations)
    selected, test_selected, controls, clustering, cluster_stability = {}, {}, {}, {}, []
    for method, score in methods.items():
        selected[method] = quota_select(score, rows, eligible, cfg['full_candidate_quotas'], n)
        test_selected[method] = quota_select(score, rows, holdout, cfg['test_candidate_quotas'], n)
    random_selected = {'full': [], 'test': []}
    for seed in range(cfg['random_references']):
        scores = np.random.default_rng(10000 + seed).random(len(rows))
        random_selected['full'].append(quota_select(scores, rows, eligible, cfg['full_candidate_quotas'], n))
        random_selected['test'].append(quota_select(scores, rows, holdout, cfg['test_candidate_quotas'], n))
    for method, ids in selected.items():
        # Fit centers only on eligible training windows, then assign candidates.
        # Neither test embeddings nor test labels choose centers or K.
        z = embeddings[feature_for_method[method]]
        for k in cfg['clusters']:
            assignments = []
            for seed in cfg['clustering_seeds']:
                best = float('inf')
                for attempt in range(cfg['kmeans_initializations']):
                    centers, labels = kmeans2(z[train], k, minit='++', seed=seed * 100 + attempt, iter=100)
                    loss = np.square(z[train] - centers[labels]).sum()
                    if loss < best:
                        best, best_centers = loss, centers
                labels = np.square(z[ids, None, :] - best_centers[None, :, :]).sum(2).argmin(1)
                clustering[f'{method}:{k}:{seed}'] = dict(labels=labels.tolist(), centers=best_centers.tolist())
                assignments.append(labels)
            cluster_stability.append(dict(method=method, k=k, mean_pairwise_ari=float(np.mean([adjusted_rand(a, b) for a, b in itertools.combinations(assignments, 2)]))))
    all_selected = sorted({i for ids in selected.values() for i in ids})
    for i in all_selected:
        r = rows[i]
        occupied = np.zeros(n, bool)
        occupied[(r['start_bin'] + np.arange(r['width_bins'])) % n] = True
        matches = [j for j, s in enumerate(rows) if s['eligible'] and not s['known_ids'] and s['width_bins'] == r['width_bins']
                   and not occupied[(s['start_bin'] + np.arange(s['width_bins'])) % n].any()]
        matches.sort(key=lambda j: abs(np.log1p(rows[j]['mean_oe']) - np.log1p(r['mean_oe'])) + abs(rows[j]['valid_fraction'] - r['valid_fraction']))
        controls[str(i)] = disjoint_select(matches, rows, n)[:cfg['controls_per_candidate']]
    frozen = dict(selected=selected, test_selected=test_selected, controls=controls, clustering=clustering,
                  random_selected=random_selected, feature_for_method=feature_for_method, interpretation=cfg['interpretation'])
    (out / 'frozen_before_rep2_reload.json').write_text(json.dumps(frozen, indent=2))
    np.savez_compressed(out / 'ranking_scores.npz', **methods)
    print('Frozen', len(methods), 'methods;', len(all_selected), 'unique candidate windows; now reloading previously seen rep2', flush=True)
    b2 = train_background(load_band(data['samples'][1], n, max(cfg['window_bins'])), roles)
    np.savez_compressed(out / 'rep2_preprocessing.npz', expected=b2['expected'], valid=b2['valid'], threshold=b2['threshold'])
    used = sorted(set(all_selected) | {j for values in controls.values() for j in values})
    correlations = {}
    for i in used:
        r = rows[i]
        _, a, ma = extract(b1, r['start_bin'], r['width_bins'])
        _, b, mb = extract(b2, r['start_bin'], r['width_bins'])
        correlations[i] = correlation(a, b, ma & mb)
    write_csv(out / 'replicate_correlations.csv', [dict(window_index=i, correlation=correlations[i]) for i in used])
    support, validation = {}, []
    for i in all_selected:
        r = rows[i]
        values = [correlations[j] for j in controls[str(i)] if correlations[j] is not None]
        cutoff = max(cfg['replicate_min_correlation'], float(np.quantile(values, cfg['control_quantile']))) if len(values) >= cfg['min_controls'] else None
        corr = correlations[i]
        support[i] = cutoff is not None and corr is not None and corr > cutoff
        validation.append(dict(window_index=i, window_id=r['window_id'], start_bp=(r['start_bin'] % n) * 100,
                               window_length_bp=r['width_bins'] * 100, wraps_origin=r['start_bin'] < 0 or r['start_bin'] + r['width_bins'] > n,
                               known_ids=r['known_ids'], split=r['split'], correlation=corr, matched_controls=len(values), cutoff=cutoff,
                               replicate_supported=support[i], coordinate_status='provisional'))
    write_csv(out / 'candidate_validation.csv', validation)
    cluster_results = []
    for key, result in clustering.items():
        method, k, seed = key.split(':')
        ids = selected[method]
        labels = np.array(result['labels'])
        for cluster in range(int(k)):
            positions = set(np.flatnonzero(labels == cluster).tolist())
            members = [ids[j] for j in sorted(positions)]
            if not members:
                continue
            annotated = sum(bool(rows[i]['known_ids']) for i in members)
            passes = sum(support[i] for i in members)
            similarities = []
            for other in cfg['clustering_seeds']:
                if other == int(seed):
                    continue
                other_labels = np.array(clustering[f'{method}:{k}:{other}']['labels'])
                alternatives = [set(np.flatnonzero(other_labels == label).tolist()) for label in np.unique(other_labels)]
                similarities.append(max(len(positions & s) / len(positions | s) for s in alternatives))
            stability = min(similarities)
            operational = annotated == 0 and passes >= cfg['min_new_cluster_members']
            cluster_results.append(dict(method=method, k=int(k), seed=int(seed), cluster=cluster, members=len(members), annotated_members=annotated,
                                        supported_members=passes, min_seed_jaccard=stability,
                                        passes_original_rule=operational, stable_candidate_new_cluster=operational and stability >= cfg['cluster_min_jaccard']))
    write_csv(out / 'clusters.csv', cluster_results)
    write_csv(out / 'cluster_stability.csv', cluster_stability)
    # Test annotations must lie fully within a test region; no train-side fragments.
    test_known = {r['structure_id'] for r in known if (roles[int(r['start']) // 100:(int(r['end']) + 99) // 100] == 2).all()}
    recalls = []
    for domain, choices in [('full', selected), ('test', test_selected)]:
        allowed = test_known if domain == 'test' else None
        for method, ids in choices.items():
            for kind, values in known_recall(ids, rows, known, allowed).items():
                recalls.append(dict(domain=domain, method=method, type=kind, **values, windows=len(ids), covered_bins=sum(rows[i]['width_bins'] for i in ids)))
        for seed, ids in enumerate(random_selected[domain]):
            for kind, values in known_recall(ids, rows, known, allowed).items():
                recalls.append(dict(domain=domain, method=f'random_{seed}', type=kind, **values, windows=len(ids), covered_bins=sum(rows[i]['width_bins'] for i in ids)))
    write_csv(out / 'known_recall.csv', recalls)
    method_summary = []
    for method, ids in selected.items():
        relevant = [r for r in cluster_results if r['method'] == method]
        method_summary.append(dict(method=method, candidates=len(ids), covered_bins=sum(rows[i]['width_bins'] for i in ids),
                                   unannotated=sum(not rows[i]['known_ids'] for i in ids), replicate_supported=sum(support[i] for i in ids),
                                   unannotated_supported=sum(support[i] and not rows[i]['known_ids'] for i in ids),
                                   passing_cluster_configurations=sum(r['passes_original_rule'] for r in relevant),
                                   stable_cluster_configurations=sum(r['stable_candidate_new_cluster'] for r in relevant)))
    write_csv(out / 'method_summary.csv', method_summary)
    # Plot all unique unannotated supported windows plus primary-method top 12.
    unannotated_supported = [i for i in all_selected if support[i] and not rows[i]['known_ids']]
    primary = selected[cfg['primary_method']]
    plot_ids = list(dict.fromkeys(unannotated_supported + sorted(primary, key=lambda i: -methods[cfg['primary_method']][i])[:12]))
    (out / 'heatmaps').mkdir(exist_ok=True)
    limit = float(np.quantile(np.concatenate([np.log1p(extract(b1, rows[i]['start_bin'], rows[i]['width_bins'])[1])[extract(b1, rows[i]['start_bin'], rows[i]['width_bins'])[2]] for i in primary]), .99))
    for i in plot_ids:
        r = rows[i]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
        for ax, b, name in zip(axes, [b1, b2], ['rep1', 'rep2']):
            _, oe, mask = extract(b, r['start_bin'], r['width_bins'])
            lo, hi = r['start_bin'] * .1, (r['start_bin'] + r['width_bins']) * .1
            im = ax.imshow(np.where(mask, np.log1p(oe), np.nan), origin='lower', vmin=0, vmax=limit, cmap='magma', extent=(lo, hi, lo, hi))
            ax.set(title=name, xlabel='Unwrapped binned position (kb)', ylabel='Position (kb)')
        fig.colorbar(im, ax=axes, label='log1p(O/E)', shrink=.7)
        fig.suptitle(f"{r['window_id']} | r={correlations[i]:.3f} | supported={support[i]}\nKnown overlap: {bool(r['known_ids'])}; exploratory, not independent confirmation", fontsize=10)
        fig.savefig(out / 'heatmaps' / f'{r["window_id"]}.png', dpi=110)
        plt.close(fig)
    # Cross-method union can overlap: report an explicitly disjoint descriptive subset.
    independent = disjoint_select(sorted(unannotated_supported, key=lambda i: (-correlations[i], i)), rows, n)
    rows_by_i = {r['window_index']: r for r in validation}
    write_csv(out / 'unannotated_supported_union.csv', [rows_by_i[i] for i in unannotated_supported], list(validation[0]))
    write_csv(out / 'unannotated_supported_disjoint.csv', [rows_by_i[i] for i in independent], list(validation[0]))
    primary_clusters = [r for r in cluster_results if r['method'] == cfg['primary_method'] and r['k'] == cfg['primary_k'] and r['seed'] == cfg['primary_cluster_seed']]
    report = dict(config=cfg, scan_windows=len(rows), eligible_windows=int(eligible.sum()),
                  partition_windows={name: int(((splits == j) & eligible).sum()) for j, name in enumerate(['train', 'validation', 'test'])},
                  training=training, methods=method_summary, unique_selected_windows=len(all_selected),
                  unannotated_supported_union=len(unannotated_supported), unannotated_supported_disjoint=len(independent),
                  primary_candidate_new_clusters=sum(r['stable_candidate_new_cluster'] for r in primary_clusters),
                  all_passing_cluster_configurations=sum(r['stable_candidate_new_cluster'] for r in cluster_results),
                  cluster_runs=len(clustering), heatmaps=len(plot_ids), primary_color_limit=limit,
                  test_known_counts={kind: sum(r['structure_id'] in test_known and r['type'] == kind for r in known) for kind in ['OPCID', 'CHIN', 'CHID']},
                  frozen_sha256=hashlib.sha256((out / 'frozen_before_rep2_reload.json').read_bytes()).hexdigest(),
                  limitations=[cfg['interpretation'], 'Unannotated does not establish a new biological type.',
                               'Pixel correlation thresholds are descriptive, not multiplicity-adjusted p-values.',
                               'Cross-method candidate union overlaps; only disjoint subset is spatially independent.',
                               'QC uses local contacts with same-partition endpoints; its threshold and distance expected values fit training bins only.',
                               'No annotated members is a restrictive operational novelty convention, not a biological definition.',
                               'Coordinates provisional; local diagonal scan only.'])
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print('COMPLETE', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
