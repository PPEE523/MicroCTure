"""Versioned AE/DAE rescanning with matched budgets and exploratory replication."""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()
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
from discover_task2 import load_band, extract, tensor, correlation, disjoint_select, write_csv, annotations
from discovery_v2_core import (bin_roles, window_role, train_background, fit_model,
                               training_percentiles, quota_select, validate_config, file_sha256,
                               load_model_artifacts, _array_identity, _json_sha256, _atomic_json)
from ml_baselines import PCA, StandardScaler
from clustering_metrics import adjusted_rand
from optional_common import experiment_arguments, input_path, signatures, runtime_paths, temporary_plotting

ROOT = _layout_root


def validate_scan_identity(records, known, n, cfg):
    """Validate v1 identity and labels; mismatches never silently change labels."""
    original = {}
    for row in records:
        name = row['window_id']
        if name in original:
            raise ValueError(f'Duplicate scan window_id: {name}')
        original[name] = row
    expected = {f'w{width}_c{center}': (center - width // 2, width, center)
                for width in cfg['window_bins'] for center in range(0, n, cfg['stride_bins'])}
    if set(original) != set(expected):
        raise ValueError('Scan window grid mismatch: missing or unexpected window IDs')
    for name, (start, width, center) in expected.items():
        row = original[name]
        if tuple(int(row[key]) for key in ['start_bin', 'width_bins', 'center_bin']) != (start, width, center):
            raise ValueError(f'Scan window geometry mismatch: {name}')
        hits, recalled = annotations(start, width, n, known)
        for key, values in [('known_ids', hits), ('recalled_ids', recalled)]:
            previous = list(filter(None, row[key].split(';')))
            if len(previous) != len(set(previous)) or set(previous) != set(values):
                raise ValueError(f'Scan annotation mismatch ({key}): {name}; regenerate v1 with the current annotation')
    return original


def save_preprocessing(path, band, roles, splits, eligible, sample_id):
    """Keep the same fitted-background evidence for both WT replicates."""
    fit = (roles == 0) & band['valid']
    identity = dict(sample_id=sample_id, rule='same-partition coverage; training-endpoint expected and QC threshold',
                    input_band=_array_identity(band['band']), bin_roles=_array_identity(roles),
                    training_fit_bins=_array_identity(fit),
                    training_qc_fit_bins=_array_identity((roles == 0) & (band['coverage'] > 0)))
    np.savez_compressed(path, bin_roles=roles, window_roles=splits, eligible=eligible,
                        expected=band['expected'], opportunities=band['opportunities'], valid=band['valid'],
                        threshold=band['threshold'], coverage=band['coverage'], training_fit_bins=fit,
                        training_qc_fit_bins=(roles == 0) & (band['coverage'] > 0),
                        fit_identity=json.dumps(identity, sort_keys=True), fit_identity_sha256=_json_sha256(identity))


def assess_replicate(corr, controls, cfg):
    values = [float(v) for v in controls if v is not None and np.isfinite(v)]
    corr = float(corr) if corr is not None and np.isfinite(corr) else None
    cutoff = max(cfg['replicate_min_correlation'], float(np.quantile(values, cfg['control_quantile']))) if len(values) >= cfg['min_controls'] else None
    reason = ('missing_or_nonfinite_correlation' if corr is None else
              'insufficient_valid_controls' if cutoff is None else None)
    supported = None if reason else corr > cutoff
    explanation = reason or ('correlation_above_cutoff' if supported else 'correlation_not_above_cutoff')
    return dict(correlation=corr, matched_controls=len(values), cutoff=cutoff,
                replicate_supported=supported,
                replicate_status='not_evaluable' if reason else 'supported' if supported else 'not_supported',
                replicate_reason=explanation)


def assess_cluster(annotated, supported, not_evaluable, minimum, stability, threshold):
    if annotated or supported + not_evaluable < minimum:
        original = False
    elif supported >= minimum:
        original = True
    else:
        original = None
    stable = False if stability < threshold or original is False else original
    return original, stable


def complete_run(out, expected_models=()):
    """Publish completion last; a failed fresh run retains all partial evidence."""
    if any((out / name).exists() for name in ['ARTIFACTS.json', 'COMPLETED.json']):
        raise FileExistsError('Run completion artifacts already exist; refusing to overwrite')
    for name in expected_models:
        metadata = json.loads((out / f'{name}_artifacts.json').read_text())
        load_model_artifacts(out, name, metadata['identity'])
    files = {p.relative_to(out).as_posix(): dict(sha256=file_sha256(p), size_bytes=p.stat().st_size)
             for p in sorted(out.rglob('*')) if p.is_file() and p.name not in {'ARTIFACTS.json', 'COMPLETED.json'}}
    artifacts = dict(schema_version=1, status='completed', files=files,
                     checkpoint_count=sum(name.endswith('.pt') for name in files))
    _atomic_json(out / 'ARTIFACTS.json', artifacts)
    _atomic_json(out / 'COMPLETED.json', dict(schema_version=1, status='completed',
                  artifacts_sha256=file_sha256(out / 'ARTIFACTS.json'), files=len(files)))


def read_csv(path, *, data_root=ROOT):
    with input_path(data_root, path).open() as handle:
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


def run_experiment(args):
    cfg = json.loads(args.config.read_text())
    validate_config(cfg)
    data = json.loads((_layout_resolve(ROOT / 'configs/data.json')).read_text())
    out = args.output_dir
    sources = ['configs/data.json', 'scripts/discover_task2_v2.py', 'scripts/discovery_v2_core.py',
               'scripts/discover_task2.py', 'scripts/contact_band.py', 'scripts/spatial_tools.py',
               'scripts/ml_baselines.py', 'scripts/optional_common.py', 'scripts/clustering_metrics.py']
    inputs = ['data/processed/structures.csv']
    scan_file = args.scan_file.resolve() if args.scan_file else input_path(args.data_root,'results/task2/scan_rep1.csv')
    cache_paths = [args.band_cache_dir.resolve()/(sample['sample_id']+'.npz') if args.band_cache_dir else input_path(args.data_root,'data/cache/discovery/'+sample['sample_id']+'.npz') for sample in data['samples']]
    manifest = dict(config=cfg, runtime=runtime_paths(args), sha256=signatures([args.config]+[_layout_resolve(ROOT/s) for s in sources],
                    [input_path(args.data_root, name) for name in inputs]+[scan_file]+cache_paths),
                    samples={sample['sample_id']: dict(path=str(input_path(args.data_root,sample['path'])),
                        size=input_path(args.data_root,sample['path']).stat().st_size,
                        mtime_ns=input_path(args.data_root,sample['path']).stat().st_mtime_ns) for sample in data['samples']})
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    n = (data['chrom_length'] + 99) // 100
    roles = bin_roles(n, cfg)
    b1 = train_background(load_band(data['samples'][0], n, max(cfg['window_bins']), data_root=args.data_root, cache_root=args.band_cache_dir, allow_build=False), roles)
    with scan_file.open() as handle:
        original_records = list(csv.DictReader(handle))
    known = read_csv('data/processed/structures.csv', data_root=args.data_root)
    original = validate_scan_identity(original_records, known, n, cfg)
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
    save_preprocessing(out / 'spatial_preprocessing.npz', b1, roles, splits, eligible, data['samples'][0]['sample_id'])
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
                statistic = float(spearmanr(scores[holdout], values[holdout]).statistic)
                quality_correlations.append(dict(model=name, quality_metric=metric,
                    test_spearman=statistic if np.isfinite(statistic) else None))
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
    b2 = train_background(load_band(data['samples'][1], n, max(cfg['window_bins']), data_root=args.data_root, cache_root=args.band_cache_dir, allow_build=False), roles)
    save_preprocessing(out / 'rep2_preprocessing.npz', b2, roles, splits, eligible, data['samples'][1]['sample_id'])
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
        assessment = assess_replicate(correlations[i], [correlations[j] for j in controls[str(i)]], cfg)
        support[i] = assessment['replicate_supported']
        validation.append(dict(window_index=i, window_id=r['window_id'], start_bp=(r['start_bin'] % n) * 100,
                               window_length_bp=r['width_bins'] * 100, wraps_origin=r['start_bin'] < 0 or r['start_bin'] + r['width_bins'] > n,
                               known_ids=r['known_ids'], split=r['split'], **assessment, coordinate_status='provisional'))
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
            passes = sum(support[i] is True for i in members)
            not_evaluable = sum(support[i] is None for i in members)
            similarities = []
            for other in cfg['clustering_seeds']:
                if other == int(seed):
                    continue
                other_labels = np.array(clustering[f'{method}:{k}:{other}']['labels'])
                alternatives = [set(np.flatnonzero(other_labels == label).tolist()) for label in np.unique(other_labels)]
                similarities.append(max(len(positions & s) / len(positions | s) for s in alternatives))
            stability = min(similarities)
            operational, stable = assess_cluster(annotated, passes, not_evaluable, cfg['min_new_cluster_members'], stability, cfg['cluster_min_jaccard'])
            cluster_results.append(dict(method=method, k=int(k), seed=int(seed), cluster=cluster, members=len(members), annotated_members=annotated,
                                        supported_members=passes, not_evaluable_members=not_evaluable, min_seed_jaccard=stability,
                                        passes_original_rule=operational, stable_candidate_new_cluster=stable,
                                        cluster_status='not_evaluable' if stable is None else 'passes_rule' if stable else 'does_not_pass_rule'))
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
                                   unannotated=sum(not rows[i]['known_ids'] for i in ids), replicate_supported=sum(support[i] is True for i in ids),
                                   not_evaluable_candidates=sum(support[i] is None for i in ids),
                                   unannotated_supported=sum(support[i] is True and not rows[i]['known_ids'] for i in ids),
                                   passing_cluster_configurations=sum(r['passes_original_rule'] is True for r in relevant),
                                   stable_cluster_configurations=sum(r['stable_candidate_new_cluster'] is True for r in relevant),
                                   not_evaluable_cluster_configurations=sum(r['stable_candidate_new_cluster'] is None for r in relevant)))
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
        corr_text = f'{correlations[i]:.3f}' if correlations[i] is not None and np.isfinite(correlations[i]) else 'NA'
        supported_text = 'NA' if support[i] is None else str(support[i])
        fig.suptitle(f"{r['window_id']} | r={corr_text} | supported={supported_text}\nKnown overlap: {bool(r['known_ids'])}; exploratory, not independent confirmation", fontsize=10)
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
                  primary_candidate_new_clusters=(None if any(r['stable_candidate_new_cluster'] is None for r in primary_clusters)
                                                  else sum(r['stable_candidate_new_cluster'] is True for r in primary_clusters)),
                  all_passing_cluster_configurations=(None if any(r['stable_candidate_new_cluster'] is None for r in cluster_results)
                                                     else sum(r['stable_candidate_new_cluster'] is True for r in cluster_results)),
                  confirmed_passing_cluster_configurations=sum(r['stable_candidate_new_cluster'] is True for r in cluster_results),
                  not_evaluable_cluster_configurations=sum(r['stable_candidate_new_cluster'] is None for r in cluster_results),
                  cluster_runs=len(clustering), heatmaps=len(plot_ids), primary_color_limit=limit,
                  test_known_counts={kind: sum(r['structure_id'] in test_known and r['type'] == kind for r in known) for kind in ['OPCID', 'CHIN', 'CHID']},
                  frozen_sha256=hashlib.sha256((out / 'frozen_before_rep2_reload.json').read_bytes()).hexdigest(),
                  limitations=[cfg['interpretation'], 'Unannotated does not establish a new biological type.',
                               'Pixel correlation thresholds are descriptive, not multiplicity-adjusted p-values.',
                               'Cross-method candidate union overlaps; disjoint subset is spatially non-overlapping, not biologically independent.',
                               'QC uses local contacts with same-partition endpoints; its threshold and distance expected values fit training bins only.',
                               'No annotated members is a restrictive operational novelty convention, not a biological definition.',
                               'Coordinates provisional; local diagonal scan only.'])
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    complete_run(out, [f'{family}_{seed}' for family in ['ae', 'dae'] for seed in cfg['seeds']])
    print('COMPLETE', json.dumps(report), flush=True)


def main(argv=None):
    args = experiment_arguments(__doc__, 'configs/discovery_v2.json', argv,
                                extra_paths=[('--scan-file','Read-only v1 scan_rep1.csv, including a fresh v1 output'),
                                             ('--band-cache-dir','Read-only directory of existing sample_id.npz bands; never builds missing caches')])
    if args is None:
        return
    with temporary_plotting() as pyplot:
        global plt
        plt = pyplot
        run_experiment(args)


if __name__ == '__main__':
    main()
