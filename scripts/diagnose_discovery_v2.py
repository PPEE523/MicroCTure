"""Frozen-candidate control and representation ablations on previously seen WT data."""
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2
from discover_task2 import load_band, extract, tensor, correlation, disjoint_select, write_csv
from task1_baselines import PCA, StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with (ROOT / path).open() as handle:
        return list(csv.DictReader(handle))


def adjusted_rand(a, b):
    """Chance-adjusted pair agreement, invariant to cluster label permutations."""
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)
    pairs = lambda x: np.sum(x * (x - 1) / 2)
    total = len(a) * (len(a) - 1) / 2
    expected = pairs(table.sum(0)) * pairs(table.sum(1)) / total
    ceiling = (pairs(table.sum(0)) + pairs(table.sum(1))) / 2
    return float((pairs(table) - expected) / (ceiling - expected)) if ceiling != expected else 1.0


def main():
    cfg = json.loads((ROOT / 'configs/discovery_v2_diagnostic.json').read_text())
    data = json.loads((ROOT / 'configs/data.json').read_text())
    out = ROOT / cfg['output_dir']
    out.mkdir(parents=True, exist_ok=True)
    sources = ['configs/discovery_v2_diagnostic.json', 'scripts/diagnose_discovery_v2.py',
               'scripts/discover_task2.py', 'scripts/task1_baselines.py', 'configs/data.json',
               'results/task2/scan_rep1.csv', cfg['candidate_source'], 'results/task2/representation.npz',
               'results/task2/frozen_before_rep2.json']
    manifest = dict(config=cfg, sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
                    inputs={s['sample_id']: dict(size=(ROOT / s['path']).stat().st_size, mtime_ns=(ROOT / s['path']).stat().st_mtime_ns) for s in data['samples']})
    path = out / 'run_manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Inputs changed: use a new versioned output directory')
    path.write_text(json.dumps(manifest, indent=2))
    rows = read_csv('results/task2/scan_rep1.csv')
    candidates = read_csv(cfg['candidate_source'])
    for r in rows:
        for key in ['start_bin', 'width_bins', 'center_bin']:
            r[key] = int(r[key])
        for key in ['valid_fraction', 'zero_fraction', 'mean_oe']:
            r[key] = float(r[key])
        r['eligible'] = r['eligible'] == 'True'
    indices = [int(r['window_index']) for r in candidates]
    assert len(indices) == 189 and len(set(indices)) == 189
    n = (data['chrom_length'] + 99) // 100
    b1 = load_band(data['samples'][0], n, 256)
    with np.load(ROOT / 'results/task2/representation.npz') as representation:
        np.testing.assert_array_equal(representation['selected_indices'], indices)
        ae = representation['pca'][indices]
    # Direct standardized shape, pooled to 16x16 without annotation or rep2 inputs.
    shapes = []
    for i in indices:
        r = rows[i]
        _, oe, mask = extract(b1, r['start_bin'], r['width_bins'])
        x = tensor(oe, mask)
        weights = x[1].reshape(16, 4, 16, 4).sum((1, 3))
        values = (x[0] * x[1]).reshape(16, 4, 16, 4).sum((1, 3))
        shapes.append(np.divide(values, weights, out=np.zeros_like(values), where=weights > 0).ravel())
    shape = PCA(8).fit(np.array(shapes)).transform(np.array(shapes))
    quality = np.array([[np.log1p(rows[i]['mean_oe']), rows[i]['valid_fraction'], rows[i]['zero_fraction']] for i in indices])
    quality = StandardScaler().fit(quality).transform(quality)
    features = dict(ae_pca8=ae, shape_pca8=shape, quality_only=quality)
    # Fixed alternative: allow background to overlap OTHER candidates, never the focal
    # candidate. Backgrounds are unannotated references, not guaranteed negatives.
    controls = {}
    for i in indices:
        r = rows[i]
        occupied = np.zeros(n, bool)
        occupied[(r['start_bin'] + np.arange(r['width_bins'])) % n] = True
        matches = [j for j, s in enumerate(rows) if s['eligible'] and not s['known_ids']
                   and s['width_bins'] == r['width_bins']
                   and not occupied[(s['start_bin'] + np.arange(s['width_bins'])) % n].any()]
        matches.sort(key=lambda j: abs(np.log1p(rows[j]['mean_oe']) - np.log1p(r['mean_oe']))
                     + abs(rows[j]['valid_fraction'] - r['valid_fraction']))
        controls[str(i)] = disjoint_select(matches, rows, n)[:cfg['controls_per_candidate']]
    labels = {}
    for name, x in features.items():
        for k in cfg['clusters']:
            for seed in cfg['seeds']:
                best = float('inf')
                for attempt in range(cfg['kmeans_initializations']):
                    centers, assignment = kmeans2(x, k, minit='++', seed=seed * 100 + attempt, iter=100)
                    loss = np.square(x - centers[assignment]).sum()
                    if loss < best:
                        best, chosen = loss, assignment
                labels[f'{name}:{k}:{seed}'] = chosen.tolist()
    frozen = dict(candidate_indices=indices, controls=controls, assignments=labels,
                  warning=cfg['interpretation'], description='Frozen before reloading rep2 in this run; rep2 was already viewed in v1.')
    (out / 'frozen_before_rep2_reload.json').write_text(json.dumps(frozen, indent=2))
    np.savez_compressed(out / 'representations.npz', **features)
    print('Frozen 36 cluster runs and focal-candidate background matches; reloading previously seen rep2', flush=True)
    b2 = load_band(data['samples'][1], n, 256)
    used = sorted(set(indices) | {j for value in controls.values() for j in value})
    correlations = {}
    for i in used:
        r = rows[i]
        _, a, ma = extract(b1, r['start_bin'], r['width_bins'])
        _, b, mb = extract(b2, r['start_bin'], r['width_bins'])
        correlations[i] = correlation(a, b, ma & mb)
    support = {'v1_exclude_all_candidates': [r['replicate_supported'] == 'True' for r in candidates], 'exclude_focal_candidate': []}
    results = []
    for i, candidate in zip(indices, candidates):
        values = [correlations[j] for j in controls[str(i)] if correlations[j] is not None]
        cutoff = max(cfg['replicate_min_correlation'], float(np.quantile(values, cfg['control_quantile']))) if len(values) >= cfg['min_controls'] else None
        corr = correlations[i]
        if candidate['replicate_correlation']:
            np.testing.assert_allclose(corr, float(candidate['replicate_correlation']), rtol=1e-6)
        passes = cutoff is not None and corr is not None and corr > cutoff
        support['exclude_focal_candidate'].append(passes)
        results.append(dict(candidate_id=candidate['candidate_id'], window_index=i, known_ids=candidate['known_ids'],
                            width_bins=rows[i]['width_bins'], correlation=corr, original_controls=candidate['matched_controls'],
                            original_cutoff=candidate['control_threshold'], original_supported=candidate['replicate_supported'],
                            focal_controls=len(values), focal_cutoff=cutoff, focal_supported=passes))
    write_csv(out / 'candidate_control_ablation.csv', results)
    write_csv(out / 'reference_correlations.csv', [dict(window_index=i, correlation=correlations[i]) for i in used])
    comparisons = []
    memberships = []
    annotated = np.array([bool(r['known_ids']) for r in candidates])
    for key, assignments in labels.items():
        name, k, seed = key.split(':')
        assignment = np.array(assignments)
        for candidate, cluster in zip(candidates, assignment):
            memberships.append(dict(representation=name, k=k, seed=seed, candidate_id=candidate['candidate_id'], cluster=int(cluster)))
        for method, passes in support.items():
            pure = supported_clusters = 0
            for cluster in range(int(k)):
                member = assignment == cluster
                if member.any() and not annotated[member].any():
                    pure += 1
                    supported_clusters += int(np.array(passes)[member].sum() >= cfg['min_new_cluster_members'])
            comparisons.append(dict(representation=name, k=int(k), seed=int(seed), control_method=method,
                                    pure_unannotated_clusters=pure, clusters_passing_v1_operational_rule=supported_clusters))
    stability = []
    for name in features:
        for k in cfg['clusters']:
            scores = [adjusted_rand(labels[f'{name}:{k}:{a}'], labels[f'{name}:{k}:{b}']) for a, b in itertools.combinations(cfg['seeds'], 2)]
            stability.append(dict(representation=name, k=k, mean_pairwise_ari=float(np.mean(scores)), min_pairwise_ari=min(scores)))
    write_csv(out / 'cluster_ablation.csv', comparisons)
    write_csv(out / 'cluster_memberships.csv', memberships)
    write_csv(out / 'cluster_stability.csv', stability)
    stages = []
    for group, subset in [('all', np.ones(len(candidates), bool)), ('unannotated', ~annotated)]:
        for method in support:
            enough = np.array([int(r['original_controls']) >= cfg['min_controls'] if method.startswith('v1') else r['focal_controls'] >= cfg['min_controls'] for r in results])
            stages.append(dict(group=group, method=method, candidates=int(subset.sum()),
                               correlation_at_least_0_5=int(sum(subset[j] and r['correlation'] is not None and r['correlation'] >= .5 for j, r in enumerate(results))),
                               insufficient_controls=int((subset & ~enough).sum()), supported=int((subset & support[method]).sum())))
    summary = dict(interpretation=cfg['interpretation'], candidates=len(candidates), cluster_runs=len(labels),
                   control_cluster_combinations=len(comparisons), stages=stages, stability=stability,
                   max_clusters_passing_original_rule=max(r['clusters_passing_v1_operational_rule'] for r in comparisons),
                   frozen_sha256=hashlib.sha256((out / 'frozen_before_rep2_reload.json').read_bytes()).hexdigest())
    (out / 'report.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
