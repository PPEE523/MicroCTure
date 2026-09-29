"""Check the frozen-candidate v2 diagnostic without selecting new parameters."""
import hashlib
import json
import numpy as np
from diagnose_discovery_v2 import ROOT, read_csv


def main():
    config = json.loads((ROOT / 'configs/discovery_v2_diagnostic.json').read_text())
    out = ROOT / config['output_dir']
    manifest = json.loads((out / 'run_manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    frozen = json.loads((out / 'frozen_before_rep2_reload.json').read_text())
    report = json.loads((out / 'report.json').read_text())
    assert hashlib.sha256((out / 'frozen_before_rep2_reload.json').read_bytes()).hexdigest() == report['frozen_sha256']
    rows = read_csv('results/task2/scan_rep1.csv')
    original = read_csv('results/task2/candidates.csv')
    checks = read_csv(f"{config['output_dir']}/candidate_control_ablation.csv")
    data = json.loads((ROOT / 'configs/data.json').read_text())
    n = (data['chrom_length'] + 99) // 100
    correlations = {int(r['window_index']): float(r['correlation']) if r['correlation'] else None
                    for r in read_csv(f"{config['output_dir']}/reference_correlations.csv")}
    assert len(checks) == len(original) == 189
    def bins(i):
        return (int(rows[i]['start_bin']) + np.arange(int(rows[i]['width_bins']))) % n
    for r, old in zip(checks, original):
        i = int(r['window_index'])
        assert i == int(old['window_index']) and r['candidate_id'] == old['candidate_id']
        assert r['original_supported'] == old['replicate_supported']
        occupied = np.zeros(n, bool)
        occupied[bins(i)] = True
        selected = frozen['controls'][str(i)]
        assert len(selected) <= config['controls_per_candidate']
        for j in selected:
            assert not occupied[bins(j)].any()
            occupied[bins(j)] = True
            assert rows[j]['width_bins'] == rows[i]['width_bins']
            assert rows[j]['eligible'] == 'True' and not rows[j]['known_ids']
        values = [correlations[j] for j in selected if correlations[j] is not None]
        assert len(values) == int(r['focal_controls'])
        enough = len(values) >= config['min_controls']
        cutoff = max(config['replicate_min_correlation'], np.quantile(values, config['control_quantile'])) if enough else None
        if enough:
            np.testing.assert_allclose(cutoff, float(r['focal_cutoff']))
        expected = enough and correlations[i] is not None and correlations[i] > cutoff
        assert r['focal_supported'] == str(bool(expected))
    annotated = np.array([bool(r['known_ids']) for r in original])
    comparisons = read_csv(f"{config['output_dir']}/cluster_ablation.csv")
    assert len(comparisons) == 72 and len(frozen['assignments']) == 36
    for row in comparisons:
        assignment = np.array(frozen['assignments'][f"{row['representation']}:{row['k']}:{row['seed']}"])
        key = 'original_supported' if row['control_method'].startswith('v1') else 'focal_supported'
        support = np.array([r[key] == 'True' for r in checks])
        pure = passing = 0
        for cluster in np.unique(assignment):
            selected = assignment == cluster
            if not annotated[selected].any():
                pure += 1
                passing += int(support[selected].sum() >= config['min_new_cluster_members'])
        assert pure == int(row['pure_unannotated_clusters'])
        assert passing == int(row['clusters_passing_v1_operational_rule'])
    for stage in report['stages']:
        subset = [r for r in checks if stage['group'] == 'all' or not r['known_ids']]
        prefix = 'original' if stage['method'].startswith('v1') else 'focal'
        assert len(subset) == stage['candidates']
        assert sum(r[f'{prefix}_supported'] == 'True' for r in subset) == stage['supported']
        assert sum(int(r[f'{prefix}_controls']) < config['min_controls'] for r in subset) == stage['insufficient_controls']
    result = dict(status='passed', fixed_candidates=189, checked_control_sets=189, cluster_runs=36,
                  checked_control_cluster_combinations=72, source_hashes_match=True,
                  focal_and_reference_windows_disjoint=True, independent_validation=False)
    (ROOT / 'reports/task2_v2_diagnostic_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
