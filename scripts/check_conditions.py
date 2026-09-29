"""Independently verify task 3/4 tables, decisions, coverage and image files."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with (ROOT / path).open() as handle:
        return list(csv.DictReader(handle))


def main():
    config = json.loads((ROOT / 'configs/conditions.json').read_text())
    report = json.loads((ROOT / 'reports/task34_results.json').read_text())
    audit = json.loads((ROOT / 'reports/conditions_audit.json').read_text())
    for key, path in [('config', 'configs/conditions.json'), ('samples_audit', 'reports/conditions_audit.json')]:
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == report['input_sha256'][key]
    assert len(audit['samples']) == 6
    assert len({s['bin_coordinates_sha256'] for s in audit['samples']}) == 1
    groups = [[i for i, s in enumerate(config['samples']) if s['condition'] == c] for c in config['conditions']]
    assert all(len(g) == 2 for g in groups)
    pages = rows('results/task3/pages.csv')
    cursor = 0
    for page in pages:
        assert int(page['start']) == cursor
        cursor = min(cursor + config['track_window_bp'], config['chrom_length'])
        assert int(page['end']) == cursor
    assert cursor == config['chrom_length'] and len(pages) == report['track_pages'] == 465
    with np.load(ROOT / 'results/task3/signals.npz') as data:
        signals, common = data['signals'], data['common_valid']
        assert signals.shape == (6, report['total_bins'])
        assert common.sum() == report['common_valid_bins']
        assert np.isfinite(signals[:, common]).all()
        assert np.isnan(signals[:, ~common]).all()
        np.testing.assert_allclose(data['condition_means'], [signals[g].mean(0) for g in groups])
        np.testing.assert_array_equal(data['positions'], np.arange(report['total_bins']) * config['resolution_bp'])
    values = rows('results/task4/structure_signals.csv')
    lookup = {(r['structure_id'], r['condition'], int(r['replicate'])): float(r['signal']) for r in values}
    assert len(values) == len(lookup) == 344 * 6
    differences = rows('results/task4/differential_structures.csv')
    assert len(differences) == len({(r['structure_id'], r['condition']) for r in differences}) == 688
    p, fold = config['pseudocount_per_million'], config['fold_change_threshold']
    for r in differences:
        wt = np.array([float(r['wt_rep1']), float(r['wt_rep2'])])
        treatment = np.array([float(r['treated_rep1']), float(r['treated_rep2'])])
        for cond, pair in [('WT_37C', wt), (r['condition'], treatment)]:
            np.testing.assert_allclose(pair, [lookup[r['structure_id'], cond, rep] for rep in [1, 2]])
        ratios = (treatment[:, None] + p) / (wt[None, :] + p)
        direction = 'increased' if (ratios > fold).all() else 'decreased' if (ratios < 1 / fold).all() else 'inconsistent_or_small'
        quality = int(r['valid_pairs']) >= config['structure_min_pairs'] and float(r['valid_fraction']) >= config['structure_min_valid_fraction']
        assert r['quality_pass'] == str(quality)
        assert r['direction'] == direction
        assert r['descriptive_change'] == str(quality and direction != 'inconsistent_or_small')
        np.testing.assert_allclose(float(r['log2_fold_change']), np.log2((treatment.mean() + p) / (wt.mean() + p)))
        np.testing.assert_allclose([float(r['min_cross_repeat_log2_fc']), float(r['max_cross_repeat_log2_fc'])], np.log2([ratios.min(), ratios.max()]))
    for row in report['summary']:
        subset = [r for r in differences if r['condition'] == row['condition'] and r['type'] == row['type']]
        good = [r for r in subset if r['quality_pass'] == 'True']
        assert len(subset) == row['total'] and len(good) == row['quality_pass']
        for direction in ['increased', 'decreased']:
            assert sum(r['descriptive_change'] == 'True' and r['direction'] == direction for r in subset) == row[direction]
        np.testing.assert_allclose(row['median_log2_fc'], np.median([float(r['log2_fold_change']) for r in good]))
    pictures = [ROOT / 'results/task3' / p['file'] for p in pages]
    heatmaps = list((ROOT / 'results/task4/heatmaps').glob('*.png'))
    assert len(heatmaps) == report['heatmaps'] == 20
    pictures += heatmaps + [ROOT / 'results/task4/effects.png']
    for path in pictures:
        with Image.open(path) as image:
            image.verify()
    checks = dict(status='passed', samples=6, contiguous_track_pages=len(pages),
                  structure_sample_values=len(values), independently_checked_comparisons=len(differences),
                  pngs_verified=len(pictures), source_hashes_match=True, summary_counts_match=True)
    (ROOT / 'reports/task34_verification.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
