"""Read-only input checks/review, or an explicitly requested new task 3/4 run.

``check`` reads Cooler metadata and existing cache headers, never raw pixels.
``review`` additionally verifies preserved numeric tables, without refitting.
``analyze`` is the separate, potentially long full plotting operation.  It only
uses existing caches and writes exclusively into a new directory in this clone.
"""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from contact_band import inspect_band_cache

ROOT = _layout_root
SOURCE_FILES = ['scripts/contact_band.py', 'scripts/run_conditions_local.py',
                'scripts/analyze_conditions.py', 'scripts/discover_task2.py']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=sha256(path))


def input_path(data_root, relative):
    from optional_common import input_path as resolve_input
    return resolve_input(data_root, relative)


def new_run_path(value, *, root=ROOT, data_root=None):
    from optional_common import fresh_output_path
    return fresh_output_path(value, root=root, data_root=data_root)


def inspect_inputs(data_root, config_path, band_cache_dir=None):
    data_root = Path(data_root).resolve()
    config_path = _layout_resolve(config_path).resolve()
    config = json.loads(config_path.read_text())
    audit_path = input_path(data_root, 'reports/conditions_audit.json')
    audit = json.loads(audit_path.read_text())
    if not audit['bin_coordinates_identical']:
        raise ValueError('Historical sample audit has incompatible coordinates')
    samples = {s['sample_id']: s for s in audit['samples']}
    if len(samples) != len(config['samples']) or len(samples) != 6:
        raise ValueError('Expected six unique audited samples')
    conditions = config['conditions']
    if any(sorted(s['replicate'] for s in config['samples'] if s['condition'] == c) != [1, 2] for c in conditions):
        raise ValueError('Each condition requires biological repeats 1 and 2')
    n = (config['chrom_length'] + 99) // 100
    records = []
    for sample in config['samples']:
        saved = samples[sample['sample_id']]
        if any(sample[k] != saved[k] for k in ['condition', 'replicate', 'path']):
            raise ValueError(f'Sample identity differs from audit: {sample["sample_id"]}')
        path = input_path(data_root, sample['path'])
        if path.stat().st_size != saved['size_bytes']:
            raise ValueError(f'Raw file size differs from audit: {path}')
        with h5py.File(path, 'r') as handle:
            chroms = [s.decode() if isinstance(s, bytes) else str(s) for s in handle['chroms/name'][:]]
            if chroms != [config['chrom']] or handle['chroms/length'][:].tolist() != [config['chrom_length']]:
                raise ValueError(f'Unexpected genome: {path}')
            if int(handle.attrs['bin-size']) != 10 or handle.attrs['genome-assembly'] != saved['assembly'] or handle.attrs['storage-mode'] != 'symmetric-upper':
                raise ValueError(f'Unexpected Cooler format: {path}')
            if int(handle.attrs['sum']) != saved['metadata_count_sum'] or int(handle.attrs['nnz']) != saved['nnz']:
                raise ValueError(f'Count metadata differs from audit: {path}')
            digest = hashlib.sha256()
            for key in ['chrom', 'start', 'end']:
                digest.update(handle[f'bins/{key}'][:].astype('<i8').tobytes())
            if digest.hexdigest() != saved['bin_coordinates_sha256']:
                raise ValueError(f'Bin coordinates differ from audit: {path}')
        cache = inspect_band_cache(sample, n, 256, data_root=data_root, cache_root=band_cache_dir)
        if cache['status'] != 'missing':
            cache['sha256'] = sha256(cache['path'])
        records.append(dict(sample_id=sample['sample_id'], path=str(path), size_bytes=path.stat().st_size,
                            mtime_ns=path.stat().st_mtime_ns, historical_audit_sha256=saved['sha256'],
                            fresh_raw_sha256_verified=False, bin_coordinates_sha256=digest.hexdigest(), cache=cache))
    metadata = input_path(data_root, 'data/metadata/GSE272159_family.soft')
    if sha256(metadata) != audit['metadata_sha256']:
        raise ValueError('GEO metadata differs from the audited source')
    small_inputs = [config_path, audit_path, metadata,
                    input_path(data_root, config['annotation_path']),
                    input_path(data_root, 'data/processed/structures.csv')]
    return dict(status='passed', samples=records, data_root=str(data_root), config_path=str(config_path),
                raw_pixels_scanned=False, caches_built=0, band_pixels_loaded=False,
                analysis_ready=all(s['cache']['status'] == 'ready' for s in records),
                input_files=[file_record(p) for p in small_inputs],
                coordinate_status=config['coordinate_status'],
                scope='Metadata/cache inspection; historical whole-file raw hashes are referenced, not reverified')


def csv_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def review_preserved(data_root, config_path):
    """Recompute historical decisions from saved signal tables, not raw pixels."""
    from analyze_conditions import compare
    config = json.loads(_layout_resolve(config_path).read_text())
    data_root = Path(data_root).resolve()
    values_path = input_path(data_root, 'results/task4/structure_signals.csv')
    differences_path = input_path(data_root, 'results/task4/differential_structures.csv')
    summary_path = input_path(data_root, 'results/task4/summary.csv')
    values = csv_rows(values_path)
    differences = csv_rows(differences_path)
    saved_report_path = _layout_resolve(data_root / 'reports/task34_results.json', root=data_root)
    saved_report = json.loads(saved_report_path.read_text())
    if sha256(config_path) != saved_report['input_sha256']['config'] or sha256(_layout_resolve(data_root / 'reports/conditions_audit.json', root=data_root)) != saved_report['input_sha256']['samples_audit']:
        raise ValueError('Preserved result input hashes differ from the configuration/audit')
    known = {r['structure_id']: r for r in csv_rows(_layout_resolve(data_root / 'data/processed/structures.csv', root=data_root))}
    lookup = {(r['structure_id'], r['condition'], int(r['replicate'])): r for r in values}
    if len(known) != 344 or len(values) != len(lookup) or len(values) != 344 * 6 or len(differences) != 344 * 2:
        raise ValueError('Preserved table coverage or unique identity mismatch')
    identities = {(r['structure_id'], r['condition']) for r in differences}
    if len(identities) != len(differences):
        raise ValueError('Duplicate structure/condition comparisons')
    max_error = 0.
    for row in differences:
        if any(row[k] != known[row['structure_id']][k] for k in ['type', 'chrom', 'start', 'end']):
            raise ValueError('Comparison coordinates differ from author annotation')
        def pair(condition, column):
            return [float(lookup[row['structure_id'], condition, rep][column]) for rep in [1, 2]]
        wt = pair(config['conditions'][0], 'signal')
        treatment = pair(row['condition'], 'signal')
        np.testing.assert_allclose(wt, [float(row['wt_rep1']), float(row['wt_rep2'])], rtol=1e-12, atol=0)
        np.testing.assert_allclose(treatment, [float(row['treated_rep1']), float(row['treated_rep2'])], rtol=1e-12, atol=0)
        effect = compare(wt, treatment, config['pseudocount_per_million'], config['fold_change_threshold'])
        oe = compare(pair(config['conditions'][0], 'mean_oe'), pair(row['condition'], 'mean_oe'), .01, config['fold_change_threshold'])
        quality = int(row['valid_pairs']) >= config['structure_min_pairs'] and float(row['valid_fraction']) >= config['structure_min_valid_fraction']
        expected = dict(quality_pass=str(quality), direction=effect['direction'],
                        descriptive_change=str(quality and effect['direction'] != 'inconsistent_or_small'),
                        oe_direction=oe['direction'],
                        oe_direction_agrees=str(bool(np.sign(effect['log2_fold_change']) == np.sign(oe['log2_fold_change']))))
        if any(row[k] != v for k, v in expected.items()):
            raise ValueError(f'Decision mismatch: {row["structure_id"]}')
        for key in ['log2_fold_change', 'min_cross_repeat_log2_fc', 'max_cross_repeat_log2_fc', 'wt_repeat_log2_ratio', 'treatment_repeat_log2_ratio']:
            delta = abs(float(row[key]) - effect[key])
            max_error = max(max_error, delta)
            np.testing.assert_allclose(float(row[key]), effect[key], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(float(row['oe_log2_fold_change']), oe['log2_fold_change'], rtol=1e-12, atol=1e-12)
    summary = csv_rows(summary_path)
    expected_summary_ids = {(c, kind) for c in config['conditions'][1:] for kind in ['OPCID', 'CHIN', 'CHID']}
    if len(summary) != 6 or {(r['condition'], r['type']) for r in summary} != expected_summary_ids:
        raise ValueError('Summary must contain all six condition/type combinations')
    for row in summary:
        subset = [r for r in differences if r['condition'] == row['condition'] and r['type'] == row['type']]
        good = [r for r in subset if r['quality_pass'] == 'True']
        expected = dict(total=len(subset), quality_pass=len(good),
                        increased=sum(r['descriptive_change'] == 'True' and r['direction'] == 'increased' for r in subset),
                        decreased=sum(r['descriptive_change'] == 'True' and r['direction'] == 'decreased' for r in subset))
        if any(int(row[k]) != value for k, value in expected.items()):
            raise ValueError('Summary count mismatch')
        np.testing.assert_allclose(float(row['median_log2_fc']), np.median([float(r['log2_fold_change']) for r in good]), rtol=1e-12, atol=1e-12)
    pages_path = input_path(data_root, 'results/task3/pages.csv')
    signals_path = input_path(data_root, 'results/task3/signals.npz')
    pages = csv_rows(pages_path)
    cursor = 0
    for page in pages:
        if int(page['start']) != cursor:
            raise ValueError('Non-contiguous track pages')
        cursor = min(cursor + config['track_window_bp'], config['chrom_length'])
        if int(page['end']) != cursor:
            raise ValueError('Unexpected track page end')
    if cursor != config['chrom_length']:
        raise ValueError('Track pages do not cover the chromosome')
    with np.load(signals_path, allow_pickle=False) as z:
        signals, common = z['signals'], z['common_valid']
        n = (config['chrom_length'] + 99) // 100
        if signals.shape != (6, n) or not np.isfinite(signals[:, common]).all() or not np.isnan(signals[:, ~common]).all():
            raise ValueError('Signal mask/coverage mismatch')
        groups = [[i for i, s in enumerate(config['samples']) if s['condition'] == c] for c in config['conditions']]
        np.testing.assert_allclose(z['condition_means'], [signals[g].mean(0) for g in groups])
        np.testing.assert_array_equal(z['positions'], np.arange(n) * config['resolution_bp'])
        valid_count = int(common.sum())
    if valid_count != saved_report['common_valid_bins'] or len(pages) != saved_report['track_pages']:
        raise ValueError('Track coverage differs from the preserved report')
    files = [values_path, differences_path, summary_path, pages_path, signals_path, saved_report_path]
    return dict(status='passed', comparisons_checked=len(differences), structure_sample_values=len(values),
                max_log2_fc_error=max_error, track_pages=len(pages), common_valid_bins=valid_count,
                summary=summary, input_files=[file_record(p) for p in files],
                raw_pixels_scanned=False, models_fitted=0, images_generated=0,
                interpretation='Descriptive relative contact changes in author-annotated intervals; no p/FDR, causal or morphology certification')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'review', 'analyze'])
    parser.add_argument('--data-root', type=Path, required=True, help='Read-only existing workspace; data are not copied')
    parser.add_argument('--run-dir', type=Path, required=True, help='New version directory under this clone/results/')
    parser.add_argument('--config', type=Path, default=_layout_resolve(ROOT / 'configs/conditions.json'))
    parser.add_argument('--band-cache-dir', type=Path, help='Existing six-sample band directory from prepare_band_cache.py')
    args = parser.parse_args(argv)
    run_dir = new_run_path(args.run_dir, data_root=args.data_root)
    args.config = _layout_resolve(args.config).resolve()
    if args.band_cache_dir is not None:
        args.band_cache_dir = _layout_resolve(args.band_cache_dir, root=args.data_root.resolve()).resolve()
    check = inspect_inputs(args.data_root, args.config, args.band_cache_dir)
    review = review_preserved(args.data_root, args.config) if args.action == 'review' else None
    if args.action == 'analyze':
        if not check['analysis_ready']:
            raise ValueError('Existing caches are missing/stale; no raw preprocessing was started')
        from analyze_conditions import main as analyze
        previous_mpl_config = os.environ.get('MPLCONFIGDIR')
        try:
            with TemporaryDirectory(prefix='microcture-task34-mpl-') as mpl_config:
                os.environ['MPLCONFIGDIR'] = mpl_config
                analyze(data_root=args.data_root, run_dir=run_dir, config_path=args.config, cache_root=args.band_cache_dir)
        finally:
            if previous_mpl_config is None:
                os.environ.pop('MPLCONFIGDIR', None)
            else:
                os.environ['MPLCONFIGDIR'] = previous_mpl_config
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'input_check.json').write_text(json.dumps(check, indent=2) + '\n')
    if review is not None:
        (run_dir / 'preserved_review.json').write_text(json.dumps(review, indent=2) + '\n')
    outputs = [file_record(p) for p in sorted(run_dir.rglob('*')) if p.is_file()]
    manifest = dict(schema_version=1, action=args.action, created_utc=datetime.now(timezone.utc).isoformat(),
                    run_dir=str(run_dir), data_root=str(args.data_root.resolve()),
                    source_files=[file_record(_layout_resolve(ROOT / p)) for p in SOURCE_FILES],
                    input_files=check['input_files'] + (review['input_files'] if review else []),
                    raw_samples=check['samples'], output_files=outputs,
                    scientific_protocol='Legacy fixed author intervals, 100 bp bin-index distances 2–100, original common mask/depth normalization',
                    caveats=['Historical raw whole-file hashes are recorded, not freshly verified',
                             'Known-interval relative signal changes do not establish structure discovery or morphological reliability',
                             'Coordinate convention remains provisional; two repeats per condition, no significance/causality test'])
    (run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(status='passed', action=args.action, run_dir=str(run_dir),
                          analysis_ready=check['analysis_ready'], raw_pixels_scanned=False,
                          caches_built=0, comparisons_checked=review['comparisons_checked'] if review else 0), indent=2))


if __name__ == '__main__':
    main()
