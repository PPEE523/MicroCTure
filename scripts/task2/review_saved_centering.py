"""Review saved centering tables with explicit NA; no contact data or fitting."""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()
import argparse
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = _layout_root


def output_path(value, data_root, root=ROOT):
    from optional_common import fresh_report_path
    return fresh_report_path(value, root=root, data_root=data_root)


def inventory(data_root):
    data_root = Path(data_root).resolve()
    from optional_common import input_path
    directories = {'primary': input_path(data_root, 'results/task2_centering'),
                   'supplement': input_path(data_root, 'results/task2_centering_supplement')}
    required = ['report.json', 'candidate_variants.csv', 'candidate_summary.csv', 'thresholds.csv']
    files, cohorts = [], {}
    for name, directory in directories.items():
        for filename in required + (['variants.csv'] if name == 'primary' else []):
            path = directory / filename
            if not path.is_file():
                raise FileNotFoundError(f'Missing saved input: {path}')
            files.append(dict(path=str(path), size_bytes=path.stat().st_size,
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        report = json.loads((directory / 'report.json').read_text())
        cohorts[name] = dict(path=str(directory), calibration_windows=report['calibration_windows'],
                             candidates=report['candidates'], exploratory=True)
    return dict(status='saved_inputs_available', cohorts=cohorts, input_files=files,
                training_performed=False, raw_pixels_read=False, historical_source_hash_match_claimed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'review'])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, default=_layout_resolve(ROOT / 'reports/task2_centering_retained_review_v1'))
    args = parser.parse_args(argv)
    output = output_path(args.report_dir, args.data_root)
    inputs = inventory(args.data_root)
    cfg = json.loads((_layout_resolve(ROOT / 'configs/centering_sensitivity.json')).read_text())
    # Set the per-run plotting cache before importing the rendering module.
    results = {}
    if args.action == 'review':
        previous = os.environ.get('MPLCONFIGDIR')
        try:
            with TemporaryDirectory(prefix='microcture-centering-review-', dir='/tmp') as temporary:
                os.environ['MPLCONFIGDIR'] = temporary
                from report_centering import read_rows, write_review
                settings = read_rows(Path(inputs['cohorts']['primary']['path']) / 'variants.csv')
                for name in ('primary', 'supplement'):
                    results[name] = write_review(inputs['cohorts'][name]['path'], output / name, cfg,
                                                settings=settings, saved_tables_only=True,
                                                title=f'{name}: retained centering evidence')
        finally:
            if previous is None:
                os.environ.pop('MPLCONFIGDIR', None)
            else:
                os.environ['MPLCONFIGDIR'] = previous
    else:
        output.mkdir(parents=True)
    manifest = dict(action=args.action, **inputs, results=results,
                    source_files={str(_layout_resolve(ROOT / 'scripts' / name)): hashlib.sha256((_layout_resolve(ROOT / 'scripts' / name)).read_bytes()).hexdigest()
                                  for name in ('review_saved_centering.py', 'centering_status.py', 'report_centering.py')},
                    config=cfg, interpretation='NA is missing evidence, never a negative result. Supplement reuses viewed regions; no independent biological confirmation.')
    (output / 'review.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='passed', action=args.action, report_dir=str(output),
                          primary_assessment=results.get('primary', {}).get('assessment_status'),
                          primary_robust_anomalies=results.get('primary', {}).get('robust_recentered_anomalies')),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
