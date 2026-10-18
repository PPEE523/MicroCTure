"""Small shared I/O helpers for optional experiments; no fitted preprocessing."""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()
import csv
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = _layout_root


def read_csv(path):
    with _layout_resolve(path).open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def sha256(path):
    digest = hashlib.sha256()
    with _layout_resolve(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def input_path(data_root, relative):
    """Resolve historical names before testing namespaces in the organized tree."""
    base = Path(data_root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Input names must be relative to --data-root')
    # A canonical results/task5 directory is a namespace, not the old experiment.
    candidates = [_layout_resolve(base / 'data/legacy_experiments' / relative, root=base),
                  _layout_resolve(base / relative, root=base)]
    for candidate in candidates:
        path = candidate.resolve()
        if not path.is_relative_to(base):
            raise ValueError('Input symlink escapes --data-root')
        if path.exists():
            return path
    raise FileNotFoundError(f'Missing read-only input: {candidates[-1]}')


def _protected_output_trees(root):
    """Read explicit migration/freeze indices without rewriting scientific data."""
    from microcture.layout import mappings
    paths = []
    for row in mappings(root):
        if row.get('kind') in {'result', 'report', 'cache', 'raw'}:
            candidate = root / row['new']
            if candidate.is_dir():
                paths.append(candidate.resolve())
    for manifest in (root / 'evidence').glob('*/freeze_manifest.json'):
        frozen = json.loads(manifest.read_text())
        for name in frozen.get('directories', []) + frozen.get('frozen_trees', []):
            paths.append(_layout_resolve(root / name, root=root).resolve())
        for key in ('run_dir', 'report_dir', 'cache_dir'):
            if frozen.get(key):
                paths.append(_layout_resolve(root / frozen[key], root=root).resolve())
    return paths


def fresh_output_path(value, *, root=ROOT, data_root=None, allow_reports=False, allow_cache=False):
    """Select a new task/experiment/version without entering retained evidence."""
    root = Path(root).resolve()
    path = _layout_resolve(value, root=root)
    if any(parent.is_symlink() for parent in (path,*path.parents)):
        raise ValueError('Output cannot pass through a symlink')
    path = path.resolve()
    tops = [root / 'results'] + ([root / 'reports'] if allow_reports else [])
    inside_output = any(path.is_relative_to(top) and path != top for top in tops)
    if path.is_relative_to(root / 'results'):
        parts = path.relative_to(root / 'results').parts
        inside_output = len(parts) >= 3 and parts[0] in {f'task{i}' for i in range(1, 7)}
    if allow_cache and path.is_relative_to(root / 'data'):
        parts = path.relative_to(root / 'data').parts
        inside_output = (len(parts) >= 3 and parts[0] in {'shared', *(f'task{i}' for i in range(1, 7))}
                         and (parts[1] == 'cache' or parts[:2] == ('task1', 'ablation')))
    if not inside_output:
        raise ValueError('Output must be a fresh project results/reports version or an explicitly allowed data cache')
    if data_root is not None and Path(data_root).resolve() != root and path.is_relative_to(Path(data_root).resolve()):
        raise ValueError('Cannot write into the external input workspace')
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {path}')
    if any(path == frozen or path.is_relative_to(frozen) or frozen.is_relative_to(path)
           for frozen in _protected_output_trees(root)):
        raise ValueError('Output overlaps retained or frozen scientific evidence; choose a new version')
    return path


def fresh_report_path(value, *, root=ROOT, data_root=None):
    return fresh_output_path(value, root=root, data_root=data_root, allow_reports=True)


def experiment_arguments(description, default_config, argv=None, *, extra_paths=(), allow_cache=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data-root', type=Path, required=True, help='Read-only existing data/cache/result root')
    parser.add_argument('--output-dir', type=Path, required=True, help='Fresh project data cache version' if allow_cache else 'Fresh version under this project/results/')
    parser.add_argument('--config', type=Path, default=_layout_resolve(ROOT / default_config))
    for option, help_text in extra_paths:
        parser.add_argument(option, type=Path, help=help_text)
    values = sys.argv[1:] if argv is None else argv
    if not values:
        parser.print_help()
        return None
    args = parser.parse_args(values)
    args.data_root = args.data_root.resolve()
    args.config = _layout_resolve(args.config).resolve()
    args.output_dir = fresh_output_path(args.output_dir, data_root=args.data_root, allow_cache=allow_cache)
    for name in ('split_file', 'cache_dir', 'band_cache_dir', 'scan_file'):
        if getattr(args, name, None) is not None:
            setattr(args, name, _layout_resolve(getattr(args, name), root=args.data_root).resolve())
    return args


def signatures(source_paths, input_paths):
    """Bind code and actual resolved inputs without copying external data."""
    return {str(_layout_resolve(path).resolve()): sha256(path) for path in list(source_paths) + list(input_paths)}


def runtime_paths(args):
    value = dict(data_root=str(args.data_root), output_dir=str(args.output_dir), config=str(args.config),
                 cache_policy='existing read-only caches; no implicit raw preprocessing', resume=False)
    for name in ['split_file', 'cache_dir', 'band_cache_dir', 'scan_file']:
        if getattr(args, name, None) is not None:
            value[name] = str(getattr(args, name).resolve())
    return value


@contextmanager
def temporary_plotting():
    previous = os.environ.get('MPLCONFIGDIR')
    try:
        with TemporaryDirectory(prefix='microcture-plot-') as folder:
            os.environ['MPLCONFIGDIR'] = folder
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            yield plt
    finally:
        if previous is None:
            os.environ.pop('MPLCONFIGDIR', None)
        else:
            os.environ['MPLCONFIGDIR'] = previous


def correlation(a, b):
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3 or np.std(a[good]) < 1e-12 or np.std(b[good]) < 1e-12:
        return None
    return float(np.corrcoef(a[good], b[good])[0, 1])
