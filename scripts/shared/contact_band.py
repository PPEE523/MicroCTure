"""NumPy/HDF5 circular contact bands, independent of learned models.

The count aggregation, coverage rule and O/E arithmetic are the historical
100 bp protocol. Cache loading is read-only; explicit preparation uses a fresh directory.
"""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np

ROOT = _layout_root


def make_band(path, n, width, chunk=2_000_000):
    """Coarsen 10 bp upper-triangle counts into the historical 100 bp band."""
    band = np.zeros((n, width), np.float64)
    coverage = np.zeros(n)
    total = 0
    with h5py.File(path, 'r') as f:
        p = f['pixels']
        for off in range(0, len(p['count']), chunk):
            sl = slice(off, off + chunk)
            a = p['bin1_id'][sl] // 10
            b = p['bin2_id'][sl] // 10
            v = p['count'][sl]
            if (v < 0).any():
                raise ValueError('Negative counts')
            total += int(v.sum())
            coverage += np.bincount(a, weights=v, minlength=n) + np.bincount(b, weights=v, minlength=n)
            linear = b - a
            d = np.minimum(linear, n - linear)
            anchor = np.where(linear <= n // 2, a, b)
            use = (d >= 2) & (d < width)
            np.add.at(band, (anchor[use], d[use]), v[use])
        assert total == int(f.attrs['sum'])
        length = int(f['chroms/length'][0])
    threshold = float(np.median(coverage[coverage > 0]) * .1)
    valid = coverage >= threshold
    if length % 100:
        valid[-1] = False
    expected = np.zeros(width)
    opportunities = np.zeros(width, int)
    for d in range(2, width):
        good = valid & np.roll(valid, -d)
        opportunities[d] = good.sum()
        expected[d] = band[good, d].sum() / max(good.sum(), 1)
    return dict(band=band.astype('float32'), valid=valid, expected=expected,
                coverage=coverage, threshold=threshold, total=total, opportunities=opportunities)


def cache_signature(path, width):
    stat = _layout_resolve(path).stat()
    return dict(version=1, size=stat.st_size, mtime_ns=stat.st_mtime_ns, width=width)


def inspect_band_cache(sample, n, width, *, data_root=None, cache_root=None):
    """Inspect ZIP/NPY headers and the small signature without loading pixels."""
    base = Path(data_root or ROOT)
    path = _layout_resolve(base / sample['path'], root=base)
    cache = _layout_resolve(cache_root, root=base) if cache_root is not None else _layout_resolve(base / 'data/cache/discovery', root=base)
    file = cache / (sample['sample_id'] + '.npz')
    result = dict(path=str(file.resolve()), status='missing', band_pixels_loaded=False)
    if not file.exists():
        return result
    with np.load(file, allow_pickle=False) as z:
        signature = json.loads(str(z['signature']))
    with zipfile.ZipFile(file) as archive, archive.open('band.npy') as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError('Unsupported band NPY format')
    expected = cache_signature(path, width)
    result.update(signature=signature, expected_signature=expected, shape=list(shape), dtype=str(dtype))
    result['status'] = 'ready' if signature == expected and shape == (n, width) and dtype == np.dtype('float32') else 'stale_or_invalid'
    return result


def load_band(sample, n, width, *, data_root=None, cache_root=None, allow_build=False):
    """Read matching saved bands; the compatibility flag never permits scanning."""
    base = Path(data_root or ROOT)
    path = _layout_resolve(base / sample['path'], root=base)
    cache = _layout_resolve(cache_root, root=base) if cache_root is not None else _layout_resolve(base / 'data/cache/discovery', root=base)
    file = cache / (sample['sample_id'] + '.npz')
    signature = cache_signature(path, width)
    if file.exists():
        with np.load(file, allow_pickle=False) as z:
            if json.loads(str(z['signature'])) == signature:
                return {k: z[k].copy() for k in z.files if k != 'signature'}
    raise FileNotFoundError(
        f'No matching read-only band cache: {file}; no raw scan was started and existing bytes were preserved. '
        'Use python -B run.py prepare-bands --help and choose a fresh cache version.')


def extract(b, start, width):
    n = len(b['valid'])
    ids = (start + np.arange(width)) % n
    a = ids[:, None]
    c = ids[None, :]
    delta = np.abs(a - c)
    d = np.minimum(delta, n - delta)
    anchor = np.where(delta <= n // 2, np.minimum(a, c), np.maximum(a, c))
    raw = b['band'][anchor, d]
    valid = b['valid'][ids]
    mask = valid[:, None] & valid[None, :] & (d >= 2)
    oe = np.divide(raw, b['expected'][d], out=np.zeros_like(raw), where=b['expected'][d] > 0)
    return raw, oe, mask
