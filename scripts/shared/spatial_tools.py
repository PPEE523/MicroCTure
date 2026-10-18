"""Pure NumPy spatial safeguards and pooling shared by discovery/reconstruction."""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()
import numpy as np


def distance(a, b, n):
    delta = np.abs(a-b)
    return np.minimum(delta, n-delta)


def assert_disjoint(footprints, split, n, guard):
    occupancy = [np.zeros(n, bool) for _ in range(3)]
    for ids, role in zip(footprints, split):
        occupancy[role][ids] = True
    for i in range(3):
        for j in range(i):
            for shift in range(-guard, guard+1):
                if (occupancy[i] & np.roll(occupancy[j], shift)).any():
                    raise ValueError('Cross-split footprint/guard overlap')
    return occupancy


def pool_masked(array, valid, size):
    factor = len(array)//size
    shape = (size, factor, size, factor)
    count = valid.reshape(shape).sum(axis=(1, 3))
    summed = np.where(valid, array, 0).reshape(shape).sum(axis=(1, 3))
    value = np.divide(summed, count, out=np.zeros_like(summed), where=count > 0)
    return np.stack([value, count/(factor*factor)]).astype('float32')
