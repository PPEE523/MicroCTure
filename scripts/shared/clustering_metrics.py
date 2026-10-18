"""Small clustering agreement metric used by discovery validation."""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()
import numpy as np


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
