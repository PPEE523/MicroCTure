"""Small shared I/O helpers for optional experiments; no fitted preprocessing."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def correlation(a, b):
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3 or np.std(a[good]) < 1e-12 or np.std(b[good]) < 1e-12:
        return None
    return float(np.corrcoef(a[good], b[good])[0, 1])
