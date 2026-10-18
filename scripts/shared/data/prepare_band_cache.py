"""Explicitly build fresh 100 bp contact bands; never runs implicitly during training."""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from contact_band import make_band, cache_signature
from optional_common import ROOT, fresh_output_path, input_path, sha256, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True, help='Read-only raw matrix workspace')
    parser.add_argument('--output-dir', type=Path, required=True, help='Fresh band directory under project data/<task>/cache/ or results/<task>/<experiment>/')
    parser.add_argument('--conditions', action='store_true', help='Build six samples instead of only the two WT samples')
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--chunk-size', type=int, default=2_000_000)
    values = sys.argv[1:] if argv is None else argv
    if not values:
        parser.print_help()
        return 0
    args = parser.parse_args(values)
    if args.width < 3 or args.chunk_size <= 0:
        parser.error('width must be >=3 and chunk-size positive')
    config = _layout_resolve(ROOT, 'configs/conditions.json' if args.conditions else 'configs/data.json')
    spec = json.loads(config.read_text())
    output = fresh_output_path(args.output_dir, data_root=args.data_root, allow_cache=True)
    inputs = []
    for sample in spec['samples']:
        path = input_path(args.data_root, sample['path'])
        with h5py.File(path, 'r') as handle:
            if int(handle.attrs['bin-size']) != 10:
                raise ValueError('This historical band algorithm requires a 10 bp input')
            length = int(handle['chroms/length'][0])
        if length != spec['chrom_length']:
            raise ValueError('Chromosome length differs from data configuration')
        inputs.append((sample, path, (length + 99)//100))
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for sample, path, n in inputs:
        print('Streaming', sample['sample_id'], flush=True)
        values = make_band(path, n, args.width, chunk=args.chunk_size)
        target = output/(sample['sample_id']+'.npz')
        with target.open('xb') as stream:
            np.savez_compressed(stream, **values, signature=json.dumps(cache_signature(path, args.width)))
        records.append(dict(sample_id=sample['sample_id'], source=str(path), size=path.stat().st_size,
                            mtime_ns=path.stat().st_mtime_ns, cache=target.name, cache_sha256=sha256(target)))
    write_json(output/'manifest.json', dict(samples=records, width=args.width, resolution_bp=100,
               code_sha256=sha256(_layout_resolve(ROOT, 'scripts/contact_band.py')), config_sha256=sha256(config),
               training_performed=False, total_pixel_scan=True))
    print('Prepared', len(records), 'read-only training inputs at', output, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
