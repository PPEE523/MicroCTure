"""Stream each WT upper triangle into a compact whole-genome 5 kb matrix."""
import json
import h5py
import numpy as np
from optional_common import ROOT, write_json, sha256


def aggregate_upper(a, b, counts, n):
    upper = np.bincount(a*n+b, weights=counts, minlength=n*n).reshape(n, n)
    # Each original molecule is counted once on the coarsened diagonal.
    return upper+upper.T-np.diag(np.diag(upper))


def main():
    config = json.loads((ROOT/'configs/optional_tasks.json').read_text())['task6']
    directory = ROOT/'data/cache/task6'
    directory.mkdir(parents=True, exist_ok=True)
    samples = json.loads((ROOT/'configs/conditions.json').read_text())['samples'][:2]
    audit = []
    for sample in samples:
        source = ROOT/sample['path']
        dest = directory/(sample['sample_id']+'.npz')
        signature = dict(source=sample['path'], bytes=source.stat().st_size,
                         mtime_ns=source.stat().st_mtime_ns, resolution_bp=config['resolution_bp'],
                         source_sha256=sha256(ROOT/'scripts/prepare_task6.py'))
        if dest.exists():
            with np.load(dest) as z:
                if json.loads(str(z['signature'])) != signature:
                    raise ValueError('Task 6 source/cache mismatch; preserve old outputs before rebuilding')
                audit.append(json.loads(str(z['audit'])))
            print('Verified existing coarse cache', sample['sample_id'], flush=True)
            continue
        with h5py.File(source, 'r') as f:
            length = int(f['chroms/length'][0])
            source_bin = int(f.attrs['bin-size'])
            assert config['resolution_bp'] % source_bin == 0
            factor = config['resolution_bp']//source_bin
            n = (length+config['resolution_bp']-1)//config['resolution_bp']
            upper, total = np.zeros((n, n)), 0
            pixels = f['pixels']
            for off in range(0, len(pixels['count']), 2_000_000):
                sl = slice(off, off+2_000_000)
                a, b = pixels['bin1_id'][sl]//factor, pixels['bin2_id'][sl]//factor
                values = pixels['count'][sl]
                if (values < 0).any():
                    raise ValueError('Negative contacts')
                upper += np.bincount(a*n+b, weights=values, minlength=n*n).reshape(n, n)
                total += int(values.sum())
                if off % 50_000_000 == 0:
                    print(sample['sample_id'], off, '/', len(pixels['count']), flush=True)
            assert total == int(f.attrs['sum']) == int(upper.sum())
        matrix = upper+upper.T-np.diag(np.diag(upper))
        row = dict(sample_id=sample['sample_id'], bins=n, resolution_bp=config['resolution_bp'],
                   chromosome_length=length, total_unique_contacts=total,
                   last_bin_bp=length-(n-1)*config['resolution_bp'], symmetric=True)
        np.savez_compressed(dest, counts=matrix, chromosome_length=length, total=total,
                            signature=json.dumps(signature), audit=json.dumps(row))
        audit.append(row)
        print('Prepared', sample['sample_id'], n, 'bins', flush=True)
    write_json(ROOT/'reports/task6_data_audit.json', audit)


if __name__ == '__main__':
    main()
