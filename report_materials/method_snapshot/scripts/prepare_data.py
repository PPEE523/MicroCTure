"""Read-only audit of raw inputs and reproducible annotation export.

Run from any directory: .venv/bin/python scripts/prepare_data.py
Only bins/indexes and a small contact window are loaded, never all pixels.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import platform

import cooler
import h5py
import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def integer(value):
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f'Expected numeric coordinate, got {value!r}')
    require(np.isfinite(value) and int(value) == value, f'Noninteger coordinate: {value}')
    return int(value)


def convert_interval(start, end, center, convention):
    start, end = integer(start), integer(end)
    center = integer(center) if center is not None else None
    if convention == '1-based-closed':
        start -= 1
        if center is not None:
            center -= 1
    else:
        require(convention == '0-based-half-open', 'Unsupported coordinate convention')
    if center is None:
        center = (start + end) / 2
    require(0 <= start < end and start <= center < end, 'Invalid interval or center')
    return start, end, center


def annotations(config):
    spec = config['annotations']
    workbook = openpyxl.load_workbook(ROOT / spec['path'], read_only=True, data_only=True)
    records, table_stats, seen = [], {}, set()
    try:
        for kind, number in spec['tables'].items():
            sheet_name = f'Supplementary Table {number}'
            sheet = workbook[sheet_name]
            rows = sheet.iter_rows(values_only=True)
            headers = next(rows)
            cols = {str(v).strip(): i for i, v in enumerate(headers) if v is not None}
            required = [f'{kind}_ID', 'Chr', 'Start', 'End']
            require(all(k in cols for k in required), f'Missing columns in {sheet_name}')
            blank = 0
            for line, row in enumerate(rows, 2):
                raw = {key: row[cols[key]] for key in required}
                if all(value is None for value in raw.values()):
                    blank += 1
                    continue
                sid = raw[f'{kind}_ID']
                require(isinstance(sid, str) and sid.startswith(kind + '_'), f'Invalid ID at {sheet_name}:{line}')
                require(sid not in seen, f'Duplicate structure ID: {sid}')
                seen.add(sid)
                require(raw['Chr'] == spec['source_chrom'], f'Unexpected chromosome: {raw["Chr"]}')
                source_center = row[cols['Center']] if 'Center' in cols else None
                start, end, center = convert_interval(raw['Start'], raw['End'], source_center,
                                                      spec['source_coordinate_system'])
                require(end <= config['chrom_length'], f'Out-of-bounds interval: {sid}')
                records.append(dict(structure_id=sid, type=kind, chrom=config['chrom'],
                                    start=start, end=end, center=center, length_bp=end-start,
                                    source_file=spec['path'], source_sheet=sheet_name, source_row=line,
                                    source_chrom=raw['Chr'], source_start=raw['Start'],
                                    source_end=raw['End'], source_center=source_center,
                                    coordinate_system='0-based-half-open',
                                    coordinate_status=spec['coordinate_status']))
            table_stats[kind] = {'empty_annotation_rows_skipped': blank}
    finally:
        workbook.close()
    counts = dict(Counter(row['type'] for row in records))
    require(counts == spec['expected_counts'], f'Unexpected label counts: {counts}')
    for kind in counts:
        lengths = [r['length_bp'] for r in records if r['type'] == kind]
        table_stats[kind].update(count=counts[kind], min_length_bp=min(lengths),
                                 median_length_bp=float(np.median(lengths)), max_length_bp=max(lengths))
    overlaps = []
    for i, first in enumerate(records):
        for second in records[i+1:]:
            size = min(first['end'], second['end']) - max(first['start'], second['start'])
            if size > 0:
                overlaps.append(dict(structure_id_a=first['structure_id'], type_a=first['type'],
                                     structure_id_b=second['structure_id'], type_b=second['type'],
                                     overlap_bp=size,
                                     identical_interval=first['start'] == second['start'] and first['end'] == second['end']))
    return records, overlaps, {'counts': counts, 'tables': table_stats,
                               'overlap_pairs': len(overlaps),
                               'identical_interval_pairs': sum(r['identical_interval'] for r in overlaps),
                               'duplicate_ids': 0, 'invalid_intervals': 0}


def audit_sample(sample, config):
    path = ROOT / sample['path']
    with h5py.File(path, 'r') as handle:
        require(handle.attrs['bin-size'] == config['bin_size'], 'Unexpected bin size')
        require(handle.attrs['genome-assembly'] == config['assembly'], 'Unexpected assembly')
        require(handle.attrs['storage-mode'] == 'symmetric-upper', 'Unexpected storage mode')
        names = [v.decode() if isinstance(v, bytes) else str(v) for v in handle['chroms/name'][:]]
        require(names == [config['chrom']], f'Unexpected chromosomes: {names}')
        require(handle['chroms/length'][:].tolist() == [config['chrom_length']], 'Chromosome length mismatch')
        starts = np.arange(0, config['chrom_length'], config['bin_size'], dtype=np.int64)
        bins = {key: handle[f'bins/{key}'][:] for key in ('chrom', 'start', 'end')}
        require(np.array_equal(bins['start'], starts), 'Invalid bin starts')
        require(np.array_equal(bins['end'], np.minimum(starts+config['bin_size'], config['chrom_length'])), 'Invalid bin ends')
        require(np.all(bins['chrom'] == 0), 'Invalid chromosome IDs')
        nnz, nbins = int(handle.attrs['nnz']), int(handle.attrs['nbins'])
        require(nbins == len(starts), 'nbins mismatch')
        for key in ('bin1_id', 'bin2_id', 'count'):
            require(handle[f'pixels/{key}'].shape == (nnz,), f'Pixel shape mismatch: {key}')
        offsets = handle['indexes/bin1_offset'][:]
        require(len(offsets) == nbins+1 and offsets[0] == 0 and offsets[-1] == nnz
                and np.all(np.diff(offsets) >= 0), 'Invalid bin offsets')
        require(handle['indexes/chrom_offset'][:].tolist() == [0, nbins], 'Invalid chromosome offsets')
        digest = hashlib.sha256()
        for key in ('chrom', 'start', 'end'):
            digest.update(bins[key].astype('<i8').tobytes())
        info = dict(sample, assembly=config['assembly'], chrom=config['chrom'],
                    chrom_length=config['chrom_length'], bin_size=config['bin_size'],
                    nbins=nbins, nnz=nnz, metadata_count_sum=int(handle.attrs['sum']),
                    has_balance_weights='weight' in handle['bins'],
                    bin_coordinates_sha256=digest.hexdigest(), size_bytes=path.stat().st_size,
                    sha256=sha256(path), source_url=config['source_url'])
    # Exercise the actual API that later plotting steps will use.
    matrix = cooler.Cooler(str(path)).matrix(balance=False).fetch((config['chrom'], 50000, 51000))
    require(matrix.shape == (100, 100) and np.array_equal(matrix, matrix.T)
            and np.isfinite(matrix).all() and (matrix >= 0).all(), 'Local matrix read failed')
    info['local_read_smoke_test'] = 'passed'
    return info


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/data.json')
    parser.add_argument('--require-verified-coordinates', action='store_true',
                        help='Fail instead of exporting provisional coordinate interpretations')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.require_verified_coordinates:
        require(config['annotations']['coordinate_status'] == 'verified', 'Source coordinate convention is not verified')
    samples = [audit_sample(sample, config) for sample in config['samples']]
    require(len({r['sample_id'] for r in samples}) == len(samples), 'Duplicate sample IDs')
    require(len({r['bin_coordinates_sha256'] for r in samples}) == 1, 'Replicate bins differ')
    records, overlaps, stats = annotations(config)
    report = dict(python=platform.python_version(), checks_passed=True,
                  audit_scope='Complete bins/indexes, file checksums, all structure annotations, local pixel smoke test; no full pixel QC',
                  coordinates=config['annotations'],
                  annotation_sha256=sha256(ROOT / config['annotations']['path']),
                  config_sha256=sha256(args.config), samples=samples, annotations=stats,
                  warnings=['Source coordinate convention remains provisional.',
                            'No balance weights: read raw counts with balance=False.',
                            'Only one biological condition is currently available.',
                            'Overlapping/nested structures are preserved; group them before train/test splitting.'])
    write_csv(ROOT / 'data/processed/structures.csv', records)
    write_csv(ROOT / 'data/processed/structure_overlaps.csv', overlaps,
              ['structure_id_a', 'type_a', 'structure_id_b', 'type_b', 'overlap_bp', 'identical_interval'])
    write_csv(ROOT / 'data/processed/samples.csv', samples)
    (ROOT / 'reports').mkdir(exist_ok=True)
    (ROOT / 'reports/data_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'samples': len(samples), **stats, 'coordinate_status': config['annotations']['coordinate_status']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
