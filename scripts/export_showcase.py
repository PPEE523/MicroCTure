"""Copy an explicit, size-limited set of research outputs into the public showcase."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify published snapshots without raw data or runtime results')
    args = parser.parse_args()
    config = json.loads((ROOT/'configs/showcase.json').read_text())
    out = ROOT/config['output_dir']
    if args.check:
        manifest = json.loads((out/'manifest.json').read_text())
        if [r['source'] for r in manifest['files']] != config['files']:
            raise ValueError('Showcase selection differs from manifest')
        total = sum(row['bytes'] for row in manifest['files'])
        if total != manifest['total_bytes'] or total > config['max_total_bytes']:
            raise ValueError('Showcase total is inconsistent or exceeds budget')
        for row in manifest['files']:
            relative = Path(row['source']).relative_to('results')
            if '..' in relative.parts or relative.as_posix() != row['file']:
                raise ValueError(f'Invalid snapshot mapping: {row["source"]}')
            path = out/row['file']
            if path.stat().st_size != row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError(f'Snapshot mismatch: {path}')
        print(f"Verified {len(manifest['files'])} snapshots ({manifest['total_bytes']:,} bytes); no raw data required")
        return
    total = sum((ROOT/name).stat().st_size for name in config['files'])
    if total > config['max_total_bytes']:
        raise ValueError(f'Showcase exceeds configured budget: {total}')
    rows = []
    # Inspect every source before writing anything, to avoid partial exports.
    for name in config['files']:
        source = ROOT/name
        relative = Path(name).relative_to('results')
        if source.suffix not in {'.png', '.csv', '.json', '.html'} or '..' in relative.parts:
            raise ValueError(f'Unsupported showcase path: {name}')
        rows.append(dict(source=name, file=relative.as_posix(), bytes=source.stat().st_size,
                         sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    for row in rows:
        destination = out/row['file']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/row['source'], destination)
    out.mkdir(parents=True, exist_ok=True)
    (out/'manifest.json').write_text(json.dumps(dict(description='Selected computed outputs; not a full runtime-results archive.',
                                                    total_bytes=total, files=rows), indent=2)+'\n')
    print(f'Exported {len(rows)} snapshots ({total:,} bytes)')


if __name__ == '__main__':
    main()
