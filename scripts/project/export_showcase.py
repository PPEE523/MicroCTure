"""Copy an explicit, size-limited set of research outputs into the public showcase."""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = _layout_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify published snapshots without raw data or runtime results')
    args = parser.parse_args()
    config = json.loads(_layout_resolve(ROOT/'configs/showcase.json').read_text())
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
    raise SystemExit('Historical snapshot export is disabled; use --check to verify the retained evidence.')


if __name__ == '__main__':
    main()
