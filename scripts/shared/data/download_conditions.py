"""Resume the four explicitly configured public GEO archives; validate gzip CRC."""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()
import argparse,concurrent.futures,gzip,json,subprocess,time
from pathlib import Path
ROOT=_layout_root


def fetch(sample):
    archive=_layout_resolve(ROOT, sample['path']+'.gz')
    archive.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1,31):
        result=subprocess.run(['curl','--silent','--show-error','--location','--fail','--continue-at','-',
                               '--connect-timeout','30','--max-time','90',sample['download_url'],'--output',str(archive)],capture_output=True,text=True)
        print(sample['sample_id'],'attempt',attempt,'bytes',archive.stat().st_size if archive.exists() else 0,'exit',result.returncode,flush=True)
        if result.returncode==0 or result.returncode==33:
            try:
                with gzip.open(archive,'rb') as f:
                    while f.read(8*1024*1024):pass
                print('VALIDATED',archive.name,flush=True);return
            except (OSError,EOFError):pass
        if result.stderr:print(result.stderr[-400:],flush=True)
        time.sleep(2)
    raise RuntimeError('Download incomplete: '+sample['sample_id'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4,choices=range(1,5),help='Concurrent archive downloads, 1 to 4')
    args=parser.parse_args()
    samples=json.loads((_layout_resolve(ROOT, 'configs/conditions.json')).read_text())['samples']
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fetch,[s for s in samples if 'download_url' in s]))


if __name__=='__main__':
    main()
