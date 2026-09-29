"""Resume the four explicitly configured public GEO archives; validate gzip CRC."""
import concurrent.futures,gzip,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def fetch(sample):
    archive=ROOT/(sample['path']+'.gz')
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


if __name__=='__main__':
    samples=json.loads((ROOT/'configs/conditions.json').read_text())['samples']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch,[s for s in samples if 'download_url' in s]))
