"""Decompress and audit downloaded public condition matrices; preserve source archives."""

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_root = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "microcture/layout.py").is_file())
_layout_sys.path.insert(0, str(_layout_root))
from microcture.layout import bootstrap as _layout_bootstrap, resolve as _layout_resolve
_layout_bootstrap()
import argparse,gzip,json,shutil
from pathlib import Path
from prepare_data import audit_sample,write_csv,sha256
from optional_common import fresh_report_path
ROOT=_layout_root


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,required=True,help='Fresh reports/shared/conditions/<version> directory; retained sample metadata is never replaced')
    args=parser.parse_args()
    out=fresh_report_path(args.report_dir,root=ROOT,data_root=ROOT)
    config=json.loads((_layout_resolve(ROOT, 'configs/conditions.json')).read_text());base=json.loads((_layout_resolve(ROOT, 'configs/data.json')).read_text())
    records=[]
    for sample in config['samples']:
        path=_layout_resolve(ROOT, sample['path'])
        if not path.exists():
            archive=Path(str(path)+'.gz');part=Path(str(path)+'.partial')
            print('Decompressing',archive.name,flush=True)
            with gzip.open(archive,'rb') as src,part.open('wb') as dst:shutil.copyfileobj(src,dst,8*1024*1024)
            part.replace(path)
        record=audit_sample(sample,base)
        archive=Path(str(path)+'.gz')
        record['archive_sha256']=sha256(archive) if archive.exists() else ''
        record['geo_accession']=sample.get('geo_accession','GSE272159')
        record['download_url']=sample.get('download_url','')
        records.append(record)
    assert len({r['bin_coordinates_sha256'] for r in records})==1
    assert all(len([s for s in records if s['condition']==c])==2 for c in config['conditions'])
    fields=sorted(set().union(*(r.keys() for r in records)))
    out.mkdir(parents=True,exist_ok=False)
    write_csv(out/'condition_samples.csv',records,fields)
    with (out/'audit.json').open('x') as stream:
        stream.write(json.dumps(dict(samples=records,bin_coordinates_identical=True,conditions=config['conditions'],metadata_sha256=sha256(_layout_resolve(ROOT, 'data/metadata/GSE272159_family.soft'))),indent=2)+'\n')
    print('Audited six samples, three conditions, identical genome/bin coordinates.',flush=True)

if __name__=='__main__':main()
