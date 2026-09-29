"""Decompress and audit downloaded public condition matrices; preserve source archives."""
import gzip,json,shutil
from pathlib import Path
from prepare_data import audit_sample,write_csv,sha256
ROOT=Path(__file__).resolve().parents[1]


def main():
    config=json.loads((ROOT/'configs/conditions.json').read_text());base=json.loads((ROOT/'configs/data.json').read_text())
    records=[]
    for sample in config['samples']:
        path=ROOT/sample['path']
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
    write_csv(ROOT/'data/processed/condition_samples.csv',records,fields)
    (ROOT/'reports/conditions_audit.json').write_text(json.dumps(dict(samples=records,bin_coordinates_identical=True,conditions=config['conditions'],metadata_sha256=sha256(ROOT/'data/metadata/GSE272159_family.soft')),indent=2)+'\n')
    print('Audited six samples, three conditions, identical genome/bin coordinates.',flush=True)

if __name__=='__main__':main()
