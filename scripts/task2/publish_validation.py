"""Publish or verify the compact frozen-candidate validation evidence."""
from pathlib import Path
import argparse
import json
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'microcture/layout.py').is_file())
sys.path.insert(0,str(ROOT))
from microcture.tasks.task2.validation_delivery import publish_snapshot,verify_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('publish','verify'))
    p.add_argument('--source',type=Path,default=ROOT/'results/task2/validation/frozen21_v1')
    p.add_argument('--snapshot',type=Path,default=ROOT/'submission/task2/validation/frozen21_v1')
    p.add_argument('--require-local-inputs',action='store_true')
    args=p.parse_args();source=args.source.absolute();snapshot=args.snapshot.absolute()
    result=publish_snapshot(ROOT,source,snapshot) if args.action=='publish' else verify_snapshot(snapshot,root=ROOT,require_local_inputs=args.require_local_inputs)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
