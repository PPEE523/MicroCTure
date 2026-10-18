#!/usr/bin/env python3
"""Fixed forty CPU model-family probes under autonomous Task 1 authorization."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    from microcture.task1_shape_baseline.experiment import validate,prepare_manifest,planned_jobs,guard,run,summarize
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/task1_shape_baseline.json');p.add_argument('--run-dir',type=Path)
    sub=p.add_subparsers(dest='command',required=True);sub.add_parser('check');sub.add_parser('summarize');a=sub.add_parser('all')
    a.add_argument('--model',choices=('all','lr_shape','svc_shape'),default='all');a.add_argument('--condition',choices=('I3','G0','V0','P0'));a.add_argument('--fold',type=int,choices=range(5))
    args=p.parse_args();c=validate(json.loads(args.config.read_text()));rd=guard(ROOT,args.run_dir or ROOT/c['run_dir']);c={**c,'run_dir':str(rd.relative_to(ROOT))}
    if args.command=='check':
        m,d,_=prepare_manifest(ROOT,c);print(json.dumps({'status':'checked','real_images_preprocessed':False,'training_launched':False,'n_structures':len(d['structures']),'fits':len(planned_jobs(c)),'features':455,
            **{k:m[k] for k in ('source_signature','experiment_signature','score_interpretation')}},indent=2));return 0
    r=summarize(rd,ROOT) if args.command=='summarize' else run(c,ROOT,rd,model=args.model,condition=args.condition,fold=args.fold)
    print(json.dumps({k:r[k] for k in ('status','completed_fits','pending_fits','execution_failure_fits','numerical_failure_fits','invalid_fits','complete_OOF')},indent=2));return int(r['status'] in ('failed','invalid'))


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,FileNotFoundError) as error:print(f'ERROR: {error}',file=sys.stderr);raise SystemExit(1)
