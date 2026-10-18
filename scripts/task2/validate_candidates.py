"""Revalidate a frozen 21-candidate cohort; no model training or raw scanning."""
from pathlib import Path
import sys
import argparse
import json

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'microcture/layout.py').is_file())
sys.path.insert(0,str(ROOT))
from microcture.layout import bootstrap
bootstrap()
from microcture.tasks.task2.validation import prepare_metadata,run_validation,verify_validation


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('preflight','run','verify'))
    p.add_argument('--config',type=Path,default=ROOT/'configs/task2/verification/candidate_evidence_v1.json')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/task2/validation/frozen21_v1')
    args=p.parse_args(argv);out=args.output_dir.absolute();config=args.config.resolve()
    if args.action=='verify':result=verify_validation(ROOT,out)
    elif args.action=='preflight':
        cfg,_,_,settings,candidates,_,primary,supplement,capacity,_=prepare_metadata(ROOT,config)
        result=dict(status='preflight_passed',candidates=len(candidates),variants=len(settings),
                    primary_calibration_capacity=capacity,known_primary_queries=len(primary[1]['calibration']),
                    known_supplement_queries=len(supplement[1]['calibration']),output_dir=str(out),
                    output_exists=out.exists(),training_performed=False,band_pixels_read=False,
                    support_controls_will_be_reselected=True,shape_calibration_will_not_be_replaced_by_background=True)
    else:
        result=run_validation(ROOT,config,out)
        result={k:v for k,v in result.items() if k!='known_retrieval_summary'}
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,FileNotFoundError,FileExistsError) as e:
        print(f'ERROR: {e}',file=sys.stderr);raise SystemExit(2)
