#!/usr/bin/env python3
"""Single-factor global Dropout0.2->0.5 on frozen R1 complete-ROI; train only under current user authorization."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from microcture.task1_dropout_v1.experiment import check_protocol, guard_output, run_jobs, summarize, validate_protocol
    from microcture.task1_formal.protocol import project_path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT / "configs/task1_dropout.json")
    parser.add_argument("--run-dir",type=Path)
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("check",help="Metadata/signatures only; no images or training")
    sub.add_parser("summarize",help="Saved scientific evidence only; no image/model inference")
    for command in ("all","run"):
        p=sub.add_parser(command,help="60 registered D5 full40 fits, optional immutable job filters")
        p.add_argument("--alignment",choices=("all","R1"),default="all")
        p.add_argument("--condition",choices=("I3","G0","V0","P0"))
        p.add_argument("--fold",type=int,choices=range(5));p.add_argument("--seed",type=int,choices=(42,43,44))
    args=parser.parse_args(); config=validate_protocol(json.loads(project_path(ROOT,args.config).read_text()))
    run_dir=guard_output(ROOT,project_path(ROOT,args.run_dir or config["run_dir"]))
    if args.command=="check":
        result=check_protocol(ROOT,config,run_dir,args.device);print(json.dumps(result,indent=2,ensure_ascii=False));return 0
    result=summarize(run_dir,ROOT) if args.command=="summarize" else run_jobs(config,ROOT,run_dir,args.device,
        alignment=args.alignment,condition=args.condition,fold=args.fold,seed=args.seed)
    print(json.dumps({k:result[k] for k in ("status","completed_jobs","pending_jobs","invalid_jobs","numerical_failure_jobs","execution_failure_jobs","complete_oof")},indent=2))
    return int(result["status"] in ("failed","invalid"))


if __name__=="__main__":
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,FileNotFoundError) as error:
        print(f"ERROR: {error}",file=sys.stderr);raise SystemExit(1)
