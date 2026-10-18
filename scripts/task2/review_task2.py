"""Audit saved task-two discovery, candidate and diagnostic evidence without fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'microcture/layout.py').is_file())
sys.path.insert(0,str(ROOT))
from microcture.layout import bootstrap, resolve
bootstrap()

from microcture.tasks.task2 import audit_discovery
from microcture.tasks.task2.delivery import (candidate_catalogue, digest, project_sources,
                                             verify_snapshot, write_csv, write_snapshot)
from microcture.tasks.task2.diagnostics import load_diagnostics
from optional_common import fresh_report_path, fresh_output_path


def preflight(args):
    """Inspect config, annotations, grid and NPZ headers without loading contact bands."""
    from contact_band import inspect_band_cache
    from discovery_v2_core import validate_config
    from discover_task2_v2 import validate_scan_identity
    from microcture.tasks.task2.delivery import read_table
    config_path=resolve(args.config)
    cfg=validate_config(json.loads(config_path.read_text()))
    out=fresh_output_path(args.output_dir,root=ROOT,data_root=ROOT)
    data=json.loads((ROOT/'configs/shared/data.json').read_text())
    n=(data['chrom_length']+99)//100
    records=[]
    scan=read_table(resolve(args.scan_file),records)
    known=read_table(ROOT/'data/shared/annotations/structures.csv',records)
    validate_scan_identity(scan,known,n,cfg)
    caches=[inspect_band_cache(s,n,max(cfg['window_bins']),data_root=ROOT,cache_root=resolve(args.band_cache_dir)) for s in data['samples']]
    if any(c['status']!='ready' for c in caches):
        raise ValueError('Missing or stale read-only band cache; prepare a new cache version explicitly')
    return dict(status='metadata_preflight_passed',scope='Config, current annotation/grid identities, raw file stats and cache headers only; no contact pixels or training',
                config=str(config_path),output_dir=str(out),scan_windows=len(scan),known_structures=len(known),caches=caches,
                neural_fits=2*len(cfg['seeds']),ranking_methods=2+4*len(cfg['seeds']),
                cluster_runs=(2+4*len(cfg['seeds']))*len(cfg['clusters'])*len(cfg['clustering_seeds']),
                main_input_tensor_bytes=len(scan)*2*64*64*4,training_performed=False,output_created=False)


def public_output(value):
    path = resolve(value)
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('Public review output cannot pass through a symlink')
    allowed = ROOT/'submission/task2/verification'
    if not path.is_relative_to(allowed) or len(path.relative_to(allowed).parts) != 1:
        raise ValueError('Public review must use submission/task2/verification/<new_version>')
    if path.exists():
        raise FileExistsError(f'Preserve existing public review: {path}')
    return path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('preflight','check','review','verify'))
    parser.add_argument('--data-root',type=Path,default=ROOT,help='Current MicroCTure directory; no raw pixels are read')
    parser.add_argument('--experiment-dir',type=Path,default=ROOT/'results/task2/discovery/v2')
    parser.add_argument('--report-dir',type=Path,help='Fresh reports/task2/<experiment>/<version> directory for review')
    parser.add_argument('--publish-dir',type=Path,help='Optional fresh submission/task2/verification/<version> snapshot')
    parser.add_argument('--snapshot',type=Path,default=ROOT/'submission/task2/verification/v2',help='verify: compact saved review')
    parser.add_argument('--require-local-inputs',action='store_true',help='verify: fail when any historical table is missing')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/task2/discovery/discovery_v2.json',help='preflight: training protocol')
    parser.add_argument('--scan-file',type=Path,default=ROOT/'results/task2/discovery/v1/scan_rep1.csv',help='preflight: saved scan identities')
    parser.add_argument('--band-cache-dir',type=Path,default=ROOT/'data/shared/cache/contact_bands_100bp',help='preflight: existing read-only WT caches')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/task2/discovery/reproducible_v3',help='preflight: fresh future training directory; never created')
    args=parser.parse_args(argv)
    if args.action=='verify':
        result=verify_snapshot(resolve(args.snapshot),require_local_inputs=args.require_local_inputs)
        print(json.dumps(result,ensure_ascii=False))
        return 0
    if args.data_root.resolve()!=ROOT:
        raise ValueError('Run the saved review inside the existing MicroCTure directory')
    if args.action=='preflight':
        print(json.dumps(preflight(args),ensure_ascii=False,allow_nan=False))
        return 0
    experiment=resolve(args.experiment_dir)
    if not experiment.is_relative_to(ROOT) or any(p.is_symlink() for p in (experiment,*experiment.parents)):
        raise ValueError('Saved discovery inputs must remain inside the project without symlinks')
    output=publish=None
    if args.action=='review':
        if args.report_dir is None:
            parser.error('review requires --report-dir with a fresh version')
        output=fresh_report_path(args.report_dir,root=ROOT,data_root=ROOT)
        if args.publish_dir is not None:
            publish=public_output(args.publish_dir)
    elif args.report_dir is not None or args.publish_dir is not None:
        parser.error('check is read-only and takes no output directories')
    data_config=ROOT/'configs/shared/data.json'
    config=json.loads(data_config.read_text())
    audit=audit_discovery(experiment,genome_bins=(config['chrom_length']+99)//100)
    diagnostics=load_diagnostics(experiment)
    catalogue,additional,sources=candidate_catalogue(
        audit['disjoint_candidate_rows'],novelty_dir=ROOT/'results/task2/novelty/v1',
        morphology_dir=ROOT/'results/task2/morphology/v1',centering_dir=ROOT/'results/task2/centering')
    sources.extend(audit['sources'])
    sources.extend(diagnostics['sources'])
    sources.append(dict(path=str(data_config),sha256=digest(data_config),bytes=data_config.stat().st_size))
    code=[ROOT/'microcture/tasks/task2'/name for name in ('review.py','diagnostics.py','delivery.py')]+[Path(__file__)]
    summary=dict(schema_version=1,status='saved_table_consistency_passed',
                 discovery=audit['summary'],checks=audit['checks'],
                 reproducibility=audit['reproducibility'],supplementary_evidence=additional,
                 sources=project_sources(sources),
                 code_sources=project_sources([dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size) for p in code]),
                 diagnostic_rows=dict(retrieval=len(diagnostics['test_retrieval_rows']),score_quality=len(diagnostics['score_quality_rows'])),
                 training_performed=False,raw_pixels_read=False,independent_confirmation=False,
                 limitations=[
                     'WT data were viewed during development; this is not external confirmation.',
                     'Saved-table consistency does not replay training, scores or raw pixel calculations.',
                     'Unannotated replicate-supported windows are not automatically new biological types.',
                     'Primary calibration is insufficient; NA is never converted to zero.',
                     'Supplementary calibrators contain no CHID; no class-complete calibration is asserted.',
                     'Random reference extrema are descriptive ranges, not biological confidence intervals.'])
    if output is not None:
        if publish is None:
            write_snapshot(output,summary,catalogue,diagnostics)
        else:
            output.mkdir(parents=True,exist_ok=False)
            (output/'review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
            write_snapshot(publish,summary,catalogue,diagnostics)
        for filename,key in (('candidate_audit.csv','candidate_rows'),('cluster_audit.csv','cluster_rows'),('method_audit.csv','method_rows')):
            write_csv(output/filename,audit[key])
        manifest=dict(schema_version=1,files=[dict(path=p.name,sha256=digest(p),bytes=p.stat().st_size)
                      for p in sorted(output.iterdir()) if p.is_file() and p.name!='ARTIFACTS.json'])
        (output/'ARTIFACTS.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
        if publish is not None:
            verify_snapshot(publish,require_local_inputs=True)
    print(json.dumps(dict(status=summary['status'],discovery=summary['discovery'],
                         reproducibility=summary['reproducibility'],
                         report_dir=str(output) if output else None,dashboard=str(publish/'index.html') if publish else str(output/'index.html') if output else None,
                         training_performed=False,raw_pixels_read=False),ensure_ascii=False,allow_nan=False))
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,FileNotFoundError,FileExistsError) as error:
        print(f'ERROR: {error}',file=sys.stderr)
        raise SystemExit(2)
