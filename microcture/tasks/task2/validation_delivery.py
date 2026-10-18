"""Small portable evidence snapshot for the frozen-candidate validation."""
from __future__ import annotations
import json
import shutil
from pathlib import Path

from .validation import (read_csv,write_csv,json_write,sha,require,number,truth,
                         support_decision,six_cells,maximum_disjoint_intervals)


def nullable_bool(value):
    return None if value in ('',None) else truth(value)


def same_number(a,b,message):
    a,b=number(a),number(b)
    require((a is None and b is None) or (a is not None and b is not None and abs(a-b)<1e-11),message)


def verify_snapshot(directory, *, root=None, require_local_inputs=False):
    directory=Path(directory)
    done=json.loads((directory/'COMPLETED.json').read_text())
    require(sha(directory/'ARTIFACTS.json')==done['artifact_manifest_sha256'],'Snapshot manifest hash mismatch')
    manifest=json.loads((directory/'ARTIFACTS.json').read_text());files=manifest['files']
    names=[r['path'] for r in files]
    require(len(names)==len(set(names)) and all(Path(p).name==p for p in names),'Invalid snapshot artifact paths')
    require(set(names)=={p.name for p in directory.iterdir() if p.name not in ('ARTIFACTS.json','COMPLETED.json')},'Snapshot file set mismatch')
    for r in files:require((directory/r['path']).is_file() and not (directory/r['path']).is_symlink() and sha(directory/r['path'])==r['sha256'],'Snapshot bytes changed: '+r['path'])
    frozen=json.loads((directory/'cohort_frozen.json').read_text());settings=frozen['variants'];candidates=frozen['candidates']
    require(len(candidates)==21 and len({r['window_id'] for r in candidates})==21,'Frozen candidate count mismatch')
    require(sha(directory/'frozen_candidates.csv')==frozen['config']['candidate_source_sha256'],'Frozen candidate bytes changed')
    controls=json.loads((directory/'controls_frozen.json').read_text())['controls']
    require(set(controls)=={r['window_id'] for r in candidates},'Frozen control-set identity mismatch')
    geometry={r['entity_id']:r for r in read_csv(directory/'centered_entities.csv')}
    from .validation import overlap
    for cid,group in controls.items():
        for j,e in enumerate(group):
            require(not any(overlap(e,geometry[c['window_id']]) for c in candidates),'Background overlaps candidate envelope')
            require(not any(overlap(e,g) for g in group[:j]),'Control envelopes overlap')
    score_rows=read_csv(directory/'replicate_scores.csv')
    scores={(r['entity_id'],r['variant_id']):r for r in score_rows}
    require(len(scores)==len(score_rows),'Duplicate cached score')
    ids={*controls}|{e['entity_id'] for g in controls.values() for e in g}
    require(set(scores)=={(i,v['variant_id']) for i in ids for v in settings},'Incomplete paired score grid')
    observed=read_csv(directory/'candidate_variants.csv')
    require(len(observed)==1134 and len({(r['candidate_id'],r['variant_id']) for r in observed})==1134,'Incomplete 21x54 grid')
    decisions={}
    for r in observed:
        cid,vid=r['candidate_id'],r['variant_id'];s=scores[cid,vid]
        d=support_decision(number(s['correlation']),[number(scores[e['entity_id'],vid]['correlation']) for e in controls[cid]])
        if not truth(s['quality_pass']):d.update(replicate_supported=None,margin=None,assessment_status='not_evaluable',assessment_reason='candidate_quality_failed')
        for k in ('correlation','control_q95','cutoff','margin'):same_number(r[k],d[k],'Recomputed threshold mismatch: '+k)
        require(nullable_bool(r['replicate_supported'])==d['replicate_supported'],'Recomputed support decision differs')
        require(int(r['matched_controls'])==d['matched_controls'] and r['assessment_status']==d['assessment_status'],'Invalid support status/count')
        decisions[cid,vid]=d
    cell_rows=read_csv(directory/'candidate_scale_evidence.csv')
    cells={(r['candidate_id'],r['anchor'],int(r['width_bins'])):r for r in cell_rows}
    require(len(cells)==len(cell_rows)==126,'Incomplete candidate scale cells')
    summary_rows=read_csv(directory/'candidate_evidence.csv');summary={r['candidate_id']:r for r in summary_rows}
    require(len(summary)==len(summary_rows)==21 and set(summary)==set(controls),'Candidate-level evidence missing')
    for c in candidates:
        cid=c['window_id'];local={v['variant_id']:decisions[cid,v['variant_id']] for v in settings}
        for expected in six_cells(settings,local):
            saved=cells[cid,expected['anchor'],expected['width_bins']]
            for k in ('assessed_variants','above_threshold','below_or_equal_threshold','not_evaluable','full_cell_above_fraction','min_margin','max_margin'):
                same_number(saved[k],expected[k],'Candidate-scale evidence mismatch: '+k)
        base=local[f'original_w{int(c["window_length_bp"])//100}_s0_none'];saved=summary[cid]
        for sk,k in [('baseline_score','correlation'),('baseline_threshold','cutoff'),('baseline_margin','margin'),('baseline_valid_controls','matched_controls')]:same_number(saved[sk],base[k],'Candidate baseline summary differs')
        require(nullable_bool(saved['baseline_support'])==base['replicate_supported'],'Candidate baseline flag differs')
        for k,val in [('assessed_variants',sum(d['replicate_supported'] is not None for d in local.values())),('above_threshold_variants',sum(d['replicate_supported'] is True for d in local.values()))]:
            require(int(saved[k])==val,'Candidate 54-variant count differs')
        require(saved['primary_shape_status']=='not_evaluable' and nullable_bool(saved['primary_robust_anomaly']) is None,'Insufficient primary morphology cannot become negative')
    pool=read_csv(directory/'formal_calibration_pool.csv');capacity=json.loads((directory/'formal_calibration_feasibility.json').read_text())
    require(len(maximum_disjoint_intervals(pool))==capacity['maximum_nonoverlapping_windows']==8,'Primary geometry capacity differs')
    retrieval=read_csv(directory/'known_retrieval_summary.csv')
    for r in retrieval:
        if int(r['query_ids'])==0:
            require(number(r['top1']) is None and number(r['top5']) is None,'Missing retrieval class must remain NA')
    checked=absent=0
    if root is not None:
        for r in frozen['input_files']:
            p=Path(root)/r['path']
            if p.is_file():require(sha(p)==r['sha256'],'Local source changed: '+r['path']);checked+=1
            else:
                absent+=1
                require(not require_local_inputs,'Missing required local source: '+r['path'])
    return dict(status='passed',files=len(files),candidates=21,threshold_decisions_recomputed=1134,
                scale_cells_recomputed=126,local_inputs_checked=checked,local_inputs_absent=absent,
                retrieval_feature_arrays_replayed=False,raw_pixels_read=False)


def publish_snapshot(root,source,destination):
    from .validation import verify_validation
    verify_validation(root,source)
    allowed=root/'submission/task2/validation'
    require(destination.is_relative_to(allowed) and len(destination.relative_to(allowed).parts)==1,'Publish to submission/task2/validation/<new_version>')
    require(not destination.exists() and not any(p.is_symlink() for p in (destination,*destination.parents)),'Snapshot exists or uses symlinks')
    destination.mkdir(parents=True,exist_ok=False)
    local_manifest=json.loads((source/'ARTIFACTS.json').read_text())
    for row in local_manifest['files']:shutil.copyfile(source/row['path'],destination/row['path'])
    cfg=json.loads((source/'cohort_frozen.json').read_text())['config']
    shutil.copyfile(root/cfg['primary_centering']/'entities.csv',destination/'centered_entities.csv')
    files=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in sorted(destination.iterdir())]
    json_write(destination/'ARTIFACTS.json',dict(schema_version=1,files=files,source_dir=str(source.relative_to(root)),
               source_artifact_manifest_sha256=sha(source/'ARTIFACTS.json'),source_completion_sha256=sha(source/'COMPLETED.json')))
    done=json.loads((source/'COMPLETED.json').read_text())
    done.update(artifact_manifest_sha256=sha(destination/'ARTIFACTS.json'),status='published_snapshot')
    json_write(destination/'COMPLETED.json',done)
    return verify_snapshot(destination,root=root,require_local_inputs=True)
