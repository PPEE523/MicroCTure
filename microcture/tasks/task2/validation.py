"""Frozen candidate replication, full-grid evidence and known retrieval checks.

Replication thresholds and known-shape calibration are separate estimands.
This module never substitutes one for the other or tunes thresholds to outcomes.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
from scipy.spatial.distance import cdist

from microcture.layout import bootstrap
bootstrap()

TYPES = ("OPCID", "CHIN", "CHID")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader(); w.writerows(rows)


def json_write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def truth(value):
    if value in (True, 'True', 'true'): return True
    if value in (False, 'False', 'false'): return False
    raise ValueError(f'Invalid boolean {value!r}')


def number(value):
    if value in ('', None): return None
    x = float(value)
    return x if math.isfinite(x) else None


def require(condition, message):
    if not condition: raise ValueError(message)


def overlap(a, b):
    return int(a['envelope_start_bin']) < int(b['envelope_end_bin']) and int(b['envelope_start_bin']) < int(a['envelope_end_bin'])


def maximum_disjoint_intervals(rows):
    """Earliest-finish scheduling gives exact maximum cardinality on a line."""
    selected = []; end = -math.inf
    for row in sorted(rows, key=lambda e: (int(e['envelope_end_bin']), int(e['envelope_start_bin']), e['entity_id'])):
        lo, hi = int(row['envelope_start_bin']), int(row['envelope_end_bin'])
        require(lo < hi, 'Invalid non-circular interval')
        if lo >= end: selected.append(row); end = hi
    return selected


def support_decision(score, controls, *, minimum=10, quantile=.95, floor=.5):
    valid = [float(x) for x in controls if x is not None and math.isfinite(float(x))]
    q95 = float(np.quantile(valid, quantile)) if len(valid) >= minimum else None
    threshold = max(floor, q95) if q95 is not None else None
    score = number(score)
    reason = 'insufficient_valid_controls' if threshold is None else 'nonfinite_candidate_score' if score is None else ''
    return dict(matched_controls=len(valid), control_q95=q95, cutoff=threshold, correlation=score,
                margin=score-threshold if not reason else None,
                replicate_supported=score > threshold if not reason else None,
                assessment_status='not_evaluable' if reason else 'assessed', assessment_reason=reason)


def match_controls(focal, pool, count):
    """No focal or control correlation/outcome is read during selection."""
    width = int(focal['width_bins'])
    def cost(r):
        return abs(math.log1p(float(r['mean_oe']))-math.log1p(float(focal['mean_oe']))) + abs(float(r['valid_fraction'])-float(focal['valid_fraction']))
    ranked = sorted((r for r in pool if int(r['width_bins']) == width),
                    key=lambda r: (cost(r), int(r['window_index'])))
    result = []
    for row in ranked:
        if not any(overlap(row, chosen) for chosen in result): result.append(row)
        if len(result) == count: break
    return [(row, cost(row)) for row in result]


def six_cells(settings, decisions):
    require(len(settings) == 54 and len({v['variant_id'] for v in settings}) == 54, 'Expected exact 54-variant grid')
    require(set(decisions) == {v['variant_id'] for v in settings}, 'Missing or extra variant decision')
    result=[]
    for anchor in ('original', 'boundary'):
        for width in (64, 128, 256):
            values=[decisions[v['variant_id']] for v in settings if v['anchor']==anchor and int(v['width_bins'])==width]
            require(len(values)==9, 'Each anchor/width cell must have nine variants')
            assessed=[v for v in values if v['replicate_supported'] is not None]
            result.append(dict(anchor=anchor, width_bins=width, variants=9, assessed_variants=len(assessed),
                               above_threshold=sum(v['replicate_supported'] for v in assessed),
                               below_or_equal_threshold=sum(not v['replicate_supported'] for v in assessed),
                               not_evaluable=9-len(assessed),
                               full_cell_above_fraction=sum(v['replicate_supported'] for v in assessed)/9 if len(assessed)==9 else None,
                               min_margin=min(v['margin'] for v in assessed) if assessed else None,
                               max_margin=max(v['margin'] for v in assessed) if assessed else None))
    return result


def rank_query(query, gallery, query_entity, gallery_entities, metric, *, identity=False):
    eligible=[j for j,e in enumerate(gallery_entities)
              if (identity and e['entity_id']==query_entity['entity_id']) or
              (e['entity_id']!=query_entity['entity_id'] and not overlap(query_entity,e))]
    if not np.isfinite(query).all() or not eligible: return []
    ds=cdist(np.asarray(query)[None], gallery[eligible], metric=metric)[0]
    return [(gallery_entities[j], float(d)) for j,d in sorted(zip(eligible,ds),key=lambda x:(x[1],gallery_entities[x[0]]['entity_id'])) if math.isfinite(d)]


def formal_capacity(entities, minimum):
    pool=[e for e in entities if e['kind']=='known' and int(e['role'])==1 and
          truth(e['boundary_reproducible']) and truth(e['all_variants_quality_pass']) and truth(e['reference_geometry_eligible'])]
    maximum=maximum_disjoint_intervals(pool)
    original=[e for e in entities if truth(e['calibration_member'])]
    return dict(pool_known_windows=len(pool), original_calibration_windows=len(original),
                maximum_nonoverlapping_windows=len(maximum), minimum_required=minimum,
                feasible=len(maximum)>=minimum, maximum_ids=[e['entity_id'] for e in maximum],
                original_ids=[e['entity_id'] for e in original],
                selection='Exact earliest-finish interval scheduling after all original full-grid eligibility filters',
                conclusion='sufficient_capacity' if len(maximum)>=minimum else 'insufficient_capacity_under_unchanged_primary_rules'), pool


def load_cohort(directory, settings):
    entities=read_csv(directory/'entities.csv')
    with np.load(directory/'features.npz', allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files}
    require(len(entities)==len(arrays['shape']) and arrays['shape'].shape[1:3]==(54,2), 'Feature/entity/grid shape mismatch')
    require(arrays['quality'].shape==(len(entities),54,2), 'Quality grid mismatch')
    for name,flag in [('training','reference_member'),('calibration','calibration_member')]:
        ids=arrays[name].tolist()
        require(len(ids)==len(set(ids)) and set(ids)=={i for i,e in enumerate(entities) if truth(e[flag])}, f'{name} identity mismatch')
    return entities, arrays


def retrieval(cohort_name, entities, arrays, settings):
    rows=[]; training=arrays['training'].tolist(); queries=arrays['calibration'].tolist()
    require(all(entities[i]['kind']=='known' for i in training+queries), 'Retrieval only accepts known queries/references')
    require(set(training).isdisjoint(queries), 'Known retrieval query leaked into training reference')
    gallery_ids=training+queries
    for metric, feature, distance in [('pca',arrays['pca'],'euclidean'),('shape',arrays['shape'],'cosine')]:
        for vi,v in enumerate(settings):
            for qi in queries:
                entity=entities[qi]
                for rep in (0,1):
                    eligible=[j for j in training if entities[j]['entity_id']!=entity['entity_id'] and not overlap(entity,entities[j])]
                    good=bool(arrays['quality'][qi,vi,rep])
                    if good:
                        eligible=[j for j in eligible if bool(arrays['quality'][j,vi,rep])]
                    matches=rank_query(feature[qi,vi,rep],feature[eligible,vi,rep],entity,[entities[j] for j in eligible],distance) if good else []
                    same=sum(e['type']==entity['type'] for e,_ in matches); n=len(matches); k=min(5,n)
                    qclass=1-math.comb(n-same,k)/math.comb(n,k) if n and n-same>=k else (1. if n else None)
                    rows.append(dict(cohort=cohort_name,check='heldout_class',metric=metric,query_id=entity['entity_id'],true_type=entity['type'],
                                     variant_id=v['variant_id'],anchor=v['anchor'],width_bins=int(v['width_bins']),replicate=rep+1,
                                     gallery_size=n,positives=same,top1_correct=matches[0][0]['type']==entity['type'] if matches else None,
                                     top5_correct=any(e['type']==entity['type'] for e,_ in matches[:5]) if matches else None,
                                     positive_rank=next((j+1 for j,(e,_) in enumerate(matches) if e['type']==entity['type']),None),
                                     nearest_id=matches[0][0]['entity_id'] if matches else '',chance_top1=same/n if n else None,
                                     chance_top5=qclass,assessment_status='assessed' if matches else 'not_evaluable'))
                # The cross-WT same-ID check has a true paired positive; it is a separate diagnostic.
                eligible=[j for j in gallery_ids if bool(arrays['quality'][j,vi,0])]
                matches=rank_query(feature[qi,vi,1],feature[eligible,vi,0],entity,[entities[j] for j in eligible],distance,identity=True) if arrays['quality'][qi,vi,1] else []
                positive=next((j+1 for j,(e,_) in enumerate(matches) if e['entity_id']==entity['entity_id']),None)
                rows.append(dict(cohort=cohort_name,check='paired_identity',metric=metric,query_id=entity['entity_id'],true_type=entity['type'],
                                 variant_id=v['variant_id'],anchor=v['anchor'],width_bins=int(v['width_bins']),replicate=2,
                                 gallery_size=len(matches),positives=1 if positive is not None else 0,
                                 top1_correct=positive==1 if positive is not None else None,top5_correct=positive<=5 if positive is not None else None,
                                 positive_rank=positive,nearest_id=matches[0][0]['entity_id'] if matches else '',
                                 chance_top1=1/len(matches) if positive is not None else None,
                                 chance_top5=min(5,len(matches))/len(matches) if positive is not None else None,
                                 assessment_status='assessed' if positive is not None else 'not_evaluable'))
    return rows


def summarize_retrieval(rows):
    groups={}
    for r in rows:
        key=tuple(r[k] for k in ('cohort','check','metric')) if 'metric' in r else None
        require(key is not None, 'Retrieval rows must identify the metric')
        groups.setdefault(key,[]).append(r)
    result=[]
    for (cohort,check,metric), values in groups.items():
        for kind in (*TYPES,'ALL'):
            subset=[r for r in values if kind=='ALL' or r['true_type']==kind]
            valid=[r for r in subset if r['assessment_status']=='assessed']
            result.append(dict(cohort=cohort,check=check,metric=metric,type=kind,
                               query_ids=len({r['query_id'] for r in subset}),grid_rows=len(subset),assessed_rows=len(valid),
                               top1=sum(r['top1_correct'] for r in valid)/len(valid) if valid else None,
                               top5=sum(r['top5_correct'] for r in valid)/len(valid) if valid else None,
                               chance_top1=float(np.mean([r['chance_top1'] for r in valid])) if valid else None,
                               chance_top5=float(np.mean([r['chance_top5'] for r in valid])) if valid else None,
                               median_positive_rank=float(np.median([r['positive_rank'] for r in valid if r['positive_rank'] is not None])) if any(r['positive_rank'] is not None for r in valid) else None))
    return result


def known_query_summaries(rows):
    groups={}
    for row in rows:
        key=tuple(row[k] for k in ('cohort','check','metric','query_id','true_type','replicate'))
        groups.setdefault(key,[]).append(row)
    result=[]
    for key,subset in groups.items():
        require(len(subset)==54 and len({r['variant_id'] for r in subset})==54, 'Incomplete known-query grid')
        valid=[r for r in subset if r['assessment_status']=='assessed']
        r=dict(zip(('cohort','check','metric','query_id','true_type','replicate'),key))
        r.update(variants=54,assessed_variants=len(valid),top1_count=sum(v['top1_correct'] for v in valid),
                 top5_count=sum(v['top5_correct'] for v in valid),
                 full_grid_top1_fraction=sum(v['top1_correct'] for v in valid)/54 if len(valid)==54 else None)
        for anchor in ('original','boundary'):
            for w in (64,128,256):
                values=[v for v in subset if v['anchor']==anchor and v['width_bins']==w]
                require(len(values)==9,'Incomplete known-query scale cell')
                r[f'{anchor}_w{w}_top1_count']=sum(v['top1_correct'] for v in values) if all(v['top1_correct'] is not None for v in values) else None
        result.append(r)
    return result


def morphology_evidence(directory, entities, settings, minimum):
    """Collapse saved distances with the unchanged nullable 80%-per-scale rule."""
    from centering_status import decision_from_row, summarize_candidate, aggregate
    source=read_csv(directory/'candidate_variants.csv')
    config=json.loads((directory/'report.json').read_text())
    count=int(config['calibration_windows'])
    vi={v['variant_id']:i for i,v in enumerate(settings)}
    flags={}
    for r in source:
        key=(r['entity_id'],vi[r['variant_id']],r['metric'])
        require(key not in flags,'Duplicate saved morphology comparison')
        flags[key]=decision_from_row(r,calibration_windows=count,minimum=minimum)
    rows=[]
    for e in entities:
        if e['kind']!='candidate':continue
        expected={(e['entity_id'],i,m) for i in range(54) for m in ('pca','shape')}
        require(expected.issubset(flags),'Incomplete candidate morphology grid')
        r=dict(entity_id=e['entity_id'],boundary_reproducible=truth(e['boundary_reproducible']),
               all_variants_quality_pass=truth(e['all_variants_quality_pass']),calibration_windows=count)
        r.update(summarize_candidate(settings,{(i,m):flags[e['entity_id'],i,m] for i in range(54) for m in ('pca','shape')},
                                     r['boundary_reproducible'],r['all_variants_quality_pass'],count>=minimum,.8))
        rows.append(r)
    require(len(rows)==21,'Candidate morphology count changed')
    return rows,aggregate(rows)


def crop_statistics(band, entity, setting, *, score=False, second=None, occlusion=None):
    from contact_band import extract
    from discover_task2 import correlation
    width=int(setting['width_bins'])
    center=int(entity['original_center_bin'] if setting['anchor']=='original' else entity['boundary_center_bin'])
    start=center+int(setting['shift_bins'])-width//2
    raw,oe,mask=extract(band,start,width)
    upper=np.triu(mask,2)
    valid=float(mask.sum()/(width*width-3*width+2))
    zero=float((raw[upper]==0).mean()) if upper.any() else 1.
    good=valid>=.8 and zero<=.8 and upper.any()
    result=dict(rep1_valid_fraction=valid,rep1_zero_fraction=zero,rep1_quality_pass=bool(good))
    if score:
        raw2,oe2,mask2=extract(second,start,width)
        u2=np.triu(mask2,2)
        valid2=float(mask2.sum()/(width*width-3*width+2))
        zero2=float((raw2[u2]==0).mean()) if u2.any() else 1.
        good2=valid2>=.8 and zero2<=.8 and u2.any()
        r=correlation(oe,oe2,mask & mask2 & occlusion)
        result.update(rep2_valid_fraction=valid2,rep2_zero_fraction=zero2,rep2_quality_pass=bool(good2),
                      correlation=number(r) if good and good2 else None,
                      quality_pass=bool(good and good2))
    return result


def build_control_geometry(scan, band, candidates, roles, cfg):
    from centering_sensitivity import anchor_position
    excluded=np.zeros(len(roles),bool)
    for e in candidates:excluded[int(e['envelope_start_bin']):int(e['envelope_end_bin'])]=True
    pool=[];census=dict(scan_rows=len(scan),source_eligible_unannotated=0,geometry_eligible=0)
    for i,r in enumerate(scan):
        if not truth(r['eligible']) or r['known_ids']:continue
        census['source_eligible_unannotated']+=1
        center=int(r['center_bin']); anchor,_,_=anchor_position(band,center,cfg)
        radius=max(cfg['width_bins'])//2+max(abs(s) for s in cfg['shift_bins'])
        lo,hi=min(center,anchor)-radius,max(center,anchor)+radius
        if not (0<=lo<hi<=len(roles)) or excluded[lo:hi].any():continue
        if not (roles[lo:hi]==roles[lo]).all() or int(roles[lo]) not in (0,1,2):continue
        pool.append(dict(r,window_index=i,entity_id=r['window_id'],original_center_bin=center,boundary_center_bin=int(anchor),
                         envelope_start_bin=lo,envelope_end_bin=hi,role=int(roles[lo])))
    census['geometry_eligible']=len(pool)
    return pool,census


def freeze_controls(focal, pool, band, settings, quality_cache, count):
    def cost(r):
        return abs(math.log1p(float(r['mean_oe']))-math.log1p(float(focal['mean_oe'])))+abs(float(r['valid_fraction'])-float(focal['valid_fraction']))
    natural=[v for v in settings if v['occlusion']=='none']
    chosen=[]
    for r in sorted((r for r in pool if int(r['width_bins'])==int(focal['width_bins'])),key=lambda r:(cost(r),int(r['window_index']))):
        if any(overlap(r,c) for c,_ in chosen):continue
        key=int(r['window_index'])
        if key not in quality_cache:
            quality_cache[key]=all(crop_statistics(band,r,v)['rep1_quality_pass'] for v in natural)
        if not quality_cache[key]:continue
        chosen.append((r,cost(r)))
        if len(chosen)==count:break
    return chosen


def input_inventory(root, config_path, cfg):
    sources=[config_path,root/cfg['source_discovery_config'],root/cfg['source_centering_config'],
             root/cfg['candidate_source'],root/cfg['source_scan'],root/'configs/shared/data.json',
             root/'data/shared/annotations/structures.csv',Path(__file__),root/'scripts/task2/validate_candidates.py']
    for name in ('primary_centering','supplement_centering'):
        directory=root/cfg[name]
        sources += [directory/f for f in ('entities.csv','features.npz','candidate_variants.csv','report.json','thresholds.csv')]
    sources += [root/cfg['primary_centering']/'variants.csv']
    sources += [root/'scripts/task2'/f for f in ('centering_sensitivity.py','centering_status.py','discovery_v2_core.py','discover_task2.py','review_novelty.py')]
    sources += [root/'scripts/shared/contact_band.py']
    data=json.loads((root/'configs/shared/data.json').read_text())
    sources += [root/cfg['band_cache_dir']/(s['sample_id']+'.npz') for s in data['samples']]
    require(all(p.is_file() and not p.is_symlink() for p in sources),'Missing or symlinked validation input')
    return [dict(path=str(p.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size) for p in sources]


def prepare_metadata(root, config_path):
    cfg=json.loads(config_path.read_text())
    require(sha(root/cfg['candidate_source'])==cfg['candidate_source_sha256'],'Frozen 21-candidate source bytes changed')
    original=json.loads((root/cfg['source_discovery_config']).read_text())
    center_cfg=json.loads((root/cfg['source_centering_config']).read_text())
    for key in ('replicate_min_correlation','control_quantile','min_controls','controls_per_candidate'):
        require(cfg[key]==original[key],f'Unchanged discovery rule violated: {key}')
    require(cfg['min_controls']==10 and cfg['control_quantile']==.95 and cfg['replicate_min_correlation']==.5,'Prespecified support rule changed')
    require(center_cfg['min_calibration_windows']==10 and center_cfg['distance_quantile']==.95 and center_cfg['robust_fraction']==.8,'Original morphology gate changed')
    settings=read_csv(root/cfg['primary_centering']/'variants.csv')
    from centering_sensitivity import variants
    require([v['variant_id'] for v in settings]==[v['variant_id'] for v in variants(center_cfg)],'Original 54 variants changed')
    candidates=read_csv(root/cfg['candidate_source']);scan=read_csv(root/cfg['source_scan'])
    require(len(candidates)==cfg['candidate_count']==21 and len({r['window_id'] for r in candidates})==21,'Frozen 21-candidate identity changed')
    primary,pa=load_cohort(root/cfg['primary_centering'],settings)
    supplement,sa=load_cohort(root/cfg['supplement_centering'],settings)
    require([e['entity_id'] for e in primary]==[e['entity_id'] for e in supplement[:len(primary)]],'Supplement changed primary entities')
    for k in ('shape','pca','quality'):require(np.array_equal(pa[k],sa[k][:len(primary)]),f'Supplement changed primary {k}')
    byid={e['entity_id']:e for e in primary if e['kind']=='candidate'}
    require(set(byid)=={r['window_id'] for r in candidates},'Frozen candidate source and features disagree')
    for r in candidates:
        row=scan[int(r['window_index'])]
        require(r['window_id']==row['window_id'] and int(r['window_length_bp'])==int(row['width_bins'])*100,'Candidate scan geometry mismatch')
        require(int(byid[r['window_id']]['original_center_bin'])==int(row['center_bin']),'Candidate center changed')
    capacity,pool=formal_capacity(primary,10)
    return cfg,original,center_cfg,settings,candidates,scan,(primary,pa),(supplement,sa),capacity,pool


def run_validation(root, config_path, output):
    started=time.monotonic()
    from optional_common import fresh_output_path
    output=fresh_output_path(output,root=root,data_root=root)
    require(not output.exists(),'Output exists; preserve it and choose a new version')
    require(output.is_relative_to(root/'results/task2/validation') and len(output.relative_to(root/'results/task2/validation').parts)==1,'Use results/task2/validation/<new_version>')
    require(not any(p.is_symlink() for p in (output,*output.parents)),'Output may not use symlinks')
    cfg,source,ccfg,settings,candidates,scan,primary,supplement,capacity,calpool=prepare_metadata(root,config_path)
    sources=input_inventory(root,config_path,cfg)
    output.mkdir(parents=True,exist_ok=False)
    frozen=dict(schema_version=1,candidates=candidates,variants=settings,config=cfg,input_files=sources,
                primary_capacity=capacity,selection_depends_on_new_candidate_results=False,
                independent_external_confirmation=False)
    json_write(output/'cohort_frozen.json',frozen)
    write_csv(output/'frozen_candidates.csv',candidates)
    write_csv(output/'formal_calibration_pool.csv',calpool)
    json_write(output/'formal_calibration_feasibility.json',capacity)
    print(f'Frozen 21 candidates / 54 variants; exact primary capacity {capacity["maximum_nonoverlapping_windows"]}/10',flush=True)
    from contact_band import load_band
    from discovery_v2_core import bin_roles,train_background
    from centering_sensitivity import occlusion_mask
    data=json.loads((root/'configs/shared/data.json').read_text());n=(data['chrom_length']+99)//100
    roles=bin_roles(n,source)
    band1=train_background(load_band(data['samples'][0],n,256,data_root=root,cache_root=root/cfg['band_cache_dir']),roles)
    candidate_entities=[e for e in primary[0] if e['kind']=='candidate']
    pool,census=build_control_geometry(scan,band1,candidate_entities,roles,ccfg)
    print(f'Control geometry pool {len(pool)}; matching uses rep1 QC/density only',flush=True)
    controls={};relations=[];quality_cache={}
    for r in candidates:
        focal=scan[int(r['window_index'])]
        matched=freeze_controls(focal,pool,band1,settings,quality_cache,cfg['controls_per_candidate'])
        controls[r['window_id']]=[e for e,_ in matched]
        for rank,(e,cost) in enumerate(matched,1):
            relations.append(dict(candidate_id=r['window_id'],rank=rank,control_id=e['entity_id'],window_index=int(e['window_index']),
                                  source_width_bins=int(e['width_bins']),match_cost=cost,
                                  rep1_mean_oe=float(e['mean_oe']),rep1_valid_fraction=float(e['valid_fraction']),
                                  envelope_start_bin=e['envelope_start_bin'],envelope_end_bin=e['envelope_end_bin'],role=e['role']))
        print(f'{r["window_id"]}: froze {len(matched)} controls',flush=True)
    json_write(output/'controls_frozen.json',dict(selection='rep1 only before new rep2 load',controls=controls,census=census,rule=cfg['control_spatial_rules']))
    write_csv(output/'matched_controls.csv',relations)
    # Only after candidate and control identities are persisted do we open rep2 pixels.
    band2=train_background(load_band(data['samples'][1],n,256,data_root=root,cache_root=root/cfg['band_cache_dir']),roles)
    unique={e['entity_id']:e for v in controls.values() for e in v}
    for e in candidate_entities:require(e['entity_id'] not in unique,'Candidate leaked into control cohort')
    masks={v['variant_id']:occlusion_mask(int(v['width_bins']),float(v['fraction']),int(v['seed'])) for v in settings}
    scores={};score_rows=[]
    for i,e in enumerate([*unique.values(),*candidate_entities]):
        for v in settings:
            row=dict(entity_id=e['entity_id'],kind='candidate' if e['entity_id'] in {c['window_id'] for c in candidates} else 'control',
                     variant_id=v['variant_id'],anchor=v['anchor'],width_bins=int(v['width_bins']))
            row.update(crop_statistics(band1,e,v,score=True,second=band2,occlusion=masks[v['variant_id']]))
            scores[e['entity_id'],v['variant_id']]=row;score_rows.append(row)
        if (i+1)%20==0:print(f'Computed matched transforms: {i+1}/{len(unique)+21} entities',flush=True)
    write_csv(output/'replicate_scores.csv',score_rows)
    primary_rows,primary_summary=morphology_evidence(root/cfg['primary_centering'],primary[0],settings,10)
    supplement_rows,supplement_summary=morphology_evidence(root/cfg['supplement_centering'],supplement[0],settings,10)
    write_csv(output/'primary_morphology_candidates.csv',primary_rows)
    write_csv(output/'supplement_morphology_candidates.csv',supplement_rows)
    pm={r['entity_id']:r for r in primary_rows};sm={r['entity_id']:r for r in supplement_rows}
    variants_out=[];cells=[];candidate_rows=[]
    for c in candidates:
        cid=c['window_id'];decisions={}
        for v in settings:
            s=scores[cid,v['variant_id']]
            d=support_decision(s['correlation'],[scores[e['entity_id'],v['variant_id']]['correlation'] for e in controls[cid]])
            if not s['quality_pass']:
                d.update(replicate_supported=None,margin=None,assessment_status='not_evaluable',assessment_reason='candidate_quality_failed')
            d.update(candidate_id=cid,variant_id=v['variant_id'],anchor=v['anchor'],width_bins=int(v['width_bins']),
                     shift_bins=int(v['shift_bins']),occlusion=v['occlusion'],quality_pass=s['quality_pass'])
            decisions[v['variant_id']]=d;variants_out.append(d)
        base=f'original_w{int(c["window_length_bp"])//100}_s0_none';b=decisions[base]
        require(abs(float(c['correlation'])-b['correlation'])<=1e-7,'Saved baseline score differs from read-only cache recomputation')
        cell_rows=six_cells(settings,decisions)
        cells.extend(dict(candidate_id=cid,**r) for r in cell_rows)
        values=list(decisions.values());valid=[r for r in values if r['replicate_supported'] is not None]
        row=dict(candidate_id=cid,start_bp=int(c['start_bp']),window_length_bp=int(c['window_length_bp']),
                 frozen_historical_correlation=float(c['correlation']),frozen_historical_cutoff=float(c['cutoff']),
                 selected_controls=len(controls[cid]),baseline_score=b['correlation'],baseline_control_q95=b['control_q95'],
                 baseline_threshold=b['cutoff'],baseline_valid_controls=b['matched_controls'],baseline_margin=b['margin'],
                 baseline_support=b['replicate_supported'],baseline_status=b['assessment_status'],
                 total_variants=54,assessed_variants=len(valid),above_threshold_variants=sum(r['replicate_supported'] for r in valid),
                 not_evaluable_variants=54-len(valid),minimum_valid_controls=min(r['matched_controls'] for r in values),
                 full_grid_above_fraction=sum(r['replicate_supported'] for r in valid)/54 if len(valid)==54 else None,
                 primary_shape_status=pm[cid]['assessment_status'],primary_robust_anomaly=pm[cid]['robust_recentered_anomaly'],
                 supplement_shape_status=sm[cid]['assessment_status'],supplement_robust_anomaly=sm[cid]['robust_recentered_anomaly'])
        for r in cell_rows:row[f'{r["anchor"]}_w{r["width_bins"]}_above_of_9']=r['above_threshold'] if r['assessed_variants']==9 else None
        candidate_rows.append(row)
    write_csv(output/'candidate_variants.csv',variants_out)
    write_csv(output/'candidate_scale_evidence.csv',cells)
    write_csv(output/'candidate_evidence.csv',candidate_rows)
    kr=[]
    for label,(ent,arr) in [('primary_calibration',primary),('supplement_calibration',supplement)]:
        kr.extend(retrieval(label,ent,arr,settings))
    known_summaries=summarize_retrieval(kr)
    write_csv(output/'known_retrieval.csv',kr)
    write_csv(output/'known_retrieval_summary.csv',known_summaries)
    write_csv(output/'known_query_evidence.csv',known_query_summaries(kr))
    # Check all protected inputs again before publishing a completion marker.
    require(sources==input_inventory(root,config_path,cfg),'Input/source changed during validation')
    summary=dict(schema_version=1,candidates=21,variants_per_candidate=54,
                 selected_control_relations=len(relations),unique_control_entities=len(unique),
                 all_baselines_have_at_least_ten_valid_controls=all(r['baseline_valid_controls']>=10 for r in candidate_rows),
                 all_variants_have_at_least_ten_valid_controls=all(r['minimum_valid_controls']>=10 for r in candidate_rows),
                 baseline_supported=sum(r['baseline_support'] is True for r in candidate_rows),
                 baseline_below_threshold=sum(r['baseline_support'] is False for r in candidate_rows),
                 baseline_not_evaluable=sum(r['baseline_support'] is None for r in candidate_rows),
                 complete_54_variant_candidates=sum(r['assessed_variants']==54 for r in candidate_rows),
                 primary_morphology_capacity=capacity,primary_morphology=primary_summary,supplement_morphology=supplement_summary,
                 known_retrieval_query_ids=len({r['query_id'] for r in kr}),known_retrieval_rows=len(kr),
                 known_query_classes={k:len({r['query_id'] for r in kr if r['true_type']==k}) for k in TYPES},
                 known_retrieval_summary=known_summaries,
                 training_performed=False,raw_genome_scan_performed=False,cached_pixels_recomputed=True,
                 independent_confirmation=False,elapsed_seconds=time.monotonic()-started,
                 peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                 interpretation=cfg['interpretation'],
                 limitations=['The 54 variants reuse the same positions and are not 54 independent samples.',
                              'Support background controls are not known-shape calibration windows.',
                              'The original primary spatial constraints allow at most eight known calibrators; the morphology endpoint stays NA.',
                              'Known calibration queries contain no CHID; class-complete retrieval is not established.',
                              'Retrieval diagnostics do not set new candidate thresholds or certify a biological type.'])
    json_write(output/'report.json',summary)
    write_presentation(output,candidate_rows,cells,known_summaries,summary)
    artifacts=[dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in sorted(output.iterdir()) if p.is_file() and p.name not in ('ARTIFACTS.json','COMPLETED.json')]
    json_write(output/'ARTIFACTS.json',dict(schema_version=1,files=artifacts))
    json_write(output/'COMPLETED.json',dict(status='completed',artifact_manifest_sha256=sha(output/'ARTIFACTS.json'),candidate_source_sha256=sha(root/cfg['candidate_source'])))
    return summary


def write_presentation(output, candidates, cells, retrieval_summary, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids=[r['candidate_id'] for r in candidates];ys=np.arange(len(ids))
    lookup={(r['candidate_id'],r['anchor'],r['width_bins']):r for r in cells}
    matrix=np.array([[lookup[cid,a,w]['full_cell_above_fraction'] if lookup[cid,a,w]['full_cell_above_fraction'] is not None else np.nan
                      for a in ('original','boundary') for w in (64,128,256)] for cid in ids])
    fig,axes=plt.subplots(1,2,figsize=(15,10),layout='constrained',gridspec_kw={'width_ratios':[1.2,1]})
    axes[0].scatter([r['baseline_score'] for r in candidates],ys,label='Candidate correlation',color='#175c9e',zorder=3)
    axes[0].scatter([r['baseline_threshold'] if r['baseline_threshold'] is not None else np.nan for r in candidates],ys,label='max(0.5, matched q95)',marker='|',s=220,color='#b64030')
    for i,r in enumerate(candidates):
        if r['baseline_threshold'] is not None:axes[0].plot([r['baseline_score'],r['baseline_threshold']],[i,i],color='#c0c4c9',zorder=1)
    axes[0].set(yticks=ys,yticklabels=ids,xlabel='Pearson r',title='Frozen 21: original source-width endpoint')
    axes[0].invert_yaxis();axes[0].legend(loc='lower left',fontsize=8);axes[0].grid(axis='x',alpha=.2)
    cmap=plt.get_cmap('Blues').copy();cmap.set_bad('#bfc3c7')
    im=axes[1].imshow(np.ma.masked_invalid(matrix),cmap=cmap,vmin=0,vmax=1,aspect='auto')
    axes[1].set(yticks=ys,yticklabels=[],xticks=range(6),xticklabels=['O 6.4k','O 12.8k','O 25.6k','B 6.4k','B 12.8k','B 25.6k'],title='All 54 variants: six fixed cells of nine')
    axes[1].tick_params(axis='x',rotation=35)
    for i in range(len(ids)):
        for j,(a,w) in enumerate((a,w) for a in ('original','boundary') for w in (64,128,256)):
            r=lookup[ids[i],a,w]
            text=f"{r['above_threshold']}/9" if r['assessed_variants']==9 else f"NA\n{r['assessed_variants']}/9"
            axes[1].text(j,i,text,ha='center',va='center',fontsize=8,color='white' if np.isfinite(matrix[i,j]) and matrix[i,j]>.6 else 'black')
    fig.colorbar(im,ax=axes[1],label='Fraction above matched support threshold; gray = incomplete cell',shrink=.7)
    fig.suptitle('Same original thresholds; O = original anchor, B = fixed boundary anchor\nVariant fractions are sensitivity descriptions, not a new morphology or cluster gate',fontsize=12)
    fig.savefig(output/'candidate_support.png',dpi=160);plt.close(fig)
    rows=[r for r in retrieval_summary if r['cohort']=='supplement_calibration' and r['type'] in TYPES]
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,check in zip(axes,('heldout_class','paired_identity')):
        selected=[r for r in rows if r['check']==check]
        positions=[];labels=[]
        for i,r in enumerate(selected):
            positions.append(i);labels.append(f"{r['metric']} / {r['type']}\n{r['query_ids']} queries")
            if r['top1'] is None:
                ax.bar(i,1,color='#eeeeee');ax.text(i,.5,'NA',ha='center')
            else:
                ax.bar(i,r['top1'],color='#2e7094' if r['metric']=='pca' else '#cf8a43')
                ax.plot([i-.32,i+.32],[r['chance_top1']]*2,color='black',lw=2)
        ax.set(xticks=positions,xticklabels=labels,ylim=(0,1.1),ylabel='Top-1 retrieval fraction over the complete grid',title=check.replace('_',' '))
        ax.tick_params(axis='x',rotation=35);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Known-structure retrieval sanity check: fixed 17-query supplement\nBlack ticks = class-prior / gallery-size chance; no CHID queries; no biological CI',fontsize=11)
    fig.savefig(output/'known_retrieval.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,7),layout='constrained',gridspec_kw={'width_ratios':[1,2]})
    axes[0].bar(['Required','Maximum\nprimary'],[10,report['primary_morphology_capacity']['maximum_nonoverlapping_windows']],color=['#aaa','#b45b48'])
    axes[0].set(ylim=(0,12),ylabel='Known calibration windows',title='Exact full-envelope capacity')
    status=np.array([[np.nan, np.nan if r['supplement_robust_anomaly'] is None else int(r['supplement_robust_anomaly'])] for r in candidates])
    cmap=plt.get_cmap('YlOrRd').copy();cmap.set_bad('#c4c6c8')
    axes[1].imshow(np.ma.masked_invalid(status),aspect='auto',cmap=cmap,vmin=0,vmax=1)
    axes[1].set(yticks=ys,yticklabels=ids,xticks=[0,1],xticklabels=['Primary: NA','Viewed supplement'],title='Original candidate-level morphology rule')
    for i,r in enumerate(candidates):
        axes[1].text(0,i,'NA',ha='center',va='center',fontsize=8)
        axes[1].text(1,i,'NA' if r['supplement_robust_anomaly'] is None else 'Not robust',ha='center',va='center',fontsize=8)
    fig.suptitle('Background replication controls do not replace known-shape calibration\nPrimary = 21 NA; supplement = 19 assessable + 2 QC failures',fontsize=11)
    fig.savefig(output/'morphology_status.png',dpi=160);plt.close(fig)
    def fmt(x):return 'NA' if x is None else f'{x:.4f}' if isinstance(x,float) else str(x)
    body=[]
    for r in candidates:
        cid=r['candidate_id']
        fields=[cid,r['baseline_valid_controls'],r['baseline_score'],r['baseline_threshold'],r['baseline_margin'],
                r['baseline_support'],f"{r['above_threshold_variants']}/{r['assessed_variants']} (+{r['not_evaluable_variants']} NA)",r['primary_robust_anomaly'],r['supplement_robust_anomaly']]
        body.append('<tr>'+''.join('<td>'+html.escape(fmt(v))+'</td>' for v in fields)+'</tr>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>固定21候选：匹配对照、54变体与已知检索</title>
<style>body{max-width:1450px;margin:25px auto;padding:20px;font:16px/1.65 sans-serif;color:#172635}img{max-width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}.card{background:#f0f4f7;padding:16px;margin:15px 0}a{color:#175c9e}</style>
<h1>固定21候选：候选级证据与已知结构检索核查</h1>
<div class="card">候选身份、54变体及原门槛保持固定。重复支持使用 max(0.5, 匹配对照相关性95%分位数)，至少10个有效对照。对照选择只使用rep1密度和质量，选择后写入冻结文件，再读取rep2；未按候选是否通过更换对照。</div>
<p>这次使用全21候选完整扰动足迹排除、对照之间完整足迹不相交；因此是更严格对照队列下的开发性重核，不覆盖历史结果。54变体属于相同位置的敏感性分析，不能增加独立样本量。</p>
<p>原形态校准在相同空间和质量规则下，最大容量仅8个；10个的门槛不变，主形态端点继续NA。匹配背景对照用于重复性，不能填入已知形态校准。探索补充仅19例可评价，完整21例端点仍NA，未确认新类型。</p>
<p><a href="candidate_evidence.csv">21例汇总</a> · <a href="candidate_scale_evidence.csv">六组尺度证据</a> · <a href="candidate_variants.csv">全部54变体</a> · <a href="matched_controls.csv">冻结匹配关系</a> · <a href="controls_frozen.json">对照冻结</a> · <a href="report.json">核验摘要</a></p>
<img src="candidate_support.png" alt="候选基线及完整变体支持">
<table><thead><tr><th>候选</th><th>基线有效对照</th><th>相关性</th><th>阈值</th><th>余量</th><th>基线支持</th><th>全部变体支持/可评估</th><th>主形态</th><th>补充稳健异常</th></tr></thead><tbody>'''+''.join(body)+'''</tbody></table>
<h2>已知结构 retrieval sanity check</h2><p>固定原8个及补充17个已知查询，按同一54变体检索。空间留出类别检索排除自身和重叠足迹，并与类别先验比较；跨WT同ID检索另报告真实配对位置能否排到前列。两项均不用于调整候选门槛，且没有CHID查询，不能认证全部三类的形态辨识。</p>
<img src="known_retrieval.png" alt="已知结构检索及先验参照"><p><a href="known_query_evidence.csv">每个已知查询54变体汇总</a> · <a href="known_retrieval_summary.csv">分类别检索汇总</a> · <a href="known_retrieval.csv">全量排名</a></p>
<img src="morphology_status.png" alt="原规则的主校准空间容量与候选状态">
<p><a href="formal_calibration_feasibility.json">原主校准最大容量证明</a> · <a href="formal_calibration_pool.csv">全部合格校准候选</a> · <a href="primary_morphology_candidates.csv">主形态汇总</a> · <a href="supplement_morphology_candidates.csv">探索补充汇总</a> · <a href="ARTIFACTS.json">产物哈希</a></p>
<p>既有WT和区域已被查看，本次不是独立外部确认。实际读取现存100 bp缓存，未训练或扫描原始10 bp全基因组。</p></html>'''
    (output/'index.html').write_text(page,encoding='utf-8')


def verify_validation(root, output):
    """Verify bytes, frozen geometry, thresholds, complete grids and retrieval."""
    complete=json.loads((output/'COMPLETED.json').read_text())
    require(complete['artifact_manifest_sha256']==sha(output/'ARTIFACTS.json'),'Completion manifest hash mismatch')
    artifacts=json.loads((output/'ARTIFACTS.json').read_text())['files']
    paths=[r['path'] for r in artifacts]
    require(len(paths)==len(set(paths)) and all(Path(p).name==p for p in paths),'Invalid artifact paths')
    require(set(paths)=={p.name for p in output.iterdir() if p.name not in ('ARTIFACTS.json','COMPLETED.json')},'Artifact file set mismatch')
    for r in artifacts:require(not (output/r['path']).is_symlink() and sha(output/r['path'])==r['sha256'],'Artifact hash mismatch: '+r['path'])
    frozen=json.loads((output/'cohort_frozen.json').read_text())
    for r in frozen['input_files']:require(sha(root/r['path'])==r['sha256'],'Protected input changed: '+r['path'])
    cfg=frozen['config'];settings=frozen['variants'];candidates=frozen['candidates']
    saved=read_csv(output/'candidate_evidence.csv')
    require([r['candidate_id'] for r in saved]==[r['window_id'] for r in candidates],'Candidate identities/order changed')
    controls=json.loads((output/'controls_frozen.json').read_text())['controls']
    entities={r['entity_id']:r for r in read_csv(root/cfg['primary_centering']/'entities.csv') if r['kind']=='candidate'}
    for cid,group in controls.items():
        for i,e in enumerate(group):
            require(not any(overlap(e,c) for c in entities.values()),'Control overlaps candidate envelope')
            require(not any(overlap(e,g) for g in group[:i]),'Controls overlap within frozen set')
            require(int(e['width_bins'])==int(next(c for c in candidates if c['window_id']==cid)['window_length_bp'])//100,'Control source width mismatch')
    scores={}
    for r in read_csv(output/'replicate_scores.csv'):
        key=(r['entity_id'],r['variant_id']);require(key not in scores,'Duplicate replicate score')
        scores[key]=r
    expected={*entities}|{e['entity_id'] for g in controls.values() for e in g}
    require(set(scores)=={(e,v['variant_id']) for e in expected for v in settings},'Incomplete score grid')
    observed=read_csv(output/'candidate_variants.csv')
    require(len(observed)==21*54 and len({(r['candidate_id'],r['variant_id']) for r in observed})==21*54,'Incomplete candidate grid')
    for r in observed:
        cid,vid=r['candidate_id'],r['variant_id']
        expected_decision=support_decision(number(scores[cid,vid]['correlation']),[number(scores[e['entity_id'],vid]['correlation']) for e in controls[cid]])
        if not truth(scores[cid,vid]['quality_pass']):expected_decision.update(replicate_supported=None,margin=None,assessment_status='not_evaluable',assessment_reason='candidate_quality_failed')
        for k in ('correlation','control_q95','cutoff','margin'):
            a,b=number(r[k]),expected_decision[k]
            require((a is None and b is None) or (a is not None and b is not None and abs(a-b)<1e-12),'Saved threshold/score mismatch: '+k)
        require(int(r['matched_controls'])==expected_decision['matched_controls'],'Control count mismatch')
        flag=None if r['replicate_supported']=='' else truth(r['replicate_supported'])
        require(flag==expected_decision['replicate_supported'] and r['assessment_status']==expected_decision['assessment_status'],'Candidate decision mismatch')
    # Full known-query grids are independently regenerated from protected features.
    kr=[]
    for label,name in [('primary_calibration','primary_centering'),('supplement_calibration','supplement_centering')]:
        ent,arr=load_cohort(root/cfg[name],settings);kr.extend(retrieval(label,ent,arr,settings))
    stored=read_csv(output/'known_retrieval.csv')
    require(len(stored)==len(kr),'Known retrieval grid length mismatch')
    for a,b in zip(stored,kr):
        for k in ('cohort','check','metric','query_id','variant_id','nearest_id','assessment_status'):
            require(a[k]==b[k],'Known retrieval identity/ranking mismatch')
        for k in ('top1_correct','top5_correct'):
            require((None if a[k]=='' else truth(a[k]))==b[k],'Known retrieval decision mismatch')
    return dict(status='passed',candidates=21,variants=54,control_sets=len(controls),threshold_decisions_recomputed=len(observed),
                known_retrieval_rows_recomputed=len(kr),protected_inputs=len(frozen['input_files']),artifacts=len(artifacts),training_performed=False)
