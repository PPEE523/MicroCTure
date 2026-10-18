#!/usr/bin/env python3
"""Read-only ROI scientific audit, exact R0 replay and conditional group uncertainty."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from microcture.task1_roi.experiment import (prepare_manifest, planned_jobs, job_directory, _load_saved,
    json_signature, sha256)
from microcture.task1_formal.evaluation import _identity, _fold_ids, _same_score, _neural_summary, _csv
from microcture.task1.evaluation import CLASS_NAMES, metrics, _read_prediction_csv

ARMS=(('R0','I3'),('R1','I3'),('R1','G0'),('R1','V0'),('R1','P0'))
SEEDS=(42,43,44)


def scores_from_confusion(cm):
    cm=np.asarray(cm,dtype=float);tp=np.diagonal(cm,axis1=-2,axis2=-1);support=cm.sum(-1);predicted=cm.sum(-2)
    precision=np.divide(tp,predicted,out=np.zeros_like(tp),where=predicted>0)
    recall=np.divide(tp,support,out=np.zeros_like(tp),where=support>0)
    f1=np.divide(2*tp,predicted+support,out=np.zeros_like(tp),where=(predicted+support)>0)
    return {'macro_f1':f1.mean(-1),'CHID_f1':f1[...,2],'CHID_precision':precision[...,2],
        'CHID_recall':recall[...,2],'balanced_accuracy':recall.mean(-1)},support,predicted


def group_bootstrap(oof, identity, repeats=2000, rng_seed=20261014):
    groups=sorted({r['group_id'] for r in identity.values()});gi={g:i for i,g in enumerate(groups)}
    cm=np.zeros((len(ARMS),len(SEEDS),len(groups),3,3),dtype=np.int64)
    for ai,arm in enumerate(ARMS):
        for si,seed in enumerate(SEEDS):
            for row in oof[(*arm,seed)]['rows']:
                cm[ai,si,gi[int(row['group_id'])],int(row['y_true']),int(row['y_pred'])]+=1
    weights=np.random.default_rng(rng_seed).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=repeats)
    raw,support,predicted=scores_from_confusion(np.einsum('bg,asgij->basij',weights,cm))
    observed,_,_=scores_from_confusion(cm.sum(2))
    per_arm={};distributions={}
    for ai,(a,c) in enumerate(ARMS):
        name=f'{a}/{c}';per_arm[name]={};distributions[name]={}
        for metric in raw:
            values=raw[metric][:,ai,:].mean(-1);seeds=observed[metric][ai];distributions[name][metric]=values
            per_arm[name][metric]={'point_mean':float(seeds.mean()),'seed_values':seeds.tolist(),
                'seed_sample_sd':float(seeds.std(ddof=1)),'ci95':np.quantile(values,[.025,.975]).tolist()}
    pairs={}
    for first,second in [('R1/I3','R0/I3'),('R1/I3','R1/G0'),('R1/I3','R1/V0'),('R1/I3','R1/P0'),('R1/V0','R1/G0'),('R1/P0','R1/V0')]:
        pair={}
        for metric in raw:
            seed_delta=np.array(per_arm[first][metric]['seed_values'])-np.array(per_arm[second][metric]['seed_values'])
            values=distributions[first][metric]-distributions[second][metric]
            pair[metric]={'point_mean':float(seed_delta.mean()),'seed_deltas':seed_delta.tolist(),
                'ci95':np.quantile(values,[.025,.975]).tolist()}
        pairs[f'{first}-{second}']=pair
    return {'scope':'Conditional saved-OOF group uncertainty, no refitting or correction for prior researcher adaptation',
        'B':repeats,'rng_seed':rng_seed,'unit':'49 spatial groups, all structures/seeds/arms sampled together',
        'seed_aggregation':'Compute metrics separately per endpoint seed, then average across seeds',
        'interval':'Unadjusted two-sided percentile95; multiple contrasts are descriptive',
        'zero_CHID_support_bootstraps':int(np.sum(np.any(support[...,2]==0,axis=(1,2)))),
        'zero_CHID_prediction_bootstraps':int(np.sum(np.any(predicted[...,2]==0,axis=(1,2)))),
        'arms':per_arm,'pairs':pairs}


def _without_time(value):
    if isinstance(value,dict):return {k:_without_time(v) for k,v in value.items() if k!='epoch_seconds'}
    if isinstance(value,list):return [_without_time(v) for v in value]
    return value


def audit(run, report):
    manifest=json.loads((run/'manifest.json').read_text());config=manifest['config'];summary=json.loads((run/'summary.json').read_text())
    current,metadata=prepare_manifest(ROOT,config,manifest['requested_device'])
    if manifest!=current:raise ValueError('ROI source/config/dependency/environment no longer matches')
    identity=_identity(metadata);folds=_fold_ids(manifest,identity,run);jobs=planned_jobs(config)
    records=[];artifact_records=[];replay=[];oof={};stats={};allcases=[]
    for job in jobs:
        record=_load_saved(run,job,manifest,identity,folds)
        if record['status']!='completed':raise ValueError('Full review requires all registered scientific fits, including completed acceptance failures')
        records.append(record);directory=job_directory(run,job)
        index=json.loads((directory/'ARTIFACTS.json').read_text())
        artifact_records.extend({'path':str((directory/row['name']).relative_to(run)),**{k:row[k] for k in ('sha256','size_bytes')}} for row in index['files'])
        if job['alignment']=='R0':
            old=ROOT/'results/task1_readout_v1/runs/I3/O0'/f"seed_{job['seed']}"/f"fold_{job['fold']}"
            newstate=torch.load(directory/'final.pt',map_location='cpu',weights_only=True)['state_dict']
            oldstate=torch.load(old/'final.pt',map_location='cpu',weights_only=True)['state_dict']
            checks={'all_state_tensors_exact':set(newstate)==set(oldstate) and all(torch.equal(newstate[k],oldstate[k]) for k in newstate),
                'numeric_history_exact':_without_time(json.loads((directory/'history.json').read_text()))==_without_time(json.loads((old/'history.json').read_text())),
                'outer_CSV_bytes_exact':(directory/'outer_predictions.csv').read_bytes()==(old/'outer_predictions.csv').read_bytes(),
                'validation_CSV_bytes_exact':(directory/'validation_predictions.csv').read_bytes()==(old/'validation_predictions.csv').read_bytes()}
            previous=json.loads((old/'metrics.json').read_text())
            checks.update({f'{k}_exact':record['metrics'][k]==previous[k] for k in ('initial_state_sha256','common_initial_state_sha256','minibatch_order_sha256','class_weights','acceptance')})
            if not all(checks.values()):raise ValueError(f'R0 exact replay failed: {job} {checks}')
            replay.append({'seed':job['seed'],'fold':job['fold'],'checks':checks})
    for alignment,condition in ARMS:
        name=f'{alignment}/{condition}';items=[r for r in records if (r['job']['alignment'],r['job']['condition'])==(alignment,condition)]
        scores=[];countrows=[]
        for seed in SEEDS:
            per=[r for r in items if r['job']['seed']==seed]
            rows=sorted([{**x,'fold':r['job']['fold']} for r in per for x in r['predictions']['rows']],key=lambda x:x['structure_id'])
            if len(rows)!=344 or {r['structure_id'] for r in rows}!=set(identity):raise ValueError('OOF must contain344 unique structures')
            y=np.array([int(r['y_true']) for r in rows]);p=np.array([[float(r[f'p_{c}']) for c in CLASS_NAMES] for r in rows]);score=metrics(y,p)
            saved=_read_prediction_csv(run/f'oof/{alignment}/{condition}/seed_{seed}/predictions.csv')
            if rows!=[{**r,'fold':int(r['fold'])} for r in saved['rows']]:raise ValueError('Derived OOF does not exactly merge sealed per-fold predictions')
            _same_score(summary['models'][name]['seeds'][str(seed)]['metrics'],score,'summary seed score')
            oof[alignment,condition,seed]={'rows':rows,'metrics':score,'by_id':{r['structure_id']:r for r in rows}}
            scores.append(score);cm=score['confusion_matrix']
            countrows.append({'seed':seed,'predicted_counts':np.bincount(p.argmax(1),minlength=3).tolist(),
                'CHID_TP':cm[2][2],'CHID_FP':cm[0][2]+cm[1][2],'CHID_FN':cm[2][0]+cm[2][1],
                'CHIN_to_CHID':cm[1][2],'CHID_to_CHIN':cm[2][1], 'WT_disagreement':sum(int(r['replicate_disagree']) for r in rows)})
        comparison=next(r for r in summary['comparison'] if (r['alignment'],r['condition'])==(alignment,condition))
        for key in ('macro_f1','accuracy','balanced_accuracy'):
            vals=np.array([s[key] for s in scores])
            if not np.allclose([comparison[f'{key}_mean'],comparison[f'{key}_seed_sample_std']],[vals.mean(),vals.std(ddof=1)],atol=1e-12,rtol=0):raise ValueError('Comparison differs')
        for cls in CLASS_NAMES:
            for measure in ('precision','recall','f1'):
                vals=np.array([s['per_class'][cls][measure] for s in scores]);key=f'{cls}_{measure}'
                if not np.allclose([comparison[f'{key}_mean'],comparison[f'{key}_seed_sample_std']],[vals.mean(),vals.std(ddof=1)],atol=1e-12,rtol=0):raise ValueError('Per-class comparison differs')
        learning=_neural_summary(items,config)
        if learning!=summary['models'][name]['learning']:raise ValueError('Learning summary differs')
        stablewrong=[];stableFN=[];stableFP=[];anywrong=[];stablecorrect=[];cases=[]
        for sid in sorted(identity):
            truth=identity[sid]['y'];rs=[oof[alignment,condition,s]['by_id'][sid] for s in SEEDS];preds=[int(r['y_pred']) for r in rs]
            if all(p!=truth for p in preds):stablewrong.append(sid)
            if any(p!=truth for p in preds):anywrong.append(sid)
            else:stablecorrect.append(sid)
            if truth==2 and all(p!=2 for p in preds):stableFN.append(sid)
            if truth!=2 and all(p==2 for p in preds):stableFP.append(sid)
            case={'arm':name,'structure_id':sid,'type':CLASS_NAMES[truth],'length_bp':identity[sid]['length_bp'],'group_id':identity[sid]['group_id'],
                'stable_wrong':all(p!=truth for p in preds),'correct_seed_count':sum(p==truth for p in preds),
                **{f'seed_{s}_prediction':CLASS_NAMES[p] for s,p in zip(SEEDS,preds)},
                **{f'seed_{s}_p_CHID':float(r['p_CHID']) for s,r in zip(SEEDS,rs)}}
            cases.append(case);allcases.append(case)
        trajectories={'train_ce_20':[r['history'][20]['weighted_train_ce'] for r in items],'train_ce_40':[r['history'][40]['weighted_train_ce'] for r in items],
            'validation_ce_20':[r['history'][20]['weighted_validation_ce'] for r in items],'validation_ce_40':[r['history'][40]['weighted_validation_ce'] for r in items]}
        failed_learning=[{'job':r['job'],'failed_checks':r['metrics']['acceptance']['primary']['failed_checks'],
            'terminal_relative_loss_regression':r['metrics']['acceptance']['primary']['terminal_relative_loss_regression']} for r in items if not r['metrics']['acceptance']['accepted']]
        stats[name]={'scores':scores,'predictions':countrows,'learning':learning,'stable_wrong_ids':stablewrong,'stable_FN_ids':stableFN,
            'stable_FP_ids':stableFP,'any_wrong_ids':anywrong,'stable_correct_ids':stablecorrect,
            'train40_CE_sample_sd':float(np.std([r['metrics']['final_train_ce'] for r in items],ddof=1)),
            'validation_CE_rising_20_to40':sum(b>a for a,b in zip(trajectories['validation_ce_20'],trajectories['validation_ce_40'])),
            'trajectory_points':{k:{'median':float(np.median(v)),'mean':float(np.mean(v))} for k,v in trajectories.items()},'learning_failures':failed_learning}
        if len(stablewrong)!=comparison['stable_wrong_count'] or stablewrong!=summary['models'][name]['stable_wrong_ids']:raise ValueError('Stable error summary differs')
    bootstrap=group_bootstrap(oof,identity)
    removed=sorted(set(stats['R0/I3']['stable_wrong_ids'])-set(stats['R1/I3']['stable_wrong_ids']))
    added=sorted(set(stats['R1/I3']['stable_wrong_ids'])-set(stats['R0/I3']['stable_wrong_ids']))
    fully_repaired=[sid for sid in removed if all(int(oof['R1','I3',s]['by_id'][sid]['y_pred'])==identity[sid]['y'] for s in SEEDS)]
    inventory={'status':'passed','experiment_signature':manifest['experiment_signature'],'source_signature':manifest['source_signature'],
        'dataset_signature':manifest['dataset_signature'],'verified_fit_indices':75,'indexed_scientific_files':len(artifact_records),
        'indexed_bytes':sum(r['size_bytes'] for r in artifact_records),'inventory_signature':json_signature(artifact_records),
        'verified_complete_OOF':15,'R0_exact_replay':replay,'old_dependency_signatures':manifest['dependency_source_signature'],
        'audit_script_sha256':sha256(Path(__file__).resolve()),
        'reading_scope':'metadata,all saved scientific files and CPU checkpoint tensor equality; no new images/weight inference/fit'}
    stats['changes']={'removed_stable_errors':removed,'added_stable_errors':added,'fully_repaired_old_stable_errors':fully_repaired}
    report.mkdir(parents=True,exist_ok=True)
    for name,value in [('integrity.json',inventory),('group_bootstrap.json',bootstrap),('error_learning.json',stats)]:
        (report/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    _csv(report/'case_review.csv',allcases)
    return {'integrity':inventory,'bootstrap':bootstrap,'stats':stats}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,default=ROOT/'results/task1_roi_v1')
    parser.add_argument('--report-dir',type=Path,default=ROOT/'reports/task1_roi_review_v1');args=parser.parse_args()
    result=audit(args.run_dir,args.report_dir)
    print(json.dumps({'status':'passed','jobs':75,'OOF':15,'indexed_scientific_files':result['integrity']['indexed_scientific_files'],
        'R1-I3-minus-R0-I3':result['bootstrap']['pairs']['R1/I3-R0/I3'],'changes':result['stats']['changes']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
