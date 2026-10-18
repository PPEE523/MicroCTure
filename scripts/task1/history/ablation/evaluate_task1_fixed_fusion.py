#!/usr/bin/env python3
"""Fixed probability arithmetic on sealed parents; no training or model inference."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from microcture.task1_shape_baseline_v2 import experiment as shape
from microcture.task1_roi import experiment as roi
from microcture.task1_formal.runner import write_artifact_index,verify_artifact_index,guard_output
from microcture.task1_formal.evaluation import _identity,_fold_ids,_same_score,_csv
from microcture.task1.evaluation import CLASS_NAMES,_read_prediction_csv,save_prediction_csv,metrics,grouped_predictions
from microcture.task1.training import write_json,run_lock
from scripts.audit_task1_roi_review import scores_from_confusion

PROTOCOL='task1_fixed_half_lr_half_m2plus_probability_fusion_v1'


def require(ok,msg):
    if not ok:raise ValueError(msg)


def validate(c):
    require(c['protocol']==PROTOCOL and c['conditions']==['I3','G0','V0','P0'] and c['folds']==list(range(5)) and c['cnn_seeds']==[42,43,44] and c['lr_seed']==42,'Freeze experiment matrix')
    require(c['lr_weight']==c['cnn_weight']==.5 and c['temperature_or_threshold_search'] is False,'No alpha/temperature/threshold search')
    require(c['macro_f1_gate']==.90 and c['per_class_precision_recall_gate']==.85 and c['bootstrap_repeats']==2000 and c['bootstrap_seed']==20261014,'Freeze gates/uncertainty')
    require(c['pooled']=='0.5 LR + 0.5 mean(CNN42,CNN43,CNN44), diagnostic only','Pooled may not replace main seeds')
    return c


def blend_rows(lr_rows,cnn_members):
    """Blend individual WT probabilities; identity and group sets must commute with averaging."""
    lr={r['structure_id']:r for r in lr_rows};members=[{r['structure_id']:r for r in rows} for rows in cnn_members]
    require(len(lr)==len(lr_rows) and all(len(d)==len(rows) for d,rows in zip(members,cnn_members)),'Duplicate IDs')
    require(bool(lr) and bool(members) and all(set(d)==set(lr) for d in members),'Member IDs differ')
    data={k:[] for k in ('structure_id','y','group_id','replicate')};prob=[]
    for sid in sorted(lr):
        allrows=[lr[sid]]+[d[sid] for d in members]
        require(len({(int(r['y_true']),int(r['group_id'])) for r in allrows})==1,'Member truth/group differs')
        wt=[]
        for r in allrows:
            records=json.loads(r['replicate_probabilities_json']);require({str(x['replicate']) for x in records}=={'1','2'} and len(records)==2,'WT identity differs')
            wt.append({str(x['replicate']):np.asarray(x['probabilities'],dtype=float) for x in records})
        for replicate in ('1','2'):
            fused=.5*wt[0][replicate]+.5*np.mean([d[replicate] for d in wt[1:]],axis=0)
            require(np.isfinite(fused).all() and np.isclose(fused.sum(),1,atol=1e-6),'Invalid fused probability')
            for k,v in [('structure_id',sid),('y',int(lr[sid]['y_true'])),('group_id',int(lr[sid]['group_id'])),('replicate',int(replicate))]:data[k].append(v)
            prob.append(fused)
    return {k:np.asarray(v) for k,v in data.items()},np.asarray(prob)


def parents(root,c):
    cr=root/c['roi_run_dir'];lr=root/c['lr_run_dir'];cm=json.loads((cr/'manifest.json').read_text());lm=json.loads((lr/'manifest.json').read_text())
    current,metadata,reference=shape.prepare_manifest(root,lm['config']);require(current==lm,'LR source/config/environment changed')
    current_roi,other=roi.prepare_manifest(root,{**cm['config'],'run_dir':'results/_fusion_metadata_reference'},cm['requested_device'])
    require(cm['source_signature']==current_roi['source_signature'] and cm['dataset_signature']==lm['dataset_signature']==current_roi['dataset_signature'],'Parent source/data differs')
    require(cm['experiment_signature']==roi.json_signature({k:v for k,v in cm.items() if k!='experiment_signature'}) and cm['fold_ids']==lm['fold_ids'],'Parent manifest/folds differ')
    require(cm['label_order']==lm['label_order']==list(CLASS_NAMES),'Parent class order differs')
    identity=_identity(metadata);folds=_fold_ids(cm,identity,cr);members={};refs=[]
    for cond in c['conditions']:
        for fold in c['folds']:
            lj=shape.planned_jobs(lm['config'],model='lr_shape',condition=cond,fold=fold)[0];ld=shape.directory(lr,lj);lv=shape.assert_resume(ld,lj,lm);require(lv['status']=='completed','Retained LR failure blocks fusion')
            pp=json.loads((ld/'protocol.json').read_text())['preprocessing'];p=_read_prediction_csv(ld/'outer_predictions.csv');shape.check_replicate_rows(p['rows']);_same_score(lv['outer_metrics'],p['metrics'],'LR member')
            require(set(p['structure_id'])==set(folds[fold]['test_ids']),'LR member outer IDs differ')
            for sid,y,g in zip(p['structure_id'],p['y'],p['group_id']):require((int(y),int(g))==(identity[str(sid)]['y'],identity[str(sid)]['group_id']),'LR truth/group differs from metadata')
            members[cond,fold,'lr']=p['rows'];refs.append({'family':'LR','job':lj,'directory':str(ld.relative_to(root)),'artifact_sha256':shape.sha256(ld/'ARTIFACTS.json'),'CSV_sha256':shape.sha256(ld/'outer_predictions.csv')})
            for seed in c['cnn_seeds']:
                cj=roi.planned_jobs(cm['config'],alignment='R1',condition=cond,fold=fold,seed=seed)[0];cd=roi.job_directory(cr,cj);cv=shape.assert_resume(cd,cj,cm);require(cv['status']=='completed','Retained CNN failure blocks fusion')
                cp=json.loads((cd/'protocol.json').read_text())['preprocessing'];require(cp==pp,'CNN/LR preprocessing differs')
                p=_read_prediction_csv(cd/'outer_predictions.csv');shape.check_replicate_rows(p['rows']);_same_score(cv['outer_metrics'],p['metrics'],'CNN member')
                require(set(p['structure_id'])==set(folds[fold]['test_ids']),'CNN member outer IDs differ')
                for sid,y,g in zip(p['structure_id'],p['y'],p['group_id']):require((int(y),int(g))==(identity[str(sid)]['y'],identity[str(sid)]['group_id']),'CNN truth/group differs from metadata')
                members[cond,fold,seed]=p['rows'];refs.append({'family':'CNN','job':cj,'directory':str(cd.relative_to(root)),'artifact_sha256':shape.sha256(cd/'ARTIFACTS.json'),'CSV_sha256':shape.sha256(cd/'outer_predictions.csv')})
    require(len(refs)==80,'Expected80 scientific parent indices')
    return members,refs,identity,cm,lm


def prepare(root,c):
    validate(c);members,refs,identity,cm,lm=parents(root,c)
    manifest={'protocol':PROTOCOL,'config':c,'source_signature':shape.sha256(Path(__file__).resolve()),'dataset_signature':cm['dataset_signature'],
        'helper_source_sha256':{'scripts/audit_task1_roi_review.py':shape.sha256(root/'scripts/audit_task1_roi_review.py')},
        'parent_manifests':{'CNN_sha256':shape.sha256(root/c['roi_run_dir']/'manifest.json'),'LR_sha256':shape.sha256(root/c['lr_run_dir']/'manifest.json')},
        'parent_sources':{'CNN':cm['source_signature'],'LR':lm['source_signature']},'members':refs,'member_inventory_signature':shape.json_signature(refs),
        'expected_parent_fits':80,'expected_derived_fold_jobs':80,'expected_main_OOF':12,'expected_pooled_OOF':4,
        'rule':'WT-level p=.5 LR+.5 CNN; pooled CNN3mean only variance diagnostic; no fitting/calibration/threshold search',
        'bias':'Known development errors suggested this family combination; all original outer data already inspected; not independent morphology validation'}
    manifest['experiment_signature']=shape.json_signature(manifest);return manifest,members,identity


def evaluate(root,c):
    manifest,members,identity=prepare(root,c);out=guard_output(root,root/c['run_dir'])
    for parent in (root/c['roi_run_dir'],root/c['lr_run_dir']):
        parent=parent.resolve();require(out!=parent and parent not in out.parents and out not in parent.parents,'Fusion output overlaps parent evidence')
    require(not any((p/'ARTIFACTS.json').exists() for p in (out,*out.parents)),'Output overlaps a sealed scientific job')
    out.mkdir(parents=True,exist_ok=True)
    with run_lock(out):
        mp=out/'manifest.json'
        if mp.exists():require(json.loads(mp.read_text())==manifest,'Existing fusion source/config/member state differs')
        else:write_json(mp,manifest)
        oof={};derived_jobs=[]
        for mode in ('main','pooled'):
            for condition in c['conditions']:
                for seed in (c['cnn_seeds'] if mode=='main' else ['pooled']):
                    rows=[]
                    for fold in c['folds']:
                        job={'mode':mode,'condition':condition,'seed':seed,'fold':fold};d=out/'runs'/mode/condition/f'seed_{seed}'/f'fold_{fold}'
                        chosen=[seed] if mode=='main' else c['cnn_seeds'];data,p=blend_rows(members[condition,fold,'lr'],[members[condition,fold,s] for s in chosen])
                        if d.exists():
                            verify_artifact_index(d,job,manifest);saved=json.loads((d/'metrics.json').read_text());require(saved['job']==job,'Saved derived job differs')
                        else:
                            d.mkdir(parents=True)
                            try:
                                score=save_prediction_csv(d/'outer_predictions.csv',data,p)['probabilities']
                                saved={'job':job,'metrics':metrics(np.array([identity[s]['y'] for s in sorted(set(data['structure_id']))]),score),
                                    'experiment_signature':manifest['experiment_signature'],'learned_parameters':0,'primary_gate_eligible':mode=='main'}
                                write_json(d/'metrics.json',saved)
                                write_json(d/'member_refs.json',{'rule':manifest['rule'],'members':[r for r in manifest['members'] if r['job']['condition']==condition and r['job']['fold']==fold and (r['family']=='LR' or r['job']['seed'] in chosen)]})
                                write_artifact_index(d,job,manifest)
                            except BaseException as error:
                                write_json(d/'failure.json',{'job':job,'error_type':type(error).__name__,'error':str(error),'preserve_partial':'No silent retry or overwrite'});raise
                        cp=_read_prediction_csv(d/'outer_predictions.csv');roi.check_replicate_rows(cp['rows'])
                        expected=grouped_predictions(data,p);require(cp['structure_id'].tolist()==expected['structure_id'].tolist() and np.array_equal(cp['probabilities'],expected['probabilities']),'Derived probabilities differ from fixed member arithmetic')
                        _same_score(saved['metrics'],cp['metrics'],'fixed fusion');rows.extend({**r,'fold':fold} for r in cp['rows']);derived_jobs.append(job)
                    rows.sort(key=lambda r:r['structure_id']);require(len(rows)==344 and {r['structure_id'] for r in rows}==set(identity),'Fusion OOF incomplete')
                    labels=np.array([int(r['y_true']) for r in rows]);p=np.array([[float(r[f'p_{cls}']) for cls in CLASS_NAMES] for r in rows]);score=metrics(labels,p)
                    dest=out/'oof'/mode/condition/f'seed_{seed}';write_json(dest/'metrics.json',{**score,'mode':mode,'seed':seed,'primary_gate_eligible':mode=='main','experiment_signature':manifest['experiment_signature']});_csv(dest/'predictions.csv',rows)
                    oof[mode,condition,seed]={'metrics':score,'rows':rows,'by_id':{r['structure_id']:r for r in rows}}
        summary={'status':'complete','derived_fold_jobs':len(derived_jobs),'main_OOF':12,'pooled_OOF':4,'experiment_signature':manifest['experiment_signature'],'source_signature':manifest['source_signature'],'rule':manifest['rule'],'bias':manifest['bias'],'comparisons':[],'conditions':{}}
        case_rows=[]
        for cond in c['conditions']:
            scores=[oof['main',cond,s]['metrics'] for s in c['cnn_seeds']];row={'condition':cond,'mode':'main','n_seeds':3}
            for key in ('macro_f1','accuracy','balanced_accuracy'):row[key+'_mean']=float(np.mean([z[key] for z in scores]));row[key+'_seed_SD']=float(np.std([z[key] for z in scores],ddof=1))
            for cls in CLASS_NAMES:
                for key in ('precision','recall','f1'):row[f'{cls}_{key}_mean']=float(np.mean([z['per_class'][cls][key] for z in scores]))
            row['development_gate_pass']=all(z['macro_f1']>=.9 and all(z['per_class'][cls][key]>=.85 for cls in CLASS_NAMES for key in ('precision','recall')) for z in scores)
            summary['comparisons'].append(row);detail={'seeds':{str(s):oof['main',cond,s]['metrics'] for s in c['cnn_seeds']},'pooled_diagnostic':oof['pooled',cond,'pooled']['metrics'],'stable_wrong_ids':[],'stable_CHID_FN':[],'stable_CHID_FP':[]}
            for sid in sorted(identity):
                pred=[int(oof['main',cond,s]['by_id'][sid]['y_pred']) for s in c['cnn_seeds']];truth=identity[sid]['y'];wrong=all(p!=truth for p in pred)
                if wrong:detail['stable_wrong_ids'].append(sid)
                if truth==2 and all(p!=2 for p in pred):detail['stable_CHID_FN'].append(sid)
                if truth!=2 and all(p==2 for p in pred):detail['stable_CHID_FP'].append(sid)
                case_rows.append({'condition':cond,'structure_id':sid,'truth':CLASS_NAMES[truth],'group_id':identity[sid]['group_id'],'stable_wrong':wrong,**{f'seed_{s}_predicted':CLASS_NAMES[p] for s,p in zip(c['cnn_seeds'],pred)}})
            summary['conditions'][cond]=detail
        bootstrap=group_ci(c,members,oof,identity);write_json(out/'summary.json',summary);_csv(out/'comparison.csv',summary['comparisons']);_csv(out/'case_predictions.csv',case_rows);write_json(out/'group_bootstrap.json',bootstrap)
        return summary


def group_ci(c,members,oof,identity):
    groups=sorted({z['group_id'] for z in identity.values()});gi={g:i for i,g in enumerate(groups)};cm=np.zeros((3,4,3,len(groups),3,3),int)
    for ci,cond in enumerate(c['conditions']):
        for si,seed in enumerate(c['cnn_seeds']):
            for kind,rows in enumerate([oof['main',cond,seed]['rows'],[r for f in c['folds'] for r in members[cond,f,seed]],[r for f in c['folds'] for r in members[cond,f,'lr']]]):
                for r in rows:cm[kind,ci,si,gi[int(r['group_id'])],int(r['y_true']),int(r['y_pred'])]+=1
    weights=np.random.default_rng(c['bootstrap_seed']).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=c['bootstrap_repeats'])
    sampled,support,predicted=scores_from_confusion(np.einsum('bg,tcsgij->btcsij',weights,cm));point,_,_=scores_from_confusion(cm.sum(3))
    pairs={}
    for name,first,second in [('fusionI3-P0',(0,0),(0,3)),('fusionI3-CNNI3',(0,0),(1,0)),('fusionI3-LRI3',(0,0),(2,0))]:
        pairs[name]={m:{'point':float((point[m][first]-point[m][second]).mean()),'ci95':np.quantile(sampled[m][:,first[0],first[1]].mean(1)-sampled[m][:,second[0],second[1]].mean(1),[.025,.975]).tolist()} for m in point}
    return {'B':2000,'seed':c['bootstrap_seed'],'unit':'49 whole spatial groups, all models and seeds jointly resampled','scope':'Conditional saved-prediction percentile95; no refitting or multiplicity correction; known development data','pooled_not_used_for_primary_CI':True,'pairs':pairs}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/task1_fixed_fusion.json');p.add_argument('command',choices=('check','evaluate'));args=p.parse_args();c=validate(json.loads(args.config.read_text()))
    if args.command=='check':
        m,_,_=prepare(ROOT,c);print(json.dumps({'status':'checked','source_signature':m['source_signature'],'experiment_signature':m['experiment_signature'],'verified_parent_fit_indices':80,'training_or_weight_inference':False},indent=2))
    else:
        s=evaluate(ROOT,c);print(json.dumps({'status':s['status'],'main_OOF':s['main_OOF'],'pooled_OOF':s['pooled_OOF'],'comparisons':s['comparisons']},indent=2))


if __name__=='__main__':main()
