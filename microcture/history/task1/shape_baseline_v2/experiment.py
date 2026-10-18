"""Fixed forty CPU fits; immutable scientific files and no outer model selection."""
from __future__ import annotations
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import warnings
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from microcture.task1.data import load_dataset
from microcture.task1.training import run_lock,score_samples,write_json
from microcture.task1.evaluation import CLASS_NAMES,metrics,save_prediction_csv,_read_prediction_csv,grouped_predictions
from microcture.task1_formal.protocol import _package_signature,project_path,json_signature,sha256
from microcture.task1_formal.runner import guard_output as old_guard,assert_resume,write_artifact_index,frozen_fold
from microcture.task1_formal.evaluation import _identity,_fold_ids,_same_score,_csv,_guard_run
from microcture.task1_reevaluation.runner import publish_fold
from microcture.task1_roi.data import prepare_roi_inner,prepare_roi_outer
from microcture.task1_roi.experiment import prepare_manifest as roi_manifest,apply_condition,check_replicate_rows
from .features import features,FEATURE_PROTOCOL

PROTOCOL='task1_fixed_shape455_lr_rbf_svc_stable_v2'
ROI_SOURCE='a1ed97051ee9d24ba0198bed07fc2884ed56d3ca629f3ed40744f8d2eede6c94'
MODELS=('lr_shape','svc_shape');CONDITIONS=('I3','G0','V0','P0');PREFIX='microcture-shape-v2-fit-'
FIXED_MODELS={'lr_shape':{'C':1.,'solver':'lbfgs','tol':1e-4,'max_iter':2000},
    'svc_shape':{'C':1.,'kernel':'rbf','gamma':'scale','break_ties':True,'decision_function_shape':'ovr','probability':False}}


def require(ok,message):
    if not ok:raise ValueError(message)


def validate(config):
    require(config['protocol']==PROTOCOL and config['models']==FIXED_MODELS and config['conditions']==list(CONDITIONS),'Frozen shape model/control configuration differs')
    require(config['seed']==42 and type(config['seed']) is int and config['folds']==list(range(5)) and config['features']==FEATURE_PROTOCOL,'Freeze seed/folds/features')
    require(config['cpu_threads']==2 and config['checkpoint_selection']=='fixed_fit_no_selection' and config['svc_score_transform']=='softmax_temperature1_then_WTmean_uncalibrated','Freeze resources/selection/scores')
    return config


def planned_jobs(config,model='all',condition=None,fold=None):
    validate(config);require(model=='all' or model in MODELS,'Unknown model');require(condition is None or condition in CONDITIONS,'Unknown condition')
    require(fold is None or type(fold) is int and fold in range(5),'Unknown fold')
    return [{'kind':'baseline','stage':'shape','model':m,'condition':c,'alignment':'R1','seed':42,'fold':f}
        for c in CONDITIONS for m in MODELS for f in range(5) if (model=='all' or model==m) and (condition is None or c==condition) and (fold is None or f==fold)]


def directory(run,job):
    require(job['model'] in MODELS and job['condition'] in CONDITIONS and type(job['fold']) is int and job['fold'] in range(5) and job['seed']==42,'Unsafe shape job')
    return run/'runs'/job['condition']/job['model']/'seed_42'/f"fold_{job['fold']}"


def guard(root,run):
    path=old_guard(root,run)
    for name in ('task1_formal_v1','task1_readout_v1','task1_reevaluation_v1','task1_roi_v1','task1_morphology_probe_v1'):
        p=(root/'results'/name).resolve();require(path!=p and p not in path.parents and path not in p.parents,'Protected scientific evidence')
    if (path/'manifest.json').exists():require(json.loads((path/'manifest.json').read_text())['protocol']==PROTOCOL,'Cannot reuse other protocol')
    return path


def prepare_manifest(root,config):
    root=root.resolve();validate(config);guard(root,project_path(root,config['run_dir']))
    rp=project_path(root,config['roi_run_dir']);reference=json.loads((rp/'manifest.json').read_text())
    require(reference['experiment_signature']==json_signature({k:v for k,v in reference.items() if k!='experiment_signature'}),'ROI manifest changed')
    require(reference['source_signature']==ROI_SOURCE==_package_signature(root,'task1_roi','run_task1_roi.py'),'Frozen ROI source changed')
    # Metadata-only reference view; this path is never created or used for fits.
    dep,metadata=roi_manifest(root,{**reference['config'],'run_dir':'results/_shape_metadata_reference'},reference['requested_device'])
    require(dep['dataset_signature']==reference['dataset_signature'] and dep['fold_ids']==reference['fold_ids'],'ROI metadata/folds changed')
    result={'protocol':PROTOCOL,'config':config,'source_signature':_package_signature(root,'task1_shape_baseline_v2','run_task1_shape_baseline_v2.py'),
        'roi_reference':{'manifest_sha256':sha256(rp/'manifest.json'),'source_signature':ROI_SOURCE,'experiment_signature':reference['experiment_signature']},
        **{k:dep[k] for k in ('dataset_signature','fold_ids','fold_signatures','label_order','environment_record','dependency_source_signature')},
        'compute_device':'cpu','expected_fits':40,'expected_OOF':8,'feature_count':455,'selection':'All parameters fixed before outer; no refit/validation selection',
        'score_interpretation':'SVC softmax(decision_function) gives uncalibrated ranking scores, averaged across WT before scoring',
        'evaluation_scope':'Previously inspected candidate structures; developmental model-family probe, no independent biological certification'}
    result['experiment_signature']=json_signature(result);return result,metadata,reference


def build_classifier(name,weights,seed=42):
    params=FIXED_MODELS[name]
    return (LogisticRegression(**params,class_weight=weights,random_state=seed) if name=='lr_shape' else SVC(**params,class_weight=weights,random_state=seed))


def predict_scores(payload,values):
    x=payload['scaler'].transform(values);model=payload['model']
    if payload['model_name']=='lr_shape':p=model.predict_proba(x)
    else:
        z=model.decision_function(x);z=z-z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True)
    require(model.classes_.tolist()==[0,1,2] and np.isfinite(p).all(),'Class order/score invalid')
    return p


def fit_one(job,train,validation,preprocessing,fold,dataset,config,manifest,output):
    started=time.perf_counter();result={};stage='inner_fit';outer_used=False
    write_json(output/'protocol.json',{'job':job,'preprocessing':preprocessing,'config':config,'selection':manifest['selection']})
    try:
        # The fitted model and scaler see inner training only. Validation is transformed after fixation.
        tx,names=features(apply_condition(train,job['condition']));scaler=StandardScaler().fit(tx)
        counts=np.bincount(train['y'],minlength=3);require(np.all(counts>0),'Missing inner class')
        weights={i:len(train['y'])/(3*int(counts[i])) for i in range(3)}
        model=build_classifier(job['model'],weights)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning);model.fit(scaler.transform(tx),train['y'])
        payload={'model':model,'scaler':scaler,'model_name':job['model'],'feature_names':names,'feature_protocol':FEATURE_PROTOCOL,'job':job}
        checkpoint=output/'model.joblib';joblib.dump(payload,checkpoint);fixed_hash=sha256(checkpoint)
        write_json(output/'feature_columns.json',{'names':names,'n_features':len(names),'reversal':'mean(f(x),f(flip_both_axes(x)))'})
        write_json(output/'endpoint.json',{'job':job,'checkpoint':'model.joblib','checkpoint_sha256':fixed_hash,'training_and_selection_complete':True,'outer_images_constructed':False})
        vp,nv=features(apply_condition(validation,job['condition']));require(names==nv,'Validation columns changed')
        vprob=predict_scores(payload,vp);save_prediction_csv(output/'validation_predictions.csv',validation,vprob)
        stage='outer_evaluation';outer=prepare_roi_outer(dataset,fold,preprocessing,'R1');outer_used=True
        ox,no=features(apply_condition(outer,job['condition']));require(names==no,'Outer columns changed')
        prob=predict_scores(payload,ox);require(sha256(checkpoint)==fixed_hash,'Endpoint changed during outer')
        save_prediction_csv(output/'outer_predictions.csv',outer,prob);score=score_samples(outer,prob);structure_prob=grouped_predictions(outer,prob)['probabilities']
        result={'status':'completed','checkpoint_sha256':fixed_hash,'outer_metrics':score,'validation_metrics':score_samples(validation,vprob),
            'class_weights':weights,'feature_count':len(names),'outer_fit_or_selection':False,'outer_constructed_after_fixed_endpoint':True,
            'prediction_counts':np.asarray(score['confusion_matrix']).sum(0).tolist(),
            'constant_probability':float(np.ptp(structure_prob,axis=0).max())<=1e-6,'single_class_argmax':len(np.unique(structure_prob.argmax(1)))==1,
            'iterations':model.n_iter_.tolist(),'svc_gamma':float(model._gamma) if job['model']=='svc_shape' else None}
    except (Exception,KeyboardInterrupt) as error:
        result={'status':'execution_failure' if isinstance(error,(ValueError,OSError,KeyboardInterrupt)) else 'numerical_failure',
            'failure_stage':stage,'error_type':type(error).__name__,'error':str(error) or type(error).__name__}
        write_json(output/'failure.json',{**result,'job':job})
    result.update(job=job,**{k:manifest[k] for k in ('experiment_signature','source_signature','dataset_signature')},
        preprocessing_signature=preprocessing['signature'],runtime_seconds=time.perf_counter()-started,outer_images_used=outer_used,
        **{f'{s}_structures':len(fold[f'{s}_ids']) for s in ('train','validation','test')})
    write_json(output/'metrics.json',result);write_artifact_index(output,job,manifest);return result


def run(config,root,run_dir,**filters):
    root=root.resolve();run_dir=guard(root,run_dir);config={**config,'run_dir':str(run_dir.relative_to(root))}
    manifest,metadata,reference=prepare_manifest(root,config);jobs=planned_jobs(config,**filters);require(bool(jobs),'No matching jobs')
    run_dir.mkdir(parents=True,exist_ok=True)
    with run_lock(run_dir),threadpool_limits(limits=config['cpu_threads']):
        mp=run_dir/'manifest.json'
        if mp.exists():require(json.loads(mp.read_text())==manifest,'Source/config/environment changed; use new run')
        else:write_json(mp,manifest)
        for p in Path('/tmp').glob(PREFIX+'*'):
            marker=p/'STAGING_CONTEXT.json'
            if not p.is_symlink() and marker.is_file():
                context=json.loads(marker.read_text());require(context.get('run_dir')!=str(run_dir),'Unpublished scientific staging blocks automatic retry')
        pending=[]
        for job in jobs:
            require(not (run_dir/'failed_staging'/directory(run_dir,job).relative_to(run_dir/'runs')).exists(),'Failed staging blocks automatic retry')
            saved=assert_resume(directory(run_dir,job),job,manifest)
            if saved is None:pending.append(job)
            else:print('Skip saved',saved['status'],job,flush=True)
        if pending:
            dataset=load_dataset(project_path(root,reference['config']['cache_dir']));require(dataset['signature']==manifest['dataset_signature'],'Cache changed')
            by_fold=defaultdict(list)
            for job in pending:by_fold[job['fold']].append(job)
            for number,items in sorted(by_fold.items()):
                fold=frozen_fold(root,reference['config'],manifest,number);train,validation,pp=prepare_roi_inner(dataset,fold,'R1')
                prior=json.loads((project_path(root,config['roi_run_dir'])/'folds/R1'/f'fold_{number}.json').read_text())
                require(prior['frozen_fold']==fold and prior['preprocessing']==pp,'R1 input preprocessing differs from signed reference')
                for job in items:
                    dest=directory(run_dir,job);dest.parent.mkdir(parents=True,exist_ok=True)
                    output=Path(tempfile.mkdtemp(prefix=PREFIX,dir='/tmp'));published=False
                    try:
                        write_json(output/'STAGING_CONTEXT.json',{'owner':PROTOCOL,'run_dir':str(run_dir),'job':job})
                        print('Start fixed CPU shape',job,flush=True)
                        result=fit_one(job,train,validation,pp,fold,dataset,config,manifest,output)
                        publish_fold(output,dest);published=True
                        require(result['status']=='completed',f'Failed fit retained at {dest}; no retry')
                    except BaseException as error:
                        if not published:
                            retained=run_dir/'failed_staging'/dest.relative_to(run_dir/'runs')
                            try:retained.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(output),str(retained))
                            except (OSError,KeyboardInterrupt):retained=output
                        else:retained=dest
                        raise RuntimeError(f'Stopped; scientific evidence retained at {retained}') from error
                    finally:
                        if published and output.exists():shutil.rmtree(output)
        return summarize(run_dir,root)


def summarize(run_dir,root):
    run_dir=guard(root,run_dir);_guard_run(run_dir);require(not (run_dir/'FROZEN.json').exists(),'Frozen results are read-only')
    manifest=json.loads((run_dir/'manifest.json').read_text());current,metadata,_=prepare_manifest(root,manifest['config'])
    require(current==manifest,'Source/config/reference changed');identity=_identity(metadata);folds=_fold_ids(manifest,identity,run_dir)
    grouped=defaultdict(list);records=[];errors=[];comparisons=[];allcases=[]
    for job in planned_jobs(manifest['config']):
        try:
            require(not (run_dir/'failed_staging'/directory(run_dir,job).relative_to(run_dir/'runs')).exists(),'Unpublished failed staging retained')
            value=assert_resume(directory(run_dir,job),job,manifest);r={'job':job,'status':'pending'}
            if value is not None:
                r.update(status=value['status'],metrics=value)
                if value['status']=='completed':
                    endpoint=json.loads((directory(run_dir,job)/'endpoint.json').read_text())
                    require(endpoint['job']==job and endpoint['checkpoint']=='model.joblib' and endpoint['checkpoint_sha256']==value['checkpoint_sha256'] and endpoint['training_and_selection_complete'] and endpoint['outer_images_constructed'] is False and value['outer_fit_or_selection'] is False and value['outer_constructed_after_fixed_endpoint'],'Endpoint/outer boundary differs')
                    p=_read_prediction_csv(directory(run_dir,job)/'outer_predictions.csv');check_replicate_rows(p['rows'])
                    require(set(p['structure_id'])==set(folds[job['fold']]['test_ids']),'Outer IDs differ')
                    for sid,y,g in zip(p['structure_id'],p['y'],p['group_id']):require((int(y),int(g))==(identity[str(sid)]['y'],identity[str(sid)]['group_id']),'Outer identity differs')
                    _same_score(value['outer_metrics'],p['metrics'],'outer_metrics');r['predictions']=p
                    validation=_read_prediction_csv(directory(run_dir,job)/'validation_predictions.csv');check_replicate_rows(validation['rows'])
                    require(set(validation['structure_id'])==set(folds[job['fold']]['validation_ids']),'Validation IDs differ')
                    _same_score(value['validation_metrics'],validation['metrics'],'validation_metrics')
        except (OSError,ValueError,KeyError,TypeError) as error:r={'job':job,'status':'invalid','error':str(error)};errors.append(r)
        records.append(r);grouped[job['condition'],job['model']].append(r)
    complete=0;details={}
    for (condition,model),items in grouped.items():
        dest=run_dir/'oof'/condition/model/'seed_42';ok=all(r['status']=='completed' for r in items)
        write_json(dest/'status.json',{'status':'complete' if ok else 'incomplete','oof_current':ok,'fit_statuses':dict(Counter(r['status'] for r in items))})
        row={'condition':condition,'model':model,'status':'complete' if ok else 'incomplete','n_structures':None}
        if ok:
            rows=sorted([{**x,'fold':r['job']['fold']} for r in items for x in r['predictions']['rows']],key=lambda x:x['structure_id'])
            require(len(rows)==len(identity) and {x['structure_id'] for x in rows}==set(identity),'OOF duplicated/incomplete')
            labels=np.array([int(x['y_true']) for x in rows]);p=np.array([[float(x[f'p_{c}']) for c in CLASS_NAMES] for x in rows]);score=metrics(labels,p);complete+=1
            row.update({k:score[k] for k in ('n_structures','macro_f1','accuracy','balanced_accuracy')})
            for c in CLASS_NAMES:
                for m in ('precision','recall','f1'):row[f'{c}_{m}']=score['per_class'][c][m]
            row['development_gate_pass']=score['macro_f1']>=.90 and all(score['per_class'][c][m]>=.85 for c in CLASS_NAMES for m in ('precision','recall'))
            row['constant_fit_count']=sum(r['metrics']['constant_probability'] or r['metrics']['single_class_argmax'] for r in items)
            cm=score['confusion_matrix'];row.update(CHID_TP=cm[2][2],CHID_FP=cm[0][2]+cm[1][2],CHID_FN=cm[2][0]+cm[2][1],CHIN_to_CHID=cm[1][2],CHID_to_CHIN=cm[2][1])
            write_json(dest/'metrics.json',{**score,'experiment_signature':manifest['experiment_signature'],'score_interpretation':manifest['score_interpretation']});_csv(dest/'predictions.csv',rows)
            errors_rows=[{**x,'condition':condition,'model':model} for x in rows if int(x['y_true'])!=int(x['y_pred'])]
            allcases.extend({**x,'condition':condition,'model':model} for x in rows)
            details[f'{condition}/{model}']={'metrics':score,'errors':errors_rows,'n_errors':len(errors_rows),'CHID_FN':[x['structure_id'] for x in errors_rows if int(x['y_true'])==2],
                'CHID_FP':[x['structure_id'] for x in errors_rows if int(x['y_pred'])==2],'WT_disagreements':sum(int(x['replicate_disagree']) for x in rows)}
        comparisons.append(row)
    counts=Counter(r['status'] for r in records);result={'status':'invalid' if errors else 'complete' if counts['completed']==40 else 'failed' if counts['execution_failure']+counts['numerical_failure'] else 'pending',
        'expected_fits':40,'completed_fits':counts['completed'],'pending_fits':counts['pending'],'execution_failure_fits':counts['execution_failure'],'numerical_failure_fits':counts['numerical_failure'],'invalid_fits':counts['invalid'],
        'complete_OOF':complete,'expected_OOF':8,'comparison':comparisons,'models':details,'errors':errors,'source_signature':manifest['source_signature'],
        'experiment_signature':manifest['experiment_signature'],'dataset_signature':manifest['dataset_signature'],'score_interpretation':manifest['score_interpretation'],'evaluation_scope':manifest['evaluation_scope']}
    write_json(run_dir/'summary.json',result);_csv(run_dir/'comparison.csv',comparisons);_csv(run_dir/'case_predictions.csv',allcases)
    _csv(run_dir/'fit_review.csv',[{**r['job'],'status':r['status'],**{k:r.get('metrics',{}).get(k) for k in ('runtime_seconds','constant_probability','single_class_argmax','svc_gamma','iterations')}} for r in records])
    return result
