#!/usr/bin/env python3
"""Read-only audit, paired group uncertainty and figures for the fixed455 CPU probe."""
from __future__ import annotations
import argparse,json,sys,tempfile,os
from pathlib import Path
import joblib
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from microcture.task1_shape_baseline_v2 import experiment as e
from microcture.task1_formal.evaluation import _identity,_fold_ids,_same_score,_csv
from microcture.task1.evaluation import CLASS_NAMES,metrics,_read_prediction_csv
from scripts.audit_task1_roi_review import scores_from_confusion


def require(ok,msg):
    if not ok:raise ValueError(msg)


def audit(run,report):
    manifest=json.loads((run/'manifest.json').read_text());current,metadata,reference=e.prepare_manifest(ROOT,manifest['config'])
    require(current==manifest,'Current science/source/environment differs');identity=_identity(metadata);folds=_fold_ids(manifest,identity,run)
    source_summary=json.loads((run/'summary.json').read_text());artifact_rows=[];model_checks=[];oof={};rows_by={}
    for job in e.planned_jobs(manifest['config']):
        d=e.directory(run,job);v=e.assert_resume(d,job,manifest);require(v is not None and v['status']=='completed','Need all40 retained scientific fits')
        index=json.loads((d/'ARTIFACTS.json').read_text());artifact_rows.extend({'path':str((d/f['name']).relative_to(run)),**f} for f in index['files'])
        protocol=json.loads((d/'protocol.json').read_text());saved_pp=json.loads((ROOT/manifest['config']['roi_run_dir']/f"folds/R1/fold_{job['fold']}.json").read_text())['preprocessing']
        require(protocol['preprocessing']==saved_pp,'R1 inner preprocessing differs')
        ep=json.loads((d/'endpoint.json').read_text());require(ep['checkpoint_sha256']==v['checkpoint_sha256'] and ep['outer_images_constructed'] is False and ep['training_and_selection_complete'],'Endpoint boundary differs')
        payload=joblib.load(d/'model.joblib');model=payload['model'];scaler=payload['scaler'];names=payload['feature_names']
        require(payload['job']==job and len(names)==len(set(names))==455 and scaler.n_features_in_==455,'Saved model/features differ')
        require(int(scaler.n_samples_seen_)==len(folds[job['fold']]['train_ids'])*2,'Scaler fitted-sample count differs from inner WT cohort')
        params=model.get_params()
        for k,val in e.FIXED_MODELS[job['model']].items():require(params[k]==val,'Unregistered model hyperparameter')
        expected={i:len(folds[job['fold']]['train_ids'])/(3*sum(identity[s]['y']==i for s in folds[job['fold']]['train_ids'])) for i in range(3)}
        require(params['class_weight']==expected,'Class weighting differs')
        resid=np.array([i for i,n in enumerate(names) if '_residual_' in n]);zero=None
        if job['condition']=='P0':
            zero=bool(np.all(scaler.mean_[resid]==0) and np.all(scaler.var_[resid]==0));require(zero,'P0 residual roundoff was amplified')
            require(np.all((model.coef_ if job['model']=='lr_shape' else model.support_vectors_)[:,resid]==0),'P0 residual model values nonzero')
        model_checks.append({'job':job,'scaler_fitted_WT_rows':int(scaler.n_samples_seen_),'P0_residual128_exactly_zero':zero,'iterations':model.n_iter_.tolist()})
        for filename,split,metricname in [('outer_predictions.csv','test_ids','outer_metrics'),('validation_predictions.csv','validation_ids','validation_metrics')]:
            p=_read_prediction_csv(d/filename);e.check_replicate_rows(p['rows']);require(set(p['structure_id'])==set(folds[job['fold']][split]),'Prediction IDs differ')
            for sid,y,g in zip(p['structure_id'],p['y'],p['group_id']):require((int(y),int(g))==(identity[str(sid)]['y'],identity[str(sid)]['group_id']),'Truth/group differs')
            _same_score(v[metricname],p['metrics'],metricname)
            if filename.startswith('outer'):rows_by.setdefault((job['condition'],job['model']),[]).extend({**r,'fold':job['fold']} for r in p['rows'])
    for (condition,model),rows in rows_by.items():
        rows.sort(key=lambda r:r['structure_id']);require(len(rows)==344 and {r['structure_id'] for r in rows}==set(identity),'OOF coverage differs')
        saved=_read_prediction_csv(run/f'oof/{condition}/{model}/seed_42/predictions.csv');require(rows==[{**r,'fold':int(r['fold'])} for r in saved['rows']],'OOF merge differs')
        _same_score(source_summary['models'][f'{condition}/{model}']['metrics'],saved['metrics'],'complete_OOF')
        oof[f'{model}/{condition}']={'rows':rows,'metrics':saved['metrics'],'by_id':{r['structure_id']:r for r in rows}}
    for condition in e.CONDITIONS:
        p=_read_prediction_csv(ROOT/f'results/task1_roi_v1/oof/R1/{condition}/seed_42/predictions.csv')
        oof[f'cnn42/{condition}']={'rows':p['rows'],'metrics':p['metrics'],'by_id':{r['structure_id']:r for r in p['rows']}}
    groups=sorted({r['group_id'] for r in identity.values()});gidx={g:i for i,g in enumerate(groups)};names=list(oof);cm=np.zeros((len(names),len(groups),3,3),int)
    for i,name in enumerate(names):
        for r in oof[name]['rows']:cm[i,gidx[int(r['group_id'])],int(r['y_true']),int(r['y_pred'])]+=1
    weights=np.random.default_rng(20261014).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
    sampled,support,predicted=scores_from_confusion(np.einsum('bg,agij->baij',weights,cm));point,_,_=scores_from_confusion(cm.sum(1))
    intervals={name:{m:{'point':float(point[m][i]),'ci95':np.quantile(sampled[m][:,i],[.025,.975]).tolist()} for m in point} for i,name in enumerate(names)}
    pairs={}
    for a,b in [(f'{m}/I3',f'{m}/P0') for m in e.MODELS]+[('lr_shape/I3','svc_shape/I3'),('lr_shape/I3','cnn42/I3'),('svc_shape/I3','cnn42/I3')]:
        ai,bi=names.index(a),names.index(b);pairs[f'{a}-{b}']={m:{'point':float(point[m][ai]-point[m][bi]),'ci95':np.quantile(sampled[m][:,ai]-sampled[m][:,bi],[.025,.975]).tolist()} for m in point}
    integrity={'status':'passed','scientific_fit_indices':40,'indexed_files':len(artifact_rows),'indexed_bytes':sum(f['size_bytes'] for f in artifact_rows),
        'inventory_signature':e.json_signature(artifact_rows),'source_signature':manifest['source_signature'],'experiment_signature':manifest['experiment_signature'],
        'verified_complete_OOF':8,'model_checks':model_checks,'zero_P0_residual_checks':sum(x['P0_residual128_exactly_zero'] is True for x in model_checks),
        'script_sha256':e.sha256(Path(__file__).resolve()),'new_fits_or_weight_inference_in_audit':False}
    bootstrap={'B':2000,'rng_seed':20261014,'unit':'All structures from each of49 spatial groups sampled together across models/conditions',
        'scope':'Conditional current OOF percentile95, no refitting/model selection or multiplicity correction; CNN comparator is predeclared seed42',
        'zero_CHID_support_repeats':int(np.sum(np.any(support[...,2]==0,axis=1))),'zero_CHID_prediction_repeats':int(np.sum(np.any(predicted[...,2]==0,axis=1))),
        'arms':intervals,'pairs':pairs}
    report.mkdir(parents=True,exist_ok=True);(report/'integrity.json').write_text(json.dumps(integrity,indent=2)+'\n');(report/'group_bootstrap.json').write_text(json.dumps(bootstrap,indent=2)+'\n')
    return oof,bootstrap,source_summary,identity,integrity


def figures(report,oof,bootstrap,identity):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    import matplotlib as mpl
    mpl.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    colors={'lr_shape':'#2878B5','svc_shape':'#CC6633','cnn42':'#777777'};labels={'lr_shape':'Fixed descriptors + LR','svc_shape':'Fixed descriptors + RBF SVC','cnn42':'M2+ seed42 (same R1 input)'}
    fig=Figure(figsize=(12,8),layout='constrained');FigureCanvasAgg(fig);axes=fig.subplots(2,2)
    for ax,metric,title in zip(axes.flat,['macro_f1','CHID_precision','CHID_recall','CHID_f1'],['Macro F1','CHID precision','CHID recall','CHID F1']):
        for j,model in enumerate(['lr_shape','svc_shape','cnn42']):
            x=np.arange(4)+(j-1)*.12;means=np.array([bootstrap['arms'][f'{model}/{c}'][metric]['point'] for c in e.CONDITIONS]);cis=np.array([bootstrap['arms'][f'{model}/{c}'][metric]['ci95'] for c in e.CONDITIONS])
            ax.errorbar(x,means,yerr=np.stack((means-cis[:,0],cis[:,1]-means)),marker=['o','s','^'][j],color=colors[model],capsize=3,label=labels[model])
        if metric!='CHID_f1':ax.axhline(.9 if metric=='macro_f1' else .85,ls='--',color='#BB4444',lw=1)
        ax.set(xticks=np.arange(4),xticklabels=['I3 full','G0 geometry','V0 validity','P0 profile'],ylim=(0,1.02),title=title,ylabel='Unique-structure OOF score');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fixed shape descriptors: developmental OOF performance and controls',fontsize=15)
    handles,legend=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend,loc='outside lower center',ncol=3,fontsize=9,
        title='Conditional95% group-bootstrap intervals; one fixed traditional seed, no refitting. SVC scores uncalibrated.',title_fontsize=9)
    for ext in ('png','pdf'):fig.savefig(report/f'performance.{ext}',dpi=220,bbox_inches='tight')
    fig=Figure(figsize=(12,5.6),layout='constrained');FigureCanvasAgg(fig);axes=fig.subplots(1,2)
    pairnames=list(bootstrap['pairs']);plabels=['LR: full minus profile','SVC: full minus profile','Full: LR minus SVC','Full: LR minus M2+42','Full: SVC minus M2+42']
    for ax,metric,title in zip(axes,['macro_f1','CHID_f1'],['Macro F1 difference','CHID F1 difference']):
        for i,name in enumerate(pairnames):
            z=bootstrap['pairs'][name][metric];v=z['point']*100;lo,hi=np.array(z['ci95'])*100
            ax.errorbar(v,i,xerr=np.array([[v-lo],[hi-v]]),fmt='o',color='#2878B5',capsize=4)
        ax.axvline(0,color='#BB4444',ls='--');ax.set(yticks=range(len(pairnames)),yticklabels=plabels,xlabel='Paired effect (percentage points)',title=title);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.suptitle('Conditional paired uncertainty across49 spatial groups (2000 bootstrap draws)',fontsize=13)
    for ext in ('png','pdf'):fig.savefig(report/f'paired_effects.{ext}',dpi=220,bbox_inches='tight')
    selected=[sid for sid in identity if identity[sid]['y']==2]
    selected.sort(key=lambda sid:int(sid.split('_')[-1]));fps={r['structure_id'] for model in ['lr_shape','svc_shape'] for r in oof[f'{model}/I3']['rows'] if int(r['y_true'])!=2 and int(r['y_pred'])==2}
    selected+=sorted(fps|{'CHIN_51','CHIN_59','CHIN_73','OPCID_66'})
    columns=['cnn42/I3','lr_shape/I3','svc_shape/I3'];values=np.array([[int(oof[c]['by_id'][sid]['y_pred']) for c in columns] for sid in selected])
    fig=Figure(figsize=(10,15),layout='constrained');FigureCanvasAgg(fig);ax=fig.subplots();ax.imshow(values,cmap=ListedColormap(['#BCD2E3','#F4D4B4','#CEE4D8']),vmin=0,vmax=2,aspect='auto')
    for i,sid in enumerate(selected):
        for j,col in enumerate(columns):
            r=oof[col]['by_id'][sid];ax.text(j,i,CLASS_NAMES[int(r['y_pred'])]+f"\nCHID score={float(r['p_CHID']):.3f}",ha='center',va='center',fontsize=8,fontweight='bold' if int(r['y_pred'])!=identity[sid]['y'] else 'normal')
    ax.set(xticks=range(3),xticklabels=['M2+ seed42','Fixed455 + LR','Fixed455 + RBF SVC'],yticks=range(len(selected)),yticklabels=[f"{sid} ({identity[sid]['length_bp']/1000:.2f} kb)" for sid in selected]);ax.xaxis.tick_top();ax.tick_params(length=0)
    fig.suptitle('All26 CHID annotations and systematic false-positive candidates\nBold=incorrect; SVC scores are uncalibrated rankings',fontsize=13)
    for ext in ('png','pdf'):fig.savefig(report/f'case_predictions.{ext}',dpi=220,bbox_inches='tight')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,default=ROOT/'results/task1_shape_baseline_v2');p.add_argument('--report-dir',type=Path,default=ROOT/'reports/task1_shape_baseline_review_v2');args=p.parse_args()
    oof,b,s,identity,integrity=audit(args.run_dir,args.report_dir)
    with tempfile.TemporaryDirectory(prefix='microcture-shape-plot-',dir='/tmp') as tmp:
        previous=os.environ.get('MPLCONFIGDIR');os.environ['MPLCONFIGDIR']=tmp
        try:figures(args.report_dir,oof,b,identity)
        finally:
            if previous is None:os.environ.pop('MPLCONFIGDIR',None)
            else:os.environ['MPLCONFIGDIR']=previous
    print(json.dumps({'audit':'passed','fits':40,'OOF':8,'P0_zero_checks':integrity['zero_P0_residual_checks'],'pairs':b['pairs']},indent=2))


if __name__=='__main__':main()
