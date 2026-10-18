#!/usr/bin/env python3
"""Verify fixed-fusion lineage/arithmetic and plot all registered outcomes."""
import json,sys,os,tempfile
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import evaluate_task1_fixed_fusion as f
from microcture.task1.evaluation import CLASS_NAMES,_read_prediction_csv,metrics
from microcture.task1_formal.runner import verify_artifact_index
from microcture.task1_formal.evaluation import _same_score


def check_summary(s,oof,members,identity):
    """Recompute case lists and paired intervals independently of evaluator helpers."""
    groups=sorted({v['group_id'] for v in identity.values()});gi={g:i for i,g in enumerate(groups)}
    f.require(len(groups)==49,'Expected49 spatial bootstrap units')
    cm=np.zeros((4,3,len(groups),3,3),int)
    for ki in range(4):
        for si,seed in enumerate((42,43,44)):
            rows=(oof['main','I3',seed]['rows'] if ki==0 else oof['main','P0',seed]['rows'] if ki==1 else [r for fold in range(5) for r in members['I3',fold,seed if ki==2 else 'lr']])
            for r in rows:cm[ki,si,gi[int(r['group_id'])],int(r['y_true']),int(r['y_pred'])]+=1
    weights=np.random.default_rng(20261014).multinomial(49,np.full(49,1/49),size=2000)
    def scores(a):
        d=np.diagonal(a,axis1=-2,axis2=-1);truth=a.sum(-1);pred=a.sum(-2)
        precision=np.divide(d,pred,out=np.zeros_like(d,dtype=float),where=pred!=0)
        recall=np.divide(d,truth,out=np.zeros_like(d,dtype=float),where=truth!=0)
        f1=np.divide(2*d,truth+pred,out=np.zeros_like(d,dtype=float),where=truth+pred!=0)
        return {'macro_f1':f1.mean(-1),'CHID_f1':f1[...,2],'CHID_precision':precision[...,2],'CHID_recall':recall[...,2],'balanced_accuracy':recall.mean(-1)}
    sampled=scores(np.einsum('bg,ksgij->bksij',weights,cm));point=scores(cm.sum(2))
    bootstrap=json.loads((ROOT/'results/task1_fixed_fusion_v1/group_bootstrap.json').read_text());maxdiff=0.
    for second,name in ((1,'fusionI3-P0'),(2,'fusionI3-CNNI3'),(3,'fusionI3-LRI3')):
        for metric,saved in bootstrap['pairs'][name].items():
            actual=np.array([float((point[metric][0]-point[metric][second]).mean()),*np.quantile(sampled[metric][:,0].mean(1)-sampled[metric][:,second].mean(1),[.025,.975])])
            expected=np.array([saved['point'],*saved['ci95']]);maxdiff=max(maxdiff,float(abs(actual-expected).max()))
            f.require(np.allclose(actual,expected,rtol=0,atol=1e-15),'Independent paired bootstrap differs')
    for condition in ('I3','G0','V0','P0'):
        actual={k:[] for k in ('stable_wrong_ids','stable_CHID_FN','stable_CHID_FP')}
        for sid,z in sorted(identity.items()):
            pred=[int(oof['main',condition,seed]['by_id'][sid]['y_pred']) for seed in (42,43,44)]
            if all(p!=z['y'] for p in pred):actual['stable_wrong_ids'].append(sid)
            if z['y']==2 and all(p!=2 for p in pred):actual['stable_CHID_FN'].append(sid)
            if z['y']!=2 and all(p==2 for p in pred):actual['stable_CHID_FP'].append(sid)
        for key,value in actual.items():f.require(value==s['conditions'][condition][key],'Independent stable-case list differs')
    return {'independent_bootstrap_max_abs_difference':maxdiff,'independent_stable_case_conditions':4,'bootstrap_spatial_groups':len(groups),'bootstrap_repeats':2000}


def audit():
    run=ROOT/'results/task1_fixed_fusion_v1';m=json.loads((run/'manifest.json').read_text());current,members,identity=f.prepare(ROOT,m['config']);f.require(current==m,'Fusion source/parent lineage changed')
    summary=json.loads((run/'summary.json').read_text());inventory=[];maxdiff=0.;oof={}
    for mode in ('main','pooled'):
        for condition in m['config']['conditions']:
            for seed in (m['config']['cnn_seeds'] if mode=='main' else ['pooled']):
                merged=[]
                for fold in range(5):
                    job={'mode':mode,'condition':condition,'seed':seed,'fold':fold};d=run/'runs'/mode/condition/f'seed_{seed}'/f'fold_{fold}';verify_artifact_index(d,job,m)
                    index=json.loads((d/'ARTIFACTS.json').read_text());inventory.extend({'path':str((d/z['name']).relative_to(run)),**z} for z in index['files'])
                    p=_read_prediction_csv(d/'outer_predictions.csv');f.roi.check_replicate_rows(p['rows'])
                    lr={r['structure_id']:r for r in members[condition,fold,'lr']};chosen=[seed] if mode=='main' else [42,43,44]
                    cn=[{r['structure_id']:r for r in members[condition,fold,s]} for s in chosen]
                    for row in p['rows']:
                        sid=row['structure_id'];actual={int(r['replicate']):np.array(r['probabilities']) for r in json.loads(row['replicate_probabilities_json'])}
                        lv={int(r['replicate']):np.array(r['probabilities']) for r in json.loads(lr[sid]['replicate_probabilities_json'])}
                        cv=[{int(r['replicate']):np.array(r['probabilities']) for r in json.loads(v[sid]['replicate_probabilities_json'])} for v in cn]
                        for wt in (1,2):
                            expected=(lv[wt]+np.mean([v[wt] for v in cv],axis=0))/2
                            diff=float(abs(actual[wt]-expected).max());maxdiff=max(maxdiff,diff);f.require(diff<=1e-15,'Independent half-mixture arithmetic differs')
                    saved=json.loads((d/'metrics.json').read_text());_same_score(saved['metrics'],p['metrics'],'fusion fold');merged.extend({**r,'fold':fold} for r in p['rows'])
                merged.sort(key=lambda r:r['structure_id']);saved=_read_prediction_csv(run/f'oof/{mode}/{condition}/seed_{seed}/predictions.csv');f.require(merged==[{**r,'fold':int(r['fold'])} for r in saved['rows']],'Derived OOF merge differs')
                f.require(len(merged)==344 and {r['structure_id'] for r in merged}==set(identity),'OOF coverage differs')
                oof[mode,condition,seed]={'rows':merged,'metrics':saved['metrics'],'by_id':{r['structure_id']:r for r in merged}}
                declared=summary['conditions'][condition]['seeds'][str(seed)] if mode=='main' else summary['conditions'][condition]['pooled_diagnostic'];_same_score(declared,saved['metrics'],'fusion summary')
    root=ROOT/'reports/task1_fixed_fusion_review_v1';root.mkdir(parents=True,exist_ok=True)
    result={'status':'passed','parent_scientific_indices':80,'derived_scientific_indices':80,'indexed_derived_files':len(inventory),'indexed_derived_bytes':sum(z['size_bytes'] for z in inventory),
        'derived_inventory_signature':f.shape.json_signature(inventory),'source_signature':m['source_signature'],'experiment_signature':m['experiment_signature'],
        'verified_main_OOF':12,'verified_pooled_OOF':4,'independent_WT_half_mixture_max_abs_difference':maxdiff,'no_training_or_weight_inference':True,
        **check_summary(summary,oof,members,identity)}
    (root/'integrity.json').write_text(json.dumps(result,indent=2)+'\n')
    return root,summary,oof,members,identity


def plots(root,s,oof,members,identity):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    import matplotlib as mpl
    mpl.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    cnn=json.loads((ROOT/'results/task1_roi_v1/summary.json').read_text());lr=json.loads((ROOT/'results/task1_shape_baseline_v2/summary.json').read_text());conditions=['I3','G0','V0','P0']
    fig=Figure(figsize=(12,5.5),layout='constrained');FigureCanvasAgg(fig);axes=fig.subplots(1,2)
    for ax,metric,title in zip(axes,['macro_f1','CHID_f1'],['Macro F1','CHID F1']):
        fusion=[next(r for r in s['comparisons'] if r['condition']==c) for c in conditions]
        mean=np.array([r[metric+'_mean'] if metric=='macro_f1' else r['CHID_f1_mean'] for r in fusion]);sd=np.array([r['macro_f1_seed_SD'] if metric=='macro_f1' else np.std([s['conditions'][c]['seeds'][str(z)]['per_class']['CHID']['f1'] for z in [42,43,44]],ddof=1) for r,c in zip(fusion,conditions)])
        ax.errorbar(np.arange(4)-.1,mean,yerr=sd,color='#2878B5',marker='o',capsize=3,label='Fixed0.5 fusion: three-seed mean ± SD')
        cnnvalues=[next(r for r in cnn['comparison'] if r['alignment']=='R1' and r['condition']==c)[metric+'_mean'] for c in conditions]
        ax.plot(np.arange(4),cnnvalues,'^-',color='#777777',label='M2+ three-seed mean')
        lrvalues=[next(r for r in lr['comparison'] if r['model']=='lr_shape' and r['condition']==c)[metric] for c in conditions]
        ax.plot(np.arange(4)+.1,lrvalues,'s-',color='#CC6633',label='Fixed455 LR (one seed)')
        pooled=[s['conditions'][c]['pooled_diagnostic'][metric] if metric=='macro_f1' else s['conditions'][c]['pooled_diagnostic']['per_class']['CHID']['f1'] for c in conditions]
        ax.plot(np.arange(4)-.1,pooled,'D',color='#2878B5',markerfacecolor='white',label='Pooled diagnostic, excluded from main gate')
        for j,z in enumerate([42,43,44]):
            values=[s['conditions'][c]['seeds'][str(z)][metric] if metric=='macro_f1' else s['conditions'][c]['seeds'][str(z)]['per_class']['CHID']['f1'] for c in conditions]
            ax.scatter(np.arange(4)-.1,values,s=15,marker=['o','s','^'][j],color='#2878B5',alpha=.75)
        ax.set(xticks=range(4),xticklabels=['I3 full','G0 geometry','V0 validity','P0 profile'],ylim=(0,1.02),title=title,ylabel='Unique-structure OOF score');ax.grid(axis='y',alpha=.2)
        if metric=='macro_f1':ax.axhline(.9,color='#BB4444',ls='--')
    handles,names=axes[0].get_legend_handles_labels();fig.legend(handles,names,loc='outside lower center',ncol=2,fontsize=9,title='Developmental data; SD across CNN member seeds is not a biological confidence interval',title_fontsize=9)
    fig.suptitle('Fixed LR + M2+ probability fusion: primary seeds and pooled diagnostic',fontsize=14)
    for ext in ('png','pdf'):fig.savefig(root/f'performance.{ext}',dpi=220,bbox_inches='tight')
    b=json.loads((ROOT/'results/task1_fixed_fusion_v1/group_bootstrap.json').read_text());fig=Figure(figsize=(11,4.5),layout='constrained');FigureCanvasAgg(fig);axes=fig.subplots(1,2)
    for ax,metric,title in zip(axes,['macro_f1','CHID_f1'],['Macro F1 effect','CHID F1 effect']):
        for i,(name,z) in enumerate(b['pairs'].items()):
            q=z[metric];v=q['point']*100;lo,hi=np.array(q['ci95'])*100;ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color='#2878B5',capsize=4)
        ax.axvline(0,color='#BB4444',ls='--');ax.set(yticks=range(3),yticklabels=['Fusion full minus profile','Fusion full minus M2+','Fusion full minus LR'],xlabel='Paired effect (percentage points)',title=title);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.suptitle('Conditional95% paired intervals: 2000 whole-group draws; no refitting',fontsize=13)
    for ext in ('png','pdf'):fig.savefig(root/f'paired_effects.{ext}',dpi=220,bbox_inches='tight')
    selected=sorted([sid for sid in identity if identity[sid]['y']==2],key=lambda z:int(z.split('_')[-1]))
    selected+=sorted(set(s['conditions']['I3']['stable_CHID_FP'])|{'CHIN_51','CHIN_59','CHIN_73','OPCID_66'})
    columns=[('CNN',z) for z in [42,43,44]]+[('Fusion',z) for z in [42,43,44]]+[('LR',42)]
    cnnby={z:{r['structure_id']:r for fold in range(5) for r in members['I3',fold,z]} for z in [42,43,44]};lrby={r['structure_id']:r for fold in range(5) for r in members['I3',fold,'lr']}
    vals=np.array([[int((cnnby[z] if kind=='CNN' else oof['main','I3',z]['by_id'] if kind=='Fusion' else lrby)[sid]['y_pred']) for kind,z in columns] for sid in selected])
    fig=Figure(figsize=(13.5,13),layout='constrained');FigureCanvasAgg(fig);ax=fig.subplots();ax.imshow(vals,cmap=ListedColormap(['#BCD2E3','#F4D4B4','#CEE4D8']),vmin=0,vmax=2,aspect='auto')
    for i,sid in enumerate(selected):
        for j,(kind,z) in enumerate(columns):
            predicted=vals[i,j];ax.text(j,i,CLASS_NAMES[predicted],ha='center',va='center',fontsize=8,fontweight='bold' if predicted!=identity[sid]['y'] else 'normal')
    ax.set(xticks=range(len(columns)),xticklabels=[f'{a}\nseed{b}' for a,b in columns],yticks=range(len(selected)),yticklabels=selected);ax.xaxis.tick_top();ax.tick_params(length=0)
    fig.suptitle('All26 CHID annotations and persistent false-positive anchors\nBold=incorrect; no selective case exclusion or relabeling',fontsize=14)
    for ext in ('png','pdf'):fig.savefig(root/f'case_predictions.{ext}',dpi=220,bbox_inches='tight')


def main():
    root,s,oof,members,identity=audit()
    with tempfile.TemporaryDirectory(prefix='microcture-fusion-plot-',dir='/tmp') as tmp:
        old=os.environ.get('MPLCONFIGDIR');os.environ['MPLCONFIGDIR']=tmp
        try:plots(root,s,oof,members,identity)
        finally:
            if old is None:os.environ.pop('MPLCONFIGDIR',None)
            else:os.environ['MPLCONFIGDIR']=old
    print(json.dumps({'audit':'passed','derived_fold_jobs':80,'main_OOF':12,'pooled_diagnostic_OOF':4,'stable_I3_errors':len(s['conditions']['I3']['stable_wrong_ids'])},indent=2))


if __name__=='__main__':main()
