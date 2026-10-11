"""Frozen Task 1 comparison. Train/choose on validation, then evaluate all on test."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/microcture-mpl')
import argparse,csv,hashlib,json,pickle,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from task1_baselines import PCA,LogisticRegression,make_pipeline,StandardScaler,accuracy_score,confusion_matrix,classification_report,f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/task1'
LABELS=['OPCID','CHIN','CHID']
SPECS={'M1':('X_oe',[1],True,True),'M2':('X_oe',[0,1],True,True),
       'A1':('X_oe',[0],True,True),'A3':('X_depth',[0,1],True,True),
       'A4':('X_oe',[0,1],True,False),'A5':('X_oe',[0,1],False,True)}

class CNN(nn.Module):
    def __init__(self,scales=2,mask=True):
        super().__init__()
        layers=[];cin=2 if mask else 1
        for cout in [16,32,64]:
            layers += [nn.Conv2d(cin,cout,3,padding=1),nn.GroupNorm(4,cout),nn.ReLU(),nn.MaxPool2d(2)]
            cin=cout
        self.encoder=nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1))
        self.head=nn.Sequential(nn.Linear(64*scales,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,3))
    def forward(self,x):
        n,s,c,h,w=x.shape
        z=self.encoder(x.reshape(n*s,c,h,w)).reshape(n,s*64)
        return self.head(z)


def inputs(data,spec):
    key,scales,mask,_=spec
    x=data[key][:,scales].copy()
    return x if mask else x[:,:,:1]


def grouped(data,p):
    ids=np.unique(data['structure_id']);ys=[];ps=[]
    for sid in ids:
        use=data['structure_id']==sid
        if len(np.unique(data['y'][use]))!=1:raise ValueError('Conflicting labels')
        ys.append(data['y'][use][0]);ps.append(p[use].mean(axis=0))
    return ids,np.array(ys),np.array(ps)


def metrics(data,p):
    ids,y,prob=grouped(data,p);pred=prob.argmax(axis=1)
    return dict(accuracy=float(accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),
                confusion_matrix=confusion_matrix(y,pred,labels=[0,1,2]).tolist(),
                per_class=classification_report(y,pred,labels=[0,1,2],target_names=LABELS,output_dict=True,zero_division=0),
                per_replicate={str(int(r)):dict(accuracy=float(accuracy_score(data['y'][data['replicate']==r],p[data['replicate']==r].argmax(1))),
                    macro_f1=float(f1_score(data['y'][data['replicate']==r],p[data['replicate']==r].argmax(1),labels=[0,1,2],average='macro',zero_division=0))) for r in np.unique(data['replicate'])})


def predict(model,x):
    model.eval()
    with torch.no_grad():return torch.cat([model(torch.from_numpy(x[i:i+64])).softmax(1) for i in range(0,len(x),64)]).numpy()


def train(name,seed,train,val,weights,cfg):
    path=OUT/f'{name}_{seed}.pt'
    if path.exists():
        checkpoint=torch.load(path,weights_only=False)
        if 'epochs_run' in checkpoint['summary']:return checkpoint['summary']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    spec=SPECS[name];model=CNN(len(spec[1]),spec[2]);x=inputs(train,spec);v=inputs(val,spec)
    xt=torch.from_numpy(x);y=torch.from_numpy(train['y']).long()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    lossfn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32) if spec[3] else None)
    best=-1.;stale=0;history=[];start=time.monotonic()
    shuffle=torch.Generator().manual_seed(seed)
    for epoch in range(1,cfg['max_epochs']+1):
        model.train();total=0
        for batch in torch.randperm(len(y),generator=shuffle).split(cfg['batch_size']):
            opt.zero_grad();loss=lossfn(model(xt[batch]),y[batch]);loss.backward();opt.step();total+=float(loss.detach())*len(batch)
        score=metrics(val,predict(model,v))['macro_f1']
        history.append(dict(epoch=epoch,train_loss=total/len(y),validation_macro_f1=score))
        if score>best+1e-10:
            best=score;stale=0
            summary=dict(model=name,seed=seed,best_epoch=epoch,validation_macro_f1=score,parameters=sum(p.numel() for p in model.parameters()))
            torch.save(dict(state_dict=model.state_dict(),summary=summary,spec=spec),path)
        else:stale+=1
        if epoch%10==0:print(name,seed,'epoch',epoch,'validation',round(score,4),'best',round(best,4),flush=True)
        if stale>=cfg['early_stopping_patience']:break
    checkpoint=torch.load(path,weights_only=False)
    checkpoint['summary'].update(epochs_run=epoch,training_seconds=time.monotonic()-start)
    torch.save(checkpoint,path)
    (OUT/f'{name}_{seed}_history.json').write_text(json.dumps(history,indent=2))
    print('FINISHED',checkpoint['summary'],flush=True)
    return checkpoint['summary']


def features(data,kind):
    if kind=='B2':return data['X_oe'][:,1].reshape(len(data['y']),-1)
    rows=list(csv.DictReader((ROOT/'data/processed/classification_examples.csv').open()))
    lookup={(r['structure_id'],int(r['replicate'])):r for r in rows}
    return np.array([[np.log1p(float(lookup[sid,int(rep)][k])) for k in ('length_bp','local_mean_count')] for sid,rep in zip(data['structure_id'],data['replicate'])])


def confusion_figure(cm,path,title):
    fig,ax=plt.subplots(figsize=(5,4),layout='constrained');im=ax.imshow(cm,cmap='Blues',vmin=0)
    for i in range(3):
        for j in range(3):ax.text(j,i,str(cm[i][j]),ha='center',va='center')
    ax.set(xticks=range(3),yticks=range(3),xticklabels=LABELS,yticklabels=LABELS,xlabel='Predicted',ylabel='True',title=title)
    fig.colorbar(im,ax=ax);fig.savefig(path,dpi=160);plt.close(fig)


def explain(name,seed,test):
    spec=SPECS[name];checkpoint=torch.load(OUT/f'{name}_{seed}.pt',weights_only=False)
    model=CNN(len(spec[1]),spec[2]);model.load_state_dict(checkpoint['state_dict']);model.eval()
    x=inputs(test,spec);p=predict(model,x);ids,y,prob=grouped(test,p)
    manifest={r['structure_id']:r for r in csv.DictReader((ROOT/'data/processed/classification_split.csv').open())}
    records=[]
    for label in range(3):
        for correct in (True,False):
            choices=np.where((y==label)&((prob.argmax(1)==y)==correct))[0]
            if not len(choices):continue
            # Deterministic examples, not selected by visually pleasing attribution.
            sid=ids[choices[0]];idx=np.where(test['structure_id']==sid)[0][0]
            tensor=torch.from_numpy(x[idx:idx+1]).requires_grad_(True)
            logits=model(tensor);target=int(logits.argmax(1));logits[0,target].backward()
            sal=tensor.grad[0,:,0].abs().numpy();pred=int(prob[choices[0]].argmax())
            fig,axes=plt.subplots(len(spec[1]),2,figsize=(8,4*len(spec[1])),squeeze=False,layout='constrained')
            perturb=[]
            r=manifest[sid];center=np.floor(float(r['center'])/100)*100
            for local,scale in enumerate(spec[1]):
                image=x[idx,local,0];mask=test[spec[0]][idx,scale,1]>0
                axes[local,0].imshow(np.where(mask,image,np.nan),origin='lower',cmap='magma')
                axes[local,1].imshow(np.where(mask,sal[local],np.nan),origin='lower',cmap='inferno')
                width=[6400,25600][scale];pos=center-width/2+(np.arange(64)+.5)*width/64
                inside=(pos>=float(r['start']))&(pos<float(r['end']))
                structure=inside[:,None]&inside[None,:]&mask
                outside=mask&~structure
                rng=np.random.default_rng(42);count=min(structure.sum(),outside.sum())
                for region,where in [('structure',structure),('background',outside)]:
                    chosen=rng.choice(np.flatnonzero(where),int(count),replace=False)
                    changed=x[idx:idx+1].copy();changed[0,local,0].flat[chosen]=0
                    after=predict(model,changed)[0,target]
                    perturb.append(dict(scale_bp=width,region=region,pixels=int(count),probability_drop=float(p[idx,target]-after)))
                axes[local,0].set_title(f'{width/1000:g} kb standardized input')
                axes[local,1].set_title('Absolute input gradient (predicted class)')
            fig.suptitle(f'{sid} | true {LABELS[label]} | structure prediction {LABELS[pred]}\nrep1 attribution; single-example predicted class: {LABELS[target]}')
            fig.savefig(OUT/f'explanation_{sid}.png',dpi=150);plt.close(fig)
            records.append(dict(structure_id=str(sid),true_class=LABELS[label],prediction=LABELS[pred],occlusion=perturb))
    (OUT/'explanations.json').write_text(json.dumps(records,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--threads',type=int,default=4);args=parser.parse_args()
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'configs/experiments.json').read_text())
    dataset=ROOT/cfg['dataset']
    fingerprints={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'configs/experiments.json',ROOT/'scripts/train_task1.py',ROOT/'reports/classification_dataset.json',ROOT/'data/processed/classification_examples.csv',*sorted(dataset.glob('*.npz'))]}
    run=OUT/'run_manifest.json'
    if run.exists() and json.loads(run.read_text())['sha256']!=fingerprints:raise ValueError('Inputs/code changed; use a fresh output directory, do not mix experiments')
    run.write_text(json.dumps(dict(sha256=fingerprints,torch=torch.__version__,threads=args.threads,augmentation='none for every run',device='cpu'),indent=2))
    with np.load(dataset/'train.npz') as f:train={k:f[k] for k in f.files}
    with np.load(dataset/'validation.npz') as f:val={k:f[k] for k in f.files}
    weights=json.loads((ROOT/'reports/classification_dataset.json').read_text())['train_class_weights']
    majority=int(np.bincount(train['y']).argmax());baselines={};summaries=[]
    for name in ['B0','B1','B2']:
        if name=='B0':p=np.eye(3)[np.full(len(val['y']),majority)]
        else:
            steps=[StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42)] if name=='B1' else [PCA(n_components=16,svd_solver='randomized',random_state=42),StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42)]
            model=make_pipeline(*steps);model.fit(features(train,name),train['y']);baselines[name]=model
            p=model.predict_proba(features(val,name))
            with (OUT/f'{name}.pkl').open('wb') as f:pickle.dump(model,f)
        summaries.append(dict(model=name,seed=42,validation_macro_f1=metrics(val,p)['macro_f1']))
    for name in SPECS:
        for seed in cfg['training_seeds']:summaries.append(train_fn(name,seed,train,val,weights,cfg['cnn_defaults']))
    # Lock model-family choice before opening held-out tensors.
    means={name:float(np.mean([s['validation_macro_f1'] for s in summaries if s['model']==name])) for name in SPECS}
    selected=max(means,key=means.get)
    selected_seed=cfg['training_seeds'][0]  # predetermined, never choose best test seed
    selection=dict(selected_cnn=selected,explanation_seed=selected_seed,validation_means=means,summaries=summaries)
    (OUT/'selection_before_test.json').write_text(json.dumps(selection,indent=2))
    with np.load(dataset/'test.npz') as f:test={k:f[k] for k in f.files}
    evaluations=[]
    for summary in summaries:
        name,seed=summary['model'],summary['seed']
        if name=='B0':p=np.eye(3)[np.full(len(test['y']),majority)]
        elif name in baselines:p=baselines[name].predict_proba(features(test,name))
        else:
            spec=SPECS[name];model=CNN(len(spec[1]),spec[2]);model.load_state_dict(torch.load(OUT/f'{name}_{seed}.pt',weights_only=False)['state_dict'])
            p=predict(model,inputs(test,spec))
        score=metrics(test,p);evaluations.append(dict(**summary,test=score))
        ids,y,prob=grouped(test,p)
        with (OUT/f'{name}_{seed}_predictions.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['structure_id','true','predicted',*LABELS]);w.writerows([sid,int(t),int(q.argmax()),*q.tolist()] for sid,t,q in zip(ids,y,prob))
        confusion_figure(score['confusion_matrix'],OUT/f'{name}_{seed}_confusion.png',f'{name} / seed {seed} / held-out structures')
    (OUT/'evaluation.json').write_text(json.dumps(evaluations,indent=2))
    table=[]
    for name in ['B0','B1','B2',*SPECS]:
        runs=[r for r in evaluations if r['model']==name]
        table.append(dict(model=name,runs=len(runs),validation_macro_f1=np.mean([r['validation_macro_f1'] for r in runs]),test_macro_f1_mean=np.mean([r['test']['macro_f1'] for r in runs]),test_macro_f1_std=np.std([r['test']['macro_f1'] for r in runs]),test_accuracy_mean=np.mean([r['test']['accuracy'] for r in runs])))
    with (OUT/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    explain(selected,selected_seed,test)
    print('COMPLETE',json.dumps(table),flush=True)

train_fn=train
if __name__=='__main__':main()
