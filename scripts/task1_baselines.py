"""Small NumPy/SciPy baselines; avoids an unavailable optional sklearn dependency."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax,logsumexp


def confusion_matrix(y,p,labels=(0,1,2)):
    out=np.zeros((len(labels),len(labels)),int)
    for a,b in zip(y,p):out[list(labels).index(a),list(labels).index(b)]+=1
    return out


def accuracy_score(y,p):return np.mean(np.asarray(y)==np.asarray(p))


def classification_report(y,p,labels=(0,1,2),target_names=('OPCID','CHIN','CHID'),output_dict=True,zero_division=0):
    cm=confusion_matrix(y,p,labels);support=cm.sum(1);predicted=cm.sum(0);tp=cm.diagonal()
    precision=np.divide(tp,predicted,out=np.zeros(3,float),where=predicted>0)
    recall=np.divide(tp,support,out=np.zeros(3,float),where=support>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(3,float),where=(precision+recall)>0)
    return {name:dict(precision=float(precision[i]),recall=float(recall[i]),**{'f1-score':float(f1[i])},support=int(support[i])) for i,name in enumerate(target_names)}


def f1_score(y,p,labels=(0,1,2),average='macro',zero_division=0):
    return np.mean([r['f1-score'] for r in classification_report(y,p,labels).values()])


class StandardScaler:
    def fit(self,x,y=None):
        self.mean=x.mean(0);self.std=x.std(0);self.std[self.std<1e-12]=1;return self
    def transform(self,x):return (x-self.mean)/self.std


class PCA:
    def __init__(self,n_components=16,**kwargs):self.n=n_components
    def fit(self,x,y=None):
        self.mean=x.mean(0)
        # Exact thin SVD, deterministic. No randomized approximation is needed here.
        _,_,v=np.linalg.svd(x-self.mean,full_matrices=False)
        self.components=v[:self.n];return self
    def transform(self,x):return (x-self.mean)@self.components.T


class LogisticRegression:
    def __init__(self,max_iter=2000,class_weight='balanced',random_state=42):self.max_iter=max_iter;self.balanced=class_weight=='balanced'
    def fit(self,x,y):
        x=np.c_[x,np.ones(len(x))];n,d=x.shape
        onehot=np.eye(3)[y];counts=np.bincount(y,minlength=3)
        sample=(n/(3*counts))[y] if self.balanced else np.ones(n)
        def objective(flat):
            w=flat.reshape(d,3);logits=x@w;p=softmax(logits,axis=1)
            loss=np.sum(sample*(logsumexp(logits,axis=1)-np.sum(onehot*logits,axis=1)))/n+np.square(w[:-1]).sum()/(2*n)
            grad=x.T@((p-onehot)*sample[:,None])/n;grad[:-1]+=w[:-1]/n
            return loss,grad.ravel()
        result=minimize(objective,np.zeros(d*3),jac=True,method='L-BFGS-B',options={'maxiter':self.max_iter,'ftol':1e-10})
        if not result.success:raise RuntimeError('Logistic regression failed: '+result.message)
        self.weights=result.x.reshape(d,3);self.iterations=result.nit;return self
    def predict_proba(self,x):return softmax(np.c_[x,np.ones(len(x))]@self.weights,axis=1)


class Pipeline:
    def __init__(self,steps):self.steps=steps
    def fit(self,x,y):
        for step in self.steps[:-1]:step.fit(x,y);x=step.transform(x)
        self.steps[-1].fit(x,y);return self
    def predict_proba(self,x):
        for step in self.steps[:-1]:x=step.transform(x)
        return self.steps[-1].predict_proba(x)


def make_pipeline(*steps):return Pipeline(steps)
