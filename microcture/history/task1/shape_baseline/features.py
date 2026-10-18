"""455 predefined descriptors; no labels, fitting or per-dataset feature selection."""
from __future__ import annotations
import numpy as np

FEATURE_PROTOCOL="roi_shape455_reversal_invariant_v1"


def _weighted(values, weights, rms=False):
    total=weights.sum(-1);mean=np.divide((values*weights).sum(-1),total,out=np.zeros_like(total),where=total>0)
    second=np.divide((values*values*weights).sum(-1),total,out=np.zeros_like(total),where=total>0)
    return mean,np.sqrt(np.maximum(second if rms else second-mean*mean,0))


def _one(x,length):
    n=len(x);parts=[length.reshape(n,1).astype(float)];names=['length_inner_standardized']
    for scale in range(2):
        s,v,m=x[:,scale,0],x[:,scale,1],x[:,scale,2];prefix=f'scale{scale}'
        coverage=np.divide((m*v).sum((1,2)),m.sum((1,2)),out=np.zeros(n),where=m.sum((1,2))>0)
        parts.append(np.column_stack((v.mean((1,2)),m.mean((1,2)),coverage)))
        names.extend(f'{prefix}_{name}' for name in ('valid_mean','mask_mean','target_validity'))
    offsets=np.abs(np.arange(64)[:,None]-np.arange(64)[None,:]);anti=(np.add.outer(np.arange(64),np.arange(64))*16)//127
    for scale in range(2):
        s,v,m=x[:,scale,0],x[:,scale,1],x[:,scale,2];prefix=f'scale{scale}'
        for grid in (4,8):
            k=64//grid;sv=(s*v).reshape(n,grid,k,grid,k).sum((2,4));amount=v.reshape(n,grid,k,grid,k).sum((2,4))
            parts.append(np.divide(sv,amount,out=np.zeros_like(sv),where=amount>0).reshape(n,-1))
            names.extend(f'{prefix}_signal_grid{grid}_{i}_{j}' for i in range(grid) for j in range(grid))
        for key,array in [('valid',v),('mask',m)]:
            parts.append(array.reshape(n,4,16,4,16).mean((2,4)).reshape(n,-1))
            names.extend(f'{prefix}_{key}_grid4_{i}_{j}' for i in range(4) for j in range(4))
        residual=np.zeros_like(s)
        for offset in range(64):
            take=offsets==offset;mean,_=_weighted(s[:,take],v[:,take]);residual[:,take]=np.where(v[:,take]>0,s[:,take]-mean[:,None],0)
        for layer in range(16):
            take=(offsets//4)==layer;mean,std=_weighted(s[:,take],v[:,take])
            parts.append(np.column_stack((mean,std,v[:,take].mean(1))))
            names.extend(f'{prefix}_distance{layer}_{k}' for k in ('mean','std','valid'))
        for kind,positions in [('anti',anti),('row',np.broadcast_to((np.arange(64)//4)[:,None],(64,64)))]:
            for band in range(16):
                take=positions==band;mean,rms=_weighted(residual[:,take],v[:,take],rms=True)
                parts.append(np.column_stack((mean,rms)));names.extend(f'{prefix}_residual_{kind}{band}_{k}' for k in ('mean','rms'))
    result=np.concatenate(parts,axis=1)
    if result.shape!=(n,455) or len(names)!=455 or len(set(names))!=455 or not np.isfinite(result).all():raise ValueError('Invalid fixed455 feature output')
    return result,names


def features(samples):
    x=np.asarray(samples['x'],dtype=np.float64);length=np.asarray(samples['length'],dtype=np.float64)
    if x.ndim!=5 or x.shape[1:]!=(2,3,64,64) or length.shape!=(len(x),1) or not np.isfinite(x).all():raise ValueError('Expected finite R1 N,2,3,64,64 and length N,1')
    if np.any((x[:,:,1:]<0)|(x[:,:,1:]>1)):raise ValueError('Invalid validity/mask range')
    original,names=_one(x,length);reversed_values,other_names=_one(x[...,::-1,::-1],length)
    if names!=other_names:raise ValueError('Feature names differ under reversal')
    return (original+reversed_values)*.5,names
