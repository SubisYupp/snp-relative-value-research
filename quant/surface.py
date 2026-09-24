import numpy as np
import pandas as pd

def local_smile(k,iv,neighbors=13):
    """Leave-one-strike-out local quadratic; all derivative inputs are contemporaneous."""
    k=np.asarray(k,float); iv=np.asarray(iv,float); n=len(k)
    if n<5: return np.full((n,3),np.nan)
    distance=np.abs(k[:,None]-k[None,:])
    # Every same-strike quote excluded, not merely the diagonal.
    distance[distance<1e-12]=np.inf
    idx=np.argsort(distance,axis=1)[:,:min(neighbors,n-1)]
    usable=np.isfinite(np.take_along_axis(distance,idx,axis=1))&np.isfinite(iv[idx])
    dx=k[idx]-k[:,None]; y=np.where(usable,iv[idx],0.)
    scale=np.maximum(np.max(abs(dx),axis=1),.005)
    x=dx/scale[:,None]; w=np.where(usable,np.exp(-2*x*x),0.)
    a=np.stack([np.ones_like(x),x,x*x],axis=-1)
    gram=np.einsum('nki,nk,nkj->nij',a,w,a)
    # Small fixed ridge on curvature, no fitted hyperparameter or future observations.
    gram[:,2,2]+=1e-5
    rhs=np.einsum('nki,nk,nk->ni',a,w,y)
    try: b=np.linalg.solve(gram,rhs[...,None])[...,0]
    except np.linalg.LinAlgError: b=np.einsum('nij,nj->ni',np.linalg.pinv(gram),rhs)
    result=np.column_stack([b[:,0],b[:,1]/scale,2*b[:,2]/scale**2])
    invalid=(usable.sum(axis=1)<4)|(result[:,0]<=0)|~np.isfinite(result).all(axis=1)
    result[invalid]=np.nan
    return result

def add_surface(d,neighbors=13):
    cols=['expected_iv','local_skew','curvature']
    out=np.full((len(d),3),np.nan)
    groups=['timestamp','contract','expiry','option_type']
    for ids in d.groupby(groups,sort=False,observed=True).indices.values():
        out[ids]=local_smile(d.moneyness.to_numpy()[ids],d.iv.to_numpy()[ids],neighbors)
    d[cols]=out; d['surface_residual']=d.iv-d.expected_iv
    # ATM uses closest observed moneyness, averaged across call and put of that series.
    atm=d.assign(distance=d.moneyness.abs()).sort_values('distance').groupby(groups,observed=True,sort=False).head(1)
    atm=atm.groupby(['timestamp','contract','expiry'],observed=True).iv.mean().rename('atm_iv').reset_index()
    d=d.merge(atm,on=['timestamp','contract','expiry'],how='left')
    d['relative_iv']=d.iv-d.atm_iv
    unique_series=d[['timestamp','contract','expiry','atm_iv','dte']].drop_duplicates(['timestamp','contract','expiry'])
    term=unique_series.groupby(['timestamp','expiry'],observed=True)[['atm_iv','dte']].median().reset_index().sort_values(['timestamp','dte'])
    term['tv']=term.atm_iv**2*term.dte/365.25
    g=term.groupby('timestamp',sort=False)
    term['forward_variance']=(g.tv.shift(-1)-term.tv)/((g.dte.shift(-1)-term.dte)/365.25)
    term=term[['timestamp','expiry','forward_variance']]
    return d.merge(term,on=['timestamp','expiry'],how='left')
