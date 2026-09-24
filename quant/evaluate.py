"""Metrics on frozen predictions. Running this module never fits or tunes a model."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,pearsonr
from .config import save_json,REGIMES

def labels(prediction,threshold):
    p=np.asarray(prediction)
    return np.where(p>threshold,'CHEAP',np.where(p < -threshold,'RICH','NEUTRAL'))

def conviction(prediction,validation_abs):
    reference=np.sort(np.asarray(validation_abs))
    return 100*np.searchsorted(reference,np.abs(prediction),side='right')/len(reference)

def regression(y,p):
    y=np.asarray(y); p=np.asarray(p); valid=np.isfinite(y)&np.isfinite(p); y=y[valid]; p=p[valid]
    if not len(y): return {'n':0}
    nonconstant=np.std(y)>0 and np.std(p)>0
    return dict(n=len(y),mae=np.mean(abs(p-y)),rmse=np.sqrt(np.mean((p-y)**2)),r2=1-np.sum((p-y)**2)/np.sum((y-y.mean())**2) if np.std(y)>0 else None,pearson=pearsonr(p,y).statistic if nonconstant else None,spearman=spearmanr(p,y).statistic if nonconstant else None,sign_hit_rate=np.mean(p*y>0),average_prediction=np.mean(p))

def ranking(d,fraction=.1):
    keys=['timestamp','expiry','contract','option_type']
    g=d.groupby(keys,observed=True,sort=False)
    n=g.prediction.transform('size')
    rank=g.prediction.rank(method='average',pct=True)
    # At least two names per leg, and distinct predicted tails.
    valid=n*fraction>=2
    top=d.loc[valid&(rank>1-fraction)].groupby(keys,observed=True).target.mean().rename('top')
    bottom=d.loc[valid&(rank<=fraction)].groupby(keys,observed=True).target.mean().rename('bottom')
    pairs=pd.concat([top,bottom],axis=1).dropna().reset_index()
    pairs['spread']=pairs.top-pairs.bottom
    series=pairs.groupby('timestamp').spread.mean().reset_index()
    if not len(series): return {'mean':None,'median':None,'standard_error':None,'t_statistic':None,'n_days':0,'series':[]}
    # Cluster cross-section at timestamp, then day. HAC on daily means addresses serial dependence.
    daily=series.groupby(series.timestamp.dt.date).spread.mean().to_numpy()
    nday=len(daily); centered=daily-daily.mean(); lag=min(5,nday-1)
    longvar=np.dot(centered,centered)/nday
    for j in range(1,lag+1): longvar+=2*(1-j/(lag+1))*np.dot(centered[j:],centered[:-j])/nday
    se=np.sqrt(max(longvar,0)/nday) if nday>=10 else np.nan
    series['cumulative']=series.spread.cumsum()
    return dict(mean=daily.mean(),median=np.median(daily),standard_error=se,t_statistic=daily.mean()/se if np.isfinite(se) and se>0 else None,n_days=nday,n_groups=len(pairs),inference='Newey-West daily-mean standard error, up to five lags. Short test period; exploratory evidence only.',series=[{**r,'timestamp':str(r['timestamp'])} for r in series.to_dict('records')])

def evaluate_frame(d,threshold):
    d=d[d.target.notna()].copy(); y=d.target.to_numpy(); p=d.prediction.to_numpy()
    out=regression(y,p); signal=labels(p,threshold); correct=p*y>0
    cheap=signal=='CHEAP'; rich=signal=='RICH'; active=cheap|rich
    hit=lambda mask: float(correct[mask].mean()) if mask.any() else None
    out.update(cheap_hit_rate=hit(cheap),rich_hit_rate=hit(rich),non_neutral_hit_rate=hit(active),class_counts={k:int((signal==k).sum()) for k in ['CHEAP','NEUTRAL','RICH']},balanced_directional_accuracy=float(np.mean([(p[y>0]>0).mean(),(p[y<0]<0).mean()])) if (y>0).any() and (y<0).any() else None)
    out['direction_table']=[dict(signal=k,realized_positive=int(((signal==k)&(y>0)).sum()),realized_negative=int(((signal==k)&(y<0)).sum()),realized_zero=int(((signal==k)&(y==0)).sum())) for k in ['CHEAP','NEUTRAL','RICH']]
    out['conviction_thresholds']=[dict(label=name,count=int((active&(d.conviction.to_numpy()>=q)).sum()),hit_rate=hit(active&(d.conviction.to_numpy()>=q))) for name,q in [('all non-neutral',0),('above 50',50),('above 75',75),('above 90 / top 10%',90),('above 95 / top 5%',95)]]
    out['high_conviction_hit_rate']=hit(active&(d.conviction.to_numpy()>=90))
    d['correct']=correct; d['absolute_prediction']=abs(p); d['absolute_realized']=abs(y); d['signed_realized']=np.sign(p)*y
    d['decile']=np.minimum((d.conviction/10).astype(int),9)+1
    out['conviction_deciles']=d.groupby('decile',observed=True).agg(count=('target','size'),mean_absolute_prediction=('absolute_prediction','mean'),hit_rate=('correct','mean'),mean_realized=('target','mean'),median_realized=('target','median'),mean_predicted=('prediction','mean'),mean_absolute_realized=('absolute_realized','mean'),conditional_performance=('signed_realized','mean')).reset_index().to_dict('records')
    d['prediction_bucket']=pd.qcut(d.prediction.rank(method='first'),10,labels=False,duplicates='drop')+1
    out['calibration']=d.groupby('prediction_bucket',observed=True).agg(count=('target','size'),predicted=('prediction','mean'),realized=('target','mean')).reset_index().to_dict('records')
    out['ranking']={str(f):ranking(d,f) for f in [.2,.1,.05]}
    out['regimes']=[]
    for regime in REGIMES:
        r=d[d.regime==regime]; metrics=regression(r.target,r.prediction)
        metrics.update(regime=regime,average_conviction=r.conviction.mean(),cheap_rich_spread=ranking(r,.1)['mean'])
        out['regimes'].append(metrics)
    ic=[]
    for stamp,r in d.groupby('timestamp',observed=True):
        if len(r)>=30 and r.prediction.nunique()>1: ic.append({'timestamp':str(stamp),'ic':spearmanr(r.prediction,r.target).statistic})
    out['rank_ic']=ic
    hits=[r['hit_rate'] for r in out['conviction_deciles']]
    out['conviction_monotone']=bool(len(hits)>1 and np.all(np.diff(hits)>=0))
    out['conviction_trend_spearman']=float(spearmanr(range(len(hits)),hits).statistic) if len(hits)>2 else None
    return out

def otm_diagnostics(d):
    p=d[(d.option_type=='P')&d.target.notna()].copy()
    p['bucket']=pd.cut(p.delta.abs(),[0,.1,.2,.35,.65,1.001],labels=['deep OTM <10d','10–20 delta','20–35 delta','ATM 35–65d','ITM >65d'],include_lowest=True)
    p['correct']=p.prediction*p.target>0
    rows=p.groupby('bucket',observed=True).agg(count=('target','size'),iv=('iv','mean'),prediction=('prediction','mean'),realized=('target','mean'),surface_residual=('surface_residual','mean'),hit_rate=('correct','mean')).reset_index().to_dict('records')
    for row in rows:
        b=p[p.bucket==row['bucket']]
        row['iv_prediction_corr']=float(b.iv.corr(b.prediction)); row['residual_prediction_corr']=float(b.surface_residual.corr(b.prediction)); row['rich_share']=float(b.label.eq('RICH').mean())
    deep=p[p.delta.abs()<.1]
    flag=bool(len(deep)>100 and deep.iv.corr(deep.prediction)<-.5 and deep.label.eq('RICH').mean()>.7)
    return dict(buckets=rows,potential_raw_iv_failure=flag,conclusion='Potential raw-IV failure: strong negative IV association and predominant rich labels in deep OTM puts.' if flag else 'The coarse raw-IV failure screen did not trigger. This is a diagnostic, not proof that confounding is absent.',sample=p.sample(min(2500,len(p)),random_state=42)[['moneyness','iv','relative_iv','surface_residual','prediction','delta','target','bucket']].to_dict('records'))

def main():
    path=Path('results/metrics/metrics.json')
    if not path.exists(): raise SystemExit('Run python -m quant.train first.')
    metrics=json.loads(path.read_text()); print(json.dumps(metrics['test'],indent=2)[:5000])
if __name__=='__main__': main()
