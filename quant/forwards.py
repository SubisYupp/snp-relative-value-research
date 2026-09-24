import numpy as np
import pandas as pd

def parity_forward(strike,call,put,rate,years):
    return strike+np.exp(rate*years)*(call-put)

def add_forwards(d):
    keys=['timestamp','contract','expiry','strike']
    pairs=d.pivot(index=keys,columns='option_type',values='option_price').reset_index()
    for side in ['C','P']:
        if side not in pairs: pairs[side]=np.nan
    rates=d.groupby(keys,observed=True,sort=False)[['rate','dte','underlying']].median().reset_index()
    pairs=pairs.merge(rates,on=keys)
    pairs['f']=parity_forward(pairs.strike,pairs.C,pairs.P,pairs.rate,pairs.dte/365.25)
    pairs.loc[(pairs.f<=0)|((pairs.f/pairs.underlying-1).abs()>.2),'f']=np.nan
    surface=['timestamp','contract','expiry']
    f=pairs.groupby(surface,observed=True,sort=False).f.agg(['median','count']).reset_index().rename(columns={'median':'parity_forward','count':'parity_pairs'})
    d=d.merge(f,on=surface,how='left')
    d['forward_fallback']=d.parity_pairs.lt(3)|d.parity_forward.isna()
    d['forward']=np.where(d.forward_fallback,d.underlying*np.exp(d.rate*d.dte/365.25),d.parity_forward)
    d['moneyness']=np.log(d.strike/d.forward)
    return d
