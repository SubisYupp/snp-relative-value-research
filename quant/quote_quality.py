"""Conservative same-timestamp checks; never use subsequent prices or returns."""
import numpy as np
import pandas as pd
from .cleaning import deduplicate, identity_invalid

def flag_quote_outliers(d, relative_tolerance=.05):
    """Quarantine only extreme, independently corroborated cross-strike errors.

    Require BOTH a monotonic price-envelope breach exceeding 5% of underlying
    and a parity-forward deviation exceeding max(5% of underlying, 10*MAD).
    This is a coarse data-error screen, not an execution/arbitrage test.
    It does not select thresholds using prediction targets.
    """
    out=d.copy(); out['suspect_price']=False
    unique,_,_=deduplicate(d)
    valid=(~identity_invalid(unique) & (unique.dte>0) & (unique.option_price>=0)
           & np.isfinite(unique.option_price) & (unique.underlying>0)
           & np.isfinite(unique.underlying) & np.isfinite(unique.rate))
    u=unique.loc[valid].copy()
    if u.empty: return out
    keys=['timestamp','contract','expiry','strike']; group=keys[:-1]
    pair=u.pivot(index=keys,columns='option_type',values='option_price').reset_index()
    if 'C' not in pair or 'P' not in pair: return out
    meta=u.groupby(keys,as_index=False)[['rate','dte','underlying']].median()
    pair=pair.merge(meta,on=keys)
    pair['parity']=pair.strike+np.exp(pair.rate*pair.dte/365.25)*(pair.C-pair.P)
    pair['median']=pair.groupby(group).parity.transform('median')
    pair['deviation']=abs(pair.parity-pair['median'])
    pair['mad']=pair.groupby(group).deviation.transform('median')
    pair['pair_count']=pair.groupby(group).parity.transform('count')
    pair['parity_outlier']=(pair.pair_count>=5)&(pair.deviation>np.maximum(relative_tolerance*pair.underlying,10*pair.mad))
    u=u.merge(pair[keys+['parity_outlier']],on=keys,how='left').sort_values(group+['option_type','strike'])
    g=u.groupby(group+['option_type'])
    lower=g.option_price.shift(); upper=g.option_price.shift(-1)
    neighbor_min=np.minimum(lower,upper); neighbor_max=np.maximum(lower,upper)
    u['breach']=np.maximum(u.option_price-neighbor_max,neighbor_min-u.option_price)
    bad=u.loc[(u.breach>relative_tolerance*u.underlying)&u.parity_outlier.fillna(False),['contract_id','timestamp']]
    if len(bad):
        bad_index=pd.MultiIndex.from_frame(bad)
        out['suspect_price']=pd.MultiIndex.from_frame(out[['contract_id','timestamp']]).isin(bad_index)
    return out
