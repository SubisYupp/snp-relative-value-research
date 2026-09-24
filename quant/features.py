import numpy as np
import pandas as pd

DYNAMIC=['iv','atm_iv','local_skew','curvature','surface_residual']

def add_lags(d,tolerance=10):
    if not 0<=tolerance<60: raise ValueError('Lag tolerance must be >=0 and less than one hour')
    left=d.copy(); left['lag_time']=left.timestamp-pd.Timedelta(hours=1)
    right=d[['contract_id','timestamp']+DYNAMIC].rename(columns={c:'past_'+c for c in ['timestamp']+DYNAMIC})
    out=pd.merge_asof(left.sort_values('lag_time'),right.sort_values('past_timestamp'),left_on='lag_time',right_on='past_timestamp',by='contract_id',direction='nearest',tolerance=pd.Timedelta(minutes=tolerance))
    assert ((out.past_timestamp<out.timestamp)|out.past_timestamp.isna()).all()
    same_session=out.timestamp.dt.normalize().eq(out.past_timestamp.dt.normalize())
    out.loc[~same_session,['past_'+c for c in DYNAMIC]]=np.nan
    for c in DYNAMIC: out[c+'_change']=out[c]-out['past_'+c]
    return out.drop(columns=['lag_time','past_timestamp']+['past_'+c for c in DYNAMIC])

def aggregate_market(d):
    """One ATM observation per series/expiry, independent of strike density."""
    series=d[['timestamp','contract','expiry','atm_iv','underlying']].drop_duplicates(['timestamp','contract','expiry'])
    return series.groupby('timestamp',as_index=False).agg(underlying=('underlying','median'),atm_iv=('atm_iv','median'))

def underlying_features(market,train_end):
    m=market.sort_values('timestamp').copy()
    training=m[m.timestamp<train_end]
    per_day=int(training.groupby(training.timestamp.dt.date).size().median())
    intervals=max(per_day-1,1)
    dt=m.timestamp.diff().dt.total_seconds()/60
    returns=np.log(m.underlying/m.underlying.shift()).where(dt.between(50,70))
    m['underlying_return']=returns
    m['realized_vol']=np.sqrt(252*intervals*returns.pow(2).rolling(per_day*5,min_periods=intervals*3).mean())
    return m,dict(observations_per_day=per_day,within_session_intervals=intervals,annualization=252*intervals,rolling_observations=per_day*5)
