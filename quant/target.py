import numpy as np
import pandas as pd

def match_horizon(d,minutes=60,tolerance=10,outcomes=None):
    if tolerance>=minutes or tolerance<0: raise ValueError('Tolerance must be >=0 and less than horizon')
    left=d.copy().sort_values('timestamp'); left['desired']=left.timestamp+pd.Timedelta(minutes=minutes)
    source=d if outcomes is None else outcomes
    if source.duplicated(['contract_id','timestamp']).any():
        raise ValueError('Ambiguous outcome contract/timestamp observations must be resolved before matching')
    right=source[['contract_id','timestamp','option_price','underlying']].rename(columns={'timestamp':'target_timestamp','option_price':'future_price','underlying':'future_underlying'}).sort_values('target_timestamp')
    out=pd.merge_asof(left.sort_values('desired'),right,left_on='desired',right_on='target_timestamp',by='contract_id',direction='nearest',tolerance=pd.Timedelta(minutes=tolerance))
    out['actual_horizon_minutes']=(out.target_timestamp-out.timestamp).dt.total_seconds()/60
    same_day=out.target_timestamp.dt.normalize().eq(out.timestamp.dt.normalize())
    out.loc[~same_day,['future_price','future_underlying','actual_horizon_minutes']]=np.nan
    out.loc[~same_day,'target_timestamp']=pd.NaT
    out['target']=((out.future_price-out.option_price)-out.delta*(out.future_underlying-out.underlying))/out.vega
    return out.drop(columns=['desired','future_price','future_underlying'])
