import pandas as pd

def vertical_candidates(d):
    keys=['timestamp','contract','expiry','option_type']
    cheap=d[d.label=='CHEAP'].sort_values('prediction',ascending=False).groupby(keys,observed=True).head(1)
    rich=d[d.label=='RICH'].sort_values('prediction').groupby(keys,observed=True).head(1)
    fields=['strike','prediction','iv','relative_iv','surface_residual','vega','conviction','target','row_id']
    out=cheap[keys+fields+['regime']].merge(rich[keys+fields],on=keys,suffixes=('_long','_short'))
    out['prediction_spread']=out.prediction_long-out.prediction_short
    out['realized_spread']=out.target_long-out.target_short
    out['conviction']=out[['conviction_long','conviction_short']].min(axis=1)
    out['short_contracts_per_long']=out.vega_long/out.vega_short
    return out.sort_values('prediction_spread',ascending=False)
