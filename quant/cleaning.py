import numpy as np

IDENTITY = ['contract_id', 'timestamp']

def deduplicate(d):
    """Keep one identical economic record; reject remaining ambiguous identities."""
    fields=[c for c in d if c!='source_file']
    exact=d.duplicated(fields,keep='first')
    unique=d.loc[~exact].copy()
    conflicts=unique.duplicated(IDENTITY,keep=False)
    return unique.loc[~conflicts].copy(), int(exact.sum()), int(conflicts.sum())

def identity_invalid(d):
    return (d.timestamp.isna() | d.expiry.isna() | d.contract.isna()
            | d.contract.astype('string').str.strip().eq('').fillna(True)
            | ~d.option_type.isin(['C','P']) | ~(d.strike>0) | ~np.isfinite(d.strike))

def outcome_quotes(d):
    """Future labels require identity, price and underlying, not future Greeks."""
    d,_,_=deduplicate(d)
    # Source zeros are ambiguous missing-price sentinels without bid/ask evidence.
    # Require a strictly positive observed price at both ends, not valid future Greeks.
    valid=(~identity_invalid(d) & (d.dte>0) & (d.option_price>0)
           & np.isfinite(d.option_price) & (d.underlying>0) & np.isfinite(d.underlying))
    if 'suspect_price' in d: valid &= ~d.suspect_price
    return d.loc[valid].copy()

def clean(d, min_vega=.01):
    report={'input_option_rows':len(d),'missing_pct':(d.isna().mean()*100).to_dict(),
            'exact_duplicates':int(d.duplicated([c for c in d if c!='source_file']).sum()),
            'duplicate_contract_timestamp':int(d.duplicated(IDENTITY).sum())}
    d,identical,ambiguous=deduplicate(d)
    drops={'identical_duplicate_copies':identical,'ambiguous_duplicate_rows':ambiguous}
    rules={
        'invalid_identity': identity_invalid(d),
        'expired': ~(d.dte>0),
        'invalid_price': ~(d.option_price>=0)|~np.isfinite(d.option_price),
        'zero_price_unusable': d.option_price.eq(0),
        'invalid_underlying': ~(d.underlying>0)|~np.isfinite(d.underlying),
        'invalid_iv': ~(d.iv>0)|~np.isfinite(d.iv),
        'nonpositive_vega': ~(d.vega>0)|~np.isfinite(d.vega),
        'small_positive_vega': (d.vega>0)&(d.vega<min_vega),
        'invalid_delta': ~np.isfinite(d.delta)|(d.delta.abs()>1.001),
        'invalid_rate': ~np.isfinite(d.rate),
        'corroborated_price_outlier': d.suspect_price if 'suspect_price' in d else np.zeros(len(d),dtype=bool),
    }
    report['flags']={k:int(v.sum()) for k,v in rules.items()}
    report['flags_population']='After duplicate resolution, before sequential validity screens.'
    report['iv_above_300pct_input_flag']=int((d.iv>3).sum())
    keep=np.ones(len(d),dtype=bool)
    for k,mask in rules.items(): drops[k]=int((keep&mask).sum()); keep &= ~mask
    d=d.loc[keep].copy()
    invalid_gamma=(d.gamma<0)|~np.isfinite(d.gamma)
    invalid_theta=~np.isfinite(d.theta)
    report['optional_features_set_missing']={'gamma':int(invalid_gamma.sum()),'theta':int(invalid_theta.sum())}
    d.loc[invalid_gamma,'gamma']=np.nan; d.loc[invalid_theta,'theta']=np.nan
    report['iv_above_300pct_retained_flag']=int((d.iv>3).sum())
    d=d.sort_values(['contract_id','timestamp'])
    previous=d.groupby('contract_id',sort=False).option_price.shift()
    report['unchanged_consecutive_prices']=int(d.option_price.eq(previous).sum())
    report['stale_note']='Unchanged consecutive prices are a proxy only; no quote update timestamps exist.'
    report['drops']=drops; report['cleaned_rows']=len(d)
    assert len(d)+sum(drops.values())==report['input_option_rows']
    return d.reset_index(drop=True),report
