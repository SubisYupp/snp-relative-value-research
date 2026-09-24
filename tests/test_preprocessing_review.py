from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from quant.ingestion import standardize
from quant.cleaning import clean,outcome_quotes
from quant.features import aggregate_market,add_lags,DYNAMIC
from quant.target import match_horizon
from quant.surface import local_smile
from quant.quote_quality import flag_quote_outliers
from quant.config import Config
from quant.preprocess import pipeline_fingerprint,source_digest,validate_source_session

def quotes():
    raw=pd.DataFrame({'time':['2025-01-02 10:00']*7,'expiry':['2025-02-02']*7,
                      'contract':['S']*7,'strike':np.arange(85,120,5,dtype=float),'underlying':[100.]*7,
                      'call_price':[18.,14.,11.,9.,7.,5.,4.],
                      'call_iv':[.2]*7,'put_iv':[.2]*7,'call_delta':[.5]*7,'put_delta':[-.5]*7,
                      'call_vega':[5.]*7,'put_vega':[5.]*7,'call_gamma':[.01]*7,'put_gamma':[.01]*7,
                      'call_theta':[-.1]*7,'put_theta':[-.1]*7})
    raw['put_price']=raw.call_price-100+raw.strike
    return standardize(raw)

def test_identical_duplicates_keep_one_and_reconcile():
    d=quotes(); copied=d.iloc[[0]].copy();copied.source_file='another.csv'
    out,report=clean(pd.concat([d,copied],ignore_index=True))
    assert len(out)==len(d)
    assert report['drops']['identical_duplicate_copies']==1
    assert len(out)+sum(report['drops'].values())==report['input_option_rows']

def test_conflicting_duplicate_excluded_even_if_one_has_invalid_iv():
    d=quotes(); conflict=d.iloc[[0]].copy();conflict['iv']=0
    out,report=clean(pd.concat([d,conflict],ignore_index=True))
    assert len(out)==len(d)-1
    assert report['drops']['ambiguous_duplicate_rows']==2

def test_future_greeks_do_not_control_price_target_eligibility():
    d=quotes().iloc[[0]].copy();future=d.copy();future.timestamp+=pd.Timedelta(hours=1)
    future['iv']=0.;future['vega']=0.;future['delta']=np.nan;future['option_price']+=2
    raw=pd.concat([d,future],ignore_index=True)
    start,_=clean(raw)
    assert len(start)==1
    matched=match_horizon(start,outcomes=outcome_quotes(raw))
    assert matched.iloc[0].target==pytest.approx(.4)

def test_zero_price_is_unusable_even_with_positive_iv():
    d=quotes();d.loc[0,'option_price']=0
    cleaned,report=clean(d)
    assert report['drops']['zero_price_unusable']==1
    assert (cleaned.option_price>0).all()
    assert (outcome_quotes(d).option_price>0).all()

def test_invalid_identity_rate_and_optional_greeks():
    d=quotes();d.loc[0,'strike']=np.inf;d.loc[1,'contract']=None;d.loc[2,'rate']=np.inf
    d.loc[3,'gamma']=-1;d.loc[4,'theta']=np.inf
    out,r=clean(d)
    assert len(out)==len(d)-3
    assert out.gamma.isna().sum()==1 and out.theta.isna().sum()==1

def test_market_atm_has_one_vote_per_series():
    base=pd.DataFrame({'timestamp':pd.to_datetime(['2025-01-01']*3),'contract':['a','b','c'],
                       'expiry':pd.to_datetime(['2025-02-01']*3),'atm_iv':[.1,.2,.3],'underlying':[100.]*3})
    dense=pd.concat([base,*[base.iloc[[0]] for _ in range(100)]],ignore_index=True)
    assert aggregate_market(base).atm_iv.iloc[0]==aggregate_market(dense).atm_iv.iloc[0]==.2

def test_surface_excludes_all_same_strike_quotes():
    k=np.array([-.2,-.1,0,0,0,.1,.2,.3]);iv=.2+k*k
    first=local_smile(k,iv);iv[2:5]=99
    changed=local_smile(k,iv)
    np.testing.assert_allclose(first[2:5],changed[2:5])

def test_surface_does_not_export_negative_expected_iv():
    k=np.linspace(-.2,.2,9);iv=np.array([.01,.01,.01,.01,.01,.01,.01,2.,9.])
    result=local_smile(k,iv)
    assert ((result[:,0]>0)|np.isnan(result[:,0])).all()

def test_severe_price_error_requires_two_contemporaneous_checks():
    d=quotes();d.loc[3,'option_price']=400
    flagged=flag_quote_outliers(d)
    assert flagged.suspect_price.sum()==1 and flagged.loc[3,'suspect_price']
    out,report=clean(flagged)
    assert report['drops']['corroborated_price_outlier']==1
    assert not outcome_quotes(flagged).contract_id.eq(flagged.loc[3,'contract_id']).any()
    assert not flag_quote_outliers(quotes()).suspect_price.any()

def test_lags_do_not_cross_source_calendar_day():
    d=pd.DataFrame({'contract_id':['a','a'],'timestamp':pd.to_datetime(['2025-01-01 23:30','2025-01-02 00:30'])})
    for c in DYNAMIC:d[c]=[1.,2.]
    assert add_lags(d).iv_change.isna().all()
    with pytest.raises(ValueError):add_lags(d,tolerance=60)

def test_side_specific_source_fields_preserved():
    raw=pd.DataFrame({'time':['2025-01-01'],'expiry':['2025-02-01'],'contract':['x'],'strike':[100],
                      'underlying':[100],'call_price':[5],'put_price':[5],'call_bid':[4.8],'put_bid':[4.7]})
    assert standardize(raw).bid.tolist()==[4.8,4.7]

def test_cache_invalidation_by_configuration_and_source_bytes(tmp_path):
    c=Config()
    assert pipeline_fingerprint(c)!=pipeline_fingerprint(replace(c,min_vega=.1))
    p=tmp_path/'source';p.write_bytes(b'123');before=source_digest(p);p.write_bytes(b'124')
    assert source_digest(p)!=before

def test_overlapping_or_multiday_files_fail_explicitly():
    d=quotes();seen=set();validate_source_session(d,seen,'one.parquet')
    with pytest.raises(ValueError,match='overlapping'):validate_source_session(d,seen,'two.parquet')
    d.loc[0,'timestamp']+=pd.Timedelta(days=1)
    with pytest.raises(ValueError,match='partition'):validate_source_session(d,set(),'mixed.parquet')
