import numpy as np
import pandas as pd
from quant.target import match_horizon
from quant.features import add_lags,DYNAMIC
from quant.forwards import parity_forward
from quant.surface import local_smile
from quant.regimes import filter_probabilities
from quant.config import calendar_split
from quant.evaluate import labels,conviction
from quant.ingestion import standardize

def frame():
    return pd.DataFrame({'contract_id':['a']*4,'timestamp':pd.to_datetime(['2025-01-01 10:00','2025-01-01 11:05','2025-01-01 12:00','2025-01-02 10:00']),'option_price':[10.,15.,12.,90.],'underlying':[100.,102.,103.,110.],'delta':[.5]*4,'vega':[2.]*4})

def test_target_and_vega_units():
    d=match_horizon(frame()); assert d.iloc[0].target==2.; assert d.iloc[0].actual_horizon_minutes==65
    assert pd.isna(d.iloc[2].target)

def test_no_contract_cross_match():
    d=frame(); d.loc[1,'contract_id']='b'; assert pd.isna(match_horizon(d).iloc[0].target)

def test_lags_future_invariance():
    d=frame()
    for c in DYNAMIC: d[c]=[1.,2.,3.,999.]
    a=add_lags(d); d.loc[2:,DYNAMIC]=1e6; b=add_lags(d)
    np.testing.assert_allclose(a.iloc[:2].iv_change,b.iloc[:2].iv_change,equal_nan=True)
    assert a.iloc[1].iv_change==1

def test_forward_and_log_moneyness():
    assert parity_forward(100,10,5,0,1)==105
    assert np.isclose(np.log(105/parity_forward(100,10,5,0,1)),0)

def test_surface_leave_one_out():
    k=np.linspace(-.2,.2,31); iv=.2-.3*k+2*k*k
    first=local_smile(k,iv); altered=iv.copy(); altered[15]+=3
    second=local_smile(k,altered)
    assert np.isclose(first[15,0],second[15,0]); assert np.isclose(first[15,0],iv[15],atol=1e-5)

def test_hmm_filter_prefix_and_continuation():
    from hmmlearn.hmm import GaussianHMM
    h=GaussianHMM(n_components=4,covariance_type='diag'); h.startprob_=np.ones(4)/4; h.transmat_=np.full((4,4),.1)+np.eye(4)*.6; h.means_=np.arange(4).reshape(-1,1); h.covars_=np.ones((4,1))*.2
    x=np.array([0.,1.,1.,2.,3.]); allp=filter_probabilities(h,x)
    np.testing.assert_allclose(allp[:3],filter_probabilities(h,x[:3]))
    np.testing.assert_allclose(allp[3:],filter_probabilities(h,x[3:],initial=allp[2]))
    np.testing.assert_allclose(allp.sum(axis=1),1)

def test_calendar_split_and_purge():
    dates=pd.date_range('2025-01-01','2025-12-31',freq='D'); v,t=calendar_split(dates)
    assert v==pd.Timestamp('2025-11-01'); assert t==pd.Timestamp('2025-12-01')
    assert dates[dates<v].max()<dates[(dates>=v)&(dates<t)].min()
    assert dates[(dates>=v)&(dates<t)].max()<dates[dates>=t].min()

def test_labels_and_conviction():
    assert labels([-2,-1,0,1,2],1).tolist()==['RICH','NEUTRAL','NEUTRAL','NEUTRAL','CHEAP']
    np.testing.assert_allclose(conviction([-3,0,1],[0,1,2,3]),[100,25,50])

def test_mixed_put_signs():
    d=pd.DataFrame({'time':['2025-01-01']*2,'expiry':['2025-02-01']*2,'strike':[100,101],'underlying':[100]*2,'contract':['a']*2,'call_price':[2]*2,'put_price':[2]*2,'call_delta':[.5]*2,'put_delta':[.5,-.5],'call_vega':[1]*2,'put_vega':[1]*2})
    result=standardize(d); assert (result[result.option_type=='P'].delta==-.5).all()
