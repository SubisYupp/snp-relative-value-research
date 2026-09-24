import numpy as np
from scipy.special import logsumexp
from hmmlearn.hmm import GaussianHMM

def filter_probabilities(model,x,initial=None):
    """Forward recursion only. Unlike predict_proba, never invokes backward smoothing."""
    x=np.asarray(x).reshape(-1,1)
    emission=model._compute_log_likelihood(x)
    transition=np.log(np.maximum(model.transmat_,1e-300))
    previous=np.log(np.maximum(model.startprob_ if initial is None else initial,1e-300))
    result=[]
    for i,row in enumerate(emission):
        prior=previous if i==0 and initial is None else logsumexp(previous[:,None]+transition,axis=0)
        previous=prior+row; previous-=logsumexp(previous)
        result.append(np.exp(previous))
    return np.asarray(result)

def fit_regimes(atm,warmup_end,seed=42):
    train=atm[atm.timestamp<warmup_end].atm_iv.to_numpy()
    if len(train)<80: raise ValueError('Insufficient warm-up observations for four-state HMM')
    x=np.log(train).reshape(-1,1)
    models=[]
    for offset in range(3):
        model=GaussianHMM(n_components=4,covariance_type='diag',n_iter=200,tol=1e-4,random_state=seed+offset,min_covar=1e-4)
        model.fit(x); models.append((model.score(x),model))
    model=max(models,key=lambda z:z[0])[1]
    order=np.argsort(model.means_.ravel())
    probabilities=filter_probabilities(model,np.log(atm.atm_iv.to_numpy()))[:,order]
    result=atm[['timestamp','atm_iv']].copy()
    for j,name in enumerate(['p_low','p_normal','p_high','p_crisis']): result[name]=probabilities[:,j]
    result['regime_entropy']=-np.sum(probabilities*np.log(np.maximum(probabilities,1e-300)),axis=1)
    result['regime']=np.array(['LOW','NORMAL','HIGH','CRISIS'])[probabilities.argmax(axis=1)]
    history=list(model.monitor_.history)
    delta=history[-1]-history[-2] if len(history)>1 else None
    metadata={'fit_end_exclusive':str(warmup_end),'state_mean_iv':np.exp(model.means_.ravel()[order]),'transition_matrix':model.transmat_[order][:,order], 'fit_observations':len(x),'converged':bool(model.monitor_.converged and delta is not None and delta>=0),'library_converged':bool(model.monitor_.converged),'last_log_likelihood_change':delta,'convergence_note':'EM stopped after a small likelihood decrease; treat convergence as imperfect.' if delta is not None and delta<0 else 'EM convergence diagnostic recorded.','method':'Training-only two-month warm-up; parameters frozen before supervised training; forward filtering only.'}
    return model,result,metadata
