import numpy as np
import pandas as pd
import shap

def explain(model,x,batch=10000):
    explainer=shap.TreeExplainer(model)
    values=[]
    for start in range(0,len(x),batch):
        values.append(explainer.shap_values(x.iloc[start:start+batch],check_additivity=False))
    values=np.concatenate(values).astype('float32')
    baseline=float(np.asarray(explainer.expected_value).ravel()[0])
    reconstructed=baseline+values.sum(axis=1)
    if not np.allclose(reconstructed,model.predict(x),atol=1e-4,rtol=1e-4): raise AssertionError('SHAP reconstruction failed')
    importance=pd.DataFrame({'feature':x.columns,'mean_abs_shap':np.abs(values).mean(axis=0)}).sort_values('mean_abs_shap',ascending=False)
    return values,baseline,importance
