"""Single reproducible training, validation selection, frozen test evaluation."""
import argparse
import gc
import hashlib
import json
import pickle
import platform
import shutil
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from .config import Config,BASE_FEATURES,REGIME_FEATURES,save_json,calendar_split
from .preprocess import preprocess
from .features import underlying_features
from .regimes import fit_regimes
from .evaluate import regression,evaluate_frame,labels,conviction,otm_diagnostics
from .verticals import vertical_candidates
from .shap_analysis import explain

def archive_previous():
    """Explicit reruns preserve the entire prior published experiment first."""
    marker=Path('results/artifacts/test_evaluation_complete.json')
    if not marker.exists(): return
    previous=json.loads(Path('results/artifacts/design_lock.json').read_text())
    destination=Path('experiments')/previous['config']['version']
    if destination.exists(): raise FileExistsError(f'Archive already exists: {destination}; refusing to overwrite it.')
    destination.mkdir(parents=True)
    for source in ['models','results','dashboard/public/data']:
        shutil.copytree(source,destination/source)
    for source in ['reports/model_report.html','reports/model_report.json','README.md']:
        target=destination/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    archived=destination/'models/xgboost/with_regime.json'
    assert hashlib.sha256(archived.read_bytes()).hexdigest()==previous['model_sha256']
    # The completion marker is archived above; removing only the active copy enables
    # this explicitly requested new experiment, not an implicit repeated holdout.
    marker.unlink()
    print(f'Previous experiment archived at {destination}',flush=True)

def run(config,historical_reevaluation=False):
    for folder in ['models/hmm','models/xgboost','results/predictions','results/metrics','results/artifacts','reports']:
        Path(folder).mkdir(parents=True,exist_ok=True)
    if Path('results/artifacts/test_evaluation_complete.json').exists():
        print('Completed experiment exists. Serving saved results; test is not re-evaluated. Use a new output workspace for a new experiment.',flush=True)
        return
    market,quality=preprocess(config)
    save_json('results/metrics/data_quality.json',quality)
    val_start,test_start=calendar_split(market.timestamp)
    warmup_end=(market.timestamp.min().to_period('M')+config.warmup_months).start_time
    market,rv_metadata=underlying_features(market,val_start)
    hmm,regimes,hmm_metadata=fit_regimes(market[['timestamp','atm_iv']],warmup_end,config.seed)
    with open('models/hmm/model.pkl','wb') as f: pickle.dump(hmm,f)
    regimes.to_parquet('results/artifacts/regimes.parquet',index=False)
    time_features=market[['timestamp','underlying_return','realized_vol']].merge(regimes.drop(columns='atm_iv'),on='timestamp')
    feature_names=BASE_FEATURES+REGIME_FEATURES
    training=[]; validation=[]; test_paths=[]; train_dates=[]
    # The current source manifest prevents accidentally loading orphaned stale cache files.
    data_files=sorted(Path(p) for p in quality['processed_files'])
    for p in data_files:
        # Read only timestamps first; final test targets/features are not loaded for model selection.
        t=pd.read_parquet(p,columns=['timestamp']).timestamp
        if t.min()>=test_start: test_paths.append(p); continue
        d=pd.read_parquet(p).merge(time_features,on='timestamp',how='left')
        end=test_start if t.min()>=val_start else val_start
        d=d[(d.timestamp>=warmup_end)&d.target.notna()&(d.target_timestamp<end)]
        if not len(d): continue
        if t.min()>=val_start: validation.append(d)
        else:
            training.append(d[feature_names+['target']].astype('float32')); train_dates.extend([d.timestamp.min(),d.timestamp.max()])
    train=pd.concat(training,ignore_index=True); del training; gc.collect()
    val=pd.concat(validation,ignore_index=True); del validation; gc.collect()
    assert max(train_dates)<val.timestamp.min()<test_start
    x=train[feature_names]; y=train.target
    print(f'Training {len(train):,}; validation {len(val):,}; test remains unopened for selection',flush=True)
    candidates=[]; models={}
    # Tune B only on validation; ablation A then uses exactly the same settings and tree count.
    for depth in [3,5]:
        model=XGBRegressor(objective='reg:pseudohubererror',huber_slope=1.,base_score=float(y.median()),n_estimators=config.n_estimators,max_depth=depth,learning_rate=.04,min_child_weight=100,subsample=.8,colsample_bytree=.9,reg_lambda=20,reg_alpha=1,gamma=.01,tree_method='hist',max_bin=128,n_jobs=6,random_state=config.seed,eval_metric='mae',early_stopping_rounds=config.early_stopping)
        model.fit(x,y,eval_set=[(val[feature_names],val.target)],verbose=False)
        score=regression(val.target,model.predict(val[feature_names]))
        candidates.append({'depth':depth,'best_iteration':model.best_iteration,**score}); models[depth]=model
        print(f'Validation depth {depth}: MAE {score["mae"]:.6f}, trees {model.best_iteration+1}',flush=True)
    selected=min(candidates,key=lambda r:r['mae']); model=models[selected['depth']]
    # Physically truncate early-stopped trees so SHAP and predict explain the identical model.
    model._Booster=model.get_booster()[:model.best_iteration+1]
    params=model.get_params(); params.update(n_estimators=selected['best_iteration']+1,early_stopping_rounds=None)
    ablation=XGBRegressor(**params); ablation.fit(train[BASE_FEATURES],y,verbose=False)
    baselines={}
    for name,features in [('ridge',feature_names),('relative_iv',['relative_iv']),('surface_residual',['surface_residual'])]:
        pipeline=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=100.))
        pipeline.fit(train[features],y); baselines[name]=(pipeline,features)
    model.save_model('models/xgboost/with_regime.json'); ablation.save_model('models/xgboost/without_regime.json')
    with open('models/baselines.pkl','wb') as f: pickle.dump(baselines,f)
    val['prediction']=model.predict(val[feature_names]); reference=np.sort(abs(val.prediction.to_numpy()))
    threshold=float(np.quantile(reference,config.neutral_quantile))
    np.save('models/conviction_reference.npy',reference)
    val['conviction']=conviction(val.prediction,reference); val['label']=labels(val.prediction,threshold)
    val['row_id']=np.arange(len(val)); val_metrics=evaluate_frame(val,threshold)
    baseline_validation={'zero':regression(val.target,np.zeros(len(val)))}
    for name,(pipeline,features) in baselines.items(): baseline_validation[name]=regression(val.target,pipeline.predict(val[features]))
    val_a=val.copy(); val_a['prediction']=ablation.predict(val_a[BASE_FEATURES]); ref_a=abs(val_a.prediction.to_numpy()); threshold_a=float(np.quantile(ref_a,config.neutral_quantile)); val_a['conviction']=conviction(val_a.prediction,ref_a)
    ablation_validation=evaluate_frame(val_a,threshold_a)
    distribution={}
    for c in feature_names:
        distribution[c]={'train':train[c].quantile([.05,.25,.5,.75,.95]).tolist(),'validation':val[c].quantile([.05,.25,.5,.75,.95]).tolist()}
    train_corr=train[feature_names].corr().to_numpy().tolist()
    lock={'config':asdict(config),'features':feature_names,'selected':selected,'candidates':candidates,'neutral_threshold':threshold,'neutral_method':'Median absolute validation prediction; symmetric 50th percentile band. No claim of economic cost coverage.','rv':rv_metadata,'hmm':hmm_metadata,'validation_start':str(val_start),'test_start':str(test_start),'warmup_end':str(warmup_end),'train_start':str(min(train_dates)),'train_end':str(max(train_dates)),'dataset_start':str(market.timestamp.min()),'dataset_end':str(market.timestamp.max()),'validation_end':str(val.timestamp.max()),'python':platform.python_version(),'training_rows':len(train),'validation_rows':len(val),'test_read_for_selection':False}
    lock['model_sha256']=hashlib.sha256(Path('models/xgboost/with_regime.json').read_bytes()).hexdigest()
    lock['historical_reevaluation']=historical_reevaluation
    lock['evaluation_status']='Revised historical evaluation; September was inspected previously, not a fresh untouched holdout.' if historical_reevaluation else 'Original chronological holdout evaluation.'
    save_json('results/artifacts/design_lock.json',lock)
    print('Design locked to disk. Opening final test for the single frozen evaluation.',flush=True)
    del x,y,train,models,val_a; gc.collect()
    test=pd.concat([pd.read_parquet(p) for p in test_paths],ignore_index=True).merge(time_features,on='timestamp',how='left')
    assert val.timestamp.max()<test.timestamp.min()
    test['row_id']=np.arange(len(test)); test['prediction']=model.predict(test[feature_names]); test['conviction']=conviction(test.prediction,reference); test['label']=labels(test.prediction,threshold)
    test_metrics=evaluate_frame(test,threshold)
    test_a=test.copy(); test_a['prediction']=ablation.predict(test[BASE_FEATURES]); test_a['conviction']=conviction(test_a.prediction,ref_a)
    ablation_test=evaluate_frame(test_a,threshold_a); del test_a
    matched=test.target.notna(); baseline_test={'zero':regression(test.loc[matched,'target'],np.zeros(matched.sum()))}
    for name,(pipeline,features) in baselines.items(): baseline_test[name]=regression(test.target,pipeline.predict(test[features]))
    for c in feature_names: distribution[c]['test']=test[c].quantile([.05,.25,.5,.75,.95]).tolist()
    metrics={'validation':val_metrics,'test':test_metrics,'baselines':{'validation':baseline_validation,'test':baseline_test},'ablation':{'validation_without_regime':ablation_validation,'test_without_regime':ablation_test}}
    save_json('results/metrics/metrics.json',metrics)
    print('Computing SHAP for every test observation',flush=True)
    shap_values,baseline,importance=explain(model,test[feature_names])
    np.save('results/artifacts/shap_test.npy',shap_values); importance.to_csv('results/metrics/feature_importance.csv',index=False)
    test.to_parquet('results/predictions/predictions_test.parquet',index=False)
    val.to_parquet('results/predictions/predictions_validation.parquet',index=False)
    verticals=vertical_candidates(test); verticals.to_csv('results/metrics/vertical_candidates.csv',index=False)
    pd.DataFrame(test_metrics['conviction_deciles']).to_csv('results/metrics/conviction_analysis.csv',index=False)
    pd.DataFrame(test_metrics['regimes']).to_csv('results/metrics/regime_metrics.csv',index=False)
    otm=otm_diagnostics(test)
    save_json('results/artifacts/explanation.json',{'baseline':baseline,'features':feature_names})
    save_json('results/artifacts/research.json',{'distribution':distribution,'correlation':train_corr,'otm':otm})
    from .export_results import export_results
    export_results(lock,metrics,quality,test,regimes,verticals,shap_values,baseline,importance,distribution,train_corr,otm)
    save_json('results/artifacts/test_evaluation_complete.json',{'model_sha256':lock['model_sha256'],'test_rows':len(test),'matched_rows':int(matched.sum()),'complete':True})
    print('Research run complete. Open reports/model_report.html or start the dashboard.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--raw-dir',default='SNP data'); parser.add_argument('--tolerance-minutes',type=int,default=10)
    parser.add_argument('--archive-and-rerun',action='store_true')
    args=parser.parse_args()
    if args.archive_and_rerun: archive_previous()
    run(Config(raw_dir=args.raw_dir,tolerance_minutes=args.tolerance_minutes),historical_reevaluation=args.archive_and_rerun)
