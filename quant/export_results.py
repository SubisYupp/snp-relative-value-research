from pathlib import Path
import html
import json
import numpy as np
import pandas as pd
from .config import save_json

def records(d):
    return json.loads(d.to_json(orient='records',date_format='iso'))

def research_conclusions(metrics,otm,importance,quality):
    t=metrics['test']; b=metrics['baselines']['test']; a=metrics['ablation']['test_without_regime']
    fmt=lambda x: 'unavailable' if x is None else f'{x:.4f}'
    out=[]
    for name in ['zero','ridge','relative_iv','surface_residual']:
        out.append(f"XGBoost {'beats' if t['rmse']<b[name]['rmse'] else 'does not beat'} {name} on test RMSE ({fmt(t['rmse'])} versus {fmt(b[name]['rmse'])}).")
    out.append(f"Regime features {'improved' if t['rmse']<a['rmse'] else 'did not improve'} test RMSE; with regimes {fmt(t['rmse'])}, without {fmt(a['rmse'])}. Validation comparison is reported separately.")
    out.append(f"Unseen-test Spearman IC is {fmt(t['spearman'])}; overall directional hit rate is {t['sign_hit_rate']:.1%}.")
    out.append('Conviction-decile hit rates increase monotonically.' if t['conviction_monotone'] else 'Hit rate does not increase monotonically with conviction; magnitude is not a calibrated probability.')
    hi=t['high_conviction_hit_rate']; active=t['non_neutral_hit_rate']
    out.append(f"Top-10%-conviction active hit rate: {fmt(hi)}; all active signals: {fmt(active)}. {'Higher conviction is descriptively stronger.' if hi is not None and active is not None and hi>active else 'Higher conviction has not established stronger directional performance.'} No independent probability calibration or significance claim is made.")
    spread=t['ranking']['0.1']; out.append(f"Cheap-minus-rich daily-mean decile spread is {fmt(spread['mean'])}, HAC standard error {fmt(spread['standard_error'])}, across {spread['n_days']} days. Cumulative chart is an additive research score, not portfolio wealth.")
    valid=[r for r in t['regimes'] if r.get('spearman') is not None]
    if valid:
        out.append(f"By Spearman IC, strongest observed regime: {max(valid,key=lambda r:r['spearman'])['regime']}; weakest: {min(valid,key=lambda r:r['spearman'])['regime']}. Sparse and absent regimes cannot be generalized.")
    out.append(otm['conclusion'])
    out.append('Largest global SHAP drivers: '+', '.join(importance.feature.head(5))+'. These explain model behavior, not causal effects.')
    out.append(f"Cleaning removed {quality['dropped']:,} of {quality['input_option_rows']:,} option rows. One-hour matches: {quality['target_match_pct']:.2f}% of clean observations. Unmatched observations remain inspectable but are excluded from outcome metrics.")
    out.append('Failure boundaries: short partial test month; potentially extreme low-Vega targets; unverified product identity, timestamps and vendor Greek conventions; no bid/ask, costs or execution model; frozen warm-up HMM may not represent later volatility regimes.')
    works=t['rmse']<b['zero']['rmse'] and t['spearman'] is not None and t['spearman']>0 and spread['mean'] is not None and spread['mean']>0
    out.append('Evidence supports further controlled research, not a profitability claim.' if works else 'The experiment does not establish a reliable advantage over the simple benchmarks. Treat the signals as exploratory; the model is not validated for trading.')
    return out

def export_results(lock,metrics,quality,test,regimes,verticals,shap_values,baseline,importance,distribution,correlation,otm):
    dest=Path('dashboard/public/data'); dest.mkdir(parents=True,exist_ok=True)
    conclusions=research_conclusions(metrics,otm,importance,quality)
    if lock.get('historical_reevaluation'):
        conclusions=[lock['evaluation_status']]+[c.replace('Unseen-test','Revisited September') for c in conclusions]
    if lock['hmm'].get('last_log_likelihood_change',0)<0:
        conclusions.append('HMM numerical diagnostic: the selected warm-up fit stopped after a small log-likelihood decrease. Its fitted parameters and all predictions remain frozen; convergence is imperfect and should be investigated in a separate future experiment.')
    sample=test.sample(min(4000,len(test)),random_state=42)
    chart_columns=['timestamp','strike','moneyness','iv','relative_iv','surface_residual','prediction','target','conviction','regime','delta','label']
    manifest=[]
    # Separate each complete timestamp: browser downloads only the selected cross section.
    for stamp,g in test.groupby('timestamp',sort=True):
        key=stamp.strftime('%Y%m%dT%H%M%S'); ids=g.index.to_numpy()
        payload={'rows':records(g),'shap':np.round(shap_values[ids],7).tolist(),'baseline':baseline,'features':lock['features']}
        save_json(dest/'snapshots'/f'{key}.json',payload)
        manifest.append({'timestamp':str(stamp),'key':key,'rows':len(g),'expiries':sorted(g.expiry.astype(str).unique().tolist())})
    # Full source quality is also retained outside the compact browser bundle.
    raw_audit=json.loads(Path('reports/raw_audit.json').read_text()) if Path('reports/raw_audit.json').exists() else {}
    quality_browser={k:v for k,v in quality.items() if k not in ['daily','source_files']}
    daily_quantiles=quality.get('daily',[])
    quality_browser['daily_distributions']=[{'file':Path(f).stem,'dte':d['dte_quantiles'],'vega':d['vega_quantiles']} for f,d in zip(quality.get('source_files',[]),daily_quantiles)]
    strike_counts={}
    for d in daily_quantiles:
        for e,n in d['strikes_per_expiry'].items(): strike_counts[str(e)]=max(n,strike_counts.get(str(e),0))
    quality_browser['strikes_per_expiry']=strike_counts
    quality_browser['raw_audit']={k:v for k,v in raw_audit.items() if k not in ['schemas','put_delta_signs']}
    sample_ids=sample.index.to_numpy()
    global_shap={'features':lock['features'],'importance':records(importance),'values':np.round(shap_values[sample_ids],7).tolist(),'feature_values':records(sample[lock['features']]),'baseline':baseline,'sample_note':f'Deterministic sample of {len(sample):,} test rows; global importance uses every test observation.'}
    shifts=[]
    for feature,parts in distribution.items():
        lo,_,median,_,hi=parts['train']; testmedian=parts['test'][2]
        shifts.append({'feature':feature,'train_median':median,'test_median':testmedian,'shift_over_train_90pct_range':abs(testmedian-median)/(hi-lo) if hi>lo else None})
    bundle={'lock':lock,'metrics':metrics,'quality':quality_browser,'conclusions':conclusions,'snapshots':manifest,'sample':records(sample[chart_columns]),'importance':records(importance),'distribution':distribution,'correlation':correlation,'shifts':shifts,'otm':otm,'test_total_rows':len(test),'units':'Delta-hedged option-price points / source Vega (assumed per 1 volatility percentage point)','limitations':['Research signals before costs; no execution or profitability claim.','Source wall-clock timezone and precise SNP product identity are unverified.','September is a partial month. Regime names describe fitted warm-up states, not fixed risk thresholds.']}
    save_json(dest/'summary.json',bundle); save_json(dest/'shap.json',global_shap)
    save_json(dest/'regimes.json',{'rows':records(regimes),'metadata':lock['hmm']})
    save_json(dest/'verticals.json',records(verticals))
    save_json('reports/model_report.json',{'conclusions':conclusions,**bundle})
    rows=''.join(f'<tr><td>{html.escape(name)}</td><td>{m["rmse"]:.6f}</td><td>{m["mae"]:.6f}</td><td>{m.get("spearman") if m.get("spearman") is not None else "—"}</td></tr>' for name,m in {'XGBoost + regime':metrics['test'],**metrics['baselines']['test'],'XGBoost without regime':metrics['ablation']['test_without_regime']}.items())
    page='<!doctype html><html><meta charset="utf-8"><title>SNP relative-value research</title><style>body{background:#111519;color:#dce2e5;font:16px/1.7 system-ui;max-width:1040px;margin:60px auto;padding:20px}h1{font-size:38px}small{color:#93a5b0}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #35404a;text-align:left}li{margin:16px 0}a{color:#8db9cb}</style>'
    page+=f'<small>RESEARCH / {lock["config"]["version"]}</small><h1>SNP options · relative value</h1><p>Validation: {lock["validation_start"]} – {lock["validation_end"]}<br>Test: {lock["test_start"]} – {lock["dataset_end"]}<br>Source period: {lock["dataset_start"]} – {lock["dataset_end"]}</p>'
    page+='<h2>Findings</h2><ol>'+''.join('<li>'+html.escape(c)+'</li>' for c in conclusions)+'</ol><h2>Unseen-test benchmarks</h2><table><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>Spearman</th></tr>'+rows+'</table><h2>Method and audit</h2><p>All choices frozen before final-test evaluation. Same-session nearest one-hour matching within ten minutes. HMM trained on the initial two months only; supervised training starts afterward. Leave-one-strike-out local quadratic surfaces. Robust XGBoost regression. Neutral band and conviction ranks use validation predictions only. Daily HAC spread errors account for cross-sectional dependence and short-term serial correlation.</p><p>See README.md, DATA_AUDIT.md, model_report.json and results/artifacts/design_lock.json for exact configuration, assumptions and complete metrics.</p></html>'
    if lock.get('historical_reevaluation'):
        page=page.replace('Unseen-test benchmarks','Revised historical benchmarks').replace('All choices frozen before final-test evaluation.','Training and validation remain chronological. September was inspected in the earlier experiment; this rerun is not a fresh untouched test.')
    Path('reports/model_report.html').write_text(page,encoding='utf-8')

def main():
    import pandas as pd
    load=lambda p:json.loads(Path(p).read_text())
    lock=load('results/artifacts/design_lock.json'); research=load('results/artifacts/research.json'); explanation=load('results/artifacts/explanation.json')
    export_results(lock,load('results/metrics/metrics.json'),load('results/metrics/data_quality.json'),pd.read_parquet('results/predictions/predictions_test.parquet'),pd.read_parquet('results/artifacts/regimes.parquet'),pd.read_csv('results/metrics/vertical_candidates.csv'),np.load('results/artifacts/shap_test.npy'),explanation['baseline'],pd.read_csv('results/metrics/feature_importance.csv'),research['distribution'],research['correlation'],research['otm'])
if __name__=='__main__': main()
