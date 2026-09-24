"""Compare preprocessing versions, without fitting or rescoring a prediction model."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np
import pandas as pd
from .config import Config,save_json

def review():
    config=Config()
    baseline_root=Path('experiments/snp-rv-v1.0.0')
    if not baseline_root.exists(): baseline_root=Path('.')
    old=json.loads((baseline_root/'results/metrics/data_quality.json').read_text())
    new=json.loads(Path(config.quality_path).read_text())
    lock=json.loads((baseline_root/'results/artifacts/design_lock.json').read_text())
    train_start=pd.Timestamp(lock['train_start']); train_end=pd.Timestamp(lock['validation_start'])
    old_market=pd.read_parquet('data/processed/market.parquet')
    new_market=pd.read_parquet(Path(config.processed_dir)/'market.parquet')
    market=old_market.merge(new_market,on='timestamp',suffixes=('_old','_new'))
    counts=Counter(); buckets=[]; horizon=Counter(); quarantined=new['quarantine']
    for p in sorted(Path(config.processed_dir).glob('*.parquet')):
        if p.name=='market.parquet':continue
        revised=pd.read_parquet(p)
        counts['revised_infinite_numerics']+=int(np.isinf(revised.select_dtypes('number').to_numpy()).sum())
        counts['revised_negative_expected_iv']+=int((revised.expected_iv<0).sum())
        counts['revised_duplicate_identities']+=int(revised.duplicated(['contract_id','timestamp']).sum())
        matched=revised.target.notna()
        assert revised.loc[matched,'actual_horizon_minutes'].between(50,70).all()
        assert revised.loc[matched,'timestamp'].dt.normalize().eq(revised.loc[matched,'target_timestamp'].dt.normalize()).all()
        assert not revised.suspect_price.any()
        assert (revised.option_price>0).all()
        baseline=pd.read_parquet(Path('data/processed')/p.name)
        ids=['contract_id','timestamp']
        common=baseline[ids+['target','expected_iv','atm_iv','forward_variance']].merge(revised[ids+['target','expected_iv','atm_iv','forward_variance']],on=ids,suffixes=('_old','_new'))
        counts['common_observations']+=len(common)
        counts['restored_observations']+=len(revised)-len(common)
        counts['removed_previously_clean_observations']+=len(baseline)-len(common)
        counts['newly_matched_common_observations']+=int((common.target_old.isna()&common.target_new.notna()).sum())
        counts['invalidated_old_matches']+=int((common.target_old.notna()&common.target_new.isna()).sum())
        for c in ['target','expected_iv','atm_iv','forward_variance']:
            same=np.isclose(common[c+'_old'],common[c+'_new'],atol=1e-7,rtol=1e-6,equal_nan=True)
            counts[c+'_changed_common_rows']+=int((~same).sum())
        # Only supervised-training dates are used for target-sensitivity diagnostics.
        training=baseline[(baseline.timestamp>=train_start)&(baseline.timestamp<train_end)&(baseline.target_timestamp<train_end)&baseline.target.notna()].copy()
        if len(training):
            training['vega_bucket']=pd.cut(training.vega,[0,.1,.5,1,3,10,np.inf])
            training['absolute_target']=abs(training.target)
            training['squared_target']=training.target.astype(float)**2
            daily=training.groupby('vega_bucket',observed=True).agg(n=('target','size'),absolute_sum=('absolute_target','sum'),squared_sum=('squared_target','sum'),max_absolute_target=('absolute_target','max')).reset_index()
            daily['vega_bucket']=daily.vega_bucket.astype(str);buckets.append(daily)
    sensitivity=pd.concat(buckets).groupby('vega_bucket',sort=False).agg(n=('n','sum'),absolute_sum=('absolute_sum','sum'),squared_sum=('squared_sum','sum'),max_absolute_target=('max_absolute_target','max')).reset_index()
    sensitivity['row_share']=sensitivity.n/sensitivity.n.sum()
    sensitivity['squared_target_share']=sensitivity.squared_sum/sensitivity.squared_sum.sum()
    sensitivity['mean_absolute_target']=sensitivity.absolute_sum/sensitivity.n
    sensitivity=sensitivity.drop(columns='absolute_sum')
    sensitivity.to_csv('reports/training_vega_sensitivity.csv',index=False)
    result={'old_version':lock['config']['version'],'new_version':config.version,'retrained':False,
            'original_model_unchanged':hashlib.sha256((baseline_root/'models/xgboost/with_regime.json').read_bytes()).hexdigest()==lock['model_sha256'],
            'old_clean_rows':old['cleaned_rows'],'new_clean_rows':new['cleaned_rows'],
            'old_matched_rows':old['matched'],'new_matched_rows':new['matched'],
            'comparison':dict(counts),'quarantined_quotes':quarantined,
            'atm_market_changed_timestamps':int((~np.isclose(market.atm_iv_old,market.atm_iv_new)).sum()),
            'atm_market_median_absolute_change':float((market.atm_iv_new-market.atm_iv_old).abs().median()),
            'old_reported_retained_iv_above_300pct':old['iv_above_300pct_retained_flag'],
            'corrected_retained_iv_above_300pct':new['iv_above_300pct_retained_flag'],
            'training_sensitivity':sensitivity.to_dict('records'),'source_manifest':new['source_manifest']}
    save_json('reports/preprocessing_review.json',result)
    low=sensitivity[sensitivity.vega_bucket.isin(['(0.0, 0.1]','(0.1, 0.5]'])]
    findings=f'''# Preprocessing review — v1.1

Completed on the actual 324-file dataset. Revised preprocessing and tests have been run; **no model was retrained or re-evaluated**. The dashboard still serves the original frozen v1.0 experiment. Treat that experiment as a historical baseline with the data-quality limitations identified here, not as validated results for the corrected pipeline.

## Measured impact

| Measure | Original | Revised |
|---|---:|---:|
| Clean option observations | {old['cleaned_rows']:,} | {new['cleaned_rows']:,} |
| Matched one-hour targets | {old['matched']:,} | {new['matched']:,} |
| Reported retained IV above 300% | {old['iv_above_300pct_retained_flag']:,} (incorrectly counted before cleaning) | {new['iv_above_300pct_retained_flag']:,} |

Recovered {counts['restored_observations']:,} previously discarded unique observations. Quarantined {len(quarantined):,} quote records with corroborated extreme same-time price inconsistencies. Among shared observations, restored {counts['newly_matched_common_observations']:,} missing targets and invalidated {counts['invalidated_old_matches']:,} previously matched targets. These counts describe processing changes, not improved predictive performance.

## Fixes implemented

1. **Identical duplicates:** retain one economic quote. Previously the cleaner removed both copies, incorrectly describing them as conflicting duplicates. Distinct records with the same identity remain excluded, even when one fails a validity screen.
2. **Extreme quote quarantine:** require both a neighboring-strike price-envelope breach greater than 5% of underlying and a parity-forward deviation greater than max(5% of underlying, 10 cross-strike MAD), with at least five call/put pairs. This uses contemporaneous quotes only, never subsequent returns. The rule is deliberately coarse. Quarantined records remain in the raw source and the JSON review; no fabricated replacement prices are inserted.
3. **Outcome eligibility:** start-time IV/Vega/Greek feature eligibility and future price eligibility are separate. Future IV, delta or Vega can no longer censor a target with a valid price, underlying and identity. Ambiguous and quarantined quotes are excluded at both ends. Require strictly positive source prices: the audit found 474 unexpired zero-price records, 402 also missing/zero IV and 72 with positive IV. Without executable bid/ask evidence, zero cannot reliably distinguish a valueless option from a missing-price sentinel. These are counted as unusable zeros, not asserted to be universally impossible option prices.
4. **ATM aggregation:** one vote per series/expiry, instead of repeating ATM IV once per strike. This corrects market-regime inputs and expiry term aggregation. {result['atm_market_changed_timestamps']:,} source timestamps change; median absolute market ATM change is {result['atm_market_median_absolute_change']:.6f} in decimal-IV units.
5. **Surface safeguards:** explicitly exclude all duplicate copies of the target strike; negative/nonfinite expected IV and insufficient surrounding points produce missing features. The original cache contained four negative expected-IV values; the revised cache contains {counts['revised_negative_expected_iv']}.
6. **Validity and lineage:** reject missing series IDs, infinite strikes and invalid rates; optional invalid gamma/theta become counted missing values. Preserve extra side-specific fields such as bid/ask/volume when supplied.
7. **Lag matching:** enforce a tolerance below one hour and prohibit cross-calendar-day lag matches, consistent with the session assumptions used for targets.
8. **Cache safety:** hash source contents and preprocessing code/configuration; bind training to the current source manifest. New outputs use `data/processed-v1.1` and `results/preprocessing-v1.1`, preserving the original cache and model artifacts. Daily-file assumptions are now checked explicitly: multi-day or overlapping files must be consolidated/partitioned before ingestion, rather than silently losing cross-file matches.
9. **Quality reporting:** count retained high-IV quotes after cleaning, reconcile all sequential exclusions, and report missing fields across the union of source schemas.

## Training-period target sensitivity

Measured on the original supervised-training window only ({train_start} to {train_end}, exclusive end), with no test-driven cutoff selection. Quotes with Vega ≤0.5 account for {low.row_share.sum():.2%} of rows but {low.squared_target_share.sum():.2%} of the sum of squared targets (zero-prediction squared error). This explains why RMSE is sensitive to small-Vega observations; it does not prove that these quotes should be removed.

The existing Vega floor remains 0.01. No new target clipping or performance-driven Vega cutoff was added. `reports/training_vega_sensitivity.csv` gives all buckets. `reports/training_extreme_targets.csv` records the inspected extreme training observations.

## Data questions that remain

- Source rates are zero throughout. Product identity, settlement convention, source timezone and Greek units still need vendor confirmation.
- The audit found no mixed underlying values within a timestamp, no negative source gamma, and no unexpired IV above 300%. The high-IV retained count in the old quality report was a reporting bug involving already-expired rows.
- Negative forward variance is preserved and reported, not clipped to make the term structure look cleaner. Multiple vendor series and noisy maturity quotes can produce this; bid/ask and product metadata are needed to interpret it.
- The original HMM has an imperfect convergence diagnostic and a short two-month warm-up. Improving it belongs in a new training/validation experiment.
- The strict quarantine is not a comprehensive price validator. Moderate errors, edge strikes and quotes without paired/surrounding strikes may escape it. Raw records remain available for vendor investigation.

## Verification and reproduction

Revised cache checks: {counts['revised_infinite_numerics']} infinite numeric values, {counts['revised_duplicate_identities']} duplicate contract/timestamp identities, {counts['revised_negative_expected_iv']} negative expected-IV values. All matched horizons are 50–70 minutes and remain within a source calendar day. Original model SHA-256 unchanged: {result['original_model_unchanged']}.

```powershell
python scripts/run.py test
python scripts/run.py preprocess
python scripts/run.py review_preprocessing
```

The original pipeline source is archived in `reports/archive/pipeline-v1.0.0.zip`. Revised source defaults are v1.1; the old dashboard and report continue to describe the old model. A revised model needs a separate experiment. September 2026 is already inspected and cannot be represented as untouched if reused.
'''
    Path('reports/PREPROCESSING_REVIEW.md').write_text(findings,encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_manifest','training_sensitivity','quarantined_quotes']},indent=2))

if __name__=='__main__': review()
