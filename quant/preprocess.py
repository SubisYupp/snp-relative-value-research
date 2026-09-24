from pathlib import Path
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import time
import pandas as pd
import numpy as np
from .config import Config,save_json,calendar_split
from .ingestion import read_file
from .cleaning import clean,outcome_quotes
from .quote_quality import flag_quote_outliers
from .forwards import add_forwards
from .surface import add_surface
from .features import add_lags,aggregate_market
from .target import match_horizon

def pipeline_fingerprint(config):
    digest=hashlib.sha256(json.dumps(asdict(config),sort_keys=True).encode())
    for name in ['config','ingestion','cleaning','quote_quality','forwards','surface','features','target','preprocess']:
        digest.update(Path(__file__).with_name(name+'.py').read_bytes())
    return digest.hexdigest()

def source_digest(path):
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()

def validate_source_session(d,seen,source):
    dates=d.timestamp.dropna().dt.normalize().unique()
    if len(dates)!=1:
        raise ValueError(f'{source}: expected one source-calendar day per file; partition multi-day files before preprocessing.')
    session=str(pd.Timestamp(dates[0]))
    if session in seen: raise ValueError(f'{source}: overlapping source day {session}; consolidate files so cross-file duplicates and horizon matches are not silently missed.')
    seen.add(session)
    return session

def preprocess(config=Config()):
    raw=sorted([*Path(config.raw_dir).glob('*.parquet'),*Path(config.raw_dir).glob('*.csv')])
    if not raw: raise FileNotFoundError(config.raw_dir)
    processed=Path(config.processed_dir)
    processed.mkdir(parents=True,exist_ok=True)
    if (processed/'market.parquet').exists() and not (processed/'pipeline_manifest.json').exists():
        raise ValueError('Unversioned existing output directory. Choose a new processed_dir to preserve the original experiment.')
    if len({p.stem for p in raw})!=len(raw): raise ValueError('Source filename stems must be unique across formats.')
    all_reports=[]; market=[]; total=0; start=time.time()
    fingerprint=pipeline_fingerprint(config)
    seen=set(); processed_files=[]; quarantine=[]
    for i,f in enumerate(raw):
        cache=processed/(f.stem+'.parquet'); meta=cache.with_suffix('.json')
        processed_files.append(str(cache))
        source_stat=[f.stat().st_size,f.stat().st_mtime_ns]
        source_sha256=source_digest(f)
        if cache.exists() and meta.exists():
            report=json.loads(meta.read_text())
            if report.get('fingerprint')==fingerprint and report.get('source_sha256')==source_sha256:
                if report['source_session'] in seen: raise ValueError('Overlapping cached source sessions')
                seen.add(report['source_session'])
                all_reports.append(report); market.extend(report['market']); quarantine.extend(report['quarantine']);total+=report['cleaned_rows']; continue
        raw_d,raw_count=read_file(f)
        session=validate_source_session(raw_d,seen,f)
        raw_d=flag_quote_outliers(raw_d)
        suspect=raw_d.loc[raw_d.suspect_price,['timestamp','contract','expiry','strike','option_type','option_price','underlying','iv','vega','source_file']].astype(str).to_dict('records')
        quarantine.extend(suspect)
        outcomes=outcome_quotes(raw_d)
        d,report=clean(raw_d,config.min_vega)
        d=add_forwards(d); d=add_surface(d,config.surface_neighbors); d=add_lags(d,config.tolerance_minutes)
        d=match_horizon(d,config.horizon_minutes,config.tolerance_minutes,outcomes=outcomes)
        m=aggregate_market(d)
        report.update(source_sha256=source_sha256,source_session=session,quarantine=suspect,outcome_eligible_quotes=len(outcomes),unavailable_smile_rows=int(d.expected_iv.isna().sum()),negative_forward_variance=int((d.forward_variance<0).sum()))
        report.update(raw_rows=raw_count,matched=int(d.target.notna().sum()),horizon_quantiles=d.actual_horizon_minutes.quantile([.05,.5,.95]).to_dict(),horizon_counts=d.actual_horizon_minutes.value_counts().to_dict(),forward_fallback=int(d.forward_fallback.sum()),fingerprint=fingerprint,source_stat=source_stat,market=[{**r,'timestamp':str(r['timestamp'])} for r in m.to_dict('records')],dte_quantiles=d.dte.quantile([0,.05,.5,.95,1]).to_dict(),vega_quantiles=d.vega.quantile([0,.05,.5,.95,1]).to_dict(),strikes_per_expiry=d.groupby('expiry').strike.nunique().to_dict())
        for c in d.select_dtypes('float64'): d[c]=d[c].astype('float32')
        d.to_parquet(cache,index=False); save_json(meta,report)
        all_reports.append(report); market.extend(report['market']); total+=len(d)
        if i%10==0: print(f'Preprocess {i+1}/{len(raw)} | {total:,} options | {time.time()-start:.0f}s',flush=True)
    m=pd.DataFrame(market); m['timestamp']=pd.to_datetime(m.timestamp); m=m.groupby('timestamp',as_index=False).median(numeric_only=True).sort_values('timestamp')
    m.to_parquet(processed/'market.parquet',index=False)
    agg={k:sum(r[k] for r in all_reports) for k in ['raw_rows','input_option_rows','cleaned_rows','matched','exact_duplicates','duplicate_contract_timestamp','unchanged_consecutive_prices','forward_fallback','iv_above_300pct_retained_flag']}
    for name in ['drops','flags','horizon_counts']:
        values=Counter()
        for r in all_reports: values.update(r[name])
        agg[name]=dict(values)
    fields=set().union(*(r['missing_pct'] for r in all_reports))
    agg['missing_pct']={k:sum(r['missing_pct'].get(k,100)*r['input_option_rows'] for r in all_reports)/agg['input_option_rows'] for k in fields}
    agg['target_match_pct']=100*agg['matched']/agg['cleaned_rows']
    agg['unmatched']=agg['cleaned_rows']-agg['matched']; agg['dropped']=agg['input_option_rows']-agg['cleaned_rows']
    agg['daily']=[{k:r[k] for k in ['raw_rows','cleaned_rows','matched','dte_quantiles','vega_quantiles','strikes_per_expiry']} for r in all_reports]
    agg['source_files']=[str(f) for f in raw]
    agg['processed_files']=processed_files
    agg['pipeline_fingerprint']=fingerprint
    agg['source_manifest']=[{'source':str(f),'sha256':r['source_sha256']} for f,r in zip(raw,all_reports)]
    agg['quarantine']=quarantine
    agg['iv_above_300pct_input_flag']=sum(r['iv_above_300pct_input_flag'] for r in all_reports)
    agg['unavailable_smile_rows']=sum(r['unavailable_smile_rows'] for r in all_reports)
    agg['negative_forward_variance']=sum(r['negative_forward_variance'] for r in all_reports)
    agg['optional_features_set_missing']={k:sum(r['optional_features_set_missing'][k] for r in all_reports) for k in ['gamma','theta']}
    assert sum(agg['drops'].values())+agg['cleaned_rows']==agg['input_option_rows']
    save_json(config.quality_path,agg)
    save_json(processed/'pipeline_manifest.json',{'config':asdict(config),'pipeline_fingerprint':fingerprint,'processed_files':processed_files,'source_manifest':agg['source_manifest']})
    print(f'Preprocessing complete: {total:,} clean options',flush=True)
    return m,agg

if __name__=='__main__': preprocess()
