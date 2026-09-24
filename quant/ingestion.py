from pathlib import Path
import pandas as pd
import numpy as np

ALIASES={'timestamp':['timestamp','time','datetime','quote_datetime'], 'expiry':['expiry','expiration','expiration_date'], 'strike':['strike','strike_price'], 'underlying':['underlying','spot','underlying_price'], 'contract':['contract','symbol','series'], 'option_type':['option_type','type','right'], 'option_price':['option_price','price','mid','mark'], 'iv':['iv','implied_volatility']}

def standardize(d, source=''):
    d=d.copy(); d.columns=d.columns.str.strip().str.lower()
    for dest, candidates in ALIASES.items():
        if dest not in d:
            found=next((c for c in candidates if c in d),None)
            if found: d=d.rename(columns={found:dest})
    for c in ['timestamp','expiry','strike','underlying','contract']:
        if c not in d: raise ValueError(f'Missing required column {c}: {source}')
    d['source_file']=source
    d['timestamp']=pd.to_datetime(d.timestamp,errors='coerce'); d['expiry']=pd.to_datetime(d.expiry,errors='coerce')
    if 'dte' in d: d=d.rename(columns={'dte':'source_dte'})
    if 'rate' not in d: d['rate']=0.
    if 'call_price' in d and 'put_price' in d:
        frames=[]
        for side in ['call','put']:
            # Preserve extra source fields; counterpart values stay in raw files.
            base=d[[c for c in d if not c.startswith(('call_','put_'))]].copy()
            for f in ['price','iv','delta','gamma','vega','theta']:
                base['option_price' if f=='price' else f]=d.get(side+'_'+f,np.nan)
            # Keep vendor-specific bid/ask/volume and other side-specific fields.
            for c in d:
                if c.startswith(side+'_') and c[len(side)+1:] not in ['price','iv','delta','gamma','vega','theta']:
                    base[c[len(side)+1:]]=d[c]
            base['option_type']='C' if side=='call' else 'P'; frames.append(base)
        d=pd.concat(frames,ignore_index=True)
    else:
        if 'option_type' not in d: raise ValueError('Need call/put columns or option_type')
        d['option_type']=d.option_type.astype(str).str.upper().str[0]
    for c in ['strike','underlying','rate','option_price','iv','delta','gamma','vega','theta']:
        d[c]=pd.to_numeric(d[c],errors='coerce') if c in d else np.nan
    d['source_delta']=d.delta
    d['delta']=np.where(d.option_type.eq('P'),-abs(d.delta),abs(d.delta))
    d['is_put']=d.option_type.eq('P').astype('int8')
    d['dte']=(d.expiry-d.timestamp).dt.total_seconds()/86400
    d['contract_id']=d.contract.astype(str)+'|'+d.expiry.astype(str)+'|'+d.strike.astype(str)+'|'+d.option_type
    return d

def read_file(path):
    p=Path(path)
    d=pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p)
    return standardize(d,p.name),len(d)
