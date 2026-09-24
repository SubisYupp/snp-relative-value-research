from pathlib import Path
import json
import numpy as np
import pandas as pd

def audit(root='SNP data'):
    files = sorted(Path(root).glob('*.parquet'))
    rows = 0; missing = {}; months = {}; hours = {}; schemas = {}; deltas = []; gaps = []; expiries = set(); negative_dte = 0
    first = None; last = None; examples = None
    for f in files:
        d = pd.read_parquet(f); rows += len(d)
        schemas[str(tuple(d.columns))] = schemas.get(str(tuple(d.columns)), 0) + 1
        t = pd.to_datetime(d.time); e = pd.to_datetime(d.expiry)
        first = min(first, t.min()) if first is not None else t.min()
        last = max(last, t.max()) if last is not None else t.max()
        for c in d: missing[c] = missing.get(c, 0) + int(d[c].isna().sum())
        for k,v in t.dt.strftime('%Y-%m').value_counts().items(): months[k] = months.get(k,0)+int(v)
        for k,v in t.dt.hour.value_counts().items(): hours[str(k)] = hours.get(str(k),0)+int(v)
        gaps.extend(t.drop_duplicates().sort_values().diff().dt.total_seconds().dropna().div(60).tolist())
        expiries.update(d.expiry.unique()); negative_dte += int((e <= t).sum())
        deltas.append({'file':f.name, 'positive_put_delta':int((d.put_delta>0).sum()), 'negative_put_delta':int((d.put_delta<0).sum())})
        if examples is None: examples = d.head(3).to_csv(index=False)
    result = dict(files=len(files), raw_rows=rows, start=str(first), end=str(last), missing_pct={k:100*v/rows for k,v in missing.items()}, months=months, hours=hours, expiry_count=len(expiries), expired_rows=negative_dte, timestamp_gap_minutes=dict(zip(*[a.tolist() for a in np.unique(gaps,return_counts=True)])), put_delta_signs=deltas, schemas=schemas)
    Path('reports').mkdir(exist_ok=True)
    Path('reports/raw_audit.json').write_text(json.dumps(result,indent=2))
    text = f'''# Data audit — completed before model implementation

Source: `{root}`; {len(files):,} daily Parquet files, {rows:,} wide rows ({2*rows:,} potential option observations).
Date range: {first} through {last}. These are 16 calendar months, with partial first and last months.
Primary experiment: June 2025–July 2026 training; August 2026 validation; September 2026 untouched test (through September 22). Calendar split is 14/1/1; the final month is explicitly partial.

## Actual schema
{', '.join(missing)}

Calls and puts are columns on the same row. `contract` identifies a series, not a unique strike. Identity must include contract + expiry + strike + option type. Preserve source contract, ATM offset, straddle delta, supplied DTE, and source file.

## Frequency and coverage
Unique within-file timestamp gaps (minutes): {result['timestamp_gap_minutes']}.
Observations by hour: {hours}. Timestamps have no timezone; retain source wall-clock time. Do not invent a timezone or align external market series. No VIX/SKEW/bid/ask/volume fields exist.
Distinct expiry timestamps: {len(expiries)}. Rows at or after supplied expiry: {negative_dte:,}; exclude these rather than correcting dates speculatively.

## Missingness
{json.dumps(result['missing_pct'], indent=2)}

## Critical conventions and assumptions
Put delta changes sign convention across files. Normalize to -abs(put_delta); calls to +abs(call_delta). Preserve raw delta. This is required for a signed delta hedge.
IV is stored as a decimal (typical 0.1–0.3). Vega appears to be option-price points per one volatility percentage point, consistent with typical values of 3–20; vendor documentation is absent, so this remains an assumption. Use source Vega directly. Target is price-point delta-hedged change divided by source Vega. A common contract multiplier cancels. Theta units are undocumented; preserve source values.
Rates in the inspected file are zero; use supplied rates as decimals. Forward is a same-timestamp robust put-call parity estimate within each contract series and expiry. Never mix different series merely because expiries coincide.
Instrument identifiers are SNP vendor symbols; exact SPX exchange/product identity, settlement rules, price construction and timezone are unverified. Report as supplied SNP options, not authenticated executable SPX quotes.
Source DTE is integer and can disagree with precise expiry; use fractional days from the supplied expiry timestamp and retain source DTE. No expiry-date repair is attempted.
Price/IV/Vega validity and duplicates will be counted in the cleaning report. Zero Vega cannot support this target. High positive IV is retained and flagged rather than arbitrarily clipping market stress.
No target or model performance has been inspected during this audit.

## First source rows
```csv
{examples}```
'''
    Path('DATA_AUDIT.md').write_text(text,encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['put_delta_signs','schemas']},indent=2))

if __name__ == '__main__': audit()
