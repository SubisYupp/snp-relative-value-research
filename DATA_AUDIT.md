# Data audit — completed before model implementation

Source: `SNP data`; 324 daily Parquet files, 4,177,774 wide rows (8,355,548 potential option observations).
Date range: 2025-06-05 09:00:00 through 2026-09-22 16:00:00. These are 16 calendar months, with partial first and last months.
Primary experiment: June 2025–July 2026 training; August 2026 validation; September 2026 untouched test (through September 22). Calendar split is 14/1/1; the final month is explicitly partial.

## Actual schema
time, contract, expiry, strike, underlying, rate, call_price, put_price, atm, call_iv, put_iv, call_delta, call_gamma, call_vega, call_theta, put_delta, put_gamma, put_vega, put_theta, dte, straddle_delta

Calls and puts are columns on the same row. `contract` identifies a series, not a unique strike. Identity must include contract + expiry + strike + option type. Preserve source contract, ATM offset, straddle delta, supplied DTE, and source file.

## Frequency and coverage
Unique within-file timestamp gaps (minutes): {60.0: 2229, 120.0: 2, 240.0: 1, 300.0: 3}.
Observations by hour: {'9': 525966, '10': 520769, '11': 519200, '12': 520697, '13': 520700, '14': 524443, '15': 524465, '16': 521534}. Timestamps have no timezone; retain source wall-clock time. Do not invent a timezone or align external market series. No VIX/SKEW/bid/ask/volume fields exist.
Distinct expiry timestamps: 370. Rows at or after supplied expiry: 98,589; exclude these rather than correcting dates speculatively.

## Missingness
{
  "time": 0.0,
  "contract": 0.0,
  "expiry": 0.0,
  "strike": 0.0,
  "underlying": 0.0,
  "rate": 0.0,
  "call_price": 0.0,
  "put_price": 0.0038297906971511623,
  "atm": 0.0,
  "call_iv": 0.0,
  "put_iv": 0.0038297906971511623,
  "call_delta": 0.0,
  "call_gamma": 0.0,
  "call_vega": 0.0,
  "call_theta": 0.0,
  "put_delta": 0.0038297906971511623,
  "put_gamma": 0.0038297906971511623,
  "put_vega": 0.0038297906971511623,
  "put_theta": 0.0038297906971511623,
  "dte": 0.0,
  "straddle_delta": 0.0038297906971511623
}

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
time,contract,expiry,strike,underlying,rate,call_price,put_price,atm,call_iv,put_iv,call_delta,call_gamma,call_vega,call_theta,put_delta,put_gamma,put_vega,put_theta,dte,straddle_delta
2025-06-05 09:00:00,ODSNP1AN25,2025-07-06 18:00:00,5500.0,5961.25,0.0,487.9,20.15625,-11,0.2202,0.2187,0.9001,0.0005,3.1876,-0.9858,0.0987,0.0005,3.1566,-1.0266,32,0.8014
2025-06-05 09:00:00,ODSNP1AN25,2025-07-06 18:00:00,5600.0,5961.25,0.0,395.3,26.127272727272725,-10,0.2026,0.2003,0.862,0.0006,3.9834,-1.1677,0.1356,0.0006,3.9349,-1.1841,32,0.7263999999999999
2025-06-05 09:00:00,ODSNP1AN25,2025-07-06 18:00:00,5700.0,5961.25,0.0,306.75,37.10588235294117,-9,0.1883,0.1851,0.8032,0.0008,4.9723,-1.3913,0.1933,0.0008,4.9188,-1.3843,32,0.6099
```
