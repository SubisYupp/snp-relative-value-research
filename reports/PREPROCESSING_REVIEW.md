# Preprocessing review — v1.1

Completed on the actual 324-file dataset. Revised preprocessing and tests have been run; **no model was retrained or re-evaluated**. The dashboard still serves the original frozen v1.0 experiment. Treat that experiment as a historical baseline with the data-quality limitations identified here, not as validated results for the corrected pipeline.

## Measured impact

| Measure | Original | Revised |
|---|---:|---:|
| Clean option observations | 8,101,979 | 8,127,728 |
| Matched one-hour targets | 7,049,059 | 7,073,457 |
| Reported retained IV above 300% | 4,869 (incorrectly counted before cleaning) | 0 |

Recovered 25,823 previously discarded unique observations. Quarantined 10 quote records with corroborated extreme same-time price inconsistencies. Among shared observations, restored 1,940 missing targets and invalidated 64 previously matched targets. These counts describe processing changes, not improved predictive performance.

## Fixes implemented

1. **Identical duplicates:** retain one economic quote. Previously the cleaner removed both copies, incorrectly describing them as conflicting duplicates. Distinct records with the same identity remain excluded, even when one fails a validity screen.
2. **Extreme quote quarantine:** require both a neighboring-strike price-envelope breach greater than 5% of underlying and a parity-forward deviation greater than max(5% of underlying, 10 cross-strike MAD), with at least five call/put pairs. This uses contemporaneous quotes only, never subsequent returns. The rule is deliberately coarse. Quarantined records remain in the raw source and the JSON review; no fabricated replacement prices are inserted.
3. **Outcome eligibility:** start-time IV/Vega/Greek feature eligibility and future price eligibility are separate. Future IV, delta or Vega can no longer censor a target with a valid price, underlying and identity. Ambiguous and quarantined quotes are excluded at both ends. Require strictly positive source prices: the audit found 474 unexpired zero-price records, 402 also missing/zero IV and 72 with positive IV. Without executable bid/ask evidence, zero cannot reliably distinguish a valueless option from a missing-price sentinel. These are counted as unusable zeros, not asserted to be universally impossible option prices.
4. **ATM aggregation:** one vote per series/expiry, instead of repeating ATM IV once per strike. This corrects market-regime inputs and expiry term aggregation. 2,223 source timestamps change; median absolute market ATM change is 0.000450 in decimal-IV units.
5. **Surface safeguards:** explicitly exclude all duplicate copies of the target strike; negative/nonfinite expected IV and insufficient surrounding points produce missing features. The original cache contained four negative expected-IV values; the revised cache contains 0.
6. **Validity and lineage:** reject missing series IDs, infinite strikes and invalid rates; optional invalid gamma/theta become counted missing values. Preserve extra side-specific fields such as bid/ask/volume when supplied.
7. **Lag matching:** enforce a tolerance below one hour and prohibit cross-calendar-day lag matches, consistent with the session assumptions used for targets.
8. **Cache safety:** hash source contents and preprocessing code/configuration; bind training to the current source manifest. New outputs use `data/processed-v1.1` and `results/preprocessing-v1.1`, preserving the original cache and model artifacts. Daily-file assumptions are now checked explicitly: multi-day or overlapping files must be consolidated/partitioned before ingestion, rather than silently losing cross-file matches.
9. **Quality reporting:** count retained high-IV quotes after cleaning, reconcile all sequential exclusions, and report missing fields across the union of source schemas.

## Training-period target sensitivity

Measured on the original supervised-training window only (2025-08-01 09:00:00 to 2026-08-01 00:00:00, exclusive end), with no test-driven cutoff selection. Quotes with Vega ≤0.5 account for 2.94% of rows but 83.84% of the sum of squared targets (zero-prediction squared error). This explains why RMSE is sensitive to small-Vega observations; it does not prove that these quotes should be removed.

The existing Vega floor remains 0.01. No new target clipping or performance-driven Vega cutoff was added. `reports/training_vega_sensitivity.csv` gives all buckets. `reports/training_extreme_targets.csv` records the inspected extreme training observations.

## Data questions that remain

- Source rates are zero throughout. Product identity, settlement convention, source timezone and Greek units still need vendor confirmation.
- The audit found no mixed underlying values within a timestamp, no negative source gamma, and no unexpired IV above 300%. The high-IV retained count in the old quality report was a reporting bug involving already-expired rows.
- Negative forward variance is preserved and reported, not clipped to make the term structure look cleaner. Multiple vendor series and noisy maturity quotes can produce this; bid/ask and product metadata are needed to interpret it.
- The original HMM has an imperfect convergence diagnostic and a short two-month warm-up. Improving it belongs in a new training/validation experiment.
- The strict quarantine is not a comprehensive price validator. Moderate errors, edge strikes and quotes without paired/surrounding strikes may escape it. Raw records remain available for vendor investigation.

## Verification and reproduction

Final verification: **29 tests passed**. A repeat preprocessing run successfully reused the revised cache and returned **8,127,728 clean observations**.

Revised cache checks: 0 infinite numeric values, 0 duplicate contract/timestamp identities, 0 negative expected-IV values. All matched horizons are 50–70 minutes and remain within a source calendar day. Original model SHA-256 unchanged: True.

```powershell
python scripts/run.py test
python scripts/run.py preprocess
python scripts/run.py review_preprocessing
```

The original pipeline source is archived in `reports/archive/pipeline-v1.0.0.zip`. Revised source defaults are v1.1; the old dashboard and report continue to describe the old model. A revised model needs a separate experiment. September 2026 is already inspected and cannot be represented as untouched if reused.
