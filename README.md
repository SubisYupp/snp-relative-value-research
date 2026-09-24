# SNP options — relative-value research

A reproducible Python research pipeline and a Next.js/React/TypeScript application for one-hour, delta-hedged, Vega-normalized option signals. It runs on the supplied `SNP data` files. No Streamlit, synthetic quotes or fabricated performance metrics are used.

**Current experiment: v1.1, retrained September 24, 2026.** The dashboard and active model report now use corrected preprocessing. September is a **revised historical evaluation**, since its results were inspected previously. See [the rerun comparison](reports/RERUN_V1.1.md) and [the preprocessing review](reports/PREPROCESSING_REVIEW.md). The original models, predictions, metrics, reports and dashboard data are preserved in `experiments/snp-rv-v1.0.0/`; original pipeline source is in `reports/archive/pipeline-v1.0.0.zip`. The historical v1.0 figures and methods below describe that initial experiment; the current reports/design lock are authoritative for v1.1.

```powershell
python scripts/run.py preprocess
python scripts/run.py review_preprocessing
```

## Open the completed research

The dashboard is served locally at **http://127.0.0.1:3000** when its server is running.

```powershell
# This workspace already contains downloaded local dependencies and generated results.
python scripts/run.py dev

# Production build and server
python scripts/run.py build
python scripts/run.py start

# Full pipeline, including preprocessing, fitting, evaluation and export
python scripts/run.py train

# Verify numerical properties and saved outputs
python scripts/run.py test

# With the server running, exercise all nine pages in a headless browser
python scripts/run.py test:ui
```

The convenience launcher discovers the workspace `.deps` libraries and `.tools` Node runtime. Standard commands in an activated environment also work: `python -m quant.train`, `python -m quant.evaluate`, `npm run dev`, `npm run build`. `quant.evaluate` reads frozen metrics; it does not select or refit models.

The final evaluation is guarded by `results/artifacts/test_evaluation_complete.json`: rerunning training returns saved results rather than silently repeating the final holdout experiment. To reproduce from scratch, use a **fresh checkout/output workspace** with the same raw files and pinned dependencies. To re-export presentation data only, use `python scripts/run.py export_results`. Changing the experimental design after viewing the test results requires a new experiment and new genuinely unseen data; the current test cannot become untouched again.

## Fresh environment

Python 3.12 and Node.js 22 were used. Install into an isolated environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
npm --prefix dashboard ci
python -m quant.audit
python -m quant.train --raw-dir "SNP data" --tolerance-minutes 10
npm run build
npm run start
```

On Unix, activate with `source .venv/bin/activate`. The original Windows execution environment did not permit `ensurepip` in its sandbox, so dependencies were installed under `.deps`; this does not change model logic. The local Node ZIP was checked against the official SHA-256 manifest. Dependency versions are pinned in `requirements.txt` and `dashboard/package-lock.json`. Browser tests use installed Microsoft Edge if available, otherwise Playwright Chromium (`npx playwright install chromium` from `dashboard`). The dashboard binds to loopback only and provides no authentication; it is intended as a local research application.

## Actual data and chronology

Read [DATA_AUDIT.md](DATA_AUDIT.md) first. The supplied data contains **324 daily Parquet files**, **4,177,774 wide rows**, and **16 calendar months**, from June 5, 2025 to September 22, 2026. Calls and puts expand into **8,355,548** candidate option observations. **8,101,979** survive cleaning. The last month is partial.

| Stage | Dates | Purpose |
|---|---|---|
| Regime warm-up | 2025-06-05 through 2025-07-31 | Fit four-state HMM, order states, freeze parameters |
| Supervised training | 2025-08-01 through 2026-07-31 | Fit XGBoost, Ridge and simple baselines |
| Validation | 2026-08-01 through 2026-08-31 | Select depth/early stopping and calibrate neutral band/conviction |
| Final test | 2026-09-01 through 2026-09-22 | Single evaluation of frozen models |

This is a 14/1/1 calendar-month allocation, with the first two training months reserved for regime warm-up, rather than arbitrarily throwing away supplied history to force 12 months. There are **5,512,112 supervised training rows**, **418,894 validation outcomes**, and **296,928 matched test outcomes**. Unmatched final-session observations still receive predictions and SHAP explanations and remain visible in the explorer.

The exact dates, selected hyperparameters, feature order, HMM fit period, model hash and numerical diagnostics are in `results/artifacts/design_lock.json`. The lock is written before final-test model scoring. The data audit and feature preprocessing cover all files, but no final-test outcome summaries or model comparisons enter selection. Targets ending on or after the next split boundary are purged.

## What the completed experiment found

| Final-test metric | Result |
|---|---:|
| Spearman IC | 0.1011 |
| Overall sign hit rate | 54.2% |
| Active top-10%-conviction hit rate | 61.1% |
| XGBoost RMSE | 3.4295 |
| Zero-prediction RMSE | 3.4330 |
| Daily-mean top/bottom decile spread | 0.4030 |

XGBoost's RMSE improvement is small. Regime features also improve RMSE only slightly. Conviction hit rates **do not increase monotonically**, despite better top-decile results. The test spans only **15 observed days**, contains **no CRISIS-state outcomes**, and does not establish executable profitability. The HMM's selected EM fit stopped after a small likelihood decrease (approximately −0.0375); library convergence alone would mask this. The report records the imperfect convergence without refitting after test inspection.

## Target and units

```text
DH_PNL = (price[t+h] - price[t]) - delta[t] * (underlying[t+h] - underlying[t])
target = DH_PNL / vega[t]
```

Prices are source price points. IV is a decimal. Source Vega is assumed to be price points per one volatility **percentage point**, based on observed magnitudes; vendor confirmation is unavailable. The target therefore represents Vega-equivalent volatility-point P&L under that assumption, not dollars or percent return. A consistent common contract multiplier multiplies numerator and denominator and cancels. Raw theta and other Greek units are preserved, not silently rescaled.

Put delta has mixed signs in the source and is normalized to `-abs(delta)`; call delta to `abs(delta)`. The original delta is retained. No transaction costs, bid/ask spread, funding, discrete rehedging path, dividends or slippage are modeled. These are statistical relative-value signals.

For each full contract identity (vendor series + expiry + strike + option type), the matcher selects the observation nearest `t + 60 minutes`, within ±10 minutes by default. It never crosses a source calendar day or contract identity. Observations outside tolerance have a missing outcome, not a zero outcome. Timestamps lack timezone information and remain in source wall-clock time.

## Cleaning and data lineage

- Automatic aliases map common long-form field names; the actual wide-form call/put data is converted to long form. Both Parquet and CSV readers are supported; new datasets should be audited before use.
- Keep source files untouched, original DTE, ATM offset, straddle delta, raw delta, series identifier and source filename. The processed schema adds canonical fields and engineered features.
- Exclude invalid identities, nonpositive underlying/IV, negative or nonfinite price, already-expired quotes, invalid delta and nonpositive Vega. A fixed numerical target-stability floor excludes positive Vega below 0.01. Zero option prices are retained when the other requirements hold.
- Positive IV above 300% is flagged but retained; no stress-aware clipping or test-driven target winsorization occurs.
- Drop every conflicting duplicate contract/timestamp identity, rather than arbitrarily trusting file order.
- Report unchanged consecutive prices as a possible staleness proxy, not verified stale quotes.
- Missingness, mutually exclusive drop counts, overlapping validity flags, monthly/hourly coverage, match horizons, DTE/Vega quantiles and strike coverage appear in the Data Quality page and JSON files.

## Features and causal ordering

`quant/config.py` defines 21 non-regime features and five regime features; no unavailable market indices are invented.

**Forward and moneyness.** Put-call parity gives `F = K + exp(r*T)*(C-P)`. At each timestamp, series and expiry, use the median of positive parity forwards within 20% of underlying, requiring at least three pairs. Otherwise use `underlying*exp(r*T)` and record the fallback. Rates are interpreted as supplied decimal rates. Log-moneyness is `log(K/F)`. Quotes from different vendor series with the same expiry are not silently paired.

**Smile.** Each quote's expected IV, skew and curvature come from a fixed, distance-weighted quadratic fit to up to 13 neighboring strikes of the same timestamp, series, expiry and type, excluding the quote's strike entirely. At least five strikes are required. A fixed 1e-5 ridge stabilizes curvature; there is no cross-time smoothing or selected bandwidth. Residual is actual minus expected IV. ATM IV uses the closest-to-forward observed strike per type and averages call/put values within the series. This approximation is explicit; it is not an arbitrage-free surface model.

**Term structure.** Take median ATM IV across available series of each timestamp/expiry. Forward variance is the total-variance difference to the next longer expiry divided by maturity difference. Missing neighbors remain missing; negative forward variance is retained as a diagnostic feature.

**Dynamics.** Prior-hour IV, ATM IV, skew, curvature and residual changes match only earlier observations within the configured tolerance. They never use centered windows or future fill. Underlying return is the prior hourly log return. Realized volatility uses a trailing 40-observation window and at least 21 eligible hourly returns. Training timestamps imply eight observations and seven within-session intervals/day, so annualization uses `252*7 = 1764`. Overnight and irregular-gap returns are excluded; the window spans approximately five source sessions. These cadence choices use training timestamps only.

**Regimes.** The observation is log of the cross-series median ATM IV. Fit four Gaussian states on the warm-up only, choose the best of three fixed random initializations by warm-up likelihood, order fitted means LOW/NORMAL/HIGH/CRISIS, then freeze all parameters. A manually implemented forward recursion produces probabilities and entropy. `predict_proba`/backward smoothing is not used. Warm-up feature rows are excluded from supervised training because HMM parameters were estimated using that entire warm-up. Regime labels are relative to that fit: CRISIS does not assert an externally defined crisis. The frozen short warm-up can underrepresent later volatility environments; this is disclosed.

**Missing features.** XGBoost handles missing values natively. Baseline median imputers/scalers are fitted only on supervised training data. No full-sample standardization is used.

## Models, validation and conviction

The main model is XGBoost regression with pseudo-Huber objective, fixed robustness scale 1.0, histogram trees, seed 42, learning rate .04, depth candidates 3 and 5, min-child weight 100, subsample .8, column sample .9, lambda 20, alpha 1, gamma .01, and at most 600 estimators. Validation MAE with 35-round early stopping selects the model; the final run selected depth 5 and 75 trees. No labels are clipped. Unused early-stopped trees are physically removed so SHAP and prediction explain the same fitted function.

The ablation removes only regime probabilities and entropy and otherwise uses identical settings and tree count. Both are evaluated on validation before the design is locked. Baselines are zero, Ridge over all features, Ridge on relative IV alone and Ridge on surface residual alone. Their fitting, median imputation and scaling use training only. The single-feature regressions are documented alternatives to a percentile heuristic.

The neutral threshold is the median of absolute validation predictions. Prediction above `+threshold` is CHEAP; below `-threshold` is RICH; otherwise NEUTRAL. Conviction is the right-sided empirical percentile of absolute prediction against the fixed validation absolute-prediction distribution. It is **not** a calibrated correctness probability. Each ablation uses its own validation-derived reference/threshold.

## Evaluation and inference

Validation and test separately include MAE, RMSE, R², Pearson, Spearman, sign accuracy, cheap/rich hit rate, balanced directional accuracy, class counts, directional outcomes, conviction deciles, prediction-bucket calibration, threshold hit rates and regime breakdowns. Realized zeros count as incorrect directional predictions. The zero baseline's sign hit rate is mechanically zero, since it expresses no direction; RMSE/MAE are its meaningful comparisons.

Ranking uses same timestamp, expiry, vendor series **and option type**, with top/bottom 20%, 10% and 5% tails and at least two observations per leg. Average tied prediction ranks can produce unequal legs. Average spreads across cross-sectional groups at each timestamp, then average timestamps within each day. The reported mean/median and uncertainty operate on those daily means; the chart cumulatively adds timestamp spreads. Newey-West variance of daily means uses up to five lags; fewer than ten days suppresses inferential quantities. Cross-sectional quotes are not treated as independent trades. The 15-day test offers limited statistical evidence even where exploratory t-statistics are large.

Vertical candidates pair the highest CHEAP and lowest RICH strike within an identical series/expiry/type/time. For one long contract, `short_contracts_per_long = long_vega/short_vega` approximates common-Vega sizing. Prediction and outcome differences are in normalized units. Fractional sizes, hedging, contract multipliers and costs are not an executable vertical-trade simulation.

Deep-OTM diagnostics show delta buckets, IV and residual correlations, rich share, outcomes and hit rates. A transparent failure flag requires at least 100 deep-OTM observations, IV/prediction correlation below −0.5 and rich share above 70%. Passing this screen does not prove that the model is free from confounding. The sample visualizations condition on delta/moneyness and include residual-versus-prediction views.

## Dashboard and explanations

Nine views: overview, full test-month smile explorer, option inspector, regimes, model quality, feature research, OTM put diagnostics, relative-value scanner and data quality. Next.js serves saved JSON without executing Python or retraining. Every timestamp has its own complete snapshot; the browser loads only the selected snapshot. Large tables use bounded, sortable/filterable 50-row pagination rather than rendering hundreds of thousands of DOM rows. Pagination is the chosen alternative to scroll virtualization.

Tree SHAP is computed for **every test option**, with a reconstruction assertion against the saved model prediction. The inspector shows all feature contributions and a bridge including the baseline, ten main contributions and the remaining-feature sum. Global importance uses all test observations. Overview/dependence plots use a fixed 4,000-row sample, OTM plots a 2,500-row sample and the SHAP distribution plot a 700-row subset. Histogram charts show the sample's central 98%, explicitly labeled; full outcomes remain in the Parquet exports. Feature interaction views color SHAP dependence by a second feature; these are not separately estimated SHAP interaction tensors.

No external service or LLM generates the research conclusions; they are deterministic comparisons of saved metrics. Empty states and absent regimes are displayed without fabricated values. The interface supports keyboard row selection, sticky headings, reduced motion and tablet/mobile layouts.

## Main artifacts

| Path | Content |
|---|---|
| `DATA_AUDIT.md`, `reports/raw_audit.json` | Pre-model source audit |
| `data/processed/*.parquet` | Daily standardized, cleaned quotes and features |
| `models/hmm/model.pkl` | Frozen warm-up regime model |
| `models/xgboost/*.json` | Main and ablation XGBoost models |
| `models/baselines.pkl`, `models/conviction_reference.npy` | Fitted baselines and conviction reference |
| `results/artifacts/design_lock.json` | Frozen design, split and model SHA-256 |
| `results/predictions/predictions_test.parquet` | Every clean test observation, prediction and available outcome |
| `results/predictions/predictions_validation.parquet` | Validation predictions/outcomes |
| `results/metrics/metrics.json` | Complete validation, test, baseline and ablation metrics |
| `results/metrics/conviction_analysis.csv` | Test conviction deciles |
| `results/metrics/regime_metrics.csv` | Test regime performance |
| `results/metrics/feature_importance.csv` | Full-test SHAP importance |
| `results/metrics/vertical_candidates.csv` | All candidate pairs |
| `results/metrics/data_quality.json` | Cleaning, alignment and coverage |
| `results/artifacts/shap_test.npy` | All per-observation SHAP contributions |
| `reports/model_report.html`, `reports/model_report.json` | Standalone research report |
| `reports/screenshots/`, `reports/browser_checks.json` | Browser verification evidence |
| `dashboard/public/data/` | Static application data and snapshot shards |

Do not load pickle artifacts from untrusted sources. These local artifacts are generated by this project.

## Known limits and future experiments

The vendor's SNP symbols are not independently verified as SPX contracts; some expiry dates are unusual. Settlement conventions, timezone, Greek units and quote construction require vendor confirmation. The supplied files contain observations through September 22, 2026; file contents have not been independently authenticated. The current pipeline preserves them as supplied and does not claim exchange provenance.

This V1 uses one chronological holdout, a short HMM warm-up and no trading-cost model. It supports further research, not a deployment decision. A subsequent pre-registered experiment should verify the vendor conventions, improve HMM convergence on training-only history, add walk-forward validation and independent future months, test sensitivity to small-Vega quotes and incorporate executable bid/ask costs. Do not retune these choices against the already viewed test month.

Implementation references: [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html), [Next.js installation](https://nextjs.org/docs/app/getting-started/installation). The implementation was verified against installed package behavior; lockfiles preserve the exact versions used.
