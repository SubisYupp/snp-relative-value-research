You are a senior quantitative researcher, machine-learning engineer, and full-stack product engineer.

Build a complete production-quality research project that uses one year of historical S&P 500 / SPX options data to identify whether individual option strikes are statistically CHEAP, NEUTRAL, or RICH.

This must be a serious quantitative research implementation, not a toy notebook.

The final deliverables must include:

1. a rigorous data-processing and ML pipeline,
2. a regime-detection model,
3. an XGBoost option rich/cheap prediction model,
4. strict time-series validation with no future leakage,
5. detailed performance evaluation,
6. conviction/calibration analysis,
7. strike-level and expiry-level relative-value analysis,
8. an extremely polished interactive web dashboard,
9. saved models/results so the dashboard does not need to retrain everything every time,
10. clear documentation explaining exactly what was done.

Do NOT use Streamlit.

Build an actual webpage application.

Preferred stack:

* Frontend: Next.js + React + TypeScript
* Styling: Tailwind CSS
* Charts: Apache ECharts or another high-quality interactive chart library
* Backend/API: Next.js server routes or a lightweight Node API where appropriate
* Quant/ML pipeline: Python
* Data processing: pandas / numpy / scipy
* ML: XGBoost
* Regime model: hmmlearn or a suitable custom Gaussian HMM implementation
* Model explanation: SHAP

The result should feel like an internal professional quantitative-trading research platform, not an AI-generated dashboard template.

---

## PROJECT OBJECTIVE

For every option strike at every observation timestamp, predict:

```
future 1-hour delta-hedged P&L per unit of Vega
```

Define the target as:

```
DH_PNL =
    (OptionPrice[t+1h] - OptionPrice[t])
    - Delta[t] * (Underlying[t+1h] - Underlying[t])
```

Then:

```
TARGET =
    DH_PNL / Vega[t]
```

Use the correct contract multiplier consistently if prices/P&L require it, but because the target is Vega-normalized, make sure all numerator and denominator units are internally consistent.

Explain the units clearly in the documentation.

The fundamental interpretation is:

```
predicted TARGET > 0
    -> long option is expected to outperform its delta hedge
    -> statistically CHEAP

predicted TARGET < 0
    -> long option is expected to underperform its delta hedge
    -> statistically RICH
```

Do not describe this automatically as arbitrage or guaranteed mispricing.

Use wording:

```
statistically cheap
statistically rich
relative-value signal
```

unless realistic transaction costs and execution data are available.

---

## DATA

The dataset contains approximately one year of SPX options data.

Make the ingestion layer flexible and automatically detect/map likely column names.

Expected columns may include fields similar to:

* time
* contract
* expiry
* strike
* underlying
* rate
* call_price
* put_price
* atm
* call_iv
* put_iv
* call_delta
* call_gamma
* call_vega
* call_theta
* put_delta
* put_gamma
* put_vega
* put_theta
* dte
* straddle_delta

There may be separate rows or columns for calls and puts.

Inspect the actual dataset before assuming its exact structure.

Build a clean standardized internal schema such as:

* timestamp
* expiry
* strike
* option_type
* option_price
* underlying
* rate
* dte
* iv
* delta
* gamma
* vega
* theta
* contract_id

If the source is wide-form with call and put columns in one row, convert it into long form.

Never silently discard important fields.

Produce a data-quality report showing:

* number of raw rows
* number of cleaned rows
* missing-value percentage by field
* duplicate observations
* duplicate contract/timestamp rows
* impossible prices
* impossible IV values
* stale quotes if detectable
* zero/negative Vega cases
* DTE distribution
* number of expiries
* number of strikes per expiry
* observations by month
* observations by hour

---

## CRITICAL ANTI-LEAKAGE RULE

Absolutely no future information may enter any feature.

At observation time t, every feature must use only information available at or before t.

The target may use t+1h because it is what we are predicting.

Never use:

* future IV
* future underlying prices
* future volatility surfaces
* centered rolling windows
* full-sample normalization
* full-sample HMM smoothing
* smoothed HMM state probabilities using future observations
* future-filled missing values
* statistics computed using validation/test months while training

Every preprocessing object must be fitted on training data only.

---

## TEMPORAL SPLIT

Use calendar months.

Preferred split:

* Months 1-10: training
* Month 11: validation
* Month 12: final untouched test set

The test month must remain completely unseen until all feature design and hyperparameter choices are frozen.

If the dataset does not contain exactly 12 full calendar months, automatically choose the closest valid chronological split:

approximately:

* first 80-84% of chronological data -> train
* next 8-10% -> validation
* final 8-10% -> test

but align boundaries to full calendar months wherever possible.

Show the exact train/validation/test dates prominently in the dashboard.

Do not random-shuffle the observations.

---

## ONE-HOUR TARGET MATCHING

The dataset may not contain an observation exactly one hour later.

For each option contract, find the next observation closest to t + 1 hour within a configurable tolerance.

For example:

```
desired horizon = 60 minutes
allowed tolerance = +/- 10 minutes
```

Do not match across expiry changes or different contracts.

Report:

* percentage successfully matched
* actual horizon distribution
* median horizon
* 5th/95th percentile horizon

Allow the tolerance to be configured.

---

## FEATURE ENGINEERING

Build the following FIXED V1 feature system.

Do not generate hundreds of random technical indicators.

Keep the features economically interpretable.

====================
A. CONTRACT FEATURES
====================

1. DTE

2. Option type

3. Forward log-moneyness:

   k = ln(K / F)

Prefer extracting the forward from put-call parity where matching call and put quotes exist:

```
C - P = exp(-rT) * (F - K)
```

therefore:

```
F = K + exp(rT) * (C - P)
```

Use a robust cross-strike estimate of F for the expiry rather than trusting one noisy pair.

If reliable call-put pairs are unavailable, use the best available forward approximation and document it.

Never use raw strike as the primary moneyness feature.

====================
B. VOLATILITY LEVEL
===================

4. IV of the option

5. ATM IV for the same expiry

6. Relative IV:

   relative_iv = option_iv - atm_iv

====================
C. LOCAL VOLATILITY CURVE
=========================

For every timestamp + expiry, build a smooth local smile in log-moneyness space.

Use an economically sensible smoother such as:

* weighted cubic spline,
* LOWESS,
* local polynomial regression,

with safeguards against overfitting.

Do not force an overly complex model.

From the local smile calculate:

7. local skew:

   dIV / dk

8. local curvature:

   d²IV / dk²

9. local surface residual:

   surface_residual =
   market_IV -
   IV predicted by surrounding strikes

IMPORTANT:

When calculating surface_residual for option i, avoid allowing that exact quote to dominate its own expected value.

Prefer leave-one-out fitting or strongly downweight the target strike.

This feature is critical.

It prevents deep OTM puts from automatically being called rich just because they naturally have higher IV.

A 10-delta put with 35% IV may be completely normal if the surrounding smile implies approximately 35%.

Only if its IV is unusually high relative to its moneyness and neighbouring curve should the residual become strongly positive.

====================
D. TERM STRUCTURE
=================

For ATM IV:

```
total variance = sigma² * T
```

Calculate a forward-variance feature between the closest relevant maturities:

```
FV(T1,T2) =
    [sigma²(T2) T2 - sigma²(T1) T1]
    / (T2 - T1)
```

Use only maturities available at time t.

If a valid neighbouring expiry is unavailable, leave this feature missing and let the pipeline handle it safely.

====================
E. SURFACE DYNAMICS
===================

Calculate lagged changes using ONLY past observations:

10. IV change over previous hour
11. ATM IV change over previous hour
12. local skew change over previous hour
13. local curvature change over previous hour
14. surface residual change over previous hour

Optionally include one-session lag versions if enough history exists, but keep V1 compact.

====================
F. GREEKS
=========

15. delta
16. gamma
17. vega
18. theta

Keep original units and document them.

Do not unnecessarily standardize for XGBoost.

====================
G. UNDERLYING STATE
===================

19. previous 1-hour underlying log return:

    r_t = ln(S_t / S_t-1h)

20. realized volatility

Calculate trailing realized volatility from past underlying returns.

Use a short-term measure such as approximately five trading days.

For hourly SPX data:

```
RV =
    sqrt(
        annualization_factor
        * mean(hourly_log_return²)
    )
```

Use the correct number of observations per trading day based on the actual dataset.

Do not blindly assume 6.5 if timestamps show something different.

====================
H. MARKET VOLATILITY INDEX FEATURES
===================================

If available, include:

21. VIX
22. CBOE SKEW

Potentially include other volatility-market indices only if actually available and aligned correctly:

* VVIX
* correlation index
* dispersion index

Never fabricate unavailable data.

If external series are imported, clearly document the source and ensure the timestamp was genuinely observable at the option timestamp.

====================
I. REGIME MODEL
===============

Build a 4-state Gaussian Hidden Markov Model.

The HMM should use:

```
log(ATM IV)
```

as its main regime observation.

The intention is NOT for the HMM to predict rich/cheap options.

The HMM only answers:

```
What volatility environment are we currently in?
```

Fit four hidden states.

After fitting on training data, sort states by their mean ATM IV and label them:

1. LOW
2. NORMAL
3. HIGH
4. CRISIS

Do not hard-code IV boundaries.

Generate FILTERED state probabilities:

```
P(Low | information through t)
P(Normal | information through t)
P(High | information through t)
P(Crisis | information through t)
```

Never use full-sample smoothed state probabilities.

Also calculate regime entropy:

```
H_t = - sum_j p_j * ln(p_j)
```

This represents regime uncertainty.

The final regime features are:

23. p_low
24. p_normal
25. p_high
26. p_crisis
27. regime_entropy

IMPORTANT:

The HMM itself must obey the train/validation/test chronology.

It must not be fitted on all 12 months before generating historical probabilities.

For the final test evaluation, its parameters may only come from the permitted historical training period.

---

## MODEL

Use XGBoost Regressor.

Target:

```
next 1-hour delta-hedged P&L / current Vega
```

Do not create a classifier as the main model.

This is fundamentally a regression problem.

Use a robust objective/loss where practical.

Huber-style robustness is desirable if supported appropriately, otherwise use squared-error regression while winsorizing only clearly pathological target values based solely on training-set thresholds.

Do not casually clip valid market stress observations.

Start with conservative tree complexity.

Suggested search space:

* max_depth: 2-6
* learning_rate: 0.01-0.10
* n_estimators: up to 2000 with early stopping
* min_child_weight
* subsample: approximately 0.6-1.0
* colsample_bytree: approximately 0.6-1.0
* reg_lambda
* reg_alpha
* gamma

Use Month 11 exclusively for:

* hyperparameter selection
* early stopping
* model selection
* neutral-band calibration

Do NOT repeatedly inspect Month 12 while developing.

Once finalized:

* lock preprocessing,
* lock HMM design,
* lock features,
* lock XGBoost hyperparameters,
* evaluate ONCE on Month 12.

---

## PREDICTION INTERPRETATION

For every option observation:

```
prediction > 0:
    expected positive future delta-hedged P&L/Vega

prediction < 0:
    expected negative future delta-hedged P&L/Vega
```

However, do not call tiny predictions cheap/rich.

Create a NEUTRAL region.

Determine the neutral threshold ONLY from the validation month.

For example, choose a symmetric threshold based on either:

* validation prediction distribution,
* prediction uncertainty,
* estimated noise,
* or realistic minimum economic edge.

Example:

```
prediction > +threshold -> CHEAP
prediction < -threshold -> RICH
otherwise -> NEUTRAL
```

Document exactly how threshold was selected.

---

## CONVICTION SCORE

I want to know not only what the model predicts, but how strongly it believes the signal.

Create a conviction score derived from the magnitude of the prediction relative to the validation prediction distribution.

For example:

```
conviction_percentile =
    percentile rank of |prediction|
    relative to validation predictions
```

Map approximately to:

* 0-50: low conviction
* 50-75: moderate
* 75-90: high
* 90-100: very high

Do NOT pretend this is a calibrated probability unless calibration actually supports that interpretation.

Call it:

```
model conviction score
```

not:

```
probability of being correct
```

unless explicitly calibrated.

---

## DEFINITION OF "CORRECT"

For non-neutral predictions:

CHEAP prediction is directionally correct if:

```
realized TARGET > 0
```

RICH prediction is directionally correct if:

```
realized TARGET < 0
```

Calculate:

```
direction_hit_rate
```

But also calculate conviction-based hit rates.

Example:

* all non-neutral signals
* conviction > 50
* conviction > 75
* conviction > 90
* top 10% conviction
* top 5% conviction

This is extremely important.

If the model is meaningful, ideally:

```
hit rate should rise as conviction rises.
```

If it does not, say so clearly.

---

## PRIMARY MODEL EVALUATION

Report all of the following on validation and final test separately.

====================
REGRESSION METRICS
==================

* MAE
* RMSE
* R²
* Pearson correlation(prediction, realized)
* Spearman rank correlation(prediction, realized)

Do not overemphasize R² because financial return prediction naturally has low R².

====================
DIRECTION METRICS
=================

* overall sign hit rate
* cheap-signal hit rate
* rich-signal hit rate
* balanced directional accuracy
* number of predictions in each class

====================
CONVICTION ANALYSIS
===================

Create conviction deciles.

For every decile report:

* number of observations
* mean absolute prediction
* sign hit rate
* mean realized target
* median realized target
* mean predicted target

The most important question:

```
Does greater model conviction correspond to stronger realized outcomes?
```

Plot:

```
conviction decile
vs
hit rate
```

and:

```
conviction decile
vs
realized absolute/conditional performance
```

====================
RANKING / RELATIVE-VALUE PERFORMANCE
====================================

At each timestamp and expiry:

rank strikes by prediction.

Calculate:

* top decile average realized target
* bottom decile average realized target

Then:

```
Cheap-Rich Spread =
    Top-decile realized target
    -
    Bottom-decile realized target
```

Aggregate through the test period.

Show:

* mean spread
* median spread
* standard error
* t-statistic if statistically meaningful
* cumulative spread through time

Also evaluate:

* top 20% vs bottom 20%
* top 10% vs bottom 10%
* top 5% vs bottom 5%

Do not force significance claims if sample sizes are insufficient.

---

## VERTICAL RELATIVE-VALUE ANALYSIS

For options with the same:

* timestamp
* expiry
* option type

compare strike predictions.

Identify candidate pairs where:

```
one strike is strongly CHEAP
another is strongly RICH
```

Calculate a relative signal such as:

```
prediction_difference =
    predicted_y_long -
    predicted_y_short
```

Because predictions are Vega-normalized, also show an approximate common-Vega comparison.

Do not automatically claim an executable vertical trade unless position sizing and transaction costs are modeled.

The dashboard should present these as:

```
relative-value vertical candidates
```

Provide:

* long strike
* short strike
* expiry
* individual predictions
* prediction spread
* conviction
* IVs
* relative IV
* surface residual
* regime
* realized subsequent outcome for historical/test observations

---

## REGIME EVALUATION

Evaluate model performance separately in:

* LOW
* NORMAL
* HIGH
* CRISIS

Use the highest filtered regime probability as the display regime, but retain probabilities for analysis.

For every regime show:

* observation count
* MAE
* RMSE
* Spearman IC
* sign hit rate
* cheap-rich spread
* average conviction
* average prediction

This will tell us where the strategy/model actually works.

---

## REGIME ABLATION TEST

Train two otherwise-identical XGBoost models:

MODEL A:
all features except HMM regime probabilities/entropy

MODEL B:
complete feature set including regime probabilities/entropy

Compare them ONLY using validation first.

After design is frozen, report final Month-12 comparison.

Show whether regime information improved:

* RMSE
* MAE
* Spearman IC
* sign hit rate
* high-conviction hit rate
* cheap-rich spread

If regime does not help, say so clearly.

Do not force a positive conclusion.

---

## SHAP / MODEL EXPLANATION

Use SHAP TreeExplainer.

Provide global analysis:

* top 20 features by mean absolute SHAP
* beeswarm-style visualization
* dependence plots for important variables
* moneyness
* relative_iv
* surface_residual
* local_skew
* curvature
* p_crisis
* p_high
* realized volatility
* gamma
* theta

Show important interactions where useful, especially:

* moneyness × IV
* moneyness × surface residual
* regime × relative IV
* regime × skew
* regime × surface residual
* gamma × realized volatility
* Vega × IV movement

For individual option predictions, display:

```
baseline prediction
+ each major SHAP contribution
= final prediction
```

This should make it possible to explain:

"Why did the model label this 10-delta put rich even though high IV is normal for deep OTM puts?"

The explanation must make clear whether the decision came from:

* its absolute IV,
* its moneyness,
* local surface residual,
* curve steepness,
* regime,
* recent skew movement,
* Greeks,
* or other features.

---

## OTM PUT SAFEGUARD

Explicitly verify that the model is NOT simply learning:

```
high IV = rich.
```

Create dedicated diagnostics.

Bin options by moneyness/delta.

For each bucket show:

* average IV
* average prediction
* realized target
* surface residual
* sign hit rate

Specifically test deep OTM puts.

Plot:

```
raw IV vs prediction
```

and:

```
surface residual vs prediction
```

conditional on moneyness.

If raw IV dominates predictions simply because downside puts have higher IV, flag this as a potential model failure.

The desired behaviour is:

```
IV is evaluated conditional on
moneyness + surface shape + regime + other features.
```

---

## BASELINES

Compare XGBoost against simple baselines:

1. Always predict zero

2. Linear/Ridge regression using the same features

3. Raw IV percentile or relative-IV percentile baseline if feasible

4. Surface-residual-only baseline

This tells us whether the ML complexity actually adds value.

Do not hide a result where a simple model beats XGBoost.

---

## DASHBOARD

Build an actual responsive professional web application.

Do not use Streamlit.

Use:

```
Next.js
React
TypeScript
Tailwind CSS
```

Charts should use a professional library such as Apache ECharts.

The UI should feel similar to a modern institutional analytics/trading platform:

* restrained
* dense but readable
* dark/light contrast handled professionally
* excellent typography
* subtle animations
* fast transitions
* responsive layouts
* consistent spacing
* no gaudy gradients
* no huge rounded cards everywhere
* no random emojis
* no generic "AI dashboard" styling
* no fake glowing neon UI
* no excessive glassmorphism
* no chatbot aesthetic

Prefer:

* near-black / charcoal / off-white neutral palette
* one restrained accent colour
* subtle borders
* compact cards
* information-dense tables
* excellent hover states
* smooth chart interactions
* polished skeleton loading states
* sticky navigation
* smooth page transitions

Use real data everywhere.

Never place fake placeholder metrics once the pipeline is working.

---

## DASHBOARD PAGE 1 — EXECUTIVE OVERVIEW

Top bar:

* dataset period
* train period
* validation period
* test period
* number of test observations
* model version

Hero metrics:

* Test Spearman IC
* Test sign hit rate
* High-conviction hit rate
* Test RMSE
* Test R²
* Cheap-Rich top/bottom decile spread
* percentage Cheap / Neutral / Rich

Visualizations:

1. cumulative Cheap-Rich performance
2. prediction vs realized scatter
3. hit rate by conviction decile
4. model performance by regime
5. prediction distribution
6. realized target distribution

Include a clear text conclusion generated deterministically from metrics, not from an LLM.

Example:

"Signals became more reliable with higher conviction" only if the data actually show this.

---

## DASHBOARD PAGE 2 — TEST-MONTH EXPLORER

Allow selection of:

* date
* timestamp
* expiry
* calls/puts
* DTE range

Show the full option smile.

Interactive chart:

X-axis:
log-moneyness or strike

Y-axis:
IV

Overlay:

* market IV
* smoothed expected local surface
* points coloured by Cheap / Neutral / Rich

Hover tooltip:

* strike
* IV
* ATM IV
* relative IV
* surface residual
* delta
* gamma
* vega
* theta
* model prediction
* conviction
* realized next-hour target
* whether direction was correct

Additional chart:

Prediction by strike.

Show zero line.

Positive bars/points = cheap.
Negative = rich.

---

## DASHBOARD PAGE 3 — OPTION INSPECTOR

Click any option.

Show:

* contract metadata
* price
* strike
* expiry
* DTE
* moneyness
* IV
* relative IV
* surface residual
* local skew
* curvature
* Greeks
* VIX/SKEW if available
* regime probabilities
* prediction
* conviction
* Cheap/Rich/Neutral label
* realized next-hour result
* prediction correct/incorrect

Add an excellent SHAP waterfall chart.

Example text:

"Primary drivers toward CHEAP:

1. negative surface residual
2. elevated realized volatility relative to implied
3. falling skew

Primary offset:
high-volatility regime."

The explanation must come directly from computed SHAP values.

---

## DASHBOARD PAGE 4 — VOLATILITY REGIMES

Show:

* ATM IV time series
* coloured background for most-likely regime
* stacked probability chart:
  Low / Normal / High / Crisis
* regime entropy
* transition matrix
* average ATM IV by regime
* duration statistics
* observations per regime

Also show model performance by regime.

---

## DASHBOARD PAGE 5 — MODEL QUALITY

Sections:

Regression:

* MAE
* RMSE
* R²
* Pearson
* Spearman

Direction:

* hit rate
* rich hit rate
* cheap hit rate
* confusion-style directional table

Calibration/conviction:

* hit rate vs conviction
* realized mean vs prediction decile
* predicted vs realized bucket chart

Ranking:

* top/bottom portfolio results
* cumulative Cheap-Rich spread
* rank IC through time

Add rolling 1-day or appropriate rolling rank correlation if sample sizes permit.

---

## DASHBOARD PAGE 6 — FEATURE RESEARCH

Show:

* SHAP global importance
* SHAP beeswarm
* selected feature dependence plots
* feature interaction plots
* correlation matrix for core features
* feature distributions by train/validation/test

Highlight distribution shifts.

Especially inspect:

* IV
* moneyness
* surface residual
* p_crisis
* p_high
* gamma
* vega
* realized volatility

---

## DASHBOARD PAGE 7 — OTM PUT DIAGNOSTICS

Dedicated page.

Purpose:

prove that deep OTM puts are not automatically classified as rich just because IV is high.

Show:

* moneyness vs IV
* moneyness vs prediction
* relative IV vs prediction
* surface residual vs prediction
* prediction distributions by delta/moneyness bucket
* hit rates by delta/moneyness bucket

Provide separate statistics for approximately:

* ATM
* 25-delta
* 10-delta
* deep OTM

Use actual available delta ranges rather than forcing bins with insufficient data.

---

## DASHBOARD PAGE 8 — RELATIVE-VALUE / VERTICAL SCANNER

For every timestamp and expiry, rank strikes.

Create a table of strongest relative-value pairs.

Columns:

* timestamp
* expiry
* option type
* long/cheap strike
* short/rich strike
* long prediction
* short prediction
* prediction difference
* long IV
* short IV
* relative IVs
* surface residuals
* current regime
* conviction
* realized next-hour spread outcome if available

Allow sorting/filtering.

Provide an interactive smile chart when a pair is clicked.

---

## DASHBOARD PAGE 9 — DATA QUALITY

Show:

* raw observations
* valid observations
* dropped observations and reasons
* missing values
* target match success
* quote coverage by hour
* strikes per expiry
* observations by month
* Vega distributions
* DTE coverage
* unmatched t+1h observations
* duplicates

Everything must be transparent.

---

## FINAL RESEARCH SUMMARY

Automatically generate a static research summary file after training:

```
reports/model_report.html
reports/model_report.json
```

Also export important tables:

```
predictions_test.parquet/csv
metrics.json
conviction_analysis.csv
regime_metrics.csv
feature_importance.csv
vertical_candidates.csv
data_quality.json
```

The report must answer:

1. Does the model beat zero prediction?
2. Does it beat linear regression?
3. Does it beat simple IV/surface-residual baselines?
4. Did regime features help?
5. What is final unseen-test Spearman IC?
6. What is final unseen-test direction hit rate?
7. Does hit rate improve with conviction?
8. Are high-conviction Cheap/Rich predictions meaningfully better?
9. Is Cheap-minus-Rich realized performance positive?
10. Which regimes work best/worst?
11. Does the model accidentally treat deep OTM puts as rich because IV is naturally high?
12. Which features actually drive predictions?
13. Where does the model fail?
14. How much data was lost during cleaning/target alignment?
15. Is there enough evidence to call the output useful for further research?

Do not sugar-coat weak results.

If the model does not work, the report should say so.

---

## PROJECT STRUCTURE

Use a clean structure similar to:

project/
data/
raw/
processed/

```
quant/
    config.py
    ingestion.py
    cleaning.py
    forwards.py
    surface.py
    features.py
    regimes.py
    target.py
    train.py
    evaluate.py
    shap_analysis.py
    verticals.py
    export_results.py

models/
    hmm/
    xgboost/

results/
    predictions/
    metrics/
    charts/
    artifacts/

dashboard/
    app/
    components/
    lib/
    public/

reports/

tests/

README.md
requirements.txt
package.json
```

Keep quantitative logic outside frontend code.

The frontend should consume exported results/API endpoints.

---

## PERFORMANCE

The dataset may be large.

Optimize appropriately:

* vectorized operations
* parquet caching
* avoid unnecessary row loops
* cache fitted surfaces where possible
* batch SHAP calculations
* precompute expensive dashboard metrics
* do not recalculate the entire ML pipeline every time the webpage loads

Provide commands such as:

```
python -m quant.train
python -m quant.evaluate
npm run dev
```

or one simple orchestration command.

---

## TESTS

Write unit/integration tests for at least:

* no future leakage in lag features
* t+1h target alignment
* forward calculation
* log-moneyness
* leave-one-out surface residual
* HMM filtering chronology
* train/validation/test date separation
* Vega-normalized target
* prediction labels
* conviction calculation

Add assertions that guarantee:

```
max(train_time) < min(validation_time)
max(validation_time) < min(test_time)
```

---

## IMPORTANT RESEARCH RULES

1. No random train-test split.

2. No future leakage.

3. No full-sample normalization.

4. No HMM smoothing using future observations.

5. No test-set hyperparameter tuning.

6. Do not optimize the project to create impressive-looking results.

7. If results are weak, display them honestly.

8. Do not treat high raw IV as synonymous with rich.

9. Always condition IV interpretation on moneyness and the surrounding surface.

10. Do not claim profitability without transaction-cost evidence.

11. Do not claim statistical significance unless calculated properly.

12. Avoid highly complex deep-learning models in V1.

13. XGBoost is the primary model.

14. Make every result reproducible with a fixed random seed.

15. Log the exact configuration and model version used.

---

## UI QUALITY BAR

Treat UI quality as seriously as model quality.

I want something I could show to:

* a quantitative researcher,
* an options trader,
* a portfolio manager,
* or an interview panel

without it looking like a college project.

Requirements:

* polished typography
* professional spacing
* fast and smooth transitions
* responsive
* desktop-first but usable on tablets
* interactive tooltips
* no chart clutter
* tables with sorting/filtering
* virtualization for large tables
* sticky table headers
* tasteful loading states
* informative empty states
* keyboard-friendly interactions where sensible
* no unnecessary animation
* no fake data

Use animation only where it improves comprehension.

---

## FINAL ACCEPTANCE CRITERIA

The project is complete only if I can:

1. place the one-year SPX options dataset into the project,
2. run one command to preprocess/train/evaluate,
3. get a strictly chronological 10-month train / 1-month validation / 1-month test experiment where possible,
4. see the final unseen-test results,
5. see whether high-conviction predictions were actually more accurate,
6. see whether regime information improved the model,
7. inspect any individual option and understand WHY it was classified cheap/rich,
8. verify deep OTM puts are not automatically called rich simply because of higher natural IV,
9. inspect the entire smile at a historical timestamp,
10. identify relative-value vertical candidates,
11. inspect realized outcomes afterward,
12. open a beautiful professional web dashboard,
13. reproduce the experiment from the README.

Before implementing the final ML pipeline, inspect the provided dataset and create a short DATA_AUDIT.md documenting the actual schema, timestamp frequency, date range, missingness, and any assumptions required.

Then implement the entire project end to end.

Do not stop after scaffolding.

Run the pipeline on the actual data, fix errors, validate outputs, start the dashboard, verify that the pages render correctly, and ensure all displayed metrics come from the real generated results.
