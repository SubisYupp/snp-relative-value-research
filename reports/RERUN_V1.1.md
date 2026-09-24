# Corrected-data model rerun ? September 24, 2026

The v1.1 model was retrained using corrected preprocessing. August remains the selection/calibration period; September is a **revised historical evaluation**, not a fresh untouched holdout. The original experiment is preserved in `experiments/snp-rv-v1.0.0/`.

| Metric | Original v1.0 | Corrected v1.1 |
|---|---:|---:|
| Spearman IC | 0.101108 | 0.085083 |
| Sign hit rate | 0.541508 | 0.532779 |
| High-conviction hit rate | 0.611314 | 0.577316 |
| RMSE | 3.429495 | 3.437942 |
| R-squared | 0.001993 | 0.001190 |

Matched September observations: 296,928 originally and 296,988 now. The evaluation populations differ after cleaning, so differences are not a controlled model-only comparison. Ranking spread also declined from 0.403037 to 0.289529. The corrected model still narrowly beats its own zero baseline on RMSE (3.437942 versus 3.440097).

Predictive rank correlation and hit rates are weaker. Cleaning was undertaken for data correctness, not to manufacture stronger performance. Small-Vega target sensitivity, uncertain vendor conventions and limited evaluation length remain relevant. See the current `model_report.html` and `model_report.json` for full benchmarks, regime diagnostics, conviction analysis and deterministic findings.

The completed active experiment is protected against implicit reruns. `python scripts/run.py train` serves the completion guard; an explicitly requested additional run can use `python scripts/run.py train --archive-and-rerun`, which archives the active experiment before refitting and labels the result as historical reevaluation. Archive folders are never silently overwritten.
