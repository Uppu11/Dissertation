# Model Card — Calibrated, Cost-Aware Rain Tool

*Generated 2026-09-13T06:09:08+00:00 · data: real weatherAUS · seed 42 · schema week2-v1*

## Intended use
Decision-support: given a day's weather features, output a **calibrated** probability of rain tomorrow and a **cost-based** recommendation (issue a rain warning or not), with a per-prediction explanation. Not a safety-critical forecast; a decision aid.

## What is inside the artifact (`model/rain_tool.joblib`)
- **Deployed model:** XGBoost (chosen by 5-fold CV PR-AUC).
- **Calibration:** isotonic (probabilities are calibrated, not raw scores).
- **Decision threshold:** 0.165, derived from an explicit cost assumption of **FN:FP = 5:1** (a missed rain-day is 5x costlier than a false alarm).
- **Explainer:** per-prediction SHAP on the deployed model's base estimator.

## Operating point (held-out test)
- recall (rain): 0.843  ·  precision (rain): 0.457
- false-alarm rate: 0.289  ·  missed-rain rate: 0.157

## Global drivers (mean |SHAP|)
Humidity3pm, Pressure3pm, WindGustSpeed, Pressure9am, Rainfall

## Key limitations
- The cost ratio is an **assumption**, not a measured cost — report the sensitivity sweep (Week 1) alongside it.
- Trained on weatherAUS; **regional transfer to Nepal is not validated** and is documented as future work.
- Calibration was already near-perfect on the base model, so the calibrated layer is a modest, honest gain.

## How to run
```
python3 rainfall_tool.py train
python3 rainfall_tool.py predict --example
python3 rainfall_tool.py batch --csv days.csv
```