# Calibrated, Cost-Aware Rainfall Prediction (Route A)

MSc dissertation project (CT7P01NI) — Upasana Koirala, London Metropolitan
University / Islington College.

Binary classification of `RainTomorrow` (Yes/No) on the Kaggle "Rain in
Australia" (weatherAUS) dataset, built as a deployable, calibrated,
explainable Route-A pipeline:

1. **Baseline** — Logistic Regression, Decision Tree, Random Forest, XGBoost,
   fixed seed (42), leakage-safe (imputation/scaling inside the CV pipeline).
2. **Evaluation harness** — PR-AUC as the headline metric (the target is
   ~22% positive, so accuracy is misleading), plus per-class recall,
   precision, ROC-AUC, Brier score, and 5-fold stratified CV mean ± sd.
3. **Calibration** — Platt (sigmoid) and isotonic calibration of the best
   baseline model, with reliability diagrams, Brier score, and Expected
   Calibration Error (ECE) before vs after.
4. **Cost-sensitive threshold** — decision threshold that minimises expected
   cost = C_FN·FN + C_FP·FP (a missed rain day costs more than a false
   alarm), with a sensitivity sweep across FN:FP ratios rather than a single
   assumed "true" cost.
5. **Explainability** — SHAP global summary and per-prediction explanations
   (TreeExplainer for tree models, LinearExplainer for Logistic Regression).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Download `weatherAUS.csv` from Kaggle ("Rain in Australia" dataset) and
place it at `data/weatherAUS.csv`. The raw CSV is not committed to this
repo (see `.gitignore`) due to file size and Kaggle's redistribution terms.

## Run

```bash
# Mac/Linux/WSL:
WEATHERAUS_CSV=./data/weatherAUS.csv python3 rainfall_reference.py

# Windows PowerShell:
$env:WEATHERAUS_CSV="data\weatherAUS.csv"; python rainfall_reference.py
```

Running with no `WEATHERAUS_CSV` set falls back to a synthetic proxy dataset
purely so the script always runs end to end — real results require the real
CSV.

## Outputs

- `results/` — baseline_metrics.csv, calibration_summary.csv,
  cost_threshold_sweep.csv, shap_mean_abs.csv, shap_per_prediction_examples.csv
- `figures/` — reliability_calibration.png, pr_curve_operating_point.png,
  shap_summary.png

## Fix log

- **2026-08-28**: Fixed a dtype-detection bug in `load_data()` — pandas
  versions that store text columns as the newer `StringDtype` (instead of
  legacy `object` dtype) caused the Yes/No → 1/0 mapping for `RainToday`/
  `RainTomorrow` to be silently skipped, crashing later on
  `.astype(int)`. Fixed by also checking `pd.api.types.is_string_dtype`.
