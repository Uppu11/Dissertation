# Week 2 — Deployable, Calibrated, Cost-Aware Rain Tool

MSc Dissertation — Upasana Koirala (21040045) · Module CT7P01NI · Supervisor Rajesh Chhetry

This is the Week 2 stage: turning the validated Week 1 pipeline (4-model baseline, PR-AUC harness,
calibration, cost-sensitive threshold, per-prediction SHAP — all already in this repo as
`rainfall_reference.py`) into an actual **deployable tool**. `rainfall_tool.py` trains once, saves
a single self-describing model artifact, and `RainTool.predict()` answers one day's weather with a
calibrated probability, a cost-based decision, and the top-3 SHAP reasons — the same function the
CLI and the Streamlit page both call.

## What's in this folder

- `rainfall_tool.py` — the deployable tool: `train_and_save()`, `RainTool.predict()` /
  `predict_batch()`, model card writer, CLI. Seed 42, leakage-safe, same pipeline as Week 1.
- `app_streamlit.py` — a minimal one-page front-end over `RainTool.predict()`.
- `days.csv` — example batch input (4 days).
- `requirements.txt`, `.gitignore`

Produced by running the tool on the **real** weatherAUS data (not committed, see Data below):

- `MODEL_CARD.md` — auto-generated; documents intended use, what's in the artifact, the operating
  point, global drivers, and limitations.
- `sample_prediction.json` — one worked prediction (input + full decision object).
- `batch_scored_example.csv` — the 4 days in `days.csv`, scored end-to-end.
- `model/rain_tool.joblib` — the trained artifact (git-ignored — regenerate with `train`, don't commit
  a 6MB binary to the repo).

## Data

Same as Week 1: [Kaggle "Rain in Australia" (weatherAUS.csv)](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package).
Download it yourself and place it at `data/weatherAUS.csv` (git-ignored, not redistributed here).

## Setup (adding onto the existing Week 1 repo/venv)

You already have a venv from Week 1. Just reuse it:

```
cd path\to\rainfall_project_package        # PowerShell
venv\Scripts\Activate.ps1
pip install -r requirements.txt            # adds streamlit + joblib on top of Week 1's packages
```

(Mac/Linux: `source venv/bin/activate`)

## Run

```
# 1. Train: fits 4 models, picks best by CV PR-AUC, calibrates, derives the cost threshold,
#    saves ONE artifact + a model card. Point at your real CSV:
$env:WEATHERAUS_CSV="data\weatherAUS.csv"; python rainfall_tool.py train      # PowerShell
WEATHERAUS_CSV=data/weatherAUS.csv python3 rainfall_tool.py train             # Mac/Linux

# 2. One worked prediction (built-in wet-day example) -> sample_prediction.json
python rainfall_tool.py predict --example

# 3. Batch-score a CSV of days -> batch_scored_example.csv
python rainfall_tool.py batch --csv days.csv

# 4. The actual front-end — opens in your browser
streamlit run app_streamlit.py
```

## Real results (weatherAUS, seed 42, FN:FP = 5:1)

Deployed model: **XGBoost** (best by 5-fold CV PR-AUC = 0.708). Calibration: **isotonic** chosen
(uncalibrated ECE 0.0055 → isotonic 0.0054; sigmoid made it worse at 0.0317 — the base model was
already near-calibrated, a modest but honest gain, not the dramatic textbook case). Cost-optimal
threshold at FN:FP=5:1 → **t = 0.175**, recall (rain) = 0.827, false-alarm rate = 0.267. Top global
SHAP drivers: Humidity3pm, Pressure3pm, WindGustSpeed, Pressure9am, Rainfall — meteorologically
sane. Full detail in `MODEL_CARD.md`.

## Fix log

- **2026-09-05** — same pandas 3.x `StringDtype` issue as Week 1's `rainfall_reference.py`:
  `load_data()`'s check `df[c].dtype == object` silently failed to convert `"Yes"/"No"` to `1/0` on
  real data (pandas 3.0.x stores text columns as `StringDtype`, not legacy `object`), which would
  have crashed training with `ValueError: invalid literal for int() with base 10: 'No'`. Fixed by
  also checking `pd.api.types.is_string_dtype(df[c])`. Verified end-to-end on the real 145,460-row
  weatherAUS file: `train`, `predict --example`, `batch`, and `streamlit run` all confirmed working.
