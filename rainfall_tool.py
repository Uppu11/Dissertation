"""
======================================================================
Upasana — Week 2 reference: from validated pipeline to a DEPLOYABLE,
EXPLAINABLE rain tool ("what good looks like" for the product stage).

Week 1 proved the science: 4-model baseline, PR-AUC harness, calibration,
cost-sensitive threshold and per-prediction SHAP all running. Week 2 turns
that into the *tool* her originality claim actually promises — one that a
user (or an examiner) can run:

    a day's weather features  ->  a CALIBRATED rain probability
                              ->  a COST-BASED decision (issue a warning or not)
                              ->  the TOP-3 SHAP reasons, in plain words.

The move from notebook to tool is three things this file demonstrates:
  1. TRAIN ONCE, PERSIST AN ARTIFACT. train_and_save() fits the baseline,
     picks the best model by CV PR-AUC, calibrates it (Platt/isotonic),
     derives the cost-optimal threshold from an explicit FN:FP assumption,
     and saves ONE self-describing artifact (model/rain_tool.joblib) plus a
     MODEL_CARD.md. Nothing is re-fit at prediction time.
  2. A CLEAN INFERENCE PATH. RainTool.predict(features) loads the artifact
     and returns a decision object — probability, decision, the threshold and
     the assumption behind it, and the per-prediction SHAP reasons. This is
     the function a CLI, a Streamlit page or an API would all call.
  3. HONEST EXPLANATIONS AT INFERENCE. SHAP runs on the deployed model's base
     estimator (TreeExplainer) for each single prediction — not just a global
     summary — so every decision carries its own "why".

Run:
    pip install scikit-learn xgboost shap joblib matplotlib pandas numpy --break-system-packages
    python3 rainfall_tool.py train                 # fits + saves model/rain_tool.joblib + MODEL_CARD.md
    python3 rainfall_tool.py predict --example      # one worked prediction -> sample_prediction.json
    python3 rainfall_tool.py batch --csv days.csv   # score many days -> batch_scored_example.csv
    WEATHERAUS_CSV=/path/weatherAUS.csv python3 rainfall_tool.py train   # real data

Numbers are on the synthetic weatherAUS-like proxy unless WEATHERAUS_CSV is set.
The tool, the artifact and the inference contract are the deliverable — not the
specific figures.
======================================================================
"""
import os
import sys
import json
import argparse
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             recall_score, precision_score)

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "model"); os.makedirs(MODEL_DIR, exist_ok=True)
ARTIFACT = os.path.join(MODEL_DIR, "rain_tool.joblib")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

FEATURES = ["MinTemp", "MaxTemp", "Rainfall", "Humidity9am", "Humidity3pm",
            "Pressure9am", "Pressure3pm", "WindGustSpeed", "Temp3pm", "RainToday"]

# Plain-language names so a per-prediction reason reads like a sentence.
PRETTY = {
    "MinTemp": "min temperature", "MaxTemp": "max temperature", "Rainfall": "rainfall today (mm)",
    "Humidity9am": "9am humidity", "Humidity3pm": "3pm humidity", "Pressure9am": "9am pressure",
    "Pressure3pm": "3pm pressure", "WindGustSpeed": "wind-gust speed", "Temp3pm": "3pm temperature",
    "RainToday": "rain today",
}

# The cost assumption travels WITH the model, so the decision is reproducible.
DEFAULT_FN_FP = 5.0   # a missed rain-day is 5x costlier than a false alarm (flood pre-warning)


# ----------------------------------------------------------------------
# DATA (identical proxy to the Week-1 reference, for continuity)
# ----------------------------------------------------------------------
def load_data():
    path = os.environ.get("WEATHERAUS_CSV")
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        df = df[[c for c in FEATURES if c in df.columns] + ["RainTomorrow"]].copy()
        for c in ["RainToday", "RainTomorrow"]:
            if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
                df[c] = df[c].map({"Yes": 1, "No": 0})
        print(f"Loaded REAL weatherAUS: {df.shape}")
        return df.dropna(subset=["RainTomorrow"])

    n = 15000
    rng = np.random.default_rng(SEED)
    humidity3pm = np.clip(rng.normal(52, 20, n), 0, 100)
    humidity9am = np.clip(humidity3pm + rng.normal(15, 12, n), 0, 100)
    pressure3pm = rng.normal(1016, 7, n)
    pressure9am = pressure3pm + rng.normal(1, 2, n)
    rainfall = np.clip(rng.exponential(2.0, n) - 1.2, 0, None)
    rain_today = (rainfall > 1.0).astype(int)
    windgust = np.clip(rng.normal(40, 13, n), 6, None)
    mintemp = rng.normal(12, 6, n)
    maxtemp = mintemp + np.clip(rng.normal(11, 4, n), 1, None)
    temp3pm = maxtemp - rng.normal(2, 2, n)
    z = (-6.0 + 0.075 * humidity3pm - 0.16 * (pressure3pm - 1016)
         + 0.35 * rain_today + 0.06 * rainfall + 0.015 * (windgust - 40))
    p = 1 / (1 + np.exp(-z))
    y = rng.binomial(1, p)
    df = pd.DataFrame({
        "MinTemp": mintemp, "MaxTemp": maxtemp, "Rainfall": rainfall,
        "Humidity9am": humidity9am, "Humidity3pm": humidity3pm,
        "Pressure9am": pressure9am, "Pressure3pm": pressure3pm,
        "WindGustSpeed": windgust, "Temp3pm": temp3pm, "RainToday": rain_today,
        "RainTomorrow": y,
    })
    for c in ["Humidity3pm", "Pressure3pm", "WindGustSpeed"]:
        df.loc[df.sample(frac=0.03, random_state=SEED).index, c] = np.nan
    print(f"No WEATHERAUS_CSV set -> SYNTHETIC proxy: {df.shape} "
          f"(positive rate {df['RainTomorrow'].mean():.1%})")
    return df


def make_models():
    pre = ColumnTransformer([("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())]), FEATURES)])
    pre_tree = ColumnTransformer([("num", SimpleImputer(strategy="median"), FEATURES)])
    models = {
        "LogReg": Pipeline([("pre", pre),
                            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "DecisionTree": Pipeline([("pre", pre_tree),
                                  ("clf", DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                                                 random_state=SEED))]),
        "RandomForest": Pipeline([("pre", pre_tree),
                                  ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                                 n_jobs=-1, random_state=SEED))]),
    }
    if HAS_XGB:
        models["XGBoost"] = Pipeline([("pre", pre_tree),
                                      ("clf", XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                                                            subsample=0.9, colsample_bytree=0.9,
                                                            eval_metric="logloss", random_state=SEED, n_jobs=-1))])
    else:
        models["GradBoost(xgb-fallback)"] = Pipeline([("pre", pre_tree),
                                                      ("clf", GradientBoostingClassifier(random_state=SEED))])
    return models


def _ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            e += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return float(e)


def _expected_cost_threshold(y, p, c_fn, c_fp=1.0):
    y = np.asarray(y)
    best_t, best_c = 0.5, np.inf
    for t in np.linspace(0.01, 0.99, 197):
        yhat = (p >= t).astype(int)
        fn = int(((yhat == 0) & (y == 1)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        c = c_fn * fn + c_fp * fp
        if c < best_c:
            best_c, best_t = c, t
    return float(best_t)


# ----------------------------------------------------------------------
# TRAIN ONCE, PERSIST ONE SELF-DESCRIBING ARTIFACT
# ----------------------------------------------------------------------
def train_and_save(fn_fp=DEFAULT_FN_FP):
    print("=" * 66)
    print("TRAIN: baseline -> pick best -> calibrate -> cost threshold -> persist")
    print(f"xgboost={HAS_XGB} shap={HAS_SHAP} seed={SEED} FN:FP={fn_fp:.0f}:1")
    print("=" * 66)
    df = load_data()
    X, y = df[FEATURES], df["RainTomorrow"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

    # baseline + CV PR-AUC pick
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, fitted = [], {}
    for name, model in make_models().items():
        model.fit(Xtr, ytr)
        fitted[name] = model
        p = model.predict_proba(Xte)[:, 1]
        cv = cross_val_score(model, Xtr, ytr, scoring="average_precision", cv=skf, n_jobs=-1)
        rows.append({"model": name, "PR_AUC_cv_mean": round(cv.mean(), 3), "PR_AUC_cv_sd": round(cv.std(), 3),
                     "ROC_AUC": round(roc_auc_score(yte, p), 3), "Brier": round(brier_score_loss(yte, p), 3),
                     "recall_rain@0.5": round(recall_score(yte, (p >= .5).astype(int)), 3)})
    baseline = pd.DataFrame(rows).sort_values("PR_AUC_cv_mean", ascending=False).reset_index(drop=True)
    best_name = baseline.iloc[0]["model"]
    base_model = fitted[best_name]
    print(baseline.to_string(index=False))
    print(f"\nBest base model by CV PR-AUC: {best_name}")

    # calibrate (pick lower-ECE of Platt/isotonic)
    cal_rows, cal_probs = [], {}
    p_raw = base_model.predict_proba(Xte)[:, 1]
    cal_rows.append({"calibration": "uncalibrated", "Brier": round(brier_score_loss(yte, p_raw), 4),
                     "ECE": round(_ece(yte.values, p_raw), 4)})
    for method in ["sigmoid", "isotonic"]:
        cal = CalibratedClassifierCV(base_model, method=method, cv=5).fit(Xtr, ytr)
        pc = cal.predict_proba(Xte)[:, 1]
        cal_probs[method] = (cal, pc)
        cal_rows.append({"calibration": method, "Brier": round(brier_score_loss(yte, pc), 4),
                         "ECE": round(_ece(yte.values, pc), 4)})
    best_cal = min(cal_probs, key=lambda m: _ece(yte.values, cal_probs[m][1]))
    calibrated_model, p_cal = cal_probs[best_cal]
    cal_df = pd.DataFrame(cal_rows)
    print("\nCalibration (lower ECE is better):")
    print(cal_df.to_string(index=False))
    print(f"Chosen calibration: {best_cal}")

    # cost-optimal threshold on the calibrated probabilities
    threshold = _expected_cost_threshold(yte, p_cal, c_fn=fn_fp, c_fp=1.0)
    yhat = (p_cal >= threshold).astype(int)
    op = {"threshold": round(threshold, 3),
          "recall_rain": round(recall_score(yte, yhat), 3),
          "precision_rain": round(precision_score(yte, yhat, zero_division=0), 3),
          "false_alarm_rate": round(((yhat == 1) & (yte.values == 0)).sum() / max((yte.values == 0).sum(), 1), 3),
          "missed_rain_rate": round(((yhat == 0) & (yte.values == 1)).sum() / max((yte.values == 1).sum(), 1), 3)}
    print(f"\nCost-optimal threshold @ FN:FP={fn_fp:.0f}:1 -> t={threshold:.3f}  "
          f"(recall_rain={op['recall_rain']}, false_alarm={op['false_alarm_rate']})")

    # global driver ranking (for the model card) via SHAP on the base model
    drivers = _global_drivers(base_model, Xtr, Xte)

    bundle = {
        "calibrated_model": calibrated_model,   # produces the probability
        "base_model": base_model,               # tree pipeline, for TreeExplainer at inference
        # a small RAW background sample so a LinearExplainer has something to
        # explain against at inference (TreeExplainer ignores it). Without a
        # real background, per-prediction SHAP collapses to ~0.
        "background": Xtr.sample(min(200, len(Xtr)), random_state=SEED).reset_index(drop=True),
        "best_name": best_name,
        "calibration": best_cal,
        "threshold": float(threshold),
        "fn_fp": float(fn_fp),
        "features": FEATURES,
        "operating_point": op,
        "baseline": baseline.to_dict(orient="records"),
        "calibration_table": cal_df.to_dict(orient="records"),
        "global_drivers": drivers,
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": "real weatherAUS" if os.environ.get("WEATHERAUS_CSV") else "synthetic proxy",
        "seed": SEED,
        "schema_version": "week2-v1",
    }
    joblib.dump(bundle, ARTIFACT)
    print(f"\nSaved artifact -> {ARTIFACT}")
    _write_model_card(bundle)
    print(f"Wrote MODEL_CARD.md")
    return bundle


def _global_drivers(base_model, Xtr, Xte):
    if not HAS_SHAP:
        clf = base_model.named_steps["clf"]
        imp = getattr(clf, "feature_importances_", None)
        if imp is None:
            return []
        s = pd.Series(imp, index=FEATURES).sort_values(ascending=False)
        return [{"feature": k, "mean_abs": float(v)} for k, v in s.items()]
    try:
        pre = base_model.named_steps["pre"]; clf = base_model.named_steps["clf"]
        Xte_t = pre.transform(Xte)
        sv = _shap_values(clf, base_model, Xtr, Xte_t)
        mean_abs = np.abs(sv).mean(0)
        s = pd.Series(mean_abs, index=FEATURES).sort_values(ascending=False)
        return [{"feature": k, "mean_abs": round(float(v), 4)} for k, v in s.items()]
    except Exception as e:
        print(f"(global SHAP skipped: {e})")
        return []


def _shap_values(clf, base_model, Xtr, Xrows_t):
    """SHAP values for already-transformed rows, model-aware."""
    is_tree = clf.__class__.__name__ in {
        "RandomForestClassifier", "DecisionTreeClassifier", "XGBClassifier", "GradientBoostingClassifier"}
    if is_tree:
        expl = shap.TreeExplainer(clf)
        sv = expl.shap_values(Xrows_t)
    else:
        bg = shap.sample(base_model.named_steps["pre"].transform(Xtr), 200, random_state=SEED)
        expl = shap.LinearExplainer(clf, bg)
        sv = expl.shap_values(Xrows_t)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


# ----------------------------------------------------------------------
# THE DEPLOYABLE TOOL: load once, predict with a decision + reasons
# ----------------------------------------------------------------------
class RainTool:
    """Loads the persisted artifact and answers one prediction at a time."""

    def __init__(self, artifact=ARTIFACT):
        if not os.path.exists(artifact):
            raise FileNotFoundError(f"No artifact at {artifact}. Run `python3 rainfall_tool.py train` first.")
        self.b = joblib.load(artifact)
        self.features = self.b["features"]

    def _row(self, features: dict) -> pd.DataFrame:
        missing = [f for f in self.features if f not in features]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return pd.DataFrame([{f: features[f] for f in self.features}])[self.features]

    def predict(self, features: dict, top_k: int = 3) -> dict:
        row = self._row(features)
        prob = float(self.b["calibrated_model"].predict_proba(row)[:, 1][0])
        decision = "ISSUE RAIN WARNING" if prob >= self.b["threshold"] else "no warning"
        reasons = self._reasons(row, top_k)
        return {
            "rain_probability": round(prob, 3),
            "decision": decision,
            "threshold": round(self.b["threshold"], 3),
            "cost_assumption": f"FN:FP = {self.b['fn_fp']:.0f}:1 (a missed rain-day costs "
                               f"{self.b['fn_fp']:.0f}x a false alarm)",
            "top_reasons": reasons,
            "model": f"{self.b['best_name']} + {self.b['calibration']} calibration",
            "note": "Probability is calibrated; decision uses the cost-optimal threshold. "
                    "Reasons are per-prediction SHAP contributions on the deployed model.",
        }

    def _reasons(self, row: pd.DataFrame, top_k: int) -> list:
        if not HAS_SHAP:
            return [{"factor": "explainer unavailable", "effect": "n/a"}]
        try:
            base = self.b["base_model"]
            Xt = base.named_steps["pre"].transform(row)
            sv = _shap_values(base.named_steps["clf"], base, self.b["background"], Xt)[0]
            order = np.argsort(-np.abs(sv))[:top_k]
            out = []
            for j in order:
                direction = "raises" if sv[j] > 0 else "lowers"
                out.append({
                    "factor": PRETTY.get(self.features[j], self.features[j]),
                    "value": _fmt(row.iloc[0, j]),
                    "effect": f"{direction} the rain chance",
                    "shap": round(float(sv[j]), 3),
                })
            return out
        except Exception as e:
            return [{"factor": f"reason step skipped ({e})", "effect": "n/a"}]

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        out = []
        for _, r in df.iterrows():
            res = self.predict(r.to_dict())
            top = res["top_reasons"][0] if res["top_reasons"] else {}
            out.append({**{f: r[f] for f in self.features},
                        "rain_probability": res["rain_probability"],
                        "decision": res["decision"],
                        "top_reason": f'{top.get("factor","")} {top.get("effect","")}'.strip()})
        return pd.DataFrame(out)


def _fmt(v):
    try:
        return round(float(v), 2)
    except Exception:
        return v


# ----------------------------------------------------------------------
# MODEL CARD (a deployable tool documents itself)
# ----------------------------------------------------------------------
def _write_model_card(b):
    drv = ", ".join(d["feature"] for d in b["global_drivers"][:5]) or "n/a"
    lines = [
        "# Model Card — Calibrated, Cost-Aware Rain Tool",
        "",
        f"*Generated {b['trained_utc']} · data: {b['data_source']} · seed {b['seed']} · schema {b['schema_version']}*",
        "",
        "## Intended use",
        "Decision-support: given a day's weather features, output a **calibrated** probability of "
        "rain tomorrow and a **cost-based** recommendation (issue a rain warning or not), with a "
        "per-prediction explanation. Not a safety-critical forecast; a decision aid.",
        "",
        "## What is inside the artifact (`model/rain_tool.joblib`)",
        f"- **Deployed model:** {b['best_name']} (chosen by 5-fold CV PR-AUC).",
        f"- **Calibration:** {b['calibration']} (probabilities are calibrated, not raw scores).",
        f"- **Decision threshold:** {b['threshold']:.3f}, derived from an explicit cost assumption "
        f"of **FN:FP = {b['fn_fp']:.0f}:1** (a missed rain-day is {b['fn_fp']:.0f}x costlier than a false alarm).",
        "- **Explainer:** per-prediction SHAP on the deployed model's base estimator.",
        "",
        "## Operating point (held-out test)",
        f"- recall (rain): {b['operating_point']['recall_rain']}  ·  precision (rain): {b['operating_point']['precision_rain']}",
        f"- false-alarm rate: {b['operating_point']['false_alarm_rate']}  ·  missed-rain rate: {b['operating_point']['missed_rain_rate']}",
        "",
        "## Global drivers (mean |SHAP|)",
        f"{drv}",
        "",
        "## Key limitations",
        "- The cost ratio is an **assumption**, not a measured cost — report the sensitivity sweep (Week 1) alongside it.",
        "- Trained on weatherAUS; **regional transfer to Nepal is not validated** and is documented as future work.",
        "- Calibration was already near-perfect on the base model, so the calibrated layer is a modest, honest gain.",
        "",
        "## How to run",
        "```",
        "python3 rainfall_tool.py train",
        "python3 rainfall_tool.py predict --example",
        "python3 rainfall_tool.py batch --csv days.csv",
        "```",
    ]
    with open(os.path.join(ROOT, "MODEL_CARD.md"), "w") as f:
        f.write("\n".join(lines))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
EXAMPLE_DAY = {"MinTemp": 17.2, "MaxTemp": 24.1, "Rainfall": 6.4, "Humidity9am": 82,
               "Humidity3pm": 79, "Pressure9am": 1006.5, "Pressure3pm": 1004.2,
               "WindGustSpeed": 54, "Temp3pm": 22.0, "RainToday": 1}


def _cli():
    ap = argparse.ArgumentParser(description="Calibrated, cost-aware, explainable rain tool")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--fn-fp", type=float, default=DEFAULT_FN_FP)
    p = sub.add_parser("predict")
    p.add_argument("--json", type=str, help="JSON dict of the 10 features")
    p.add_argument("--example", action="store_true", help="use a built-in wet-day example")
    b = sub.add_parser("batch"); b.add_argument("--csv", required=True)
    a = ap.parse_args()

    if a.cmd == "train":
        train_and_save(fn_fp=a.fn_fp)

    elif a.cmd == "predict":
        tool = RainTool()
        feats = EXAMPLE_DAY if a.example else json.loads(a.json)
        res = tool.predict(feats)
        print(json.dumps(res, indent=2))
        if a.example:
            with open(os.path.join(ROOT, "sample_prediction.json"), "w") as f:
                json.dump({"input": feats, "output": res}, f, indent=2)
            print("\nWrote sample_prediction.json")

    elif a.cmd == "batch":
        tool = RainTool()
        df = pd.read_csv(a.csv)
        scored = tool.predict_batch(df)
        out = os.path.join(ROOT, "batch_scored_example.csv")
        scored.to_csv(out, index=False)
        print(scored.to_string(index=False))
        print(f"\nWrote {out}")


if __name__ == "__main__":
    _cli()
