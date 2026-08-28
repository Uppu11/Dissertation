"""
======================================================================
Upasana — "Rain tomorrow?" as a CALIBRATED, COST-AWARE, EXPLAINABLE tool
Reference implementation ("what good looks like") for her committed Route A.

Route A (her words): a deployable rainfall tool that goes beyond model
comparison by delivering (1) calibrated rain probabilities, (2) a
cost-sensitive decision threshold from an explicit cost matrix, and
(3) per-prediction SHAP explanations. The four-model baseline is the shared
spine either route would use.

This script is the whole pipeline, leakage-safe, seed-fixed, end to end:
  1. Data ...... real weatherAUS if WEATHERAUS_CSV is set, else a realistic
                 synthetic proxy (~22% "rain tomorrow") so it always runs.
  2. Baseline .. LogisticRegression / DecisionTree / RandomForest / XGBoost,
                 fixed seed, in sklearn Pipelines (scaling only where needed).
  3. Harness ... PR-AUC headline (+ ROC-AUC, Brier), per-class recall/precision,
                 and k-fold mean +/- sd PR-AUC. PR-AUC is the right headline on
                 an imbalanced target, not accuracy.
  4. Calibrate . Platt (sigmoid) and isotonic on the best base model; reliability
                 curve + Brier + ECE BEFORE and AFTER. Reliability figure saved.
  5. Cost ...... expected-cost threshold from an explicit cost matrix
                 (missed-rain FN vs false-alarm FP), plus a SENSITIVITY SWEEP
                 over FN:FP ratios; chosen operating point marked on the PR curve.
  6. Explain ... per-prediction SHAP for sample instances (TreeExplainer for tree
                 models, LinearExplainer for LogReg); importance fallback if no shap.

Run:
    pip install scikit-learn xgboost shap matplotlib pandas numpy --break-system-packages
    python3 rainfall_reference.py                 # synthetic proxy
    WEATHERAUS_CSV=/path/weatherAUS.csv python3 rainfall_reference.py   # real data

Outputs: results/ (metrics, calibration, cost sweep) and figures/ (reliability,
PR curve with operating point, SHAP summary).
======================================================================
"""
import os
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             precision_recall_curve, recall_score, precision_score,
                             confusion_matrix, classification_report)

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results"); os.makedirs(RES, exist_ok=True)
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)

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


# ----------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------
def load_data():
    """Real weatherAUS if provided; else a realistic synthetic proxy."""
    path = os.environ.get("WEATHERAUS_CSV")
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        df = df[[c for c in FEATURES if c in df.columns] + ["RainTomorrow"]].copy()
        for c in ["RainToday", "RainTomorrow"]:
            if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
                df[c] = df[c].map({"Yes": 1, "No": 0})
        print(f"Loaded REAL weatherAUS: {df.shape}")
        return df.dropna(subset=["RainTomorrow"])

    # ---- synthetic proxy: correlated, imbalanced, learnable-but-not-trivial ----
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

    # true log-odds of rain tomorrow: humidity up, pressure down, rain today up
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
    # inject a little missingness so the imputation step is exercised
    for c in ["Humidity3pm", "Pressure3pm", "WindGustSpeed"]:
        df.loc[df.sample(frac=0.03, random_state=SEED).index, c] = np.nan
    print(f"No WEATHERAUS_CSV set -> using SYNTHETIC proxy: {df.shape} "
          f"(positive rate {df['RainTomorrow'].mean():.1%})")
    return df


# ----------------------------------------------------------------------
# 2. MODELS
# ----------------------------------------------------------------------
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
                                  ("clf", RandomForestClassifier(n_estimators=300, max_depth=None,
                                                                 class_weight="balanced_subsample",
                                                                 n_jobs=-1, random_state=SEED))]),
    }
    if HAS_XGB:
        models["XGBoost"] = Pipeline([("pre", pre_tree),
                                      ("clf", XGBClassifier(n_estimators=400, max_depth=5,
                                                            learning_rate=0.05, subsample=0.9,
                                                            colsample_bytree=0.9, eval_metric="logloss",
                                                            random_state=SEED, n_jobs=-1))])
    else:
        models["GradBoost(xgb-fallback)"] = Pipeline([("pre", pre_tree),
                                      ("clf", GradientBoostingClassifier(random_state=SEED))])
    return models


# ----------------------------------------------------------------------
# 3. EVALUATION HARNESS
# ----------------------------------------------------------------------
def ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            e += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return e


def evaluate(models, Xtr, ytr, Xte, yte):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        fitted[name] = model
        p = model.predict_proba(Xte)[:, 1]
        yhat = (p >= 0.5).astype(int)
        cv = cross_val_score(model, Xtr, ytr, scoring="average_precision", cv=skf, n_jobs=-1)
        rows.append({
            "model": name,
            "PR_AUC(test)": round(average_precision_score(yte, p), 3),
            "PR_AUC(cv mean)": round(cv.mean(), 3),
            "PR_AUC(cv sd)": round(cv.std(), 3),
            "ROC_AUC": round(roc_auc_score(yte, p), 3),
            "Brier": round(brier_score_loss(yte, p), 3),
            "recall_rain@0.5": round(recall_score(yte, yhat), 3),
            "prec_rain@0.5": round(precision_score(yte, yhat, zero_division=0), 3),
            "recall_norain@0.5": round(recall_score(yte, yhat, pos_label=0), 3),
        })
    res = pd.DataFrame(rows).sort_values("PR_AUC(cv mean)", ascending=False)
    res.to_csv(os.path.join(RES, "baseline_metrics.csv"), index=False)
    return res, fitted


# ----------------------------------------------------------------------
# 4. CALIBRATION
# ----------------------------------------------------------------------
def calibrate(base_model, Xtr, ytr, Xte, yte, base_name):
    p_raw = base_model.predict_proba(Xte)[:, 1]
    out = {"uncalibrated": p_raw}
    for method in ["sigmoid", "isotonic"]:
        cal = CalibratedClassifierCV(base_model, method=method, cv=5)
        cal.fit(Xtr, ytr)
        out[method] = cal.predict_proba(Xte)[:, 1]

    summary = []
    for label, p in out.items():
        summary.append({"model": base_name, "calibration": label,
                        "Brier": round(brier_score_loss(yte, p), 4),
                        "ECE": round(ece(yte.values if hasattr(yte, "values") else yte, p), 4),
                        "PR_AUC": round(average_precision_score(yte, p), 3)})
    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(RES, "calibration_summary.csv"), index=False)

    # reliability diagram
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for label in ["uncalibrated", "sigmoid", "isotonic"]:
        frac, mean_pred = calibration_curve(yte, out[label], n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac, marker="o", label=f"{label} (ECE={ece(np.asarray(yte), out[label]):.3f})")
    plt.xlabel("mean predicted probability"); plt.ylabel("observed frequency of rain")
    plt.title(f"Reliability — {base_name} (before/after calibration)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "reliability_calibration.png"), dpi=120); plt.close()

    # pick the best-calibrated probabilities (lowest ECE) for the cost step
    best = min(["sigmoid", "isotonic"], key=lambda m: ece(np.asarray(yte), out[m]))
    return out, sdf, best


# ----------------------------------------------------------------------
# 5. COST-SENSITIVE THRESHOLD + SENSITIVITY SWEEP
# ----------------------------------------------------------------------
def expected_cost_threshold(y, p, c_fn, c_fp, grid=None):
    """Threshold minimising expected cost = c_fn*FN + c_fp*FP."""
    if grid is None:
        grid = np.linspace(0.01, 0.99, 197)
    y = np.asarray(y)
    best_t, best_c = 0.5, np.inf
    for t in grid:
        yhat = (p >= t).astype(int)
        fn = int(((yhat == 0) & (y == 1)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        c = c_fn * fn + c_fp * fp
        if c < best_c:
            best_c, best_t = c, t
    return best_t


def cost_analysis(y, p):
    y = np.asarray(y)
    # sensitivity sweep over FN:FP ratios (missed rain vs false alarm), FP cost fixed = 1
    rows = []
    chosen_default = None
    for ratio in [1, 2, 3, 5, 10]:
        t = expected_cost_threshold(y, p, c_fn=ratio, c_fp=1.0)
        yhat = (p >= t).astype(int)
        rows.append({
            "FN:FP": f"{ratio}:1", "threshold": round(t, 3),
            "recall_rain": round(recall_score(y, yhat), 3),
            "precision_rain": round(precision_score(y, yhat, zero_division=0), 3),
            "false_alarm_rate": round(((yhat == 1) & (y == 0)).sum() / max((y == 0).sum(), 1), 3),
            "missed_rain_rate": round(((yhat == 0) & (y == 1)).sum() / max((y == 1).sum(), 1), 3),
        })
        if ratio == 3:
            chosen_default = (t, yhat)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(os.path.join(RES, "cost_threshold_sweep.csv"), index=False)

    # PR curve with the chosen (3:1) operating point marked
    prec, rec, _ = precision_recall_curve(y, p)
    t_star, yhat_star = chosen_default
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, lw=2, label="PR curve")
    plt.scatter([recall_score(y, yhat_star)], [precision_score(y, yhat_star, zero_division=0)],
                color="red", zorder=5, label=f"cost-optimal @ FN:FP=3:1 (t={t_star:.2f})")
    plt.xlabel("recall (rain)"); plt.ylabel("precision (rain)")
    plt.title("PR curve + cost-sensitive operating point")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "pr_curve_operating_point.png"), dpi=120); plt.close()
    return sweep, t_star


# ----------------------------------------------------------------------
# 6. SHAP EXPLANATIONS
# ----------------------------------------------------------------------
def explain(base_model, Xtr, Xte, base_name):
    if not HAS_SHAP:
        clf = base_model.named_steps["clf"]
        imp = getattr(clf, "feature_importances_", None)
        if imp is not None:
            s = pd.Series(imp, index=FEATURES).sort_values(ascending=False)
            s.to_csv(os.path.join(RES, "feature_importance_fallback.csv"))
            print("SHAP not installed -> saved model feature_importances_ as fallback.")
        return
    try:
        pre = base_model.named_steps["pre"]; clf = base_model.named_steps["clf"]
        Xtr_t = pre.transform(Xtr); Xte_t = pre.transform(Xte)
        is_tree = clf.__class__.__name__ in {
            "RandomForestClassifier", "DecisionTreeClassifier",
            "XGBClassifier", "GradientBoostingClassifier"}
        if is_tree:
            expl = shap.TreeExplainer(clf)
            sv = expl.shap_values(Xte_t)
        else:  # linear model (LogReg) -> LinearExplainer with a background sample
            bg = shap.sample(Xtr_t, 200, random_state=SEED)
            expl = shap.LinearExplainer(clf, bg)
            sv = expl.shap_values(Xte_t)
        if isinstance(sv, list):        # some versions return per-class list
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:                # (n, features, classes)
            sv = sv[:, :, 1]
        # global summary
        plt.figure()
        shap.summary_plot(sv, features=Xte_t, feature_names=FEATURES, show=False)
        plt.tight_layout(); plt.savefig(os.path.join(FIG, "shap_summary.png"), dpi=120); plt.close()
        # per-prediction reasons for 5 sample days
        mean_abs = np.abs(sv).mean(0)
        pd.Series(mean_abs, index=FEATURES).sort_values(ascending=False)\
            .to_csv(os.path.join(RES, "shap_mean_abs.csv"))
        rows = []
        for i in range(min(5, Xte_t.shape[0])):
            order = np.argsort(-np.abs(sv[i]))[:3]
            reasons = "; ".join(f"{FEATURES[j]} ({'+' if sv[i][j] > 0 else '-'}{abs(sv[i][j]):.2f})" for j in order)
            rows.append({"instance": i, "top_drivers_toward/away_from_rain": reasons})
        pd.DataFrame(rows).to_csv(os.path.join(RES, "shap_per_prediction_examples.csv"), index=False)
        print(f"SHAP done on {base_name}: saved summary + per-prediction examples.")
    except Exception as e:
        print(f"SHAP step skipped ({e}).")


# ----------------------------------------------------------------------
def main():
    print("=" * 66)
    print("Upasana Route-A reference: calibrated, cost-aware, explainable rain tool")
    print(f"xgboost={HAS_XGB}  shap={HAS_SHAP}  seed={SEED}")
    print("=" * 66)
    df = load_data()
    X, y = df[FEATURES], df["RainTomorrow"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

    print("\n[1-3] Baseline + evaluation harness")
    models = make_models()
    res, fitted = evaluate(models, Xtr, ytr, Xte, yte)
    print(res.to_string(index=False))
    best_name = res.iloc[0]["model"]
    print(f"\nBest base model by CV PR-AUC: {best_name}")

    print("\n[4] Calibration (Platt vs isotonic)")
    _, cal_sum, best_cal = calibrate(fitted[best_name], Xtr, ytr, Xte, yte, best_name)
    print(cal_sum.to_string(index=False))
    print(f"Best-calibrated probabilities: {best_cal} -> used for the cost step")

    # recompute the chosen calibrated probs for the cost analysis
    cal = CalibratedClassifierCV(fitted[best_name], method=best_cal, cv=5).fit(Xtr, ytr)
    p_cal = cal.predict_proba(Xte)[:, 1]

    print("\n[5] Cost-sensitive threshold + sensitivity sweep")
    sweep, t_star = cost_analysis(yte, p_cal)
    print(sweep.to_string(index=False))
    print(f"Recommended operating threshold at FN:FP=3:1 -> t={t_star:.2f} "
          "(state the ratio as an assumption; report the sweep, not one number).")

    print("\n[6] SHAP explanations")
    explain(fitted[best_name], Xtr, Xte, best_name)

    print("\nDone. See results/ and figures/.")
    print("NOTE: numbers here are on the SYNTHETIC proxy unless WEATHERAUS_CSV was set. "
          "The method, harness and outputs are the deliverable — rerun on real weatherAUS.")


if __name__ == "__main__":
    main()
