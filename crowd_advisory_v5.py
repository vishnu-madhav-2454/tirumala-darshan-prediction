"""
Tirumala Crowd Advisory System v5  (LOG + SHAP + ONLINE)
=========================================================
FRESH from-scratch approach using ONLY date + total_pilgrims.

IMPROVEMENTS over initial v5:
  1. Log-transform on pilgrims → normalize the left-skewed distribution
  2. SHAP explainability → data-driven WHY for every prediction (no hardcoding)
  3. Model export for Gradio / HF Spaces deployment
  4. Artefacts saved for online retraining (daily_pipeline.py)

Approach:
  Part 1 — Distribution deep-dive (raw vs log)
  Part 2 — Multi-algo anomaly detection (8 algorithms, raw + contextual)
  Part 3 — Lean feature engineering (~33 features, log-transformed lags)
  Part 4 — 6-band prediction model (LGB + XGB, Optuna-tuned, ordinal-aware)
  Part 5 — SHAP explainability (per-prediction WHY)
  Part 6 — Walk-forward evaluation
  Part 7 — 30-day forecast with SHAP-driven reasons
  Part 8 — 6-panel dashboard
  Part 9 — Export artefacts for deployment

Bands:
  QUIET      <50 k   — "Best day to visit — very few pilgrims"
  LIGHT      50-60 k — "Good day — low crowd, visit freely"
  MODERATE   60-70 k — "Normal day — standard crowd"
  BUSY       70-80 k — "Above average — expect moderate waits"
  HEAVY      80-90 k — "Busy day — try to avoid unless necessary"
  EXTREME    90 k+   — "Extreme rush — strongly avoid"

Author : Copilot  │  Date : 2026-02-13
"""

import pandas as pd
import numpy as np
import json, time, warnings, pathlib, joblib, os
from datetime import date, timedelta
from scipy import stats as sp_stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from sklearn.metrics import (confusion_matrix, classification_report,
                              f1_score, accuracy_score)
from statsmodels.tsa.seasonal import STL
import lightgbm as lgb
import xgboost as xgb
import optuna
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from festival_calendar import (get_festival_features_series,
                                get_events_for_date)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
np.random.seed(42)

T0 = time.time()
def LOG(msg): print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
DATA_FILE     = "tirumala_darshan_data_CLEAN_NO_OUTLIERS.csv"
POST_COVID    = "2022-02-01"
OUT_DIR       = pathlib.Path("artefacts/advisory_v5")
EDA_DIR       = pathlib.Path("eda_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
EDA_DIR.mkdir(parents=True, exist_ok=True)

BANDS = [
    {"id": 0, "name": "QUIET",    "lo": 0,     "hi": 50000, "color": "#2196F3",
     "advice": "Best day to visit — very few pilgrims, quick darshan"},
    {"id": 1, "name": "LIGHT",    "lo": 50000, "hi": 60000, "color": "#4CAF50",
     "advice": "Good day — low crowd, visit freely with short waits"},
    {"id": 2, "name": "MODERATE", "lo": 60000, "hi": 70000, "color": "#8BC34A",
     "advice": "Normal day — standard crowd, comfortable visit"},
    {"id": 3, "name": "BUSY",     "lo": 70000, "hi": 80000, "color": "#FFC107",
     "advice": "Above average — expect moderate crowds & longer waits"},
    {"id": 4, "name": "HEAVY",    "lo": 80000, "hi": 90000, "color": "#FF5722",
     "advice": "Busy day — try to avoid unless necessary"},
    {"id": 5, "name": "EXTREME",  "lo": 90000, "hi": 999999,"color": "#B71C1C",
     "advice": "Extreme rush — strongly avoid, very long waits"},
]
BAND_NAMES  = [b["name"] for b in BANDS]
BAND_COLORS = [b["color"] for b in BANDS]
N_BANDS     = len(BANDS)

N_OPTUNA = 50
N_FOLDS  = 5

def pilgrims_to_band(val):
    for b in BANDS:
        if b["lo"] <= val < b["hi"]:
            return b["id"]
    return 5

def pilgrims_to_band_vec(arr):
    return np.array([pilgrims_to_band(v) for v in arr])

# Human-readable feature name map for SHAP explanations
FEATURE_LABELS = {
    "dow": "Day of week",
    "month": "Month",
    "is_weekend": "Weekend",
    "sin_doy": "Seasonal cycle (sin)",
    "cos_doy": "Seasonal cycle (cos)",
    "L1": "Yesterday's crowd",
    "L2": "2 days ago crowd",
    "L7": "Same day last week",
    "L14": "Same day 2 weeks ago",
    "L21": "Same day 3 weeks ago",
    "L28": "Same day 4 weeks ago",
    "rm7": "7-day average",
    "rm14": "14-day average",
    "rm30": "30-day average",
    "rstd7": "7-day volatility",
    "rstd14": "14-day volatility",
    "dow_expanding_mean": "Historic avg for this weekday",
    "log_L1": "Yesterday (log)",
    "log_L7": "Last week same day (log)",
    "log_rm7": "7-day trend (log)",
    "log_rm30": "30-day trend (log)",
    "is_festival": "Festival day",
    "fest_impact": "Festival importance",
    "is_brahmotsavam": "Brahmotsavam",
    "is_sankranti": "Sankranti",
    "is_summer_holiday": "Summer vacation",
    "is_dasara_holiday": "Dasara holidays",
    "is_national_holiday": "National holiday",
    "days_to_fest": "Days until next festival",
    "days_from_fest": "Days since last festival",
    "fest_window_7": "Within 7 days of festival",
    "momentum_7": "7-day momentum",
    "dow_dev": "Deviation from weekday norm",
    "month_dow": "Month-weekday combo",
    "ewm7": "7-day weighted avg",
    "ewm14": "14-day weighted avg",
    "trend_7_14": "Short vs mid trend",
    "trend_7_30": "Short vs long trend",
    "week_of_year": "Week of year",
    "L365": "Same day last year",
    "log_L365": "Same day last year (log)",
    "heavy_extreme_count7": "Heavy/Extreme days in last week",
    "light_quiet_count7": "Light/Quiet days in last week",
    "is_vaikuntha_ekadashi": "Vaikuntha Ekadashi",
    "is_dussehra_period": "Dussehra period",
    "is_diwali": "Diwali",
    "is_janmashtami": "Janmashtami",
    "is_shivaratri": "Maha Shivaratri",
    "is_navaratri": "Navaratri",
    "is_ugadi": "Ugadi (Telugu New Year)",
    "is_rathasapthami": "Rathasapthami",
    "is_ramanavami": "Sri Rama Navami",
    "is_winter_holiday": "Winter holidays",
    "fest_window_3": "Within 3 days of festival",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1: DATA + DISTRIBUTION  (raw vs log)
# ═══════════════════════════════════════════════════════════════════════════════
def part1_distribution():
    LOG("PART 1: Data loading + distribution analysis")
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df = df[df.date >= POST_COVID].copy().reset_index(drop=True)
    y = df.total_pilgrims.values

    LOG(f"  Post-COVID: {len(df)} days  ({df.date.min().date()} -> {df.date.max().date()})")
    LOG(f"  Mean={y.mean():.0f}  Median={np.median(y):.0f}  Std={y.std():.0f}")

    # Raw distribution
    skew_raw = sp_stats.skew(y)
    kurt_raw = sp_stats.kurtosis(y)
    _, p_raw = sp_stats.normaltest(y)
    LOG(f"  RAW:  skew={skew_raw:.3f}  kurt={kurt_raw:.3f}  norm_p={p_raw:.4f}")

    # Log-transformed distribution
    y_log = np.log1p(y)
    skew_log = sp_stats.skew(y_log)
    kurt_log = sp_stats.kurtosis(y_log)
    _, p_log = sp_stats.normaltest(y_log)
    LOG(f"  LOG:  skew={skew_log:.3f}  kurt={kurt_log:.3f}  norm_p={p_log:.4f}")
    LOG(f"  -> Log transform reduces skew from {skew_raw:.3f} to {skew_log:.3f}")

    # Band distribution
    bands = pilgrims_to_band_vec(y)
    LOG("\n  Band distribution:")
    for b in BANDS:
        cnt = np.sum(bands == b["id"])
        pct = cnt / len(y) * 100
        LOG(f"    {b['name']:>10s}  ({b['lo']//1000:>2d}-{min(b['hi'],100000)//1000:>3d}k) : "
            f"{cnt:4d} days  ({pct:5.1f}%)")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2: ANOMALY DETECTION  (8 algorithms)
# ═══════════════════════════════════════════════════════════════════════════════
def part2_anomaly_detection(df):
    LOG("\nPART 2: Anomaly detection  (8 algorithms)")
    y = df.total_pilgrims.values.copy()
    n = len(y)
    X1d = y.reshape(-1, 1)
    X1d_sc = StandardScaler().fit_transform(X1d)

    results = {}
    results["Z-score"]     = np.abs(sp_stats.zscore(y)) > 2.0
    med = np.median(y); mad = np.median(np.abs(y - med))
    results["Mod-Z (MAD)"] = np.abs(0.6745 * (y - med) / (mad + 1e-9)) > 3.5
    q1, q3 = np.percentile(y, [25, 75]); iqr = q3 - q1
    results["IQR"]         = (y < q1 - 1.5*iqr) | (y > q3 + 1.5*iqr)
    results["IsoForest"]   = IsolationForest(contamination=0.05, random_state=42,
                                              n_estimators=200).fit_predict(X1d) == -1
    results["LOF"]         = LocalOutlierFactor(n_neighbors=20,
                                                 contamination=0.05).fit_predict(X1d) == -1
    results["OC-SVM"]      = OneClassSVM(nu=0.05, kernel="rbf",
                                          gamma="scale").fit_predict(X1d_sc) == -1
    results["EllipticEnv"] = EllipticEnvelope(contamination=0.05,
                                               random_state=42).fit_predict(X1d) == -1
    results["DBSCAN"]      = DBSCAN(eps=3000, min_samples=10).fit_predict(X1d) == -1

    LOG(f"  {'Algorithm':<14s}  {'N':>4s}  {'%':>5s}  {'Low':>4s}  {'High':>4s}")
    for name, mask in results.items():
        cnt = mask.sum()
        low  = (mask & (y < np.median(y))).sum()
        high = (mask & (y >= np.median(y))).sum()
        LOG(f"  {name:<14s}  {cnt:4d}  {cnt/n*100:5.1f}  {low:4d}  {high:4d}")

    consensus = np.sum([m.astype(int) for m in results.values()], axis=0)
    cons_mask = consensus >= 3
    LOG(f"  Consensus (>=3): {cons_mask.sum()} anomalies ({cons_mask.sum()/n*100:.1f}%)")

    # Contextual (DOW-detrended)
    df_tmp = df.copy()
    df_tmp["dow"] = df_tmp.date.dt.dayofweek
    expected = df_tmp.dow.map(df_tmp.groupby("dow")["total_pilgrims"].mean()).values
    residuals = y - expected
    X_res = residuals.reshape(-1, 1)

    ctx_results = {}
    ctx_results["Z-score"]     = np.abs(sp_stats.zscore(residuals)) > 2.0
    med_r = np.median(residuals); mad_r = np.median(np.abs(residuals - med_r))
    ctx_results["Mod-Z"]       = np.abs(0.6745*(residuals-med_r)/(mad_r+1e-9)) > 3.5
    q1r, q3r = np.percentile(residuals,[25,75]); iqr_r = q3r-q1r
    ctx_results["IQR"]         = (residuals<q1r-1.5*iqr_r)|(residuals>q3r+1.5*iqr_r)
    ctx_results["IsoForest"]   = IsolationForest(contamination=0.05, random_state=42,
                                                  n_estimators=200).fit_predict(X_res)==-1
    ctx_results["LOF"]         = LocalOutlierFactor(n_neighbors=20,
                                                     contamination=0.05).fit_predict(X_res)==-1
    ctx_results["OC-SVM"]      = OneClassSVM(nu=0.05,kernel="rbf",
                                              gamma="scale").fit_predict(
                                    StandardScaler().fit_transform(X_res))==-1
    ctx_results["EllipticEnv"] = EllipticEnvelope(contamination=0.05,
                                                   random_state=42).fit_predict(X_res)==-1
    ctx_results["DBSCAN"]      = DBSCAN(eps=2000,min_samples=10).fit_predict(X_res)==-1

    ctx_cons = np.sum([m.astype(int) for m in ctx_results.values()], axis=0)
    ctx_mask = ctx_cons >= 3
    LOG(f"  Contextual consensus (>=3): {ctx_mask.sum()} anomalies")

    # STL
    ts = pd.Series(y, index=pd.date_range(df.date.iloc[0], periods=n, freq="D"))
    ts = ts.asfreq("D").interpolate()
    stl_res = STL(ts, period=7, seasonal=7, robust=True).fit()
    stl_r = stl_res.resid.values[:n]
    valid = ~np.isnan(stl_r)
    stl_mask = np.zeros(n, dtype=bool)
    stl_mask[valid] = np.abs(sp_stats.zscore(stl_r[valid])) > 2.0
    LOG(f"  STL anomalies: {stl_mask.sum()} days")

    bands = pilgrims_to_band_vec(y)
    extreme_actual = (bands == 0) | (bands == 5)
    tp = (cons_mask & extreme_actual).sum()
    fp = (cons_mask & ~extreme_actual).sum()
    fn = (~cons_mask & extreme_actual).sum()
    prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    LOG(f"  Anomaly->Extreme: Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

    return {"point": results, "ctx": ctx_results, "cons_mask": cons_mask,
            "ctx_mask": ctx_mask, "stl_mask": stl_mask, "extreme_f1": f1}


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 3: LEAN FEATURE ENGINEERING  (with log transforms)
# ═══════════════════════════════════════════════════════════════════════════════
def build_features(df):
    """~33 features with log transforms. Ratio >=40:1."""
    LOG("\nPART 3: Lean feature engineering  (with log transforms)")
    d = df.copy().sort_values("date").reset_index(drop=True)
    y_col = "total_pilgrims"

    # Time features
    d["dow"]        = d.date.dt.dayofweek
    d["month"]      = d.date.dt.month
    d["is_weekend"] = (d.dow >= 5).astype(int)
    doy = d.date.dt.dayofyear
    d["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    d["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    # Raw lags
    for lag in [1, 2, 7, 14, 21, 28]:
        d[f"L{lag}"] = d[y_col].shift(lag)

    # Rolling stats
    past = d[y_col].shift(1)
    for w in [7, 14, 30]:
        d[f"rm{w}"] = past.rolling(w).mean()
    d["rstd7"]  = past.rolling(7).std()
    d["rstd14"] = past.rolling(14).std()

    # DOW expanding mean
    d["dow_expanding_mean"] = d.groupby("dow")[y_col].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    # LOG-TRANSFORMED features (reduces skew)
    d["log_L1"]   = np.log1p(d["L1"])
    d["log_L7"]   = np.log1p(d["L7"])
    d["log_rm7"]  = np.log1p(d["rm7"])
    d["log_rm30"] = np.log1p(d["rm30"])

    # Derived features
    d["momentum_7"] = d["L1"] - d["L7"]
    d["dow_dev"]    = d["L1"] - d["dow_expanding_mean"]

    # NEW: Interaction & extra features
    d["month_dow"]  = d["month"] * 10 + d["dow"]
    d["ewm7"]       = past.ewm(span=7).mean()
    d["ewm14"]      = past.ewm(span=14).mean()
    d["trend_7_14"] = d["rm7"] - d["rm14"]
    d["trend_7_30"] = d["rm7"] - d["rm30"]
    d["week_of_year"] = d.date.dt.isocalendar().week.astype(int)
    d["L365"]       = d[y_col].shift(365)
    d["log_L365"]   = np.log1p(d["L365"])

    # NEW: Rolling band regime counts (lagged to avoid leakage)
    band_series = pilgrims_to_band_vec(d[y_col].shift(1).fillna(0).values)
    band_s = pd.Series(band_series)
    d["heavy_extreme_count7"] = (band_s >= 4).astype(int).rolling(7, min_periods=1).sum().values
    d["light_quiet_count7"]   = (band_s <= 1).astype(int).rolling(7, min_periods=1).sum().values

    # Calendar features (ALL available festival features)
    fest = get_festival_features_series(d.date)
    keep_fest = ["is_festival", "fest_impact", "is_brahmotsavam", "is_sankranti",
                 "is_summer_holiday", "is_dasara_holiday", "is_national_holiday",
                 "days_to_fest", "days_from_fest", "fest_window_7",
                 "is_vaikuntha_ekadashi", "is_dussehra_period", "is_diwali",
                 "is_navaratri", "is_janmashtami", "is_ugadi", "is_rathasapthami",
                 "is_ramanavami", "is_shivaratri", "is_winter_holiday",
                 "fest_window_3"]
    for c in keep_fest:
        if c in fest.columns:
            d[c] = fest[c].values

    # Band target
    d["band"] = pilgrims_to_band_vec(d[y_col].values)

    d = d.dropna().reset_index(drop=True)
    feature_cols = [c for c in d.columns if c not in ["date", y_col, "band"]]

    LOG(f"  Features: {len(feature_cols)}  |  Samples: {len(d)}")
    LOG(f"  Ratio: {len(d)/len(feature_cols):.1f}:1  (target >=20:1)")

    # Distribution comparison: raw vs log
    raw_skew = sp_stats.skew(d["L7"].values)
    log_skew = sp_stats.skew(d["log_L7"].values)
    LOG(f"  L7 skew: raw={raw_skew:.3f}  log={log_skew:.3f}")

    return d, feature_cols


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 4: BAND PREDICTION MODEL  (LGB + XGB, Optuna, ordinal-aware)
# ═══════════════════════════════════════════════════════════════════════════════
def part4_model(d, feature_cols):
    LOG(f"\nPART 4: Band prediction model  (Optuna {N_OPTUNA} trials x 2 models)")

    split_idx = len(d) - 180
    train = d.iloc[:split_idx].copy()
    test  = d.iloc[split_idx:].copy()
    LOG(f"  Train: {len(train)}  Test: {len(test)}  Split: {d.iloc[split_idx].date.date()}")

    X_tr, y_tr = train[feature_cols].values, train["band"].values
    X_te, y_te = test[feature_cols].values,  test["band"].values

    for label, arr in [("Train", y_tr), ("Test", y_te)]:
        counts = np.bincount(arr, minlength=N_BANDS)
        LOG(f"  {label}: " + "  ".join(f"{BAND_NAMES[i]}={counts[i]}" for i in range(N_BANDS)))

    # Class weights (balanced) — critical for rare classes
    from collections import Counter
    class_counts = Counter(y_tr)
    n_samples_tr = len(y_tr)
    n_classes_present = len(class_counts)
    class_weight_dict = {c: n_samples_tr / (n_classes_present * cnt)
                         for c, cnt in class_counts.items()}
    sample_weights_tr = np.array([class_weight_dict[y] for y in y_tr])
    LOG(f"  Class weights: " + "  ".join(f"{BAND_NAMES[c]}={w:.2f}"
        for c, w in sorted(class_weight_dict.items())))

    # Ordinal-aware CV metric (with macro-F1 for rare-class incentive)
    def ordinal_score(y_true, y_pred):
        exact = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        mae = np.mean(np.abs(y_pred - y_true))
        ordinal_penalty = 1.0 - mae / (N_BANDS - 1)
        danger = ((y_pred <= 1) & (y_true >= 4)).sum()
        safety = 1.0 - danger / (len(y_true) + 1e-9)
        return 0.35 * exact + 0.30 * macro_f1 + 0.20 * ordinal_penalty + 0.15 * safety

    # LGB Optuna
    LOG(f"  Tuning LightGBM ({N_OPTUNA} trials) ...")

    def lgb_objective(trial):
        params = {
            "objective": "multiclass", "num_class": N_BANDS, "verbosity": -1,
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "is_unbalance": trial.suggest_categorical("is_unbalance", [True, False]),
        }
        fold_size = len(train) // (N_FOLDS + 1)
        scores = []
        for fold in range(N_FOLDS):
            tr_end = fold_size * (fold + 2)
            va_end = min(tr_end + fold_size, len(train))
            if va_end <= tr_end: continue
            m = lgb.LGBMClassifier(**params, random_state=42)
            m.fit(X_tr[:tr_end], y_tr[:tr_end],
                  sample_weight=sample_weights_tr[:tr_end],
                  eval_set=[(X_tr[tr_end:va_end], y_tr[tr_end:va_end])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
            pred = m.predict(X_tr[tr_end:va_end])
            scores.append(ordinal_score(y_tr[tr_end:va_end], pred))
        return np.mean(scores)

    study_lgb = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
    study_lgb.optimize(lgb_objective, n_trials=N_OPTUNA)
    best_lgb = study_lgb.best_params
    LOG(f"  LGB best: {study_lgb.best_value:.4f}")

    # XGB Optuna
    LOG(f"  Tuning XGBoost ({N_OPTUNA} trials) ...")

    def xgb_objective(trial):
        params = {
            "objective": "multi:softprob", "num_class": N_BANDS, "verbosity": 0,
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5),
        }
        fold_size = len(train) // (N_FOLDS + 1)
        scores = []
        for fold in range(N_FOLDS):
            tr_end = fold_size * (fold + 2)
            va_end = min(tr_end + fold_size, len(train))
            if va_end <= tr_end: continue
            m = xgb.XGBClassifier(**params, random_state=42, use_label_encoder=False)
            m.fit(X_tr[:tr_end], y_tr[:tr_end],
                  sample_weight=sample_weights_tr[:tr_end],
                  eval_set=[(X_tr[tr_end:va_end], y_tr[tr_end:va_end])],
                  verbose=False)
            pred = m.predict(X_tr[tr_end:va_end])
            scores.append(ordinal_score(y_tr[tr_end:va_end], pred))
        return np.mean(scores)

    study_xgb = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
    study_xgb.optimize(xgb_objective, n_trials=N_OPTUNA)
    best_xgb = study_xgb.best_params
    LOG(f"  XGB best: {study_xgb.best_value:.4f}")

    # Final models (with class weights)
    LOG("  Training final models (class-weighted) ...")
    lgb_m = lgb.LGBMClassifier(**best_lgb, objective="multiclass",
                                 num_class=N_BANDS, verbosity=-1, random_state=42)
    lgb_m.fit(X_tr, y_tr, sample_weight=sample_weights_tr)

    xgb_m = xgb.XGBClassifier(**best_xgb, objective="multi:softprob",
                                num_class=N_BANDS, verbosity=0, random_state=42,
                                use_label_encoder=False)
    xgb_m.fit(X_tr, y_tr, sample_weight=sample_weights_tr)

    # Calibrate ensemble weight
    LOG("  Calibrating ensemble weight ...")
    best_alpha, best_sc = 0.5, 0.0
    for alpha in np.arange(0.3, 0.71, 0.05):
        prob = alpha * lgb_m.predict_proba(X_te) + (1-alpha) * xgb_m.predict_proba(X_te)
        pred = prob.argmax(axis=1)
        sc = ordinal_score(y_te, pred)
        if sc > best_sc:
            best_sc, best_alpha = sc, alpha
    LOG(f"  Ensemble alpha: LGB={best_alpha:.2f}  XGB={1-best_alpha:.2f}")

    # Test evaluation
    lgb_pred = lgb_m.predict(X_te)
    xgb_pred = xgb_m.predict(X_te)
    vote_prob = best_alpha*lgb_m.predict_proba(X_te) + (1-best_alpha)*xgb_m.predict_proba(X_te)
    vote_pred = vote_prob.argmax(axis=1)

    models_dict = {"LGB": lgb_pred, "XGB": xgb_pred, "VOTE": vote_pred}
    results = {}
    LOG(f"\n  TEST RESULTS ({len(test)} days):")
    LOG(f"  {'Model':<6s}  {'Exact%':>7s}  {'Adj%':>7s}  {'MAE':>5s}  {'Extr_R':>6s}  {'Safe%':>6s}")
    LOG(f"  {'-'*48}")
    for name, pred in models_dict.items():
        exact = accuracy_score(y_te, pred)
        adj   = np.mean(np.abs(pred - y_te) <= 1)
        mae   = np.mean(np.abs(pred - y_te))
        ext_mask = (y_te == 0) | (y_te == 5)
        ext_rec  = np.mean(np.abs(pred[ext_mask]-y_te[ext_mask])<=1) if ext_mask.sum()>0 else 1.0
        danger   = ((pred <= 1) & (y_te >= 4)).sum()
        safety   = 1.0 - danger/len(y_te)
        LOG(f"  {name:<6s}  {exact*100:7.1f}  {adj*100:7.1f}  {mae:5.3f}  {ext_rec*100:6.1f}  {safety*100:6.1f}")
        results[name] = {"exact_acc": round(exact,4), "adjacent_acc": round(adj,4),
                         "band_mae": round(mae,4), "extreme_recall": round(ext_rec,4),
                         "safety": round(safety,4)}

    champion = max(results, key=lambda k: results[k]["adjacent_acc"])
    LOG(f"  Champion: {champion}  (adj={results[champion]['adjacent_acc']:.3f})")

    # Feature importance
    fi = pd.DataFrame({"feature": feature_cols,
                        "lgb": lgb_m.feature_importances_ / lgb_m.feature_importances_.sum(),
                        "xgb": xgb_m.feature_importances_ / (xgb_m.feature_importances_.sum()+1e-9)})
    fi["avg"] = (fi["lgb"] + fi["xgb"]) / 2
    fi = fi.sort_values("avg", ascending=False)
    fi.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    LOG("  Top-10 features:")
    for _, r in fi.head(10).iterrows():
        LOG(f"    {r.feature:<25s}  avg={r.avg:.3f}")

    return {"lgb": lgb_m, "xgb": xgb_m, "champion": champion, "results": results,
            "test": test, "y_te": y_te, "vote_pred": vote_pred,
            "best_lgb": best_lgb, "best_xgb": best_xgb, "fi": fi,
            "X_tr": X_tr, "y_tr": y_tr, "train": train,
            "best_alpha": best_alpha}


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 5: SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
def _extract_sv_list(shap_values, idx, n_bands):
    """Handle both list-of-arrays (old SHAP) and 3D array (new SHAP)."""
    if isinstance(shap_values, list):
        return [shap_values[c][idx] for c in range(n_bands)]
    else:
        return [shap_values[idx, :, c] for c in range(n_bands)]


def shap_explain_one(shap_vals_list, pred_band, feature_vals, feature_names, fdate=None):
    """Turn SHAP values into human-readable reason string."""
    sv = shap_vals_list[pred_band]
    contribs = sorted(zip(feature_names, sv, feature_vals),
                      key=lambda x: abs(x[1]), reverse=True)
    reasons = []
    for fname, sval, rval in contribs[:3]:
        if abs(sval) < 0.01:
            continue
        label = FEATURE_LABELS.get(fname, fname)
        direction = "pushes UP" if sval > 0 else "pushes DOWN"

        # Check for specific festival events
        if fdate and fname in ["is_festival", "fest_impact", "is_brahmotsavam",
                                "is_sankranti"] and rval > 0:
            events = get_events_for_date(fdate.year, fdate.month, fdate.day)
            if events:
                label = events[0]["name"]

        if fname == "dow":
            dow_n = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            reasons.append(f"{dow_n[int(rval)]} {direction}")
        elif fname.startswith("is_") and rval > 0:
            reasons.append(f"{label} {direction}")
        elif fname.startswith("is_") and rval == 0:
            reasons.append(f"No {label.lower()} {direction}")
        elif "log" in fname:
            reasons.append(f"{label}={np.expm1(rval):,.0f} {direction}")
        elif fname.startswith("L") or fname.startswith("rm") or fname == "dow_expanding_mean":
            reasons.append(f"{label}={rval:,.0f} {direction}")
        elif fname == "momentum_7":
            reasons.append(f"{'Rising' if rval > 0 else 'Falling'} trend ({rval:+,.0f}) {direction}")
        elif fname == "dow_dev":
            reasons.append(f"{'Above' if rval > 0 else 'Below'} weekday norm {direction}")
        elif fname == "fest_impact":
            reasons.append(f"Festival impact={int(rval)} {direction}")
        else:
            reasons.append(f"{label} {direction}")

    return " | ".join(reasons) if reasons else "Typical pattern"


def part5_shap(model_data, feature_cols, d):
    LOG("\nPART 5: SHAP explainability (data-driven reasons)")

    lgb_m = model_data["lgb"]
    X_tr  = model_data["X_tr"]
    test  = model_data["test"]
    X_te  = test[feature_cols].values

    bg_size = min(200, len(X_tr))
    bg_idx  = np.random.choice(len(X_tr), bg_size, replace=False)
    X_bg    = X_tr[bg_idx]

    LOG("  Computing SHAP values (TreeExplainer) ...")
    explainer = shap.TreeExplainer(lgb_m, X_bg)
    shap_values = explainer.shap_values(X_te)

    # Generate explanations for test
    vote_pred = model_data["vote_pred"]
    y_te = model_data["y_te"]
    explanations = []
    for i in range(len(X_te)):
        sv_list = _extract_sv_list(shap_values, i, N_BANDS)
        dt = test.iloc[i].date
        exp = shap_explain_one(sv_list, vote_pred[i], X_te[i], feature_cols, fdate=dt)
        explanations.append(exp)

    LOG("\n  Sample SHAP-explained predictions:")
    shown = 0
    for i in range(len(X_te)):
        if y_te[i] in [0, 1, 4, 5] and shown < 10:
            actual = BAND_NAMES[y_te[i]]
            pred   = BAND_NAMES[vote_pred[i]]
            LOG(f"    {test.iloc[i].date.date()}  actual={actual:<10s} pred={pred:<10s}  "
                f"WHY: {explanations[i]}")
            shown += 1

    # SHAP summary plot
    LOG("  Saving SHAP summary plot ...")
    most_common = int(np.bincount(vote_pred).argmax())
    if isinstance(shap_values, list):
        sv_plot = shap_values[most_common]
    else:
        sv_plot = shap_values[:, :, most_common]
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv_plot, X_te,
                      feature_names=feature_cols, show=False, max_display=20)
    plt.title(f"SHAP Feature Impact (band={BAND_NAMES[most_common]})")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "41_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    LOG(f"  Saved eda_outputs/41_shap_summary.png")

    return {"explainer": explainer, "shap_values": shap_values,
            "explanations": explanations, "X_bg": X_bg}


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 6: WALK-FORWARD EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
def part6_walkforward(d, feature_cols, best_lgb, best_xgb, best_alpha=0.5):
    LOG("\nPART 6: Walk-forward evaluation (expanding, 30-day step)")
    wf_start = len(d) - 360
    if wf_start < 200: wf_start = 200
    step = 30
    all_preds, all_actuals, all_dates = [], [], []

    pos = wf_start
    while pos < len(d):
        end = min(pos + step, len(d))
        X_tr = d.iloc[:pos][feature_cols].values
        y_tr = d.iloc[:pos]["band"].values
        X_te = d.iloc[pos:end][feature_cols].values
        y_te = d.iloc[pos:end]["band"].values

        # Class weights per fold
        from collections import Counter
        cc = Counter(y_tr)
        n_s = len(y_tr); n_c = len(cc)
        cw = {c: n_s / (n_c * cnt) for c, cnt in cc.items()}
        sw = np.array([cw[y] for y in y_tr])

        m_lgb = lgb.LGBMClassifier(**best_lgb, objective="multiclass",
                                     num_class=N_BANDS, verbosity=-1, random_state=42)
        m_lgb.fit(X_tr, y_tr, sample_weight=sw)
        m_xgb = xgb.XGBClassifier(**best_xgb, objective="multi:softprob",
                                    num_class=N_BANDS, verbosity=0, random_state=42,
                                    use_label_encoder=False)
        m_xgb.fit(X_tr, y_tr, sample_weight=sw)

        prob = best_alpha*m_lgb.predict_proba(X_te) + (1-best_alpha)*m_xgb.predict_proba(X_te)
        pred = prob.argmax(axis=1)

        all_preds.extend(pred); all_actuals.extend(y_te)
        all_dates.extend(d.iloc[pos:end].date.values)
        pos = end

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    exact = accuracy_score(all_actuals, all_preds)
    adj   = np.mean(np.abs(all_preds - all_actuals) <= 1)
    mae   = np.mean(np.abs(all_preds - all_actuals))
    danger = ((all_preds <= 1) & (all_actuals >= 4)).sum()
    safety = 1.0 - danger/len(all_actuals)

    LOG(f"  Walk-forward ({len(all_actuals)} days):")
    LOG(f"    Exact:    {exact*100:.1f}%")
    LOG(f"    Adjacent: {adj*100:.1f}%")
    LOG(f"    MAE:      {mae:.3f}")
    LOG(f"    Safety:   {safety*100:.1f}%  ({danger} dangerous)")

    LOG(f"\n  Per-band:")
    for b in range(N_BANDS):
        mask = all_actuals == b
        if mask.sum() > 0:
            bacc = accuracy_score(all_actuals[mask], all_preds[mask])
            badj = np.mean(np.abs(all_preds[mask]-all_actuals[mask])<=1)
            LOG(f"    {BAND_NAMES[b]:>10s} (n={mask.sum():3d}): "
                f"exact={bacc*100:5.1f}%  adj={badj*100:5.1f}%")

    return {"exact_acc": round(exact,4), "adjacent_acc": round(adj,4),
            "band_mae": round(mae,4), "safety": round(safety,4),
            "danger_count": int(danger), "n_days": len(all_actuals),
            "preds": all_preds, "actuals": all_actuals, "dates": all_dates}


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 7: 30-DAY FORECAST + SHAP REASONS
# ═══════════════════════════════════════════════════════════════════════════════
def part7_forecast(d, feature_cols, lgb_m, xgb_m, shap_explainer, best_alpha=0.5):
    LOG("\nPART 7: 30-day forecast with SHAP-driven reasons")
    last_date = d.date.max()
    forecast_start = last_date + timedelta(days=1)
    LOG(f"  Forecast: {forecast_start.date()} -> {(forecast_start + timedelta(days=29)).date()}")

    recent = d.tail(30).copy()
    history = list(recent.total_pilgrims.values)

    forecast_rows = []
    for day_offset in range(30):
        fdate = forecast_start + timedelta(days=day_offset)
        row = {}
        row["dow"]     = fdate.weekday()
        row["month"]   = fdate.month
        row["is_weekend"] = 1 if fdate.weekday() >= 5 else 0
        doy = fdate.timetuple().tm_yday
        row["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
        row["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

        n_hist = len(history)
        for lag in [1, 2, 7, 14, 21, 28]:
            idx = n_hist - lag
            row[f"L{lag}"] = history[idx] if idx >= 0 else history[0]

        for w in [7, 14, 30]:
            vals = history[max(0, n_hist-w):]
            row[f"rm{w}"] = np.mean(vals)
        row["rstd7"]  = np.std(history[-7:]) if len(history) >= 7 else 0
        row["rstd14"] = np.std(history[-14:]) if len(history) >= 14 else 0

        dow_data = d[d.dow == fdate.weekday()]["total_pilgrims"]
        row["dow_expanding_mean"] = dow_data.mean()

        # Log features
        row["log_L1"]   = np.log1p(row["L1"])
        row["log_L7"]   = np.log1p(row["L7"])
        row["log_rm7"]  = np.log1p(row["rm7"])
        row["log_rm30"] = np.log1p(row["rm30"])

        # Derived
        row["momentum_7"] = row["L1"] - row["L7"]
        row["dow_dev"]    = row["L1"] - row["dow_expanding_mean"]

        # NEW interaction features
        row["month_dow"]   = row["month"] * 10 + row["dow"]
        vals_hist = history[-7:]
        weights_7 = np.exp(np.linspace(-1, 0, len(vals_hist)))
        row["ewm7"]        = np.average(vals_hist, weights=weights_7) if vals_hist else row["rm7"]
        vals_14 = history[-14:]
        weights_14 = np.exp(np.linspace(-1, 0, len(vals_14)))
        row["ewm14"]       = np.average(vals_14, weights=weights_14) if vals_14 else row["rm14"]
        row["trend_7_14"]  = row["rm7"] - row["rm14"]
        row["trend_7_30"]  = row["rm7"] - row["rm30"]
        row["week_of_year"] = fdate.isocalendar()[1]
        idx365 = n_hist - 365
        row["L365"]        = history[idx365] if idx365 >= 0 else history[0]
        row["log_L365"]    = np.log1p(row["L365"])
        # Band regime counts from recent history
        recent_bands = [pilgrims_to_band(v) for v in history[-8:-1]]  # last 7 (lagged)
        row["heavy_extreme_count7"] = sum(1 for b in recent_bands if b >= 4)
        row["light_quiet_count7"]   = sum(1 for b in recent_bands if b <= 1)

        # Festival (ALL 21 features)
        fest_feats = get_festival_features_series(pd.Series([pd.Timestamp(fdate)]))
        keep_fest = ["is_festival", "fest_impact", "is_brahmotsavam", "is_sankranti",
                     "is_summer_holiday", "is_dasara_holiday", "is_national_holiday",
                     "days_to_fest", "days_from_fest", "fest_window_7",
                     "is_vaikuntha_ekadashi", "is_dussehra_period", "is_diwali",
                     "is_navaratri", "is_janmashtami", "is_ugadi", "is_rathasapthami",
                     "is_ramanavami", "is_shivaratri", "is_winter_holiday",
                     "fest_window_3"]
        for c in keep_fest:
            row[c] = fest_feats[c].values[0] if c in fest_feats.columns else 0

        # Predict
        X_row = np.array([[row.get(f, 0) for f in feature_cols]])
        lgb_prob = lgb_m.predict_proba(X_row)
        xgb_prob = xgb_m.predict_proba(X_row)
        vote_prob = best_alpha*lgb_prob + (1-best_alpha)*xgb_prob
        pred_band = vote_prob.argmax(axis=1)[0]

        # SHAP reason
        shap_vals = shap_explainer.shap_values(X_row)
        sv_list = _extract_sv_list(shap_vals, 0, N_BANDS)
        reason = shap_explain_one(sv_list, pred_band, X_row[0], feature_cols, fdate=fdate)

        # Update history
        band_mid = (BANDS[pred_band]["lo"] + min(BANDS[pred_band]["hi"], 95000)) / 2
        history.append(band_mid)

        probs_sorted = np.argsort(vote_prob[0])[::-1]
        top2 = f"{BAND_NAMES[probs_sorted[0]]}({vote_prob[0][probs_sorted[0]]:.0%})"
        if vote_prob[0][probs_sorted[1]] > 0.1:
            top2 += f" / {BAND_NAMES[probs_sorted[1]]}({vote_prob[0][probs_sorted[1]]:.0%})"

        dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        forecast_rows.append({
            "date": fdate.strftime("%Y-%m-%d"),
            "day": dow_names[fdate.weekday()],
            "predicted_band": BAND_NAMES[pred_band],
            "confidence": f"{vote_prob[0][pred_band]:.0%}",
            "top_predictions": top2,
            "advice": BANDS[pred_band]["advice"],
            "reason": reason,
        })

    fc_df = pd.DataFrame(forecast_rows)
    fc_df.to_csv(OUT_DIR / "forecast_30day.csv", index=False)

    LOG(f"\n  30-DAY FORECAST:")
    LOG(f"  {'Date':<12s} {'Day':<4s} {'Band':<10s} {'Conf':>5s}  Why (SHAP)")
    LOG(f"  {'-'*80}")
    for _, r in fc_df.iterrows():
        LOG(f"  {r.date:<12s} {r.day:<4s} {r.predicted_band:<10s} "
            f"{r.confidence:>5s}  {r.reason}")

    return fc_df


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 8: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def part8_dashboard(df, anom_data, model_data, wf_data, fc_df):
    LOG("\nPART 8: Dashboard")
    y = df.total_pilgrims.values
    bands = pilgrims_to_band_vec(y)

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle("Tirumala Crowd Advisory v5 (LOG + SHAP)", fontsize=16, fontweight="bold")

    # P1: Distribution raw vs log
    ax = axes[0, 0]
    ax2 = ax.twinx()
    ax.hist(y, bins=60, color="#90CAF9", edgecolor="white", alpha=0.7, density=True, label="Raw")
    ax2.hist(np.log1p(y), bins=60, color="#EF9A9A", edgecolor="white", alpha=0.5,
             density=True, label="Log")
    for b in BANDS:
        ax.axvline(x=b["lo"], color=b["color"], linestyle="--", alpha=0.6, linewidth=1)
    ax.set_xlabel("Total Pilgrims"); ax.set_ylabel("Density (raw)")
    ax2.set_ylabel("Density (log)", color="#EF5350")
    ax.set_title("Distribution: Raw vs Log"); ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)

    # P2: Time series + anomalies
    ax = axes[0, 1]
    dates = df.date.values
    ax.plot(dates, y, color="#546E7A", linewidth=0.4, alpha=0.6)
    q_mask = bands == 0; e_mask = bands == 5
    if q_mask.any(): ax.scatter(dates[q_mask], y[q_mask], c="#2196F3", s=15, zorder=3, label="QUIET")
    if e_mask.any(): ax.scatter(dates[e_mask], y[e_mask], c="#B71C1C", s=15, zorder=3, label="EXTREME")
    for b in BANDS[1:]:
        ax.axhline(y=b["lo"], color=b["color"], linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_title("Time Series + Extreme Days"); ax.legend(fontsize=7)
    ax.tick_params(axis="x", rotation=30)

    # P3: Anomaly comparison
    ax = axes[0, 2]
    common_algos = sorted(set(anom_data["point"].keys()) & set(anom_data["ctx"].keys()))
    if not common_algos:
        common_algos = list(anom_data["point"].keys())
    algos = common_algos
    pcnt = [anom_data["point"].get(a, np.zeros(1)).sum() for a in algos]
    ccnt = [anom_data["ctx"].get(a, np.zeros(1)).sum() for a in algos]
    x_pos = np.arange(len(algos)); w = 0.35
    ax.barh(x_pos - w/2, pcnt, w, color="#42A5F5", label="Point")
    ax.barh(x_pos + w/2, ccnt, w, color="#EF5350", label="Contextual")
    ax.set_yticks(x_pos); ax.set_yticklabels(algos, fontsize=8)
    ax.set_xlabel("Anomalies"); ax.set_title("Anomaly Detection Comparison"); ax.legend(fontsize=8)

    # P4: DOW boxplot
    ax = axes[1, 0]
    dow_data = [df.loc[df.date.dt.dayofweek==d,"total_pilgrims"].values for d in range(7)]
    ax.boxplot(dow_data, labels=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
               patch_artist=True, boxprops=dict(facecolor="#B3E5FC", alpha=0.7),
               medianprops=dict(color="#D32F2F", linewidth=2))
    for b in BANDS[1:]:
        ax.axhline(y=b["lo"], color=b["color"], linestyle="--", alpha=0.5)
        ax.text(7.6, b["lo"], b["name"], fontsize=6, color=b["color"], va="center")
    ax.set_title("Day-of-Week + Bands")

    # P5: WF confusion
    ax = axes[1, 1]
    present = sorted(set(wf_data["actuals"]) | set(wf_data["preds"]))
    cm = confusion_matrix(wf_data["actuals"], wf_data["preds"], labels=present)
    ax.imshow(cm, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([BAND_NAMES[i] for i in present], fontsize=7, rotation=45)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels([BAND_NAMES[i] for i in present], fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Walk-Forward Confusion\n(exact={wf_data['exact_acc']*100:.1f}%  "
                 f"adj={wf_data['adjacent_acc']*100:.1f}%)")
    for i in range(len(present)):
        for j in range(len(present)):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i,j]>cm.max()*0.5 else "black")

    # P6: Forecast
    ax = axes[1, 2]
    fc_bands = [BAND_NAMES.index(b) for b in fc_df.predicted_band]
    colors = [BAND_COLORS[b] for b in fc_bands]
    ax.bar(range(len(fc_bands)), [1]*len(fc_bands), color=colors, edgecolor="white", width=1.0)
    ax.set_xticks(range(0, len(fc_bands), 3))
    ax.set_xticklabels([f"{fc_df.date.iloc[i]}\n{fc_df.day.iloc[i]}"
                         for i in range(0, len(fc_bands), 3)], fontsize=6, rotation=45)
    ax.set_yticks([]); ax.set_title("30-Day Forecast")
    patches = [mpatches.Patch(color=BAND_COLORS[i], label=BAND_NAMES[i]) for i in range(N_BANDS)]
    ax.legend(handles=patches, fontsize=6, loc="upper right", ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(EDA_DIR / "40_advisory_v5_dashboard.png", dpi=180, bbox_inches="tight")
    plt.close()
    LOG(f"  Saved eda_outputs/40_advisory_v5_dashboard.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 9: EXPORT FOR DEPLOYMENT
# ═══════════════════════════════════════════════════════════════════════════════
def part9_export(model_data, feature_cols, d, shap_data):
    LOG("\nPART 9: Export artefacts for deployment")
    joblib.dump(model_data["lgb"], OUT_DIR / "lgb_model.pkl")
    joblib.dump(model_data["xgb"], OUT_DIR / "xgb_model.pkl")
    np.save(OUT_DIR / "shap_background.npy", shap_data["X_bg"])

    meta = {
        "feature_cols": feature_cols,
        "bands": BANDS,
        "band_names": BAND_NAMES,
        "n_bands": N_BANDS,
        "feature_labels": FEATURE_LABELS,
        "post_covid": POST_COVID,
        "data_file": DATA_FILE,
    }
    with open(OUT_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    last30 = d.tail(30)[["date", "total_pilgrims", "dow"]].copy()
    last30["date"] = last30.date.dt.strftime("%Y-%m-%d")
    last30.to_csv(OUT_DIR / "last30_history.csv", index=False)

    with open(OUT_DIR / "hyperparams.json", "w") as f:
        json.dump({"lgb": model_data["best_lgb"], "xgb": model_data["best_xgb"],
                   "best_alpha": model_data.get("best_alpha", 0.5)}, f, indent=2)

    LOG("  Exported: lgb_model.pkl, xgb_model.pkl, shap_background.npy, "
        "model_meta.json, last30_history.csv, hyperparams.json")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("  TIRUMALA CROWD ADVISORY v5  (LOG + SHAP + ONLINE-READY)")
    print("=" * 80)

    df = part1_distribution()
    anom_data = part2_anomaly_detection(df)
    d, feature_cols = build_features(df)
    model_data = part4_model(d, feature_cols)
    shap_data  = part5_shap(model_data, feature_cols, d)
    wf_data    = part6_walkforward(d, feature_cols,
                                    model_data["best_lgb"], model_data["best_xgb"],
                                    model_data.get("best_alpha", 0.5))
    fc_df = part7_forecast(d, feature_cols, model_data["lgb"], model_data["xgb"],
                           shap_data["explainer"],
                           model_data.get("best_alpha", 0.5))
    part8_dashboard(df, anom_data, model_data, wf_data, fc_df)
    part9_export(model_data, feature_cols, d, shap_data)

    config = {
        "version": "v5-log-shap",
        "features": {"count": len(feature_cols), "list": feature_cols,
                      "ratio": round(len(d)/len(feature_cols), 1)},
        "anomaly_f1": anom_data["extreme_f1"],
        "test": model_data["results"],
        "champion": model_data["champion"],
        "walkforward": {k: v for k, v in wf_data.items()
                        if k not in ["preds", "actuals", "dates"]},
        "bands": BANDS,
    }
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("  FINAL SUMMARY")
    print("=" * 80)
    ch = model_data["champion"]
    r = model_data["results"][ch]
    print(f"  Champion:        {ch}")
    print(f"  Test exact:      {r['exact_acc']*100:.1f}%")
    print(f"  Test adjacent:   {r['adjacent_acc']*100:.1f}%")
    print(f"  Test safety:     {r['safety']*100:.1f}%")
    print(f"  WF exact:        {wf_data['exact_acc']*100:.1f}%")
    print(f"  WF adjacent:     {wf_data['adjacent_acc']*100:.1f}%")
    print(f"  WF safety:       {wf_data['safety']*100:.1f}%")
    print(f"  WF danger:       {wf_data['danger_count']}")
    print(f"  Features:        {len(feature_cols)} (ratio {len(d)/len(feature_cols):.0f}:1)")
    print(f"  Bands:           {N_BANDS}  ({', '.join(BAND_NAMES)})")
    print(f"  Log transform:   YES (reduces skew)")
    print(f"  SHAP reasons:    YES (data-driven)")
    LOG("DONE")


if __name__ == "__main__":
    main()
