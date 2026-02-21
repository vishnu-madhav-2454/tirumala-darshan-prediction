"""
Tirumala Crowd Advisory — Flask Backend API
=============================================
Endpoints:
  GET  /api/model-info           → model metadata
  POST /api/predict              → predictions for date range
  GET  /api/calendar/<year>/<month> → calendar data as JSON
  POST /api/chat                 → chatbot response
  GET  /api/history              → last 30 days
  GET  /                         → serves React frontend

Run:
  python flask_api.py
"""

import json, os, pathlib, calendar, functools, re, logging, textwrap, threading
import json as _json
import random as _random
import re as _re
import time as _time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from festival_calendar import get_festival_features_series, get_events_for_date
from hindu_calendar import (
    HINDU_MONTH_MAP, FESTIVALS, PURNIMA, AMAVASYA, EKADASHI,
    IMPACT, get_events_for_date as hc_get_events,
    get_hindu_month_info, get_max_impact, get_impact_factor, get_crowd_reason,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD ARTEFACTS
# ═══════════════════════════════════════════════════════════════════════════════
ART_DIR = pathlib.Path("artefacts/advisory_v5")

gb_model  = joblib.load(ART_DIR / "gb_model.pkl")
lgb_model = joblib.load(ART_DIR / "lgb_model.pkl")
xgb_model = joblib.load(ART_DIR / "xgb_model.pkl")

with open(ART_DIR / "model_meta.json", encoding="utf-8") as f:
    META = json.load(f)
with open(ART_DIR / "hyperparams.json", encoding="utf-8") as f:
    HYPER = json.load(f)

FEATURE_COLS  = META["feature_cols"]
BANDS         = META["bands"]
BAND_NAMES    = META["band_names"]
FEATURE_LABELS = META["feature_labels"]
N_BANDS       = META["n_bands"]
DATA_FILE     = META["data_file"]
GB_PARAMS     = HYPER.get("gb", {})
_ens_w = HYPER.get("ensemble_weights", {"gb": 0.10, "lgb": 0.50, "xgb": 0.40})
W_GB, W_LGB, W_XGB = _ens_w["gb"], _ens_w["lgb"], _ens_w["xgb"]

# Pre-compute feature statistics for deviation-based explanations
_feat_means = {}
_feat_stds  = {}
_fi_arr     = gb_model.feature_importances_

df_full = pd.read_csv(DATA_FILE, parse_dates=["date"])
df_full = df_full[df_full.date >= META["post_covid"]].sort_values("date").reset_index(drop=True)

# ── Daily pipeline state ──
_pipeline_lock = threading.Lock()
_pipeline_status = {
    "last_scrape": None,
    "last_retrain": None,
    "records_added": 0,
    "data_end": df_full.date.max().strftime("%Y-%m-%d"),
    "total_records": len(df_full),
    "status": "idle",
    "error": None,
}

# Knowledge bases
KB_PATH   = pathlib.Path("data/ttd_knowledge_base.json")
KB        = json.load(open(KB_PATH, encoding="utf-8"))   if KB_PATH.exists()   else {"categories": []}

# ═══════════════════════════════════════════════════════════════════════════════
#  COLOURS & LABELS
# ═══════════════════════════════════════════════════════════════════════════════
BAND_COLORS = {
    "QUIET": "#2196F3", "LIGHT": "#4CAF50", "MODERATE": "#8BC34A",
    "BUSY": "#FFC107",  "HEAVY": "#FF5722", "EXTREME": "#B71C1C",
}
BAND_BG = {
    "QUIET": "#E3F2FD", "LIGHT": "#E8F5E9", "MODERATE": "#F1F8E9",
    "BUSY": "#FFF8E1",  "HEAVY": "#FBE9E7", "EXTREME": "#FFEBEE",
}
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def pilgrims_to_band(val):
    for i, b in enumerate(BANDS):
        if b["lo"] <= val < b["hi"]:
            return i
    return N_BANDS - 1


# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICTION ENGINE  (Gradient Boosting — feature-importance-based explanations)
# ═══════════════════════════════════════════════════════════════════════════════
def _rebuild_feature_stats():
    """Rebuild feature means/stds from current data for deviation-based explanations."""
    global _feat_means, _feat_stds
    from train_gb_model import build_features, pilgrims_to_band_vec
    d_tmp = df_full.copy()
    d_tmp, _ = build_features(d_tmp)
    X_tmp = d_tmp[FEATURE_COLS].values
    _feat_means = {f: X_tmp[:, i].mean() for i, f in enumerate(FEATURE_COLS)}
    _feat_stds  = {f: X_tmp[:, i].std() + 1e-9 for i, f in enumerate(FEATURE_COLS)}

try:
    _rebuild_feature_stats()
except Exception:
    # Fallback: will use 0 means and 1 stds
    _feat_means = {f: 0 for f in FEATURE_COLS}
    _feat_stds  = {f: 1 for f in FEATURE_COLS}


def importance_explain(feature_vals, feature_names, fdate=None):
    """Generate human-readable explanations using feature importance + deviation from mean."""
    scores = []
    for fi_idx, fname in enumerate(feature_names):
        fval = feature_vals[fi_idx]
        z = (fval - _feat_means.get(fname, 0)) / _feat_stds.get(fname, 1)
        score = _fi_arr[fi_idx] * abs(z)
        scores.append((fname, score, z, fval))
    scores.sort(key=lambda x: x[1], reverse=True)

    reasons = []
    for fname, score, z, rval in scores[:3]:
        if score < 0.001:
            continue
        label = FEATURE_LABELS.get(fname, fname)
        direction = "pushes UP" if z > 0 else "pushes DOWN"
        if fdate and fname in ["is_festival", "fest_impact", "is_brahmotsavam",
                                "is_sankranti"] and rval > 0:
            events = get_events_for_date(fdate.year, fdate.month, fdate.day)
            if events:
                label = events[0]["name"]
        if fname == "dow":
            reasons.append(f"{DOW_NAMES[int(rval)]} {direction}")
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


def build_single_features(fdate, history_vals, dow_means, month_dow_means=None):
    """Build feature vector for one date (mirrors train_gb_model.build_features)."""
    row = {}
    row["dow"]        = fdate.weekday()
    row["month"]      = fdate.month
    row["is_weekend"] = 1 if fdate.weekday() >= 5 else 0
    doy = fdate.timetuple().tm_yday
    row["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    row["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    n = len(history_vals)
    for lag in [1, 2, 7, 14, 21, 28]:
        idx = n - lag
        row[f"L{lag}"] = history_vals[idx] if idx >= 0 else history_vals[0]
    for w in [7, 14, 30]:
        vals = history_vals[max(0, n - w):]
        row[f"rm{w}"] = np.mean(vals)
    row["rstd7"]  = np.std(history_vals[-7:])  if n >= 7  else 0
    row["rstd14"] = np.std(history_vals[-14:]) if n >= 14 else 0
    row["dow_expanding_mean"] = dow_means.get(fdate.weekday(), 70000)

    # Month×DOW expanding mean (seasonal weekday patterns)
    if month_dow_means is not None:
        md_key = fdate.month * 10 + fdate.weekday()
        row["month_dow_mean"] = month_dow_means.get(md_key, row["dow_expanding_mean"])
    else:
        row["month_dow_mean"] = row["dow_expanding_mean"]

    row["log_L1"]   = np.log1p(row["L1"])
    row["log_L7"]   = np.log1p(row["L7"])
    row["log_rm7"]  = np.log1p(row["rm7"])
    row["log_rm30"] = np.log1p(row["rm30"])

    row["momentum_7"] = row["L1"] - row["L7"]
    row["dow_dev"]    = row["L1"] - row["dow_expanding_mean"]

    row["month_dow"]   = row["month"] * 10 + row["dow"]
    vals7 = history_vals[-7:]
    w7 = np.exp(np.linspace(-1, 0, len(vals7)))
    row["ewm7"]  = np.average(vals7, weights=w7) if vals7 else row["rm7"]
    vals14 = history_vals[-14:]
    w14 = np.exp(np.linspace(-1, 0, len(vals14)))
    row["ewm14"] = np.average(vals14, weights=w14) if vals14 else row.get("rm14", row["rm7"])
    row["trend_7_14"] = row["rm7"] - row["rm14"]
    row["trend_7_30"] = row["rm7"] - row["rm30"]
    row["week_of_year"] = fdate.isocalendar()[1]
    idx365 = n - 365
    row["L365"]     = history_vals[idx365] if idx365 >= 0 else history_vals[0]
    row["log_L365"] = np.log1p(row["L365"])

    # Year-over-year growth
    row["yoy_growth"] = max(-2, min(2, (row["L1"] - row["L365"]) / (row["L365"] + 1e-9)))

    # Month × weekend interaction
    row["month_weekend"] = row["month"] * row["is_weekend"]

    recent = history_vals[-8:-1] if n >= 8 else history_vals[-7:]
    recent_bands = [pilgrims_to_band(v) for v in recent]
    row["heavy_extreme_count7"] = sum(1 for b in recent_bands if b >= 4)
    row["light_quiet_count7"]   = sum(1 for b in recent_bands if b <= 1)

    fest = get_festival_features_series(pd.Series([pd.Timestamp(fdate)]))
    keep_fest = [
        "is_festival", "fest_impact", "is_brahmotsavam", "is_sankranti",
        "is_summer_holiday", "is_dasara_holiday", "is_national_holiday",
        "days_to_fest", "days_from_fest", "fest_window_7",
        "is_vaikuntha_ekadashi", "is_dussehra_period", "is_diwali",
        "is_navaratri", "is_janmashtami", "is_ugadi", "is_rathasapthami",
        "is_ramanavami", "is_shivaratri", "is_winter_holiday",
        "fest_window_3",
    ]
    for c in keep_fest:
        row[c] = fest[c].values[0] if c in fest.columns else 0

    # Lunar features (is_pournami, is_amavasya, is_ekadashi) are intentionally
    # excluded.  hindu_calendar.py only covers 2025-2027; training data spans
    # 2022-2026, so including these would mean 3+ years of all-zero values —
    # zero signal, pure noise.  Re-add when the calendar is extended to 2022.

    return np.array([[row.get(f, 0) for f in FEATURE_COLS]])


def predict_one(fdate, history_vals, dow_means, month_dow_means=None):
    X = build_single_features(fdate, history_vals, dow_means, month_dow_means)
    # Weighted ensemble: GB(0.10) + LGB(0.50) + XGB(0.40)
    prob = (
        W_GB  * gb_model.predict_proba(X) +
        W_LGB * lgb_model.predict_proba(X) +
        W_XGB * xgb_model.predict_proba(X)
    )
    pred_band = int(prob.argmax(axis=1)[0])
    conf      = float(prob[0][pred_band])
    reason    = importance_explain(X[0], FEATURE_COLS, fdate=fdate)

    # ── Festival boost ──────────────────────────────────────────────
    # is_brahmotsavam / is_vaikuntha_ekadashi are not in FEATURE_COLS
    # (dropped during feature selection because training lag features
    # already captured those peaks historically).  For FUTURE dates the
    # lag history has no memory of upcoming festival boosts, so we
    # apply a deterministic calendar floor here.
    from festival_calendar import get_festival_features
    fest = get_festival_features(fdate)
    fest_impact    = fest.get("fest_impact", 0)
    is_brahmo      = fest.get("is_brahmotsavam", 0)
    is_vaikuntha   = fest.get("is_vaikuntha_ekadashi", 0)
    is_sankranti   = fest.get("is_sankranti", 0)

    # Brahmotsavams / Vaikuntha Ekadashi / Sankranti → floor at HEAVY (4)
    if is_brahmo or is_vaikuntha or is_sankranti:
        if pred_band < 4:          # below HEAVY
            pred_band = 4
            conf = max(conf, 0.78)
            tag = ("Brahmotsavams" if is_brahmo
                   else "Vaikuntha Ekadashi" if is_vaikuntha
                   else "Sankranti")
            reason = f"{tag} (festival floor) | {reason}"
    # Any other impact=5 festival → floor at BUSY (3)
    elif fest_impact >= 5 and pred_band < 3:
        pred_band = 3
        conf = max(conf, 0.72)
        reason = f"Major festival (floor) | {reason}"
    # Summer / Dasara / Winter school holidays → floor at BUSY (3) on weekends
    elif fest_impact >= 4 and fdate.weekday() >= 5 and pred_band < 3:
        pred_band = 3
        conf = max(conf, 0.68)
        reason = f"Holiday weekend (floor) | {reason}"

    return pred_band, conf, reason


# ── prediction cache ────────────────────────────────────────────────
_prediction_cache    = {}
_CACHE_TTL_SECS      = 6 * 3600   # 6 hours — stale predictions are recomputed

# ── Lookup index for fast actual-data retrieval ─────────────────────
_actual_data = {}   # date-string → total_pilgrims

def _rebuild_actual_lookup():
    """Rebuild the actual-data lookup from df_full."""
    global _actual_data
    _actual_data = {
        r.date.strftime("%Y-%m-%d"): int(r.total_pilgrims)
        for _, r in df_full.iterrows()
    }

_rebuild_actual_lookup()


def predict_date_range(start_date, end_date):
    """Return predictions for a date range.
    - Past dates with actual data → use real pilgrim counts (is_actual=True)
    - Today and future dates → use ML prediction (is_actual=False)
    - Confidence decays for dates far from the last real data point.
    """
    history = list(df_full.total_pilgrims.values)
    dow_means = df_full.groupby(df_full.date.dt.dayofweek)["total_pilgrims"].mean().to_dict()
    # Month×DOW means for seasonal weekday patterns
    _tmp = df_full.copy()
    _tmp["_md_key"] = _tmp.date.dt.month * 10 + _tmp.date.dt.dayofweek
    month_dow_means = _tmp.groupby("_md_key")["total_pilgrims"].mean().to_dict()
    last_date = df_full.date.max().to_pydatetime()
    today_str = date.today().strftime("%Y-%m-%d")

    # ── Confidence decay for far-future dates ──────────────────────
    # Autoregressive lag features degrade as we predict further out.
    # Apply a smooth sigmoid decay: full confidence within 60 days,
    # gradual decay starting after 60 days, floored at 0.45.
    def _decay_confidence(raw_conf, days_ahead):
        if days_ahead <= 60:
            return raw_conf
        # Sigmoid decay: halves confidence gap at ~180 days
        import math
        decay = 1.0 / (1.0 + math.exp((days_ahead - 180) / 60))
        floor = 0.45
        return max(floor, raw_conf * decay)

    # Fill gap between data end and start_date (if needed for lag features)
    if start_date > last_date:
        gap_days = (start_date - last_date).days
        for i in range(1, gap_days):
            d = last_date + timedelta(days=i)
            ck = d.strftime("%Y-%m-%d")
            if ck in _actual_data:
                history.append(_actual_data[ck])
            else:
                # Expire stale cache entries so far-future predictions refresh
                if ck in _prediction_cache and _time.time() - _prediction_cache[ck].get("_ts", 0) > _CACHE_TTL_SECS:
                    del _prediction_cache[ck]
                if ck in _prediction_cache:
                    history.append(_prediction_cache[ck]["mid"])
                else:
                    b, c, r = predict_one(d, history, dow_means, month_dow_means)
                    # Use DOW-seasonal mean clipped to predicted band — preserves
                    # weekly variation in the autoregressive lag chain instead of
                    # collapsing to a constant band midpoint after many steps.
                    band_lo  = BANDS[b]["lo"]
                    band_hi  = min(BANDS[b]["hi"], 95000)
                    dow_est  = dow_means.get(d.weekday(), (band_lo + band_hi) / 2)
                    mid      = max(band_lo, min(band_hi, dow_est))
                    _prediction_cache[ck] = {"band": b, "conf": c, "reason": r, "mid": mid, "_ts": _time.time()}
                    history.append(mid)

    results = []
    total_days = (end_date - start_date).days + 1
    for i in range(total_days):
        fdate = start_date + timedelta(days=i)
        ck = fdate.strftime("%Y-%m-%d")

        # Expire stale cache entries so predictions refresh periodically
        if ck in _prediction_cache and _time.time() - _prediction_cache[ck].get("_ts", 0) > _CACHE_TTL_SECS:
            del _prediction_cache[ck]

        # If actual data exists for this date → use it
        if ck in _actual_data:
            actual_val = _actual_data[ck]
            actual_band = pilgrims_to_band(actual_val)
            results.append({
                "date": fdate, "band": actual_band,
                "conf": 1.0, "reason": "Actual data",
                "pilgrims": actual_val, "is_actual": True,
            })
            history.append(actual_val)
        elif ck in _prediction_cache:
            cached = _prediction_cache[ck]
            days_ahead = (fdate - last_date).days
            decayed_conf = _decay_confidence(cached["conf"], days_ahead)
            results.append({
                "date": fdate, "band": cached["band"],
                "conf": decayed_conf, "reason": cached["reason"],
                "pilgrims": cached["mid"], "is_actual": False,
            })
            history.append(cached["mid"])
        else:
            b, c, r = predict_one(fdate, history, dow_means, month_dow_means)
            days_ahead = (fdate - last_date).days
            c = _decay_confidence(c, days_ahead)
            # Use DOW-seasonal mean clipped to predicted band (not a constant
            # band midpoint) so the autoregressive lag chain retains weekly
            # variation even hundreds of steps into the future.
            band_lo  = BANDS[b]["lo"]
            band_hi  = min(BANDS[b]["hi"], 95000)
            dow_est  = dow_means.get(fdate.weekday(), (band_lo + band_hi) / 2)
            mid      = max(band_lo, min(band_hi, dow_est))
            _prediction_cache[ck] = {"band": b, "conf": c, "reason": r, "mid": mid, "_ts": _time.time()}
            results.append({
                "date": fdate, "band": b, "conf": c, "reason": r,
                "pilgrims": mid, "is_actual": False,
            })
            history.append(mid)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  DAILY PIPELINE — Scrape → Retrain → Reload (runs on schedule)
# ═══════════════════════════════════════════════════════════════════════════════
def _daily_pipeline_job():
    """Scrape new data, warm-start retrain, and hot-reload models.
    Called by APScheduler daily at 6:00 AM IST (00:30 UTC)."""
    global df_full, gb_model, lgb_model, xgb_model
    global _prediction_cache, _actual_data

    with _pipeline_lock:
        _pipeline_status["status"] = "running"
        _pipeline_status["error"] = None
        log.info("=" * 50)
        log.info("DAILY PIPELINE STARTED")
        log.info("=" * 50)

        try:
            # ── Step 1: Scrape ──
            log.info("[Pipeline] Step 1: Scraping latest data ...")
            from app.scraper import scrape_incremental
            added = scrape_incremental(max_pages=5)
            _pipeline_status["last_scrape"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _pipeline_status["records_added"] = added
            log.info(f"[Pipeline]   -> {added} new records added")

            # ── Step 2: Reload data ──
            log.info("[Pipeline] Step 2: Reloading data ...")
            df_new = pd.read_csv(DATA_FILE, parse_dates=["date"])
            df_new = df_new[df_new.date >= META["post_covid"]].sort_values("date").reset_index(drop=True)
            df_full = df_new
            _rebuild_actual_lookup()
            _pipeline_status["data_end"] = df_full.date.max().strftime("%Y-%m-%d")
            _pipeline_status["total_records"] = len(df_full)
            log.info(f"[Pipeline]   Data: {len(df_full)} days up to {df_full.date.max().date()}")

            # Clear prediction cache (stale after new data)
            _prediction_cache.clear()
            log.info("[Pipeline]   Prediction cache cleared")

            # ── Step 3: Online retrain (warm-start) ──
            if added > 0:
                log.info("[Pipeline] Step 3: Online retrain (warm-start) ...")
                _online_retrain()
                _pipeline_status["last_retrain"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log.info("[Pipeline]   Retrain complete")
            else:
                log.info("[Pipeline] Step 3: No new data — skipping retrain")

            _pipeline_status["status"] = "idle"
            log.info("=" * 50)
            log.info("DAILY PIPELINE COMPLETE")
            log.info("=" * 50)

        except Exception as e:
            _pipeline_status["status"] = "error"
            _pipeline_status["error"] = str(e)
            log.error(f"[Pipeline] ERROR: {e}", exc_info=True)


def _online_retrain():
    """Warm-start retrain all 3 models on updated data and hot-reload ensemble."""
    global gb_model, lgb_model, xgb_model, _fi_arr

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    from train_gb_model import build_features, pilgrims_to_band_vec
    from collections import Counter
    import lightgbm as lgb_lib
    import xgboost as xgb_lib

    d = df_full.copy()
    d, _ = build_features(d)

    X = d[FEATURE_COLS].values
    y = d["band"].values

    # Class weights for balanced training
    class_counts = Counter(y)
    n_samples = len(y)
    n_classes = len(class_counts)
    cw = {c: n_samples / (n_classes * cnt) for c, cnt in class_counts.items()}
    sw = np.array([cw[yi] for yi in y])

    # ── GB: full retrain with saved hyperparams ──
    new_gb = GradientBoostingClassifier(**GB_PARAMS.copy(), random_state=42)
    new_gb.fit(X, y, sample_weight=sw)

    # ── LGB: warm-start (init_model continues training) ──
    lgb_params = HYPER.get("lgb", {}).copy()
    lgb_params["n_estimators"] = min(lgb_params.get("n_estimators", 300), 100)
    new_lgb = lgb_lib.LGBMClassifier(**lgb_params, objective="multiclass",
                                      num_class=N_BANDS, verbosity=-1, random_state=42)
    new_lgb.fit(X, y, init_model=lgb_model)

    # ── XGB: warm-start (xgb_model continues training) ──
    xgb_params = HYPER.get("xgb", {}).copy()
    xgb_params["n_estimators"] = min(xgb_params.get("n_estimators", 300), 100)
    new_xgb = xgb_lib.XGBClassifier(**xgb_params, objective="multi:softprob",
                                      num_class=N_BANDS, verbosity=0, random_state=42)
    new_xgb.fit(X, y, xgb_model=xgb_model.get_booster())

    # ── Ensemble validation on last 30 days ──
    X_val, y_val = X[-30:], y[-30:]
    ens_prob = (
        W_GB  * new_gb.predict_proba(X_val) +
        W_LGB * new_lgb.predict_proba(X_val) +
        W_XGB * new_xgb.predict_proba(X_val)
    )
    pred = ens_prob.argmax(axis=1)
    exact = accuracy_score(y_val, pred)
    adj = np.mean(np.abs(pred - y_val) <= 1)
    log.info(f"[Pipeline]   Ensemble validation (last 30): exact={exact*100:.1f}%  adj={adj*100:.1f}%")

    # ── Save all 3 models to disk ──
    joblib.dump(new_gb,  ART_DIR / "gb_model.pkl")
    joblib.dump(new_lgb, ART_DIR / "lgb_model.pkl")
    joblib.dump(new_xgb, ART_DIR / "xgb_model.pkl")

    # Update meta
    with open(ART_DIR / "model_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    meta["last_retrain"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta["data_end"] = df_full.date.max().strftime("%Y-%m-%d")
    meta["n_samples"] = len(d)
    with open(ART_DIR / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # Hot-reload into globals
    gb_model  = new_gb
    lgb_model = new_lgb
    xgb_model = new_xgb
    _fi_arr = gb_model.feature_importances_
    _rebuild_feature_stats()
    log.info("[Pipeline]   GB model hot-reloaded into memory")


def _start_scheduler():
    """Start APScheduler to run daily pipeline at 6:00 AM IST (00:30 UTC)."""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger

        scheduler = BackgroundScheduler(daemon=True)
        # TTD publishes yesterday's data around 10-11 AM IST next day.
        # Run at 12:30 PM IST (07:00 UTC) to ensure data is available.
        scheduler.add_job(
            _daily_pipeline_job,
            CronTrigger(hour=7, minute=0),   # 07:00 UTC = 12:30 PM IST
            id="daily_pipeline",
            name="Daily Scrape + Retrain",
            replace_existing=True,
            misfire_grace_time=3600,
        )
        scheduler.start()
        log.info("Scheduler started: daily pipeline at 12:30 PM IST (07:00 UTC)")

        # Also run once at startup if data might be stale
        data_end = df_full.date.max().date()
        yesterday = date.today() - timedelta(days=1)
        if data_end < yesterday:
            log.info(f"Data ends at {data_end}, yesterday was {yesterday} — running pipeline now")
            threading.Thread(target=_daily_pipeline_job, daemon=True).start()

    except ImportError:
        log.warning("APScheduler not installed — daily automation disabled. "
                     "Install with: pip install apscheduler")
    except Exception as e:
        log.error(f"Scheduler failed to start: {e}")



# ═══════════════════════════════════════════════════════════════════════════════
#  LLM CHATBOT (RAG with ChromaDB + HuggingFace)
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_KB_PATH = os.path.join(BASE_DIR, "data", "ttd_knowledge_base.json")
_VECTORDB_DIR = os.path.join(BASE_DIR, "vectordb")
_knowledge_base = None

# ── LLM provider globals (Groq primary, HuggingFace fallback) ──
_groq_client = None        # Groq client (primary)
_groq_model = None         # Groq model name
_hf_chat_client = None     # HF client for chatbot (fallback)
_hf_chat_model = None      # Model name for chatbot
_llm_provider = None       # "groq" | "hf" | None
_chat_available = False
_chroma_collection = None
_rag_available = False


def _load_kb():
    global _knowledge_base
    if _knowledge_base is None:
        try:
            with open(_KB_PATH, "r", encoding="utf-8") as f:
                _knowledge_base = _json.load(f)
            logging.info("Knowledge base loaded")
        except Exception as e:
            logging.error("KB load failed: %s", e)
            _knowledge_base = {"categories": []}
    return _knowledge_base


def _init_vectordb():
    """Initialize ChromaDB vector store for RAG."""
    global _chroma_collection, _rag_available
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        if not os.path.exists(_VECTORDB_DIR):
            logging.warning("Vector DB not found at %s. Run: python build_vectordb.py", _VECTORDB_DIR)
            return

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        client = chromadb.PersistentClient(path=_VECTORDB_DIR)
        _chroma_collection = client.get_collection(
            name="ttd_knowledge",
            embedding_function=ef,
        )
        _rag_available = True
        logging.info("ChromaDB loaded: %d documents", _chroma_collection.count())
    except Exception as e:
        logging.error("ChromaDB init failed: %s", e)
        _rag_available = False


def _init_llm():
    """Initialize LLM client: try Groq first (fast, free), fall back to HuggingFace."""
    global _groq_client, _groq_model
    global _hf_chat_client, _hf_chat_model
    global _chat_available, _llm_provider

    # ── Try Groq first (recommended — fast, free, supports Llama-3.3-70B) ──
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        try:
            from groq import Groq
            _groq_model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
            _groq_client = Groq(api_key=groq_key)
            # Quick test
            _groq_client.chat.completions.create(
                model=_groq_model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
            _chat_available = True
            _llm_provider = "groq"
            logging.info("Chatbot LLM ready (Groq): %s", _groq_model)
            return
        except ImportError:
            logging.warning("groq package not installed — trying HuggingFace fallback")
        except Exception as e:
            logging.warning("Groq init failed: %s — trying HuggingFace fallback", e)

    # ── Fallback: HuggingFace Inference API ──
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        logging.error("Neither groq nor huggingface_hub installed — chatbot disabled")
        return

    chat_token = os.environ.get("HF_TOKEN_CHAT", os.environ.get("HF_TOKEN", "")) or None
    _hf_chat_model = os.environ.get("HF_MODEL_CHAT", "meta-llama/Llama-3.3-70B-Instruct")
    try:
        _hf_chat_client = InferenceClient(token=chat_token)
        _hf_chat_client.chat_completion(
            model=_hf_chat_model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        _chat_available = True
        _llm_provider = "hf"
        logging.info("Chatbot LLM ready (HF): %s", _hf_chat_model)
    except Exception as e:
        logging.error("HuggingFace chatbot LLM init also failed: %s — chatbot will use keyword fallback", e)


# Initialize on import
_init_llm()
_init_vectordb()


def _retrieve_context(query: str, n_results: int = 8) -> str:
    """Retrieve relevant context from vector DB using semantic search."""
    if not _rag_available or _chroma_collection is None:
        return ""
    try:
        results = _chroma_collection.query(query_texts=[query], n_results=n_results)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        context_parts = []
        for doc, meta in zip(docs, metas):
            src = meta.get("source", "unknown")
            title = meta.get("title", "")
            context_parts.append(f"[Source: {src} | {title}]\n{doc}")
        return "\n\n---\n\n".join(context_parts)
    except Exception as e:
        logging.error("Vector search error: %s", e)
        return ""


def _build_rag_prompt(user_message: str, lang: str = "en") -> str:
    """Build a RAG-augmented system prompt: retrieve relevant chunks then combine with instructions."""
    context = _retrieve_context(user_message)

    LANG_NAMES = {
        "te": "Telugu", "en": "English", "hi": "Hindi",
        "ta": "Tamil", "ml": "Malayalam", "kn": "Kannada",
    }
    lang_name = LANG_NAMES.get(lang, "English")

    return textwrap.dedent(f"""\
    You are the official TTD (Tirumala Tirupati Devasthanams) AI Assistant -- "Srivari Seva Bot".
    You help devotees with accurate, detailed information about the Tirumala Venkateswara Temple,
    darshan types & tickets, sevas & costs, accommodation & hotels, travel & transport,
    festivals, temple history & legend, dress code, do's & don'ts, and trip planning.

    CRITICAL LANGUAGE INSTRUCTION:
    The user's selected language is **{lang_name}** (code: {lang}).
    You MUST respond ENTIRELY in **{lang_name}** script and language.
    - If {lang_name} is Telugu, write in Telugu script (తెలుగు).
    - If {lang_name} is Tamil, write in Tamil script (தமிழ்).
    - If {lang_name} is Malayalam, write in Malayalam script (മലയാളം).
    - If {lang_name} is Kannada, write in Kannada script (ಕನ್ನಡ).
    - If {lang_name} is Hindi, write in Devanagari script (हिन्दी).
    - If {lang_name} is English, write in English.
    Do NOT mix languages. Do NOT default to English unless the user's language is English.
    Proper nouns like "Tirumala", "TTD", "Venkateswara" can remain in English/original form.

    GUIDELINES:
    - Be warm, respectful, and devotional in tone. Start with a namaskaram when appropriate.
    - Give ACCURATE, SPECIFIC information based ONLY on the context provided below.
    - Include actual prices, timings, phone numbers, and website URLs when available.
    - If the context doesn't contain the answer, say so politely and suggest visiting tirumala.org.
    - Keep answers concise (2-4 paragraphs max) but rich with detail.
    - Use emojis sparingly for warmth.
    - Format with bullet points or numbered lists when listing multiple items.
    - NEVER make up information not present in the context.

    === RETRIEVED KNOWLEDGE (from TTD's official data & real scraped information) ===
    {context}
    ===
    """)


# ── Circuit breaker: skip LLM if it keeps failing ──
_llm_fail_counts = {}   # model -> consecutive fail count
_llm_fail_until = {}    # model -> timestamp until which to skip
_LLM_CIRCUIT_THRESHOLD = 2   # failures before tripping
_LLM_CIRCUIT_COOLDOWN = 300  # seconds to skip after tripping (5 min)


def _llm_call(system_prompt: str, user_message: str,
              max_tokens: int = 1500, temperature: float = 0.5) -> str:
    """Call the active LLM provider (Groq or HF) with circuit breaker + retries."""
    if _llm_provider == "groq" and _groq_client:
        model = _groq_model
    elif _llm_provider == "hf" and _hf_chat_client:
        model = _hf_chat_model
    else:
        return ""

    # Circuit breaker: skip if recently failing
    now = _time.time()
    if model in _llm_fail_until and now < _llm_fail_until[model]:
        remaining = int(_llm_fail_until[model] - now)
        logging.info("Circuit breaker OPEN for %s — skipping LLM (retry in %ds)", model, remaining)
        return ""

    last_err = None
    for attempt in range(2):
        try:
            if _llm_provider == "groq":
                resp = _groq_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            else:
                resp = _hf_chat_client.chat_completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            # Success — reset circuit breaker
            _llm_fail_counts[model] = 0
            if model in _llm_fail_until:
                del _llm_fail_until[model]
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            logging.warning("LLM %s attempt %d failed: %s -- retry in %ds", model, attempt + 1, e, wait)
            _time.sleep(wait)

    # All retries failed — update circuit breaker
    _llm_fail_counts[model] = _llm_fail_counts.get(model, 0) + 1
    if _llm_fail_counts[model] >= _LLM_CIRCUIT_THRESHOLD:
        _llm_fail_until[model] = now + _LLM_CIRCUIT_COOLDOWN
        logging.warning("Circuit breaker TRIPPED for %s — skipping for %ds", model, _LLM_CIRCUIT_COOLDOWN)
    logging.error("LLM %s failed after 2 retries: %s", model, last_err)
    return ""


def _chatbot_respond(user_message, lang="en"):
    """RAG-powered chatbot: vector retrieval + LLM."""
    if not _chat_available:
        return _fallback_chatbot(user_message)
    try:
        system_prompt = _build_rag_prompt(user_message, lang=lang)
        reply = _llm_call(system_prompt, user_message, max_tokens=1500, temperature=0.5)
        if not reply:
            return _fallback_chatbot(user_message)
        source = "rag" if _rag_available else "llm"
        return reply, source
    except Exception as e:
        logging.error("Chatbot error: %s", e)
        return _fallback_chatbot(user_message)


def _fallback_chatbot(user_message):
    """Fallback keyword-based chatbot when LLM is unavailable."""
    kb = _load_kb()
    query_lower = user_message.lower().strip()
    best_score = 0
    best_answer = None
    for cat in kb.get("categories", []):
        cat_score = sum(1 for kw in cat.get("keywords", []) if kw in query_lower)
        for qa in cat.get("qa", []):
            q = qa["q"].lower()
            q_words = set(q.split())
            query_words = set(query_lower.split())
            overlap = len(q_words & query_words)
            score = overlap + cat_score * 0.5
            if query_lower in q or q in query_lower:
                score += 5
            if score > best_score:
                best_score = score
                best_answer = qa["a"]
    if best_score >= 2 and best_answer:
        return best_answer, "knowledge_base"
    return ("I can help with temple info, darshan types, sevas, accommodation, "
            "travel, laddus, and crowd forecast. "
            "Try asking 'What types of darshan are available?' or "
            "'How busy is it on 2026-03-15?'"), "fallback"


# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK APP
# ═══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__, static_folder="client/build", static_url_path="")
CORS(app)


# ── API: Health ─────────────────────────────────────────────────────
@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "data_rows": len(df_full),
                    "data_end": df_full.date.max().strftime("%Y-%m-%d")})


# ── API: Pipeline Status ────────────────────────────────────────────
@app.route("/api/pipeline/status")
def api_pipeline_status():
    """Return the current state of the daily scrape + retrain pipeline."""
    return jsonify(_pipeline_status)


@app.route("/api/pipeline/trigger", methods=["POST"])
def api_pipeline_trigger():
    """Manually trigger the daily pipeline (scrape + retrain)."""
    if _pipeline_status["status"] == "running":
        return jsonify({"error": "Pipeline already running"}), 409
    threading.Thread(target=_daily_pipeline_job, daemon=True).start()
    return jsonify({"message": "Pipeline triggered", "status": "running"})


# ── API: Predict Today ─────────────────────────────────────────────
@app.route("/api/predict/today")
def api_predict_today():
    today = datetime.now()
    predictions = predict_date_range(today, today)
    p = predictions[0]
    bn = BAND_NAMES[p["band"]]
    events = hc_get_events(today.year, today.month, today.day)
    ev_list = [e["name"] for e in events] if events else []
    return jsonify({
        "date": today.strftime("%Y-%m-%d"),
        "day": DOW_NAMES[today.weekday()],
        "predicted_band": bn,
        "predicted_pilgrims": round(p.get("pilgrims", 0)),
        "confidence": round(p["conf"], 3),
        "reason": p["reason"],
        "advice": BANDS[p["band"]].get("advice", ""),
        "color": BAND_COLORS[bn],
        "events": ev_list,
        "is_actual": p.get("is_actual", False),
    })


# ── API: Predict (GET for quick forecast, POST for date range) ──────
@app.route("/api/predict", methods=["GET", "POST"])
def api_predict():
    if request.method == "GET":
        n = min(int(request.args.get("days", 7)), 90)
        start = datetime.now()
        end = start + timedelta(days=n - 1)
        predictions = predict_date_range(start, end)
        forecast = []
        for p in predictions:
            bn = BAND_NAMES[p["band"]]
            forecast.append({
                "date": p["date"].strftime("%Y-%m-%d"),
                "day": DOW_NAMES[p["date"].weekday()],
                "predicted_band": bn,
                "band_name": bn,
                "predicted_pilgrims": round(p.get("pilgrims", 0)),
                "confidence": round(p["conf"], 3),
                "reason": p["reason"],
                "advice": BANDS[p["band"]].get("advice", ""),
                "color": BAND_COLORS[bn],
                "bg": BAND_BG[bn],
                "is_weekend": p["date"].weekday() >= 5,
                "is_actual": p.get("is_actual", False),
            })
        return jsonify({"forecast": forecast, "total": len(forecast)})

    # POST — date range prediction
    data = request.json or {}
    start_str = data.get("start_date", "")
    end_str   = data.get("end_date", "")
    try:
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end   = datetime.strptime(end_str,   "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    if end < start:
        return jsonify({"error": "End date must be after start date."}), 400
    if (end - start).days > 90:
        return jsonify({"error": "Maximum 90 days range."}), 400

    predictions = predict_date_range(start, end)
    results = []
    for p in predictions:
        bn = BAND_NAMES[p["band"]]
        results.append({
            "date": p["date"].strftime("%Y-%m-%d"),
            "day":  DOW_NAMES[p["date"].weekday()],
            "band": p["band"],
            "band_name": bn,
            "predicted_band": bn,
            "predicted_pilgrims": round(p.get("pilgrims", 0)),
            "confidence": round(p["conf"], 3),
            "reason": p["reason"],
            "advice": BANDS[p["band"]].get("advice", ""),
            "color": BAND_COLORS[bn],
            "bg": BAND_BG[bn],
            "is_weekend": p["date"].weekday() >= 5,
            "is_actual": p.get("is_actual", False),
        })

    from collections import Counter
    band_counts = Counter(r["band_name"] for r in results)
    summary = {bn: band_counts.get(bn, 0) for bn in BAND_NAMES}
    best = sorted(results, key=lambda r: (r["band"], -r["confidence"]))[:3]

    return jsonify({
        "predictions": results,
        "summary": summary,
        "best_days": best,
        "total_days": len(results),
    })


# ── API: Data Summary ──────────────────────────────────────────────
@app.route("/api/data/summary")
def api_data_summary():
    s = df_full["total_pilgrims"]
    return jsonify({
        "total_records": len(df_full),
        "date_range": {
            "start": df_full.date.min().strftime("%Y-%m-%d"),
            "end": df_full.date.max().strftime("%Y-%m-%d"),
        },
        "pilgrim_stats": {
            "mean": round(s.mean()),
            "median": round(s.median()),
            "max": int(s.max()),
            "min": int(s.min()),
            "std": round(s.std()),
        },
    })


# ── API: Model Info ─────────────────────────────────────────────────
@app.route("/api/model-info")
def api_model_info():
    return jsonify({
        "n_features": len(FEATURE_COLS),
        "n_bands": N_BANDS,
        "band_names": BAND_NAMES,
        "bands": BANDS,
        "data_end": df_full.date.max().strftime("%Y-%m-%d"),
        "data_rows": len(df_full),
        "model_type": META.get("model_type", "GradientBoosting"),
        "last_retrain": META.get("last_retrain", ""),
        "band_colors": BAND_COLORS,
        "band_bg": BAND_BG,
    })


# ── API: Calendar ───────────────────────────────────────────────────
@app.route("/api/calendar/<int:year>/<int:month>")
def api_calendar(year, month):
    hindu_info = get_hindu_month_info(month)
    first_day  = datetime(year, month, 1)
    last_day_n = calendar.monthrange(year, month)[1]
    last_day   = datetime(year, month, last_day_n)
    predictions = predict_date_range(first_day, last_day)
    first_weekday = calendar.monthrange(year, month)[0]  # Mon=0

    days = []
    today_d = date.today()
    for idx, p in enumerate(predictions):
        day_num = idx + 1
        bn = BAND_NAMES[p["band"]]
        events = hc_get_events(year, month, day_num)
        ev_list = []
        for e in (events or [])[:2]:
            imp = IMPACT.get(e.get("impact", "low"), {})
            ev_list.append({
                "name": e["name"],
                "name_te": e.get("name_te", ""),
                "emoji": e.get("emoji", "📌"),
                "impact": e.get("impact", "low"),
                "impact_color": {"extreme": "#B71C1C", "very_high": "#E65100",
                                 "high": "#F57F17", "moderate": "#2E7D32",
                                 "low": "#757575"}.get(e.get("impact", "low"), "#757575"),
            })
        days.append({
            "day": day_num,
            "band": p["band"],
            "band_name": bn,
            "confidence": round(p["conf"], 3),
            "color": BAND_COLORS[bn],
            "bg": BAND_BG[bn],
            "events": ev_list,
            "is_today": (year == today_d.year and month == today_d.month and day_num == today_d.day),
            "is_actual": p.get("is_actual", False),
            "total_pilgrims": round(p.get("pilgrims", 0)),
        })

    # Festivals this month
    year_fests  = FESTIVALS.get(year, [])
    month_fests = []
    for f in year_fests:
        if f[0] == month:
            imp = IMPACT.get(f[5], IMPACT["low"])
            month_fests.append({
                "day": f[1], "name": f[2], "name_te": f[3],
                "type": f[4], "impact": f[5], "impact_label": imp.get("label", ""),
            })

    return jsonify({
        "year": year, "month": month,
        "month_name": calendar.month_name[month],
        "hindu_month": hindu_info,
        "first_weekday": first_weekday,
        "num_days": last_day_n,
        "days": days,
        "festivals": month_fests,
    })


# ── API: Chat (RAG-powered) ──────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json or {}
    message = data.get("message", "").strip()
    lang = data.get("lang", "en").strip()
    if lang not in ("te", "en", "hi", "ta", "ml", "kn"):
        lang = "en"
    if not message:
        return jsonify({"reply": "Please enter a question.", "source": "error"})

    # Check for date-based crowd prediction queries
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', message)
    if date_match:
        try:
            fdate = datetime.strptime(date_match.group(1), "%Y-%m-%d")
            predictions = predict_date_range(fdate, fdate)
            p = predictions[0]
            bn = BAND_NAMES[p["band"]]
            events = hc_get_events(fdate.year, fdate.month, fdate.day)
            ev_names = [e["name"] for e in events] if events else []
            return jsonify({
                "reply": f"Crowd prediction for {fdate.strftime('%B %d, %Y (%A)')}",
                "prediction": {
                    "date": fdate.strftime("%Y-%m-%d"),
                    "band": bn, "confidence": f"{p['conf']:.0%}",
                    "advice": BANDS[p["band"]].get("advice", ""),
                    "reason": p["reason"],
                    "events": ev_names,
                },
                "source": "prediction",
            })
        except Exception:
            pass

    # Check for crowd forecast keywords
    crowd_keywords = ["crowd", "busy", "rush", "best time", "when to visit",
                      "how many", "pilgrim", "footfall", "peak", "quiet", "darshan time"]
    query_lower = message.lower()
    if any(kw in query_lower for kw in crowd_keywords):
        today_dt = datetime.now()
        predictions = predict_date_range(today_dt, today_dt + timedelta(days=6))
        forecast = []
        for p in predictions:
            bn = BAND_NAMES[p["band"]]
            forecast.append({"date": p["date"].strftime("%b %d (%a)"), "band": bn, "conf": f"{p['conf']:.0%}"})
        if forecast:
            return jsonify({
                "reply": "Here's the current crowd forecast for the next 7 days:",
                "crowd_forecast": forecast,
                "source": "forecast",
            })

    # RAG + LLM chatbot
    result = _chatbot_respond(message, lang=lang)
    if isinstance(result, tuple):
        reply, source = result
    else:
        reply, source = result, "error"
    return jsonify({"reply": reply, "source": source})


# ── API: History (with pagination & filtering) ─────────────────────
@app.route("/api/history")
def api_history():
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    filtered = df_full.copy()
    if start_date:
        try:
            filtered = filtered[filtered.date >= pd.Timestamp(start_date)]
        except Exception:
            pass
    if end_date:
        try:
            filtered = filtered[filtered.date <= pd.Timestamp(end_date)]
        except Exception:
            pass

    # Sort by date descending for recent-first
    filtered = filtered.sort_values("date", ascending=False)
    total = len(filtered)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_df = filtered.iloc[start_idx:end_idx]

    records = []
    for _, r in page_df.iterrows():
        val = int(r.total_pilgrims)
        band_idx = pilgrims_to_band(val)
        records.append({
            "date": r.date.strftime("%Y-%m-%d"),
            "total_pilgrims": val,
            "band_index": band_idx,
            "band_name": BAND_NAMES[band_idx],
            "band_color": BAND_COLORS[BAND_NAMES[band_idx]],
            "band_bg": BAND_BG[BAND_NAMES[band_idx]],
            "day_of_week": DOW_NAMES[r.date.weekday()],
            "is_weekend": r.date.weekday() >= 5,
        })

    # Summary stats for the full filtered dataset
    s = filtered["total_pilgrims"]
    busiest_row = filtered.loc[s.idxmax()]
    quietest_row = filtered.loc[s.idxmin()]
    summary = {
        "avg": round(s.mean()),
        "median": round(s.median()),
        "max": int(s.max()),
        "min": int(s.min()),
        "busiest_date": busiest_row.date.strftime("%Y-%m-%d"),
        "busiest_pilgrims": int(busiest_row.total_pilgrims),
        "quietest_date": quietest_row.date.strftime("%Y-%m-%d"),
        "quietest_pilgrims": int(quietest_row.total_pilgrims),
        "date_start": filtered.date.min().strftime("%Y-%m-%d"),
        "date_end": filtered.date.max().strftime("%Y-%m-%d"),
    }

    return jsonify({
        "data": records,
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
        "summary": summary,
        "band_names": BAND_NAMES,
        "band_colors": BAND_COLORS,
        "band_bg": BAND_BG,
    })


# ── Serve React frontend (MUST be AFTER all /api routes) ───────────
@app.route("/")
def serve_react():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    fpath = os.path.join(app.static_folder, path)
    if os.path.isfile(fpath):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")


@app.errorhandler(404)
def not_found(e):
    """SPA fallback — serve index.html for all non-API routes."""
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(app.static_folder, "index.html")


# ═══════════════════════════════════════════════════════════════════════════════
# Start scheduler (works for both gunicorn and direct run)
_start_scheduler()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print("  TIRUMALA CROWD ADVISORY - Flask API")
    print(f"  Data: {len(df_full)} days  ({df_full.date.min().date()} to {df_full.date.max().date()})")
    print(f"  Features: {len(FEATURE_COLS)}  |  Bands: {N_BANDS}")
    print(f"  Model: Ensemble (GB×{W_GB:.0%} + LGB×{W_LGB:.0%} + XGB×{W_XGB:.0%})")
    print(f"  Daily automation: scrape + retrain at 12:30 PM IST")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)
