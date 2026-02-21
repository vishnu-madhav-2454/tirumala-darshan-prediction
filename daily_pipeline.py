"""
Tirumala Daily Pipeline  — Scrape → Retrain → Export
=====================================================
Run this daily (via cron, Task Scheduler, or HF Spaces scheduler).

Steps:
  1. Scrape latest darshan data from news.tirumala.org  (incremental)
  2. Reload data + rebuild features
  3. Online retrain: LGB init_model (warm-start), XGB continue training
  4. Recompute SHAP background
  5. Re-export artefacts for Gradio app
  6. (Optional) Push to HuggingFace Spaces

Usage:
    python daily_pipeline.py                # scrape + retrain + export
    python daily_pipeline.py --no-scrape    # retrain only (if data already updated)
    python daily_pipeline.py --push-hf      # also push to HF Spaces
"""

import argparse, json, os, sys, time, pathlib, warnings
import numpy as np
import pandas as pd
import joblib
import shap
import lightgbm as lgb
import xgboost as xgb
from datetime import datetime

warnings.filterwarnings("ignore")
np.random.seed(42)

T0 = time.time()
def LOG(msg): print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)

# ─── Config ─────────────────────────────────────────────────────────
ART_DIR   = pathlib.Path("artefacts/advisory_v5")
DATA_FILE = "data/tirumala_darshan_data_CLEAN_NO_OUTLIERS.csv"
POST_COVID = "2022-02-01"

BANDS = [
    {"id": 0, "name": "QUIET",    "lo": 0,     "hi": 50000},
    {"id": 1, "name": "LIGHT",    "lo": 50000, "hi": 60000},
    {"id": 2, "name": "MODERATE", "lo": 60000, "hi": 70000},
    {"id": 3, "name": "BUSY",     "lo": 70000, "hi": 80000},
    {"id": 4, "name": "HEAVY",    "lo": 80000, "hi": 90000},
    {"id": 5, "name": "EXTREME",  "lo": 90000, "hi": 999999},
]
N_BANDS = 6

def pilgrims_to_band(val):
    for b in BANDS:
        if b["lo"] <= val < b["hi"]:
            return b["id"]
    return 5


def _clean_csv():
    """Keep only date + total_pilgrims columns (scraper may add extras)."""
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    if len(df.columns) > 2:
        df = df[["date", "total_pilgrims"]].copy()
        df.to_csv(DATA_FILE, index=False)
        LOG(f"  Cleaned CSV to 2 columns ({len(df)} rows)")



# ─── Step 1: Scrape ─────────────────────────────────────────────────
def step1_scrape():
    LOG("STEP 1: Scraping latest data ...")
    from app.scraper import scrape_incremental
    added = scrape_incremental(max_pages=5)
    LOG(f"  -> {added} new records added")
    return added


# ─── Step 2: Build features ─────────────────────────────────────────
def step2_build_features():
    LOG("STEP 2: Loading data + building features ...")
    from festival_calendar import get_festival_features_series

    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df = df[df.date >= POST_COVID].copy().sort_values("date").reset_index(drop=True)
    LOG(f"  Data: {len(df)} days  ({df.date.min().date()} -> {df.date.max().date()})")

    d = df.copy()
    y_col = "total_pilgrims"

    d["dow"]        = d.date.dt.dayofweek
    d["month"]      = d.date.dt.month
    d["is_weekend"] = (d.dow >= 5).astype(int)
    doy = d.date.dt.dayofyear
    d["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    d["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

    for lag in [1, 2, 7, 14, 21, 28]:
        d[f"L{lag}"] = d[y_col].shift(lag)

    past = d[y_col].shift(1)
    for w in [7, 14, 30]:
        d[f"rm{w}"] = past.rolling(w).mean()
    d["rstd7"]  = past.rolling(7).std()
    d["rstd14"] = past.rolling(14).std()

    d["dow_expanding_mean"] = d.groupby("dow")[y_col].transform(
        lambda s: s.shift(1).expanding().mean()
    )

    d["log_L1"]   = np.log1p(d["L1"])
    d["log_L7"]   = np.log1p(d["L7"])
    d["log_rm7"]  = np.log1p(d["rm7"])
    d["log_rm30"] = np.log1p(d["rm30"])

    d["momentum_7"] = d["L1"] - d["L7"]
    d["dow_dev"]    = d["L1"] - d["dow_expanding_mean"]

    # NEW: Interaction & extra features
    d["month_dow"]    = d["month"] * 10 + d["dow"]
    d["ewm7"]         = past.ewm(span=7).mean()
    d["ewm14"]        = past.ewm(span=14).mean()
    d["trend_7_14"]   = d["rm7"] - d["rm14"]
    d["trend_7_30"]   = d["rm7"] - d["rm30"]
    d["week_of_year"] = d.date.dt.isocalendar().week.astype(int)
    d["L365"]         = d[y_col].shift(365)
    d["log_L365"]     = np.log1p(d["L365"])

    # Rolling band regime counts (lagged)
    from train_gb_model import pilgrims_to_band_vec
    band_series = pilgrims_to_band_vec(d[y_col].shift(1).fillna(0).values)
    band_s = pd.Series(band_series)
    d["heavy_extreme_count7"] = (band_s >= 4).astype(int).rolling(7, min_periods=1).sum().values
    d["light_quiet_count7"]   = (band_s <= 1).astype(int).rolling(7, min_periods=1).sum().values

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

    d["band"] = np.array([pilgrims_to_band(v) for v in d[y_col].values])
    d = d.dropna().reset_index(drop=True)

    feature_cols = [c for c in d.columns if c not in ["date", y_col, "band"]]
    LOG(f"  Features: {len(feature_cols)}  Samples: {len(d)}")

    return d, feature_cols, df


# ─── Step 3: Online retrain (warm-start) ────────────────────────────
def step3_online_retrain(d, feature_cols):
    LOG("STEP 3: Online retrain (warm-start) ...")

    X = d[feature_cols].values
    y = d["band"].values

    # Load existing models
    old_lgb = joblib.load(ART_DIR / "lgb_model.pkl")
    old_xgb = joblib.load(ART_DIR / "xgb_model.pkl")
    LOG(f"  Loaded existing models from {ART_DIR}")

    # Load hyperparams
    with open(ART_DIR / "hyperparams.json") as f:
        hp = json.load(f)

    # LGB warm-start: init_model continues training
    LOG("  LGB warm-start training ...")
    lgb_params = hp["lgb"].copy()
    # Use fewer estimators for incremental update
    lgb_params["n_estimators"] = min(lgb_params.get("n_estimators", 200), 100)
    new_lgb = lgb.LGBMClassifier(**lgb_params, objective="multiclass",
                                   num_class=N_BANDS, verbosity=-1, random_state=42)
    new_lgb.fit(X, y, init_model=old_lgb)

    # XGB warm-start
    LOG("  XGB warm-start training ...")
    xgb_params = hp["xgb"].copy()
    xgb_params["n_estimators"] = min(xgb_params.get("n_estimators", 200), 100)
    new_xgb = xgb.XGBClassifier(**xgb_params, objective="multi:softprob",
                                  num_class=N_BANDS, verbosity=0, random_state=42,
                                  use_label_encoder=False)
    new_xgb.fit(X, y, xgb_model=old_xgb.get_booster())

    # Quick validation on last 30 days
    X_val = X[-30:]
    y_val = y[-30:]
    lgb_pred = new_lgb.predict(X_val)
    xgb_pred = new_xgb.predict(X_val)
    vote = (hp.get('best_alpha', 0.5)*new_lgb.predict_proba(X_val) + (1-hp.get('best_alpha', 0.5))*new_xgb.predict_proba(X_val)).argmax(1)

    from sklearn.metrics import accuracy_score
    adj = np.mean(np.abs(vote - y_val) <= 1)
    exact = accuracy_score(y_val, vote)
    LOG(f"  Validation (last 30): exact={exact*100:.1f}%  adj={adj*100:.1f}%")

    return new_lgb, new_xgb


# ─── Step 4: SHAP + Export ──────────────────────────────────────────
def step4_export(lgb_m, xgb_m, d, feature_cols, df):
    LOG("STEP 4: Export artefacts ...")

    # Save models
    joblib.dump(lgb_m, ART_DIR / "lgb_model.pkl")
    joblib.dump(xgb_m, ART_DIR / "xgb_model.pkl")

    # SHAP background
    X = d[feature_cols].values
    bg_size = min(200, len(X))
    bg_idx = np.random.choice(len(X), bg_size, replace=False)
    X_bg = X[bg_idx]
    np.save(ART_DIR / "shap_background.npy", X_bg)

    # Last 30 history
    last30 = d.tail(30)[["date", "total_pilgrims", "dow"]].copy()
    last30["date"] = last30.date.dt.strftime("%Y-%m-%d")
    last30.to_csv(ART_DIR / "last30_history.csv", index=False)

    # Update metadata
    with open(ART_DIR / "model_meta.json") as f:
        meta = json.load(f)
    meta["last_retrain"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta["data_end"] = df.date.max().strftime("%Y-%m-%d")
    meta["n_samples"] = len(d)
    with open(ART_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    LOG(f"  Exported to {ART_DIR}")


# ─── Step 5: Push to HuggingFace Spaces ────────────────────────────
def step5_push_hf():
    LOG("STEP 5: Push to HuggingFace Spaces ...")
    try:
        from huggingface_hub import HfApi, upload_folder
        api = HfApi()

        # These should be set via environment variables or .env
        repo_id = os.environ.get("HF_REPO_ID", "")
        if not repo_id:
            LOG("  ⚠ HF_REPO_ID not set — skipping push.")
            LOG("  Set: HF_REPO_ID=your-username/tirumala-crowd-advisory")
            LOG("  Set: HF_TOKEN=hf_xxxxx")
            return False

        token = os.environ.get("HF_TOKEN", "")
        if not token:
            LOG("  ⚠ HF_TOKEN not set — skipping push.")
            return False

        # Files needed on HF Spaces
        files_to_upload = [
            "flask_api.py",
            "festival_calendar.py",
            "hindu_calendar.py",
            "requirements.txt",
            str(ART_DIR / "lgb_model.pkl"),
            str(ART_DIR / "xgb_model.pkl"),
            str(ART_DIR / "shap_background.npy"),
            str(ART_DIR / "model_meta.json"),
            str(ART_DIR / "config.json"),
            str(ART_DIR / "hyperparams.json"),
            str(ART_DIR / "last30_history.csv"),
            DATA_FILE,
        ]

        # Upload entire project dir (simplified)
        upload_folder(
            repo_id=repo_id,
            folder_path=".",
            path_in_repo=".",
            token=token,
            repo_type="space",
            ignore_patterns=["*.pyc", "__pycache__", ".venv*", "artefacts/production/*",
                             "eda_outputs/*", "*.ipynb", ".git/*", "node_modules/*",
                             "app_gradio.py", "client/node_modules/*"],
        )
        LOG(f"  ✅ Pushed to https://huggingface.co/spaces/{repo_id}")
        return True

    except ImportError:
        LOG("  ⚠ huggingface_hub not installed. Run: pip install huggingface-hub")
        return False
    except Exception as e:
        LOG(f"  ❌ Push failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Tirumala Daily Pipeline")
    parser.add_argument("--no-scrape", action="store_true",
                        help="Skip scraping (retrain with existing data)")
    parser.add_argument("--push-hf", action="store_true",
                        help="Push updated artefacts to HF Spaces")
    parser.add_argument("--full-retrain", action="store_true",
                        help="Full Optuna retrain instead of warm-start")
    args = parser.parse_args()

    print("=" * 60)
    print("  TIRUMALA DAILY PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Step 1: Scrape
    if not args.no_scrape:
        added = step1_scrape()
        if added == 0:
            LOG("No new data — checking if retrain needed anyway ...")
        # Keep only the 2 columns needed for ML pipeline
        _clean_csv()

    # Step 2: Build features
    d, feature_cols, df = step2_build_features()

    # Step 3: Retrain
    if args.full_retrain:
        LOG("Full retrain requested — running train_gb_model.py ...")
        os.system("python train_gb_model.py --trials 80 --walkforward")
    else:
        lgb_m, xgb_m = step3_online_retrain(d, feature_cols)

        # Step 4: Export
        step4_export(lgb_m, xgb_m, d, feature_cols, df)

    # Step 5: Push to HF
    if args.push_hf:
        step5_push_hf()

    print("\n" + "=" * 60)
    LOG("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
