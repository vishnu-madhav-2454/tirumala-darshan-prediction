"""
═══════════════════════════════════════════════════════════════════════
  Tirumala Darshan Prediction — Flask REST API
═══════════════════════════════════════════════════════════════════════

Champion Model: Top5-Blend (MAE=2,354 | R²=0.7504)
  Chronos-T5  (58.7%) — Amazon pretrained foundation model
  Tuned-XGB   (21.1%) — Gradient boosting with 68 engineered features
  N-HiTS      (17.0%) — Neural hierarchical interpolation (5-seed)
  N-BEATS     ( 1.7%) — Neural basis expansion (5-seed)
  LGB-GOSS    ( 1.5%) — LightGBM gradient one-side sampling

Endpoints:
  GET  /api/health              → API health + model status
  GET  /api/predict?days=7      → Forecast next N days (1–90)
  POST /api/predict/date        → Predict for a specific date
  GET  /api/predict/range       → Predict for a date range
  GET  /api/data/summary        → Dataset summary statistics
  GET  /api/data/history        → Historical data (paginated)
  GET  /api/model/info          → Model architecture & metrics

Usage:
  python flask_api.py
  → http://localhost:5000
═══════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
from dotenv import load_dotenv
load_dotenv()
import warnings
import logging
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Cloud mode detection — skip Chronos-T5 on Render/Railway (limited RAM)
# HF Spaces has 16GB RAM so it runs ALL models (not considered limited cloud)
IS_CLOUD = (os.environ.get("RENDER") == "1" or os.environ.get("RAILWAY_ENVIRONMENT") is not None) \
           and os.environ.get("SPACE_ID") is None  # HF Spaces sets SPACE_ID

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Project imports ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.features import make_features, get_dl_features

# ── Torch imports ──
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    HAS_TORCH = False
    DEVICE = "cpu"

# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROD_DIR = os.path.join(BASE_DIR, "artefacts", "production")
ARTEFACTS_DIR = os.path.join(BASE_DIR, "artefacts")
DATA_CSV = os.path.join(BASE_DIR, "tirumala_darshan_data_CLEAN_NO_OUTLIERS.csv")

COVID_START = pd.Timestamp("2020-03-19")
COVID_END   = pd.Timestamp("2022-01-31")

# ═══════════════════════════════════════════════════════════════════
#  DL MODEL ARCHITECTURES (must match training)
# ═══════════════════════════════════════════════════════════════════
DL_SEQ = 30  # sequence length

if HAS_TORCH:
    class NBeatsBlock(nn.Module):
        def __init__(self, inp_dim, hidden, theta_dim):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(inp_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            self.theta_b = nn.Linear(hidden, theta_dim)
            self.theta_f = nn.Linear(hidden, 1)
            self.backcast_proj = nn.Linear(theta_dim, inp_dim)

        def forward(self, x):
            h = self.fc(x)
            backcast = self.backcast_proj(self.theta_b(h))
            forecast = self.theta_f(h)
            return backcast, forecast

    class NBeatsNet(nn.Module):
        def __init__(self, inp, n_blocks=4, hidden=256, theta_dim=32):
            super().__init__()
            inp_dim = DL_SEQ * inp
            self.inp_dim = inp_dim
            self.blocks = nn.ModuleList([
                NBeatsBlock(inp_dim, hidden, theta_dim) for _ in range(n_blocks)
            ])

        def forward(self, x):
            B = x.shape[0]
            x = x.reshape(B, -1)
            forecast = torch.zeros(B, 1, device=x.device)
            for block in self.blocks:
                backcast, f = block(x)
                x = x - backcast
                forecast = forecast + f
            return forecast.squeeze(-1)

    class NHiTSBlock(nn.Module):
        def __init__(self, inp_dim, hidden, pool_size):
            super().__init__()
            pooled_dim = inp_dim // pool_size + (1 if inp_dim % pool_size else 0)
            self.pool = nn.AdaptiveMaxPool1d(pooled_dim)
            self.fc = nn.Sequential(
                nn.Linear(pooled_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            self.theta_b = nn.Linear(hidden, inp_dim)
            self.theta_f = nn.Linear(hidden, 1)

        def forward(self, x):
            pooled = self.pool(x.unsqueeze(1)).squeeze(1)
            h = self.fc(pooled)
            return self.theta_b(h), self.theta_f(h)

    class NHiTSNet(nn.Module):
        def __init__(self, inp, n_blocks=3, hidden=256):
            super().__init__()
            inp_dim = DL_SEQ * inp
            self.inp_dim = inp_dim
            pool_sizes = [1, 2, 4]
            self.blocks = nn.ModuleList([
                NHiTSBlock(inp_dim, hidden, pool_sizes[i % len(pool_sizes)])
                for i in range(n_blocks)
            ])

        def forward(self, x):
            B = x.shape[0]
            x = x.reshape(B, -1)
            forecast = torch.zeros(B, 1, device=x.device)
            for block in self.blocks:
                backcast, f = block(x)
                x = x - backcast
                forecast = forecast + f
            return forecast.squeeze(-1)


# ═══════════════════════════════════════════════════════════════════
#  MODEL LOADER (singleton)
# ═══════════════════════════════════════════════════════════════════
class ModelManager:
    """Loads and manages all champion model components."""

    def __init__(self):
        self.loaded = False
        self.config = None
        self.xgb_model = None
        self.lgb_model = None
        self.tab_scaler = None
        self.tgt_scaler = None
        self.exog_scaler = None
        self.nbeats_models = []
        self.nhits_models = []
        self.chronos_pipe = None
        self.blend_weights = None
        self.selected_features = None
        self.dl_feat_cols = None

    def load(self):
        """Load all model artefacts into memory."""
        if self.loaded:
            return

        print("  Loading production artefacts ...")

        # Config
        with open(os.path.join(PROD_DIR, "config.json")) as f:
            self.config = json.load(f)

        self.selected_features = self.config["selected_features"]
        self.dl_feat_cols = self.config["dl_feat_cols"]
        self.blend_weights = self.config["blend_weights"]

        # Tabular models
        self.xgb_model = joblib.load(os.path.join(PROD_DIR, "xgb_tuned.pkl"))
        self.lgb_model = joblib.load(os.path.join(PROD_DIR, "lgb_goss.pkl"))
        self.tab_scaler = joblib.load(os.path.join(PROD_DIR, "tab_scaler.pkl"))
        print("    ✓ XGBoost + LightGBM loaded")

        # DL scalers
        self.tgt_scaler = joblib.load(os.path.join(PROD_DIR, "tgt_scaler.pkl"))
        self.exog_scaler = joblib.load(os.path.join(PROD_DIR, "exog_scaler.pkl"))

        # N-BEATS models (5 seeds local, 2 seeds on cloud to save RAM)
        if HAS_TORCH:
            max_seeds = 2 if IS_CLOUD else 5
            n_feat = len(self.dl_feat_cols) + 1  # +1 for target column
            nbeats_dir = os.path.join(ARTEFACTS_DIR, "nbeats")
            if os.path.isdir(nbeats_dir):
                for i in range(max_seeds):
                    path = os.path.join(nbeats_dir, f"seed_{i}.pt")
                    if os.path.exists(path):
                        m = NBeatsNet(n_feat).to(DEVICE)
                        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
                        m.eval()
                        self.nbeats_models.append(m)
                print(f"    ✓ N-BEATS loaded ({len(self.nbeats_models)} seeds)")

            # N-HiTS models (5 seeds local, 2 seeds on cloud to save RAM)
            nhits_dir = os.path.join(ARTEFACTS_DIR, "nhits")
            if os.path.isdir(nhits_dir):
                for i in range(max_seeds):
                    path = os.path.join(nhits_dir, f"seed_{i}.pt")
                    if os.path.exists(path):
                        m = NHiTSNet(n_feat).to(DEVICE)
                        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
                        m.eval()
                        self.nhits_models.append(m)
                print(f"    ✓ N-HiTS loaded ({len(self.nhits_models)} seeds)")

        # Chronos-T5 (skip on cloud to save RAM)
        if IS_CLOUD:
            print("    ⏭ Chronos-T5 skipped (cloud mode — saving RAM)")
            self.chronos_pipe = None
        else:
            try:
                from chronos import ChronosPipeline
                print("    Loading Chronos-T5-base (this may take a moment) ...")
                self.chronos_pipe = ChronosPipeline.from_pretrained(
                    "amazon/chronos-t5-base",
                    device_map=DEVICE,
                    dtype=torch.float32,
                )
                print("    ✓ Chronos-T5-base loaded")
            except Exception as e:
                print(f"    ⚠ Chronos-T5 unavailable: {e}")
                self.chronos_pipe = None

        self.loaded = True
        print("  ✅ All models loaded successfully!\n")

    def _get_raw_data(self):
        """Load and return COVID-free raw data."""
        raw = pd.read_csv(DATA_CSV, parse_dates=["date"])
        raw = raw[["date", "total_pilgrims"]].sort_values("date").reset_index(drop=True)
        pre = raw[raw.date < COVID_START].copy().reset_index(drop=True)
        post = raw[raw.date > COVID_END].copy().reset_index(drop=True)
        return pd.concat([pre, post], ignore_index=True)

    def _predict_tabular(self, df_feat):
        """Get XGB and LGB predictions for the last row."""
        last_row = df_feat.iloc[[-1]]
        X_row = last_row[self.selected_features].values
        if np.isnan(X_row).any():
            X_row = np.nan_to_num(X_row, nan=0.0)
        xgb_pred = float(self.xgb_model.predict(X_row)[0])
        lgb_pred = float(self.lgb_model.predict(X_row)[0])
        return xgb_pred, lgb_pred

    def _predict_dl(self, dl_df, models, model_name):
        """Get DL model ensemble prediction from sequences."""
        if not models or not HAS_TORCH:
            return None

        tgt_vals = dl_df["total_pilgrims"].values.reshape(-1, 1)
        exog_vals = dl_df[self.dl_feat_cols].values

        tgt_scaled = self.tgt_scaler.transform(tgt_vals)
        exog_scaled = self.exog_scaler.transform(exog_vals)
        combined = np.hstack([tgt_scaled, exog_scaled])

        if len(combined) < DL_SEQ:
            return None

        seq = combined[-DL_SEQ:]
        seq_t = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)

        seed_preds = []
        with torch.no_grad():
            for m in models:
                p = m(seq_t).float().cpu().numpy().flatten()
                inv = self.tgt_scaler.inverse_transform(p.reshape(-1, 1)).flatten()
                seed_preds.append(float(inv[0]))

        return float(np.mean(seed_preds))

    def _predict_chronos(self, series, n_days=1):
        """Get Chronos-T5 prediction."""
        if self.chronos_pipe is None:
            return None

        try:
            # Set seed for reproducible predictions
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            np.random.seed(42)
            
            context = torch.tensor(series, dtype=torch.float32).unsqueeze(0)
            forecast = self.chronos_pipe.predict(
                context, prediction_length=n_days, num_samples=20,
                limit_prediction_length=False,
            )
            preds = forecast.median(dim=1).values.squeeze().cpu().numpy()
            if n_days == 1:
                return float(preds) if np.isscalar(preds) else float(preds[0])
            return preds.tolist()
        except Exception as e:
            logging.warning(f"Chronos prediction failed: {e}")
            return None

    def predict_next_days(self, days=7):
        """Predict pilgrim count for the next N days using Top5-Blend."""
        self.load()

        raw_data = self._get_raw_data()
        last_date = raw_data["date"].max()
        current_raw = raw_data.copy()

        # Chronos: predict all days at once (more efficient)
        chronos_preds = None
        if self.chronos_pipe is not None:
            series = current_raw["total_pilgrims"].values.astype(float)
            chronos_preds = self._predict_chronos(series, n_days=days)
            if chronos_preds is not None and not isinstance(chronos_preds, list):
                chronos_preds = [chronos_preds]

        results = []
        for d in range(days):
            target_date = last_date + timedelta(days=d + 1)

            # ── Tabular prediction (XGB + LGB) ──
            placeholder = pd.DataFrame([{"date": target_date, "total_pilgrims": np.nan}])
            tmp = pd.concat([current_raw, placeholder], ignore_index=True)
            tmp["date"] = pd.to_datetime(tmp["date"])

            df_feat = make_features(tmp)
            xgb_pred, lgb_pred = self._predict_tabular(df_feat)

            # ── DL predictions (N-BEATS + N-HiTS) ──
            # Fill NaN target with XGB prediction for DL sequence
            tmp_dl = tmp.copy()
            tmp_dl.loc[tmp_dl.index[-1], "total_pilgrims"] = xgb_pred
            dl_df = get_dl_features(tmp_dl)

            nbeats_pred = self._predict_dl(dl_df, self.nbeats_models, "N-BEATS")
            nhits_pred = self._predict_dl(dl_df, self.nhits_models, "N-HiTS")

            # ── Chronos prediction ──
            chronos_pred = None
            if chronos_preds is not None and d < len(chronos_preds):
                chronos_pred = chronos_preds[d]

            # ── Top5-Blend ──
            w = self.blend_weights
            components = {}
            blend_val = 0.0
            total_weight = 0.0

            if chronos_pred is not None:
                components["Chronos-T5"] = round(chronos_pred)
                blend_val += w["Chronos-T5"] * chronos_pred
                total_weight += w["Chronos-T5"]

            components["Tuned-XGB"] = round(xgb_pred)
            blend_val += w["Tuned-XGB"] * xgb_pred
            total_weight += w["Tuned-XGB"]

            if nhits_pred is not None:
                components["N-HiTS"] = round(nhits_pred)
                blend_val += w["N-HiTS"] * nhits_pred
                total_weight += w["N-HiTS"]

            if nbeats_pred is not None:
                components["N-BEATS"] = round(nbeats_pred)
                blend_val += w["N-BEATS"] * nbeats_pred
                total_weight += w["N-BEATS"]

            components["LGB-GOSS"] = round(lgb_pred)
            blend_val += w["LGB-GOSS"] * lgb_pred
            total_weight += w["LGB-GOSS"]

            # Re-normalize weights if some models are missing
            if total_weight > 0:
                blend_val = blend_val / total_weight
            else:
                blend_val = xgb_pred

            blend_val = round(blend_val)

            # Confidence band (widens with days ahead)
            base_pct = 0.03
            growth_pct = 0.004
            band_pct = min(base_pct + growth_pct * d, 0.20)

            # Crowd level
            if blend_val >= 80000:
                crowd_level = "Very High"
            elif blend_val >= 65000:
                crowd_level = "High"
            elif blend_val >= 50000:
                crowd_level = "Moderate"
            elif blend_val >= 35000:
                crowd_level = "Low"
            else:
                crowd_level = "Very Low"

            results.append({
                "date": target_date.strftime("%Y-%m-%d"),
                "day": target_date.strftime("%A"),
                "predicted_pilgrims": blend_val,
                "confidence_low": round(blend_val * (1 - band_pct)),
                "confidence_high": round(blend_val * (1 + band_pct)),
                "crowd_level": crowd_level,
                "days_ahead": d + 1,
                "model_breakdown": components,
            })

            # Append prediction as pseudo-observation for next day
            pseudo = pd.DataFrame([{"date": target_date, "total_pilgrims": blend_val}])
            current_raw = pd.concat([current_raw, pseudo], ignore_index=True)

        return results

    def predict_date(self, target_date_str):
        """Predict for a specific date (past or future)."""
        self.load()

        target_date = pd.Timestamp(target_date_str)
        raw_data = self._get_raw_data()
        last_date = raw_data["date"].max()

        days_ahead = (target_date - last_date).days

        if days_ahead <= 0:
            # Past date — check if actual data exists
            actual_row = raw_data[raw_data["date"].dt.date == target_date.date()]
            if len(actual_row) > 0:
                actual_val = int(actual_row.iloc[0]["total_pilgrims"])

                # Also predict what the model would have forecasted
                subset = raw_data[raw_data["date"] <= target_date].copy().reset_index(drop=True)
                if len(subset) >= 31:
                    df_feat = make_features(subset)
                    xgb_pred, lgb_pred = self._predict_tabular(df_feat)
                    predicted = round(xgb_pred)  # simplified for past dates
                else:
                    predicted = None

                return {
                    "date": target_date.strftime("%Y-%m-%d"),
                    "day": target_date.strftime("%A"),
                    "actual_pilgrims": actual_val,
                    "predicted_pilgrims": predicted,
                    "is_past": True,
                    "error": abs(actual_val - predicted) if predicted else None,
                }
            else:
                # Date not in dataset — treat as gap, predict forward
                days_ahead = max(1, days_ahead)

        # Future date
        forecast = self.predict_next_days(max(1, days_ahead))
        result = forecast[-1]
        result["is_past"] = False
        result["actual_pilgrims"] = None
        return result

    def get_data_summary(self):
        """Return dataset summary statistics."""
        raw = self._get_raw_data()
        tp = raw["total_pilgrims"].dropna()
        return {
            "total_records": len(raw),
            "date_range": {
                "start": str(raw["date"].min().date()),
                "end": str(raw["date"].max().date()),
            },
            "covid_period_removed": {
                "start": "2020-03-19",
                "end": "2022-01-31",
            },
            "pilgrim_stats": {
                "mean": round(float(tp.mean())),
                "median": round(float(tp.median())),
                "min": round(float(tp.min())),
                "max": round(float(tp.max())),
                "std": round(float(tp.std())),
            },
            "recent_data": raw.tail(10).assign(
                date=lambda x: x["date"].dt.strftime("%Y-%m-%d")
            ).to_dict(orient="records"),
        }

    def get_history(self, page=1, per_page=50, year=None, month=None,
                     start_date=None, end_date=None):
        """Return paginated historical data with optional date range."""
        raw = self._get_raw_data()

        if start_date:
            raw = raw[raw["date"] >= pd.Timestamp(start_date)]
        if end_date:
            raw = raw[raw["date"] <= pd.Timestamp(end_date)]
        if year:
            raw = raw[raw["date"].dt.year == int(year)]
        if month:
            raw = raw[raw["date"].dt.month == int(month)]

        total = len(raw)
        start = (page - 1) * per_page
        end = start + per_page
        page_data = raw.iloc[start:end]

        return {
            "total_records": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
            "data": page_data.assign(
                date=lambda x: x["date"].dt.strftime("%Y-%m-%d")
            ).to_dict(orient="records"),
        }


# ═══════════════════════════════════════════════════════════════════
#  FLASK APP
# ═══════════════════════════════════════════════════════════════════
STATIC_DIR = os.path.join(BASE_DIR, "client", "dist")

app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Singleton model manager
manager = ModelManager()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ── Health Check ──
@app.route("/api/health", methods=["GET"])
def health():
    has_data = os.path.exists(DATA_CSV)
    has_artefacts = os.path.exists(os.path.join(PROD_DIR, "config.json"))
    return jsonify({
        "status": "ok",
        "models_loaded": manager.loaded,
        "data_available": has_data,
        "artefacts_available": has_artefacts,
        "device": DEVICE,
        "timestamp": datetime.now().isoformat(),
    })


# ── Today's Prediction ──
@app.route("/api/predict/today", methods=["GET"])
def predict_today():
    """Return prediction for today and next 6 days, anchored to current date."""
    try:
        forecast = manager.predict_next_days(7)
        today_str = datetime.now().strftime("%Y-%m-%d")

        # Find today in forecast or use day-1
        today_pred = forecast[0] if forecast else None
        for f in forecast:
            if f["date"] == today_str:
                today_pred = f
                break

        return jsonify({
            "success": True,
            "today": today_str,
            "today_prediction": today_pred,
            "week_forecast": forecast,
        })
    except Exception as e:
        logging.exception("Today prediction error")
        return jsonify({"error": str(e)}), 500


# ── Data Update + Online Learning ──
@app.route("/api/data/update", methods=["POST"])
def data_update():
    """Scrape latest data and reload models (online learning)."""
    try:
        from app.scraper import scrape_incremental

        new_count = scrape_incremental(max_pages=5)
        # Reload data in manager (force refresh)
        manager.loaded = False
        manager.nbeats_models = []
        manager.nhits_models = []
        manager.load()

        return jsonify({
            "success": True,
            "new_records": new_count,
            "message": f"Data updated with {new_count} new records. Models reloaded.",
        })
    except Exception as e:
        logging.exception("Data update error")
        return jsonify({"error": str(e)}), 500


# ── Predict Next N Days ──
@app.route("/api/predict", methods=["GET"])
def predict_days():
    days = request.args.get("days", 7, type=int)
    if days < 1 or days > 90:
        return jsonify({"error": "days must be between 1 and 90"}), 400

    try:
        forecast = manager.predict_next_days(days)
        return jsonify({
            "success": True,
            "days": days,
            "model": "Top5-Blend",
            "forecast": forecast,
        })
    except Exception as e:
        logging.exception("Prediction error")
        return jsonify({"error": str(e)}), 500


# ── Predict Specific Date ──
@app.route("/api/predict/date", methods=["POST"])
def predict_date():
    data = request.get_json()
    if not data or "date" not in data:
        return jsonify({"error": "Request body must include 'date' (YYYY-MM-DD)"}), 400

    try:
        result = manager.predict_date(data["date"])
        return jsonify({
            "success": True,
            "model": "Top5-Blend",
            "prediction": result,
        })
    except Exception as e:
        logging.exception("Date prediction error")
        return jsonify({"error": str(e)}), 500


# ── Predict Date Range ──
@app.route("/api/predict/range", methods=["GET"])
def predict_range():
    start = request.args.get("start")
    end = request.args.get("end")
    if not start or not end:
        return jsonify({"error": "Both 'start' and 'end' query params required (YYYY-MM-DD)"}), 400

    try:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        raw = manager._get_raw_data()
        last_date = raw["date"].max()

        days_to_end = (end_dt - last_date).days
        if days_to_end <= 0:
            return jsonify({"error": "End date must be in the future"}), 400

        days_from_start = max(1, (end_dt - last_date).days)
        forecast = manager.predict_next_days(days_from_start)

        # Filter to requested range
        filtered = [f for f in forecast
                    if start <= f["date"] <= end]

        return jsonify({
            "success": True,
            "model": "Top5-Blend",
            "start": start,
            "end": end,
            "forecast": filtered,
        })
    except Exception as e:
        logging.exception("Range prediction error")
        return jsonify({"error": str(e)}), 500


# ── Data Summary ──
@app.route("/api/data/summary", methods=["GET"])
def data_summary():
    try:
        summary = manager.get_data_summary()
        return jsonify({"success": True, **summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Historical Data (paginated, with date range) ──
@app.route("/api/data/history", methods=["GET"])
def data_history():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    year = request.args.get("year", type=int)
    month = request.args.get("month", type=int)
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    try:
        result = manager.get_history(
            page, min(per_page, 5000), year, month,
            start_date=start_date, end_date=end_date
        )
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Model Info ──
@app.route("/api/model/info", methods=["GET"])
def model_info():
    config_path = os.path.join(PROD_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    return jsonify({
        "success": True,
        "champion": "Top5-Blend",
        "metrics": config.get("champion_metrics", {}),
        "blend_weights": config.get("blend_weights", {}),
        "models": [
            {
                "name": "Chronos-T5",
                "type": "Foundation Model (Pretrained)",
                "weight": 0.587,
                "paper": "Ansari et al. 2024 — arXiv:2403.07815",
                "description": "Amazon's pretrained T5-based time series foundation model",
            },
            {
                "name": "Tuned-XGB",
                "type": "Gradient Boosting",
                "weight": 0.211,
                "description": "XGBoost with 68 engineered features (calendar, lags, rolling stats, Fourier)",
            },
            {
                "name": "N-HiTS",
                "type": "Deep Learning",
                "weight": 0.170,
                "paper": "Challu et al. AAAI 2023",
                "description": "Neural Hierarchical Interpolation — multi-scale time series forecasting",
            },
            {
                "name": "N-BEATS",
                "type": "Deep Learning",
                "weight": 0.017,
                "paper": "Oreshkin et al. ICLR 2020",
                "description": "Neural Basis Expansion Analysis — interpretable residual-subtraction blocks",
            },
            {
                "name": "LGB-GOSS",
                "type": "Gradient Boosting",
                "weight": 0.015,
                "description": "LightGBM with Gradient One-Side Sampling",
            },
        ],
        "data": {
            "records": "3,479 COVID-free (2013–2026)",
            "features": "68 engineered features (MI + XGB importance union)",
            "covid_removed": "Mar 19, 2020 → Jan 31, 2022",
        },
        "device": DEVICE,
    })


# ── LLM Chatbot + AI Trip Planner ──
import json as _json
import random as _random
import re as _re
import textwrap
import time as _time
from calendar import monthrange as _monthrange

from hindu_calendar import (
    get_events_for_date, get_hindu_month_info, get_max_impact, get_crowd_reason,
)

_KB_PATH = os.path.join(BASE_DIR, "ttd_knowledge_base.json")
_TRIP_PATH = os.path.join(BASE_DIR, "tirumala_trip_data.json")
_VECTORDB_DIR = os.path.join(BASE_DIR, "vectordb")
_knowledge_base = None
_trip_data = None

# ── LLM provider globals ──
_llm_provider = None       # "gemini" | "huggingface"
_gemini_model = None       # google.generativeai model
_hf_client = None          # huggingface_hub InferenceClient
_hf_model = None           # HF model name
_llm_available = False
_chroma_collection = None
_rag_available = False

def _load_kb():
    global _knowledge_base
    if _knowledge_base is None:
        try:
            with open(_KB_PATH, "r", encoding="utf-8") as f:
                _knowledge_base = _json.load(f)
            logging.info("✅ TTD Knowledge base loaded (%d categories)", len(_knowledge_base.get("categories", [])))
        except Exception as e:
            logging.error("❌ KB load failed: %s", e)
            _knowledge_base = {"categories": [], "greetings": {"keywords": [], "responses": ["🙏 Namaste!"]}, "fallback": {"responses": ["Sorry, I could not find information on that."]}}
    return _knowledge_base

def _load_trip_data():
    global _trip_data
    if _trip_data is None:
        try:
            with open(_TRIP_PATH, "r", encoding="utf-8") as f:
                _trip_data = _json.load(f)
            logging.info("✅ Trip data loaded")
        except Exception as e:
            logging.error("❌ Trip data load failed: %s", e)
            _trip_data = {}
    return _trip_data

def _init_vectordb():
    """Initialize ChromaDB vector store for RAG."""
    global _chroma_collection, _rag_available
    if not os.path.exists(_VECTORDB_DIR):
        logging.warning("⚠️ Vector DB not found at %s — run build_vectordb.py first", _VECTORDB_DIR)
        return
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        client = chromadb.PersistentClient(path=_VECTORDB_DIR)
        _chroma_collection = client.get_collection(name="ttd_knowledge", embedding_function=ef)
        _rag_available = True
        logging.info("✅ ChromaDB vector store loaded (%d documents)", _chroma_collection.count())
    except Exception as e:
        logging.error("❌ ChromaDB init failed: %s", e)


def _init_llm():
    """Initialize LLM — prefer Google Gemini (fast, reliable), fall back to HuggingFace."""
    global _gemini_model, _hf_client, _hf_model, _llm_available, _llm_provider

    # ── Try Google Gemini first ──
    google_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if google_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=google_key)
            _gemini_model = genai.GenerativeModel("gemini-2.0-flash")
            # Quick test
            _gemini_model.generate_content("hi", generation_config={"max_output_tokens": 5})
            _llm_provider = "gemini"
            _llm_available = True
            logging.info("✅ Google Gemini 2.0 Flash initialized")
            return
        except Exception as e:
            logging.warning("⚠️ Gemini init failed: %s — trying HuggingFace", e)

    # ── Fall back to HuggingFace ──
    try:
        from huggingface_hub import InferenceClient
        hf_token = os.environ.get("HF_TOKEN", "") or None
        _hf_client = InferenceClient(token=hf_token)
        _hf_model = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
        _hf_client.chat_completion(
            model=_hf_model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        _llm_provider = "huggingface"
        _llm_available = True
        logging.info("✅ HuggingFace LLM initialized (model: %s)", _hf_model)
    except Exception as e:
        logging.error("❌ LLM init failed (both Gemini & HF): %s", e)

# Initialize on import
_init_llm()
_init_vectordb()


def _retrieve_context(query: str, n_results: int = 8) -> str:
    """Retrieve relevant context from vector DB using semantic search."""
    if not _rag_available or not _chroma_collection:
        return ""
    try:
        results = _chroma_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        chunks = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            source_tag = meta.get("source", "general")
            title = meta.get("title", "")
            chunks.append(f"[Source: {source_tag} | {title}]\n{doc}")
        return "\n\n---\n\n".join(chunks)
    except Exception as e:
        logging.error("RAG retrieval error: %s", e)
        return ""


def _build_rag_prompt(user_message: str) -> str:
    """Build a RAG-augmented system prompt: retrieve relevant chunks then combine with instructions."""
    context = _retrieve_context(user_message)
    return textwrap.dedent(f"""\
    You are the official TTD (Tirumala Tirupati Devasthanams) AI Assistant — "శ్రీవారి సేవ Bot".
    You help devotees with accurate, detailed information about the Tirumala Venkateswara Temple,
    darshan types & tickets, sevas & costs, accommodation & hotels, travel & transport,
    festivals, temple history & legend, dress code, do's & don'ts, and trip planning.

    GUIDELINES:
    - Be warm, respectful, and devotional in tone. Start with 🙏 when appropriate.
    - Give ACCURATE, SPECIFIC information based ONLY on the context provided below.
    - Include actual prices (₹), timings, phone numbers, and website URLs when available.
    - If the context doesn't contain the answer, say so politely and suggest visiting tirumala.org.
    - Keep answers concise (2-4 paragraphs max) but rich with detail.
    - Use emojis sparingly for warmth (🛕, 🙏, ✅, 📌, 💰).
    - Answer in the same language as the user's question (Telugu, Hindi, or English).
    - Format with bullet points or numbered lists when listing multiple items.
    - NEVER make up information not present in the context.

    === RETRIEVED KNOWLEDGE (from TTD's official data & real scraped information) ===
    {context}
    ===
    """)

def _llm_chat(system_prompt: str, user_message: str, max_tokens: int = 1500,
              temperature: float = 0.5) -> str:
    """Unified LLM call — tries Gemini, then HuggingFace with retries."""
    # ── Gemini ──
    if _llm_provider == "gemini" and _gemini_model:
        try:
            import google.generativeai as genai
            model = genai.GenerativeModel(
                "gemini-2.0-flash",
                system_instruction=system_prompt,
            )
            resp = model.generate_content(
                user_message,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            return resp.text.strip()
        except Exception as e:
            logging.warning("Gemini call failed: %s — trying HF fallback", e)

    # ── HuggingFace (with retries) ──
    if _hf_client:
        last_err = None
        for attempt in range(3):
            try:
                resp = _hf_client.chat_completion(
                    model=_hf_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                last_err = e
                wait = 2 ** attempt
                logging.warning("HF attempt %d failed: %s — retrying in %ds", attempt + 1, e, wait)
                _time.sleep(wait)
        logging.error("HF LLM failed after 3 retries: %s", last_err)

    return ""


def _chatbot_respond(user_message):
    """RAG-powered chatbot: vector retrieval → LLM."""
    if not _llm_available:
        return "🙏 The chatbot is currently unavailable. Please try again later.", "error"
    try:
        system_prompt = _build_rag_prompt(user_message)
        reply = _llm_chat(system_prompt, user_message, max_tokens=1500, temperature=0.5)
        if not reply:
            return "🙏 I'm having trouble processing your request right now. Please try again shortly.", "error"
        source = "rag" if _rag_available else "llm"
        return reply, source
    except Exception as e:
        logging.error("Chatbot error: %s", e)
        return "🙏 I'm having trouble processing your request right now. Please try again shortly.", "error"


@app.route("/api/chat", methods=["POST"])
def chat():
    """RAG Chatbot endpoint — answers questions about TTD using vector search + Gemini AI."""
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"reply": "🙏 Please type your question about Tirumala Tirupati Devasthanams.", "source": "system"})
    reply, source = _chatbot_respond(user_message)
    return jsonify({"reply": reply, "source": source})


# ── AI Trip Planner ──

def _generate_trip_plan(params):
    """Generate a detailed trip plan for visitors already in Tirupati/Tirumala."""
    td = _load_trip_data()
    days = params.get("days", 2)
    budget = params.get("budget", "standard")
    origin = params.get("origin", "Tirupati")  # default: already there
    group_size = params.get("group_size", 2)
    interests = params.get("interests", ["temples", "darshan"])

    # Get relevant cost data
    costs = td.get("daily_costs_estimate", {}).get(budget, td.get("daily_costs_estimate", {}).get("standard", {}))

    # Get transport info
    transport = td.get("transport", {}).get("how_to_reach_tirupati", {}).get("by_road", {}).get("from_cities", {})
    origin_transport = transport.get(origin, transport.get("Chennai", {}))

    # Filter hotels by budget
    all_hotels = []
    for hotel_group in td.get("hotels", {}).values():
        for h in hotel_group:
            if h.get("category") == budget or budget == "luxury":
                all_hotels.append(h)
    if not all_hotels:
        for hotel_group in td.get("hotels", {}).values():
            all_hotels.extend(hotel_group)

    # Get attractions sorted by priority
    attractions = sorted(td.get("attractions", []), key=lambda x: x.get("priority", 99))

    # Get restaurants
    restaurants = td.get("restaurants", [])

    if _llm_available:
        try:
            # Build structured prompt for trip generation
            hotels_text = "\n".join([
                f"- {h['name']}: ₹{h['price_range']['min']}-₹{h['price_range']['max']}/night, {h.get('type', '')}, Rating: {h.get('rating', 'N/A')}"
                for h in all_hotels[:10]
            ])
            attractions_text = "\n".join([
                f"- {a['name']} ({a.get('type','')}): {a.get('description','')[:80]}... | Duration: {a.get('visit_duration_hours', 1)}h | Fee: ₹{a.get('entry_fee', 0)} | Timings: {a.get('timings','')}"
                for a in attractions[:12]
            ])
            restaurants_text = "\n".join([
                f"- {r['name']}: {r.get('cuisine', '')} | ~₹{r.get('price_per_person', 0)}/person | {r.get('timings','')}"
                for r in restaurants[:7]
            ])
            sevas_text = "\n".join([
                f"- {s['name']}: ₹{s['cost']} at {s['time']} — {s['description']}"
                for s in td.get("sevas", [])[:6]
            ])

            prompt = textwrap.dedent(f"""\
            Create a detailed {days}-day Tirumala-Tirupati itinerary for a group of {group_size} people who are ALREADY IN TIRUPATI.
            They have arrived and need a practical day-by-day plan of what to do, where to go, and how to budget their time and money.

            TRIP DETAILS:
            - The group is ALREADY in Tirupati (no travel from another city needed)
            - Budget Level: {budget} (~₹{costs.get('per_day_inr', 2500)}/person/day)
            - Interests: {', '.join(interests)}
            - Group Size: {group_size}

            AVAILABLE HOTELS:
            {hotels_text}

            ATTRACTIONS:
            {attractions_text}

            RESTAURANTS:
            {restaurants_text}

            SEVAS AVAILABLE:
            {sevas_text}

            TRANSPORT:
            - {origin} to Tirupati: Bus ~₹{origin_transport.get('bus_fare', 'N/A')}, Car fuel ~₹{origin_transport.get('fuel_cost_approx', 'N/A')}
            - Tirupati to Tirumala: Bus ₹75 (every 2 min), Taxi ₹1200, Walking (3550 steps/11 km)

            RESPOND IN THIS EXACT JSON FORMAT:
            {{
              "title": "Trip title string",
              "summary": "2-3 sentence overview",
              "recommended_hotel": {{
                "name": "Hotel Name",
                "cost_per_night": 1500,
                "reason": "Why this hotel"
              }},
              "itinerary": [
                {{
                  "day": 1,
                  "title": "Day 1 title",
                  "activities": [
                    {{
                      "time": "6:00 AM",
                      "activity": "Activity description",
                      "location": "Location name",
                      "duration": "2 hours",
                      "cost": 0,
                      "tip": "Helpful tip"
                    }}
                  ]
                }}
              ],
              "cost_breakdown": {{
                "transport_to_tirupati": 500,
                "local_transport": 300,
                "accommodation_total": 3000,
                "food_total": 1000,
                "darshan_sevas": 600,
                "attractions": 200,
                "miscellaneous": 500,
                "total_per_person": 6100,
                "total_group": 12200
              }},
              "map_points": [
                {{"name": "Place Name", "lat": 13.63, "lng": 79.42, "type": "hotel/temple/attraction", "day": 1}}
              ],
              "packing_tips": ["tip1", "tip2"],
              "best_time_tip": "Best time to visit advice"
            }}

            IMPORTANT: Return ONLY valid JSON, no markdown, no code blocks, no extra text.
            Make the itinerary practical and time-aware. Include actual costs.
            Include at least the main temple darshan and top attractions based on interests.
            The group is ALREADY in Tirupati — do NOT include travel from another city.
            Start Day 1 from their hotel in Tirupati itself.
            Include lat/lng coordinates for ALL map_points (Tirumala: ~13.68, 79.35; Tirupati: ~13.63, 79.42).
            """)

            raw = _llm_chat(
                "You are a travel planning assistant. Return ONLY valid JSON, no markdown, no code blocks.",
                prompt,
                max_tokens=4096,
                temperature=0.6,
            )
            if not raw:
                raise ValueError("LLM returned empty response")
            # Clean potential markdown wrapping
            if raw.startswith("```"):
                raw = _re.sub(r'^```\w*\n?', '', raw)
                raw = _re.sub(r'\n?```$', '', raw)
            plan = _json.loads(raw)
            plan["source"] = "llm"
            return plan

        except Exception as e:
            logging.error("Trip plan LLM error: %s", e)

    # Fallback — build a basic plan without LLM
    return _build_fallback_trip(td, params)


def _build_fallback_trip(td, params):
    """Build a basic trip plan without LLM."""
    days = params.get("days", 2)
    budget = params.get("budget", "standard")
    group_size = params.get("group_size", 2)
    origin = params.get("origin", "Chennai")

    costs = td.get("daily_costs_estimate", {}).get(budget, {})
    per_day = costs.get("per_day_inr", 2500)
    attractions = sorted(td.get("attractions", []), key=lambda x: x.get("priority", 99))

    itinerary = []
    attr_idx = 0
    for d in range(1, days + 1):
        activities = []
        if d == 1:
            activities.append({"time": "6:00 AM", "activity": f"Depart from {origin}", "location": origin, "duration": "3-5 hours", "cost": 350, "tip": "Start early to avoid traffic"})
            activities.append({"time": "11:00 AM", "activity": "Arrive Tirupati, check into hotel", "location": "Tirupati", "duration": "1 hour", "cost": 0, "tip": "Keep ID proof ready"})
            activities.append({"time": "12:00 PM", "activity": "Lunch", "location": "Tirupati", "duration": "1 hour", "cost": 150, "tip": "Try local Andhra meals"})
            activities.append({"time": "1:30 PM", "activity": "Travel to Tirumala", "location": "Tirupati → Tirumala", "duration": "45 mins", "cost": 75, "tip": "APSRTC bus every 2 mins from bus stand"})
            activities.append({"time": "3:00 PM", "activity": "Sri Venkateswara Swamy Darshan", "location": "Tirumala", "duration": "3-5 hours", "cost": 300, "tip": "Special Entry ₹300 for shorter queue"})
            activities.append({"time": "8:00 PM", "activity": "Dinner & return to hotel", "location": "Tirumala/Tirupati", "duration": "2 hours", "cost": 200, "tip": "Free Annadanam available at Tirumala"})
        else:
            activities.append({"time": "7:00 AM", "activity": "Breakfast", "location": "Hotel/Restaurant", "duration": "1 hour", "cost": 100, "tip": "Try local tiffins — idli, dosa, pongal"})
            # Add 2-3 attractions per day
            for _ in range(min(3, len(attractions) - attr_idx)):
                if attr_idx < len(attractions):
                    a = attractions[attr_idx]
                    h = 9 + attr_idx * 2
                    activities.append({
                        "time": f"{h}:00 AM" if h < 12 else f"{h-12}:00 PM",
                        "activity": f"Visit {a['name']}",
                        "location": a.get("location", "Tirupati"),
                        "duration": f"{a.get('visit_duration_hours', 1)} hours",
                        "cost": a.get("entry_fee", 0),
                        "tip": a.get("tips", "")
                    })
                    attr_idx += 1
            activities.append({"time": "1:00 PM", "activity": "Lunch", "location": "Restaurant", "duration": "1 hour", "cost": 200, "tip": "Vegetarian food widely available"})
            if d == days:
                activities.append({"time": "4:00 PM", "activity": f"Return journey to {origin}", "location": "Tirupati", "duration": "3-5 hours", "cost": 350, "tip": "Buy prasadam and souvenirs before leaving"})

        itinerary.append({"day": d, "title": f"Day {d}", "activities": activities})

    map_points = [{"name": "Tirupati Railway Station", "lat": 13.6288, "lng": 79.4192, "type": "transport", "day": 1}]
    for a in attractions[:min(days * 3, len(attractions))]:
        map_points.append({"name": a["name"], "lat": a.get("lat", 13.65), "lng": a.get("lng", 79.40), "type": a.get("type", "attraction"), "day": 1})

    total_pp = per_day * days
    return {
        "title": f"🛕 {days}-Day Tirumala Pilgrimage from {origin}",
        "summary": f"A {budget} {days}-day trip to Tirumala-Tirupati from {origin} for {group_size} people. Includes main temple darshan, sightseeing, and local cuisine.",
        "recommended_hotel": {"name": "TTD Guest House", "cost_per_night": per_day // 3, "reason": "Affordable and close to temples"},
        "itinerary": itinerary,
        "cost_breakdown": {
            "transport_to_tirupati": 500, "local_transport": 300 * days,
            "accommodation_total": (per_day // 3) * days, "food_total": 500 * days,
            "darshan_sevas": 600, "attractions": 200, "miscellaneous": 500,
            "total_per_person": total_pp, "total_group": total_pp * group_size
        },
        "map_points": map_points,
        "packing_tips": td.get("packing_essentials", [])[:6],
        "best_time_tip": "October to February is the best time — pleasant weather and fewer crowds on weekdays.",
        "source": "fallback"
    }


@app.route("/api/trip/plan", methods=["POST"])
def trip_plan():
    """AI Trip Planner — generates an itinerary for visitors already in Tirupati."""
    data = request.get_json(silent=True) or {}
    params = {
        "days": min(max(int(data.get("days", 2)), 1), 7),
        "budget": data.get("budget", "standard"),
        "origin": "Tirupati",
        "group_size": min(max(int(data.get("group_size", 2)), 1), 20),
        "interests": data.get("interests", ["temples", "darshan"]),
    }
    plan = _generate_trip_plan(params)
    return jsonify(plan)


@app.route("/api/trip/data", methods=["GET"])
def trip_data():
    """Return raw trip data — hotels, attractions, transport costs."""
    td = _load_trip_data()
    return jsonify({
        "hotels": td.get("hotels", {}),
        "attractions": td.get("attractions", []),
        "restaurants": td.get("restaurants", []),
        "transport": td.get("transport", {}),
        "daily_costs": td.get("daily_costs_estimate", {}),
        "sevas": td.get("sevas", []),
        "festivals": td.get("festivals", []),
        "tips": td.get("tips", []),
    })


# ── Hindu Calendar (predictions + festivals + lunar events) ──
@app.route("/api/calendar", methods=["GET"])
def calendar_month():
    """Return calendar data: predictions + festival annotations for a month."""
    now = datetime.now()
    year = request.args.get("year", now.year, type=int)
    month = request.args.get("month", now.month, type=int)

    if not (1 <= month <= 12) or not (2020 <= year <= 2030):
        return jsonify({"error": "Invalid year/month"}), 400

    try:
        _, days_in_month = _monthrange(year, month)
        today = now.date()

        # Historical data for past dates in this month
        raw = manager._get_raw_data()
        hist_mask = (raw["date"].dt.year == year) & (raw["date"].dt.month == month)
        hist_data = raw[hist_mask].assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
        hist_map = {r["date"]: r for _, r in hist_data.iterrows()}

        # Predictions for future dates
        last_data_date = raw["date"].max().date()
        month_end = date(year, month, days_in_month)
        days_to_end = (month_end - last_data_date).days
        pred_map = {}
        if days_to_end > 0 and days_to_end <= 90:
            forecast = manager.predict_next_days(min(days_to_end, 90))
            pred_map = {f["date"]: f for f in forecast}

        # Build day-by-day calendar
        days = []
        for d in range(1, days_in_month + 1):
            date_str = f"{year}-{month:02d}-{d:02d}"
            date_obj = date(year, month, d)
            day_name = date_obj.strftime("%A")

            entry = {
                "date": date_str,
                "day": d,
                "day_name": day_name,
                "is_weekend": date_obj.weekday() >= 5,
                "is_today": date_obj == today,
            }

            # Pilgrim count (actual or predicted)
            if date_str in hist_map:
                entry["pilgrims"] = int(hist_map[date_str]["total_pilgrims"])
                entry["source"] = "actual"
            elif date_str in pred_map:
                entry["pilgrims"] = pred_map[date_str]["predicted_pilgrims"]
                entry["confidence_low"] = pred_map[date_str].get("confidence_low")
                entry["confidence_high"] = pred_map[date_str].get("confidence_high")
                entry["source"] = "predicted"

            # Hindu calendar events
            events = get_events_for_date(year, month, d)
            entry["events"] = events
            entry["max_impact"] = get_max_impact(events) if events else None
            entry["crowd_reason"] = get_crowd_reason(events) if events else ""

            days.append(entry)

        # Hindu month info
        hindu_info = get_hindu_month_info(month)

        return jsonify({
            "success": True,
            "year": year,
            "month": month,
            "month_name": datetime(year, month, 1).strftime("%B"),
            "hindu_month_te": hindu_info["telugu"],
            "hindu_month_en": hindu_info["english"],
            "days_in_month": days_in_month,
            "first_day_weekday": date(year, month, 1).weekday(),  # 0=Mon
            "days": days,
        })
    except Exception as e:
        logging.exception("Calendar error")
        return jsonify({"error": str(e)}), 500


# ── Serve React SPA ──
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
    """Serve React build for production. All non-API routes fall through to index.html."""
    if path and os.path.exists(os.path.join(STATIC_DIR, path)):
        return send_from_directory(STATIC_DIR, path)
    return send_from_directory(STATIC_DIR, "index.html")


# ═══════════════════════════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🛕 శ్రీవారి సేవ — Tirumala Darshan Prediction API")
    print("=" * 60)
    print(f"  Device: {DEVICE}")

    # Pre-load models on startup
    manager.load()

    # Check if production build exists
    has_build = os.path.exists(os.path.join(STATIC_DIR, "index.html"))

    print(f"  Frontend build: {'✅ Found' if has_build else '❌ Not found (run: cd client && npm run build)'}")
    print(f"  Starting server on http://localhost:5000")
    if has_build:
        print(f"  🌐 Open http://localhost:5000 in your browser")
    print("  API Endpoints:")
    print("    GET  /api/health")
    print("    GET  /api/predict?days=7")
    print("    POST /api/predict/date  {\"date\": \"2026-03-01\"}")
    print("    GET  /api/predict/range?start=...&end=...")
    print("    GET  /api/data/summary")
    print("    GET  /api/data/history?page=1&per_page=50")
    print("    GET  /api/model/info")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=False)
