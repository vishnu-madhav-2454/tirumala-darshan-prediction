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
import warnings
import logging
from datetime import datetime, timedelta

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

    def get_history(self, page=1, per_page=50, year=None, month=None):
        """Return paginated historical data."""
        raw = self._get_raw_data()

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


# ── Historical Data (paginated) ──
@app.route("/api/data/history", methods=["GET"])
def data_history():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    year = request.args.get("year", type=int)
    month = request.args.get("month", type=int)

    try:
        result = manager.get_history(page, min(per_page, 200), year, month)
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
