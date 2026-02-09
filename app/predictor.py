"""
Prediction / inference module.

Loads saved artefacts and produces forecasts for future dates.

Usage (standalone):
    python -m app.predictor --days 7
"""
import json
import os
import argparse
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import joblib

# Suppress sklearn feature-name warning when passing numpy arrays
warnings.filterwarnings("ignore", message="X does not have valid feature names")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from app.config import (
    DATA_CSV, SEED, SEQ_LEN,
    BIGRU_HIDDEN, BIGRU_LAYERS, BIGRU_DROPOUT,
    LGB_MODEL_PATH, BIGRU_MODEL_DIR, SCALER_PATH,
    TGT_SCALER_PATH, EXOG_SCALER_PATH,
    FEATURES_PATH, BLEND_WEIGHTS_PATH, DEFAULT_BLEND, N_SEEDS,
)
from app.features import make_features, get_dl_features

DEVICE = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"


# ── Bi-GRU definition (must match trainer) ──────────────────────────
if HAS_TORCH:
    class BiGRU(nn.Module):
        def __init__(self, inp, hid=48, layers=2, drop=0.35):
            super().__init__()
            self.gru = nn.GRU(
                inp, hid, layers, batch_first=True, bidirectional=True,
                dropout=drop if layers > 1 else 0,
            )
            self.drop = nn.Dropout(drop)
            self.fc = nn.Linear(hid * 2, 1)
        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(self.drop(out[:, -1, :])).squeeze(-1)


def _load_artefacts():
    """Load all saved model artefacts. Returns dict."""
    arts = {}

    # LGB
    if os.path.exists(LGB_MODEL_PATH):
        arts["lgb_model"] = joblib.load(LGB_MODEL_PATH)
        arts["scaler"] = joblib.load(SCALER_PATH)
        with open(FEATURES_PATH) as f:
            arts["features"] = json.load(f)
    else:
        raise FileNotFoundError(
            "No trained model found. Run `python -m app.trainer --force` first."
        )

    # Blend weights
    if os.path.exists(BLEND_WEIGHTS_PATH):
        with open(BLEND_WEIGHTS_PATH) as f:
            arts["blend"] = json.load(f)
    else:
        arts["blend"] = DEFAULT_BLEND

    # Bi-GRU
    meta_path = os.path.join(BIGRU_MODEL_DIR, "meta.json")
    if HAS_TORCH and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        arts["bigru_meta"] = meta
        arts["tgt_scaler"] = joblib.load(TGT_SCALER_PATH)
        arts["exog_scaler"] = joblib.load(EXOG_SCALER_PATH)
        models = []
        for s in range(N_SEEDS):
            path = os.path.join(BIGRU_MODEL_DIR, f"seed_{s}.pt")
            if os.path.exists(path):
                m = BiGRU(
                    meta["n_feat"], BIGRU_HIDDEN, BIGRU_LAYERS, BIGRU_DROPOUT
                ).to(DEVICE)
                m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
                m.eval()
                models.append(m)
        arts["bigru_models"] = models

    return arts


def _predict_past_date(target_date, arts, raw):
    """Generate a model prediction for a date that is already in the dataset.

    Uses all data *before* that date to build features and predict, so
    we see what the model would have forecast for that day.
    """
    target_date = pd.Timestamp(target_date)
    idx = raw.index[raw["date"].dt.date == target_date.date()]

    if len(idx) == 0:
        return None, None  # date not in dataset

    actual_val = int(raw.loc[idx[0], "total_pilgrims"])

    # Build features using only data up to (and including) the target date
    # The model only uses *lagged* features so it never sees the current value
    subset = raw[raw["date"] <= target_date].copy().reset_index(drop=True)
    if len(subset) < 31:
        return actual_val, None  # not enough history for a prediction

    df_feat = make_features(subset)
    last_row = df_feat.iloc[[-1]]
    X_row = last_row[arts["features"]].values
    if np.isnan(X_row).any():
        X_row = np.nan_to_num(X_row, nan=0.0)

    predicted = float(arts["lgb_model"].predict(X_row)[0])

    # BiGRU prediction for past dates (if available)
    bigru_pred = None
    if "bigru_models" in arts and arts["bigru_models"]:
        dl_df = get_dl_features(subset)
        dl_feat_cols = arts["bigru_meta"]["dl_feat_cols"]
        tgt_vals = dl_df["total_pilgrims"].values.reshape(-1, 1)
        exog_vals = dl_df[dl_feat_cols].values
        tgt_sc = arts["tgt_scaler"]
        exog_sc = arts["exog_scaler"]
        tgt_scaled = tgt_sc.transform(tgt_vals)
        exog_scaled = exog_sc.transform(exog_vals)
        combined = np.hstack([tgt_scaled, exog_scaled])
        if len(combined) >= SEQ_LEN:
            seq = combined[-SEQ_LEN:]
            seq_t = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
            seed_preds = []
            with torch.no_grad():
                for m in arts["bigru_models"]:
                    p = m(seq_t).cpu().numpy().flatten()
                    inv = tgt_sc.inverse_transform(p.reshape(-1, 1)).flatten()
                    seed_preds.append(float(inv[0]))
            bigru_pred = float(np.mean(seed_preds))

    # Blend
    bw = arts["blend"]
    if bigru_pred is not None:
        blend_pred = round(bw["BiGRU"] * bigru_pred + bw["LGB-GOSS"] * predicted)
    else:
        blend_pred = round(predicted)

    return actual_val, blend_pred


def predict_single_date(target_date) -> dict:
    """Predict pilgrim count for a specific target date.

    For past dates with actual data: returns both actual and predicted values.
    For future dates: returns the forecast with confidence band.
    """
    target_date = pd.Timestamp(target_date)
    arts = _load_artefacts()
    raw = pd.read_csv(DATA_CSV, parse_dates=["date"])
    raw = raw[["date", "total_pilgrims"]].sort_values("date").dropna().reset_index(drop=True)
    last_date = raw["date"].max()

    days_ahead = (target_date - last_date).days

    if days_ahead <= 0:
        # Date is in the past — return both actual AND predicted
        actual_row = raw[raw["date"].dt.date == target_date.date()]
        if len(actual_row) > 0:
            actual_val, predicted_val = _predict_past_date(target_date, arts, raw)
            return {
                "date": target_date,
                "actual": actual_val,
                "predicted": predicted_val,
                "blend_pred": predicted_val if predicted_val else actual_val,
                "is_actual": True,
                "days_ahead": 0,
            }
        else:
            days_ahead = max(1, days_ahead)

    # Future date — forecast iteratively
    df = predict_next_days(days_ahead)
    last_row = df.iloc[-1]

    base_pct = 0.03
    growth_pct = 0.005
    band_pct = min(base_pct + growth_pct * (days_ahead - 1), 0.25)
    blend_val = last_row["blend_pred"]

    return {
        "date": target_date,
        "actual": None,
        "predicted": round(blend_val),
        "blend_pred": round(blend_val),
        "is_actual": False,
        "confidence_low": round(blend_val * (1 - band_pct)),
        "confidence_high": round(blend_val * (1 + band_pct)),
        "days_ahead": days_ahead,
    }


def predict_next_days(days: int = 7) -> pd.DataFrame:
    """Predict pilgrim count for the next `days` days.

    Returns DataFrame with columns: date, lgb_pred, bigru_pred, blend_pred.
    """
    arts = _load_artefacts()
    raw = pd.read_csv(DATA_CSV, parse_dates=["date"])
    raw = raw[["date", "total_pilgrims"]].sort_values("date").dropna().reset_index(drop=True)
    last_date = raw["date"].max()

    results = []
    # We predict one day at a time, appending the prediction as a pseudo-
    # observation so lag features are available for the next day.
    current_raw = raw.copy()

    for d in range(1, days + 1):
        target_date = last_date + timedelta(days=d)

        # --- LGB prediction ---
        # Append a placeholder row for the target date
        placeholder = pd.DataFrame(
            [{"date": target_date, "total_pilgrims": np.nan}]
        )
        tmp = pd.concat([current_raw, placeholder], ignore_index=True)
        tmp["date"] = pd.to_datetime(tmp["date"])

        df_feat = make_features(tmp)
        # The last row is our target
        last_row = df_feat.iloc[[-1]]
        X_row = last_row[arts["features"]].values

        # Handle NaNs in features with forward fill from previous values
        if np.isnan(X_row).any():
            X_row = np.nan_to_num(X_row, nan=0.0)

        lgb_pred = float(arts["lgb_model"].predict(X_row)[0])

        # --- BiGRU prediction ---
        bigru_pred = None
        if "bigru_models" in arts and arts["bigru_models"]:
            dl_df = get_dl_features(tmp)
            dl_feat_cols = arts["bigru_meta"]["dl_feat_cols"]

            tgt_vals = dl_df["total_pilgrims"].values.reshape(-1, 1)
            # Fill NaN target (last row) with the LGB prediction for sequence building
            tgt_vals[-1, 0] = lgb_pred
            exog_vals = dl_df[dl_feat_cols].values

            tgt_sc = arts["tgt_scaler"]
            exog_sc = arts["exog_scaler"]
            tgt_scaled = tgt_sc.transform(tgt_vals)
            exog_scaled = exog_sc.transform(exog_vals)
            combined = np.hstack([tgt_scaled, exog_scaled])

            if len(combined) >= SEQ_LEN:
                seq = combined[-SEQ_LEN:]
                seq_t = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)

                seed_preds = []
                with torch.no_grad():
                    for m in arts["bigru_models"]:
                        p = m(seq_t).cpu().numpy().flatten()
                        inv = tgt_sc.inverse_transform(p.reshape(-1, 1)).flatten()
                        seed_preds.append(float(inv[0]))
                bigru_pred = float(np.mean(seed_preds))

        # --- Blend ---
        bw = arts["blend"]
        if bigru_pred is not None:
            blend = bw["BiGRU"] * bigru_pred + bw["LGB-GOSS"] * lgb_pred
        else:
            blend = lgb_pred

        results.append({
            "date": target_date,
            "lgb_pred": round(lgb_pred),
            "bigru_pred": round(bigru_pred) if bigru_pred else None,
            "blend_pred": round(blend),
        })

        # Append pseudo-observation for next iteration
        pseudo = pd.DataFrame(
            [{"date": target_date, "total_pilgrims": blend}]
        )
        current_raw = pd.concat([current_raw, pseudo], ignore_index=True)

    return pd.DataFrame(results)


# ── CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict pilgrim count")
    parser.add_argument("--days", type=int, default=7, help="Days to forecast")
    args = parser.parse_args()

    df = predict_next_days(args.days)
    print("\n  📊 Tirumala Darshan Forecast")
    print("  " + "=" * 50)
    for _, row in df.iterrows():
        day_name = row["date"].strftime("%a %d-%b-%Y")
        print(f"  {day_name}  →  {row['blend_pred']:,.0f} pilgrims")
    print()
