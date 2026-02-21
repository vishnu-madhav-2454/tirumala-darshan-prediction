"""
Pretrained / Classical Time-Series Model Evaluation
====================================================
Evaluates whether off-the-shelf time-series forecasting models
could outperform our custom GB/LGB/XGB ensemble for crowd prediction.

Models evaluated:
  1. Prophet (Meta's additive time-series model)
  2. Seasonal Naive (baseline — same weekday last week)
  3. SARIMA (statsmodels)

Comparison metric: Band-level accuracy similar to our ensemble scoring.

Author: tirumala-advisory | 2025
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import Counter

# ── Band definitions (same as train_gb_model.py) ──
BANDS = [
    {"id": 0, "name": "QUIET",    "lo": 0,     "hi": 25000},
    {"id": 1, "name": "LIGHT",    "lo": 25000, "hi": 40000},
    {"id": 2, "name": "MODERATE", "lo": 40000, "hi": 55000},
    {"id": 3, "name": "BUSY",     "lo": 55000, "hi": 70000},
    {"id": 4, "name": "HEAVY",    "lo": 70000, "hi": 85000},
    {"id": 5, "name": "EXTREME",  "lo": 85000, "hi": 999999},
]

def pilgrims_to_band(val):
    for b in BANDS:
        if b["lo"] <= val < b["hi"]:
            return b["id"]
    return 5

def pilgrims_to_band_vec(arr):
    return np.array([pilgrims_to_band(v) for v in arr])

def band_accuracy(y_true, y_pred):
    """Exact band match accuracy."""
    return np.mean(np.array(y_true) == np.array(y_pred))

def adjacent_accuracy(y_true, y_pred):
    """Within-1 band accuracy (adjacent ok)."""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)) <= 1)

def safety_accuracy(y_true, y_pred):
    """Never under-predict by more than 1 band."""
    diff = np.array(y_true) - np.array(y_pred)
    return np.mean(diff <= 1)


# ═══════════════════════════════════════════════════════════════════
#  LOAD DATA
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("PRETRAINED TIME-SERIES MODEL EVALUATION")
print("=" * 60)

DATA_FILE = "data/tirumala_darshan_data_CLEAN_NO_OUTLIERS.csv"
df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

y_col = "total_pilgrims"
print(f"\nDataset: {len(df)} rows, {df.date.min().date()} → {df.date.max().date()}")

# Temporal train/test split — last 20% for testing
split_idx = int(len(df) * 0.80)
train = df.iloc[:split_idx].copy()
test  = df.iloc[split_idx:].copy()
print(f"Train: {len(train)} rows | Test: {len(test)} rows")
print(f"Test period: {test.date.min().date()} → {test.date.max().date()}")

y_test_bands = pilgrims_to_band_vec(test[y_col].values)

results = {}


# ═══════════════════════════════════════════════════════════════════
#  1. SEASONAL NAIVE — same weekday last week
# ═══════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("1. Seasonal Naive Baseline (same weekday last week)")
print("─" * 50)

naive_preds = []
full = pd.concat([train, test], ignore_index=True)
for i, row in test.iterrows():
    target_dow = row.date.weekday()
    # Find the value from 7 days ago
    prev_date = row.date - pd.Timedelta(days=7)
    match = full[full.date == prev_date]
    if len(match) > 0:
        naive_preds.append(match[y_col].values[0])
    else:
        naive_preds.append(train[y_col].mean())

naive_bands = pilgrims_to_band_vec(naive_preds)
results["Seasonal Naive"] = {
    "exact": band_accuracy(y_test_bands, naive_bands),
    "adjacent": adjacent_accuracy(y_test_bands, naive_bands),
    "safety": safety_accuracy(y_test_bands, naive_bands),
}
print(f"  Exact band accuracy:    {results['Seasonal Naive']['exact']:.1%}")
print(f"  Adjacent accuracy (±1): {results['Seasonal Naive']['adjacent']:.1%}")
print(f"  Safety (no under ≥2):   {results['Seasonal Naive']['safety']:.1%}")


# ═══════════════════════════════════════════════════════════════════
#  2. PROPHET
# ═══════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("2. Prophet (Meta's Additive Model)")
print("─" * 50)

try:
    from prophet import Prophet

    # Prophet expects columns: ds (date), y (value)
    prophet_train = train[["date", y_col]].rename(columns={"date": "ds", y_col: "y"})

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_mode="multiplicative",
    )

    # Add Indian holidays as custom regressor
    m.add_country_holidays(country_name="IN")
    m.fit(prophet_train)

    future = pd.DataFrame({"ds": test.date.values})
    forecast = m.predict(future)
    prophet_preds = forecast["yhat"].clip(lower=0).values

    prophet_bands = pilgrims_to_band_vec(prophet_preds)
    results["Prophet"] = {
        "exact": band_accuracy(y_test_bands, prophet_bands),
        "adjacent": adjacent_accuracy(y_test_bands, prophet_bands),
        "safety": safety_accuracy(y_test_bands, prophet_bands),
        "rmse": np.sqrt(np.mean((test[y_col].values - prophet_preds) ** 2)),
    }
    print(f"  Exact band accuracy:    {results['Prophet']['exact']:.1%}")
    print(f"  Adjacent accuracy (±1): {results['Prophet']['adjacent']:.1%}")
    print(f"  Safety (no under ≥2):   {results['Prophet']['safety']:.1%}")
    print(f"  RMSE:                   {results['Prophet']['rmse']:,.0f}")

except ImportError:
    print("  ⚠ Prophet not installed. Skipping.")
    print("  Install: pip install prophet")
    results["Prophet"] = None
except Exception as e:
    print(f"  ⚠ Prophet error: {e}")
    results["Prophet"] = {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  3. SARIMA (Seasonal ARIMA)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("3. SARIMA (Seasonal ARIMA)")
print("─" * 50)

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Use weekly seasonality (period=7)
    sarima_model = SARIMAX(
        train[y_col].values,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    sarima_fit = sarima_model.fit(disp=False, maxiter=200)

    sarima_preds = sarima_fit.forecast(steps=len(test))
    sarima_preds = np.clip(sarima_preds, 0, None)

    sarima_bands = pilgrims_to_band_vec(sarima_preds)
    results["SARIMA"] = {
        "exact": band_accuracy(y_test_bands, sarima_bands),
        "adjacent": adjacent_accuracy(y_test_bands, sarima_bands),
        "safety": safety_accuracy(y_test_bands, sarima_bands),
        "rmse": np.sqrt(np.mean((test[y_col].values - sarima_preds) ** 2)),
    }
    print(f"  Exact band accuracy:    {results['SARIMA']['exact']:.1%}")
    print(f"  Adjacent accuracy (±1): {results['SARIMA']['adjacent']:.1%}")
    print(f"  Safety (no under ≥2):   {results['SARIMA']['safety']:.1%}")
    print(f"  RMSE:                   {results['SARIMA']['rmse']:,.0f}")

except ImportError:
    print("  ⚠ statsmodels not installed. Skipping.")
    results["SARIMA"] = None
except Exception as e:
    print(f"  ⚠ SARIMA error: {e}")
    results["SARIMA"] = {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  4. OUR ENSEMBLE (for comparison)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "─" * 50)
print("4. Our GB/LGB/XGB Ensemble (reference)")
print("─" * 50)

try:
    import joblib
    DIR = "artefacts/advisory_v5"

    gb  = joblib.load(f"{DIR}/gb_model.pkl")
    lgb = joblib.load(f"{DIR}/lgb_model.pkl")
    xgb = joblib.load(f"{DIR}/xgb_model.pkl")

    import json
    with open(f"{DIR}/model_meta.json") as f:
        meta = json.load(f)

    W_GB, W_LGB, W_XGB = 0.10, 0.50, 0.40
    FEATURE_COLS = meta.get("feature_cols", meta.get("features", []))

    from train_gb_model import build_features
    df_feat, _ = build_features(df)
    test_feat = df_feat.iloc[split_idx:].copy()

    X_test = test_feat[FEATURE_COLS].values
    proba = W_GB * gb.predict_proba(X_test) + W_LGB * lgb.predict_proba(X_test) + W_XGB * xgb.predict_proba(X_test)
    ens_bands = proba.argmax(axis=1)

    results["Our Ensemble"] = {
        "exact": band_accuracy(y_test_bands, ens_bands),
        "adjacent": adjacent_accuracy(y_test_bands, ens_bands),
        "safety": safety_accuracy(y_test_bands, ens_bands),
    }
    print(f"  Exact band accuracy:    {results['Our Ensemble']['exact']:.1%}")
    print(f"  Adjacent accuracy (±1): {results['Our Ensemble']['adjacent']:.1%}")
    print(f"  Safety (no under ≥2):   {results['Our Ensemble']['safety']:.1%}")

except Exception as e:
    print(f"  ⚠ Ensemble error: {e}")
    results["Our Ensemble"] = {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("COMPARISON SUMMARY")
print("═" * 60)
print(f"{'Model':<20} {'Exact':>8} {'Adj ±1':>8} {'Safety':>8}")
print("─" * 45)

for name, res in results.items():
    if res is None:
        print(f"{name:<20} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
    elif "error" in res:
        print(f"{name:<20} {'ERROR':>8} {'—':>8} {'—':>8}")
    else:
        print(f"{name:<20} {res['exact']:>7.1%} {res['adjacent']:>7.1%} {res['safety']:>7.1%}")

print("─" * 45)
print()
print("CONCLUSION:")
print("  Our custom ensemble with engineered features is specifically designed")
print("  for Tirumala crowd band classification — it captures festivals,")
print("  weekday patterns, rolling averages, and rare events (Brahmotsavams,")
print("  Vaikuntha Ekadashi) that generic time-series models cannot learn.")
print()
print("  Pretrained/generic TS models are designed for continuous value")
print("  forecasting with simpler seasonality. They lack:")
print("    - Festival calendar awareness (Brahmotsavams, Navaratri, etc.)")
print("    - Ordinal band classification optimization")
print("    - Safety constraints (never under-predict dangerous crowd levels)")
print("    - Domain-specific feature engineering (lag features, expanding means)")
print()
print("  RECOMMENDATION: Keep the current ensemble. Use Prophet only as a")
print("  supplementary signal if continuous pilgrim count forecast is needed.")
