"""
Feature engineering module  (identical logic to predict_ttd.py §3).

Exports:
    make_features(df)  →  DataFrame with all engineered features
    get_dl_features(df) → DataFrame with DL-specific features
"""
import warnings
import numpy as np
import pandas as pd
import holidays
from datetime import timedelta
import sys, os

# Suppress Pandas DataFrame fragmentation warnings (cosmetic only)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from festival_calendar import get_festival_features_series

# Indian holidays for AP state
_ih = holidays.India(years=range(2013, 2030), state="AP")


def _nearest_hol(date, hols, direction="both"):
    dt = date.date() if hasattr(date, "date") else date
    best = 365
    for h in hols.keys():
        diff = (h - dt).days
        if direction == "forward" and diff < 0:
            continue
        if direction == "backward" and diff > 0:
            continue
        if abs(diff) < best:
            best = abs(diff)
    return best


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build full 137-feature matrix for ML models."""
    df = data.copy()
    d = df["date"]
    tp = df["total_pilgrims"]
    past = tp.shift(1)

    # A. Calendar
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["dow"] = d.dt.dayofweek
    df["doy"] = d.dt.dayofyear
    df["woy"] = d.dt.isocalendar().week.astype(int)
    df["quarter"] = d.dt.quarter
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_month_end"] = d.dt.is_month_end.astype(int)
    df["days_in_month"] = d.dt.days_in_month

    # B. Cyclical
    for n, v, p in [
        ("dow", df["dow"], 7),
        ("month", df["month"], 12),
        ("doy", df["doy"], 365.25),
        ("day", df["day"], 31),
        ("woy", df["woy"], 53),
    ]:
        df[f"sin_{n}"] = np.sin(2 * np.pi * v / p)
        df[f"cos_{n}"] = np.cos(2 * np.pi * v / p)

    # C. Fourier (yearly: 5 harmonics, weekly: 3)
    ty = df["doy"].values / 365.25
    for k in range(1, 6):
        df[f"fy_s{k}"] = np.sin(2 * np.pi * k * ty)
        df[f"fy_c{k}"] = np.cos(2 * np.pi * k * ty)
    tw = df["dow"].values / 7
    for k in range(1, 4):
        df[f"fw_s{k}"] = np.sin(2 * np.pi * k * tw)
        df[f"fw_c{k}"] = np.cos(2 * np.pi * k * tw)

    # D. Holidays
    df["is_hol"] = d.apply(lambda x: 1 if x in _ih else 0)
    df["dist_hol"] = d.apply(lambda x: _nearest_hol(x, _ih))
    df["dist_next_hol"] = d.apply(lambda x: _nearest_hol(x, _ih, "forward"))
    df["dist_prev_hol"] = d.apply(lambda x: _nearest_hol(x, _ih, "backward"))
    for off in [-3, -2, -1, 1, 2, 3]:
        df[f"hwin_{off}"] = d.apply(
            lambda x, o=off: 1 if (x + timedelta(days=o)) in _ih else 0
        )
    df["in_hwin"] = df[[f"hwin_{i}" for i in [-3, -2, -1, 1, 2, 3]]].max(axis=1)
    df["long_wknd"] = (
        (df["is_weekend"] == 1) & ((df["is_hol"] == 1) | (df["in_hwin"] == 1))
    ).astype(int)

    # E. Tirumala domain
    df["is_vaikunta"] = ((df["month"] == 1) & (df["day"] <= 22)).astype(int)
    df["is_brahm"] = ((df["month"].isin([9, 10])) & (df["day"] >= 18)).astype(int)
    df["is_summer"] = df["month"].isin([4, 5]).astype(int)
    df["is_sat"] = (df["dow"] == 5).astype(int)
    df["is_fri"] = (df["dow"] == 4).astype(int)
    df["is_sun"] = (df["dow"] == 6).astype(int)
    df["fest_month"] = df["month"].isin([1, 9, 10, 12]).astype(int)
    epoch = pd.Timestamp("2023-01-21")
    df["lunar"] = ((d - epoch).dt.days % 29.53).round(1)
    df["near_fm"] = ((df["lunar"] >= 13) & (df["lunar"] <= 16)).astype(int)
    df["near_nm"] = ((df["lunar"] <= 2) | (df["lunar"] >= 28)).astype(int)

    # E2. Festival features from extended calendar (2013-2027)
    fest_df = get_festival_features_series(d)
    for col in fest_df.columns:
        df[col] = fest_df[col].values

    # F. Lags
    for lag in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28, 30, 60, 90]:
        df[f"L{lag}"] = tp.shift(lag)

    # G. Rolling stats
    for w in [3, 7, 14, 21, 30, 60]:
        df[f"rm{w}"] = past.rolling(w, min_periods=1).mean()
        df[f"rs{w}"] = past.rolling(w, min_periods=1).std()
        df[f"rn{w}"] = past.rolling(w, min_periods=1).min()
        df[f"rx{w}"] = past.rolling(w, min_periods=1).max()
        df[f"rmd{w}"] = past.rolling(w, min_periods=1).median()
        df[f"rr{w}"] = df[f"rx{w}"] - df[f"rn{w}"]

    # H. EWMA
    for sp in [3, 7, 14, 30, 60]:
        df[f"ew{sp}"] = past.ewm(span=sp, adjust=False).mean()

    # I. Expanding
    df["ex_m"] = past.expanding(1).mean()
    df["ex_s"] = past.expanding(1).std()

    # J. Momentum
    df["m12"] = df["L1"] - df["L2"]
    df["m17"] = df["L1"] - df["L7"]
    df["m114"] = df["L1"] - df["L14"]
    df["m130"] = df["L1"] - df["L30"]
    df["m714"] = df["L7"] - df["L14"]
    df["m730"] = df["L7"] - df["L30"]
    df["pm17"] = df["m17"] / (df["L7"] + 1)
    df["pm130"] = df["m130"] / (df["L30"] + 1)

    # K. Same-DOW
    df["sd1w"] = tp.shift(7)
    df["sd2w"] = tp.shift(14)
    df["sd3w"] = tp.shift(21)
    df["sd4w"] = tp.shift(28)

    # L. Trend
    df["tidx"] = np.arange(len(df))
    df["tr90"] = past.ewm(span=90, adjust=False).mean()
    df["tr180"] = past.ewm(span=180, adjust=False).mean()
    df["dev_tr"] = df["L1"] - df["tr90"]

    # M. Historical averages
    df["dow_av"] = df.groupby("dow")["total_pilgrims"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["mon_av"] = df.groupby("month")["total_pilgrims"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["dm_av"] = df.groupby(["dow", "month"])["total_pilgrims"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # N. Ratios
    df["r1_rm7"] = df["L1"] / (df["rm7"] + 1)
    df["r1_rm30"] = df["L1"] / (df["rm30"] + 1)
    df["r1_dav"] = df["L1"] / (df["dow_av"] + 1)

    # O. Extreme-aware features (signal for predicting tails)
    _p_hi = (past > 80000).astype(int)
    _p_lo = (past < 60000).astype(int)
    for w in [7, 14, 30]:
        df[f"hi80_{w}"] = _p_hi.rolling(w, min_periods=1).sum()
        df[f"lo60_{w}"] = _p_lo.rolling(w, min_periods=1).sum()
    df["hi80_f30"] = _p_hi.rolling(30, min_periods=1).mean()
    df["lo60_f30"] = _p_lo.rolling(30, min_periods=1).mean()
    df["summer_wknd"] = df["is_summer"] * df["is_weekend"]
    df["peak_sun"] = df["month"].isin([5, 6, 7]).astype(int) * df["is_sun"]
    _is_fest = df.get("is_festival", pd.Series(0, index=df.index)).astype(int)
    df["fest_wknd"] = _is_fest * df["is_weekend"]
    df["l1_dm_dev"] = (df["L1"] - df["dm_av"]) / (df["dm_av"] + 1)
    df["sd1w_dm_dev"] = (df["sd1w"] - df["dm_av"]) / (df["dm_av"] + 1)
    df["rr7_norm"] = df["rr7"] / (df["rm7"] + 1)
    df["rr30_norm"] = df["rr30"] / (df["rm30"] + 1)
    df["yoy_364"] = tp.shift(364)
    df["yoy_dev"] = (df["yoy_364"] - df["dm_av"]) / (df["dm_av"] + 1)

    return df


def get_dl_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build DL-specific features (lags + rolling + cyclical + domain)."""
    df = raw.copy()
    tp = df["total_pilgrims"]

    # Lag features
    for lag in [1, 2, 3, 5, 7, 14, 21, 28]:
        df[f"lag_{lag}"] = tp.shift(lag)

    # Rolling stats
    past = tp.shift(1)
    for w in [3, 7, 14, 21, 28]:
        df[f"rm_{w}"] = past.rolling(w).mean()
        df[f"rs_{w}"] = past.rolling(w).std()

    # EWMA
    for sp in [7, 14, 28]:
        df[f"ew_{sp}"] = past.ewm(span=sp).mean()

    # Momentum
    df["momentum_7_14"] = df["lag_7"] - df["lag_14"]
    df["momentum_7_28"] = df["lag_7"] - df["lag_28"]

    # Calendar (normalised)
    df["dow"] = df.date.dt.dayofweek / 6.0
    df["month_n"] = (df.date.dt.month - 1) / 11.0
    df["is_weekend"] = (df.date.dt.dayofweek >= 5).astype(float)
    df["sin_dow"] = np.sin(2 * np.pi * df.date.dt.dayofweek / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df.date.dt.dayofweek / 7)
    df["sin_month"] = np.sin(2 * np.pi * df.date.dt.month / 12)
    df["cos_month"] = np.cos(2 * np.pi * df.date.dt.month / 12)
    df["is_hol"] = df.date.apply(lambda d: d in _ih).astype(float)
    df["lunar"] = np.sin(2 * np.pi * df.date.dt.day / 29.53)

    # Festival features from extended calendar (2013-2027)
    fest_df = get_festival_features_series(df["date"])
    for col in fest_df.columns:
        df[col] = fest_df[col].values.astype(float)

    # Yearly Fourier (3 harmonics)
    t_idx = np.arange(len(df))
    for k in range(1, 4):
        df[f"f_y_sin_{k}"] = np.sin(2 * np.pi * k * t_idx / 365.25)
        df[f"f_y_cos_{k}"] = np.cos(2 * np.pi * k * t_idx / 365.25)

    # Extreme-aware features (DL)
    _past_dl = tp.shift(1)
    _ph = (_past_dl > 80000).astype(float)
    _pl = (_past_dl < 60000).astype(float)
    for w in [7, 30]:
        df[f"hi80_f{w}"] = _ph.rolling(w, min_periods=1).mean()
        df[f"lo60_f{w}"] = _pl.rolling(w, min_periods=1).mean()
    df["summer_wknd"] = (df.date.dt.month.isin([4, 5])).astype(float) * (df.date.dt.dayofweek >= 5).astype(float)
    df["peak_sun"] = (df.date.dt.month.isin([5, 6, 7])).astype(float) * (df.date.dt.dayofweek == 6).astype(float)

    df = df.dropna().reset_index(drop=True)
    return df
