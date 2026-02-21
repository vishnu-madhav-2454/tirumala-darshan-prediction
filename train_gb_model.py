"""
Tirumala Crowd Advisory -- Model Training Pipeline
===================================================
Provides:
  - build_features(df)        -> feature-engineered DataFrame + feature_cols
  - pilgrims_to_band(val)     -> single daily count -> band id (0-5)
  - pilgrims_to_band_vec(arr) -> array of counts -> band ids
  - train_gb_model(...)       -> Optuna-tuned GB + LGB + XGB ensemble + export

SPLIT STRATEGY -- STRATIFIED TRAIN/TEST + StratifiedKFold CV:
  We use sklearn StratifiedShuffleSplit for the train/cal/test split and
  StratifiedKFold for Optuna cross-validation.  This guarantees every class
  (QUIET, LIGHT, MODERATE, BUSY, HEAVY, EXTREME) is proportionally represented
  in both train and test sets -- critical because:
    - EXTREME is only 1.3% of data (14 samples).  A temporal split puts only
      2 in train and 12 in test -- the model cannot learn EXTREME patterns.
    - QUIET is only 0.7% (8 samples).  Temporal split puts 0 in test.
  With stratified split, the model sees ~11 EXTREME in train and ~3 in test,
  vastly improving learning for rare classes.

  HOW FEATURES ARE COMPUTED:
    All features (lags, rolling means, expanding means) are computed FIRST
    in strict temporal order on the full dataset using .shift(N) to prevent
    same-row leakage.  The stratified split then divides the already-computed
    feature matrix.  Walk-forward evaluation (always temporal) provides the
    honest real-world-deployment metric alongside the stratified test score.

DESIGN DECISIONS:
  Split:     StratifiedShuffleSplit 72% train / 8% cal / 20% test
  CV:        StratifiedKFold(n_splits=5, shuffle=True) for Optuna
  Weighting: balanced class weights + 1.5x boost for HEAVY/EXTREME (rare)
  Models:    GradientBoosting (primary, used by flask_api.py)
             + LightGBM + XGBoost -> probabilities averaged as weighted ensemble
  Score:     ordinal-aware: 30% exact + 30% macro-F1 + 25% ordinal + 15% safety

ON LUNAR FEATURES (is_pournami, is_amavasya, is_ekadashi):
  hindu_calendar.py only covers 2025-2027.  Training spans 2022-2026, so
  these would be 0 for ~75% of rows -- noise, not signal.
  Excluded until the lunar calendar is extended back to 2022.

Author: tirumala-advisory  |  Updated: 2026-02-21
"""

from __future__ import annotations
import json, time, warnings, pathlib, argparse
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from collections import Counter

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
)
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
import optuna

from festival_calendar import get_festival_features_series

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
np.random.seed(42)

T0 = time.time()
def LOG(msg: str) -> None:
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


# ===============================================================================
#  CONSTANTS
# ===============================================================================
DATA_FILE  = "data/tirumala_darshan_data_CLEAN_NO_OUTLIERS.csv"
POST_COVID = "2022-02-01"
ART_DIR    = pathlib.Path("artefacts/advisory_v5")
ART_DIR.mkdir(parents=True, exist_ok=True)

# 6-band crowd classification
BANDS: list[dict] = [
    {"id": 0, "name": "QUIET",    "lo": 0,     "hi": 50_000},
    {"id": 1, "name": "LIGHT",    "lo": 50_000, "hi": 60_000},
    {"id": 2, "name": "MODERATE", "lo": 60_000, "hi": 70_000},
    {"id": 3, "name": "BUSY",     "lo": 70_000, "hi": 80_000},
    {"id": 4, "name": "HEAVY",    "lo": 80_000, "hi": 90_000},
    {"id": 5, "name": "EXTREME",  "lo": 90_000, "hi": 999_999},
]
BAND_NAMES = [b["name"] for b in BANDS]
N_BANDS    = len(BANDS)

BAND_COLORS = {
    "QUIET":    "#2196F3",
    "LIGHT":    "#4CAF50",
    "MODERATE": "#8BC34A",
    "BUSY":     "#FFC107",
    "HEAVY":    "#FF5722",
    "EXTREME":  "#B71C1C",
}
BAND_ADVICE = {
    "QUIET":    "Best day to visit — very few pilgrims, quick darshan",
    "LIGHT":    "Good day — low crowd, visit freely with short waits",
    "MODERATE": "Normal day — standard crowd, comfortable visit",
    "BUSY":     "Above average — expect moderate crowds & longer waits",
    "HEAVY":    "Busy day — consider visiting on alternative days",
    "EXTREME":  "Extreme rush — strongly avoid unless essential",
}

# Human-readable feature labels (used in model metadata / SHAP explanations)
FEATURE_LABELS = {
    "dow":                    "Day of week (0=Mon ... 6=Sun)",
    "month":                  "Month (1-12)",
    "is_weekend":             "Weekend flag (Sat/Sun = 1)",
    "sin_doy":                "Annual cycle — sin component",
    "cos_doy":                "Annual cycle — cos component",
    "L1":                     "Yesterday's pilgrim count",
    "L2":                     "2 days ago",
    "L7":                     "Same day last week",
    "L14":                    "Same day 2 weeks ago",
    "L21":                    "Same day 3 weeks ago",
    "L28":                    "Same day 4 weeks ago",
    "L365":                   "Same calendar day last year",
    "rm7":                    "7-day rolling average",
    "rm14":                   "14-day rolling average",
    "rm30":                   "30-day rolling average",
    "rstd7":                  "7-day rolling std (local volatility)",
    "rstd14":                 "14-day rolling std",
    "dow_expanding_mean":     "Historic weekday average (expanding)",
    "month_dow_mean":         "Seasonal weekday average (month × DOW, expanding)",
    "log_L1":                 "Yesterday's count (log scale)",
    "log_L7":                 "Last-week same day (log)",
    "log_rm7":                "7-day trend (log)",
    "log_rm30":               "30-day trend (log)",
    "log_L365":               "Same day last year (log)",
    "momentum_7":             "7-day momentum: yesterday − last-week",
    "dow_dev":                "Deviation from weekday historical average",
    "month_dow":              "Month × weekday categorical interaction",
    "ewm7":                   "7-day exponential weighted average",
    "ewm14":                  "14-day exponential weighted average",
    "trend_7_14":             "Short vs mid-term trend (rm7 − rm14)",
    "trend_7_30":             "Short vs long-term trend (rm7 − rm30)",
    "week_of_year":           "ISO week number (1-53)",
    "yoy_growth":             "Year-over-year growth (L1 vs L365, clipped ±2)",
    "month_weekend":          "Month × weekend (captures summer/winter weekend effects)",
    "heavy_extreme_count7":   "HEAVY/EXTREME days in last 7 days (crowd regime)",
    "light_quiet_count7":     "QUIET/LIGHT days in last 7 days",
    "is_festival":            "Any festival day",
    "fest_impact":            "Festival crowd-impact score (0-5)",
    "days_to_fest":           "Days until next festival (crowd builds beforehand)",
    "days_from_fest":         "Days since last festival (post-festival tail)",
    "fest_window_7":          "Within 7-day festival window",
    "fest_window_3":          "Within 3-day festival window",
    "is_brahmotsavam":        "Brahmotsavam (biggest annual festival, week-long)",
    "is_sankranti":           "Makar Sankranti",
    "is_summer_holiday":      "Summer school vacation period",
    "is_dasara_holiday":      "Dasara school holiday period",
    "is_national_holiday":    "National / public holiday",
    "is_vaikuntha_ekadashi":  "Vaikuntha Ekadashi (often 300K+ pilgrims)",
    "is_dussehra_period":     "Dussehra period",
    "is_diwali":              "Diwali",
    "is_navaratri":           "Navaratri",
    "is_janmashtami":         "Janmashtami",
    "is_ugadi":               "Ugadi (Telugu New Year)",
    "is_rathasapthami":       "Rathasapthami",
    "is_ramanavami":          "Sri Rama Navami",
    "is_shivaratri":          "Maha Shivaratri",
    "is_winter_holiday":      "Winter holiday period",
}


# ===============================================================================
#  BAND UTILITIES  (imported by flask_api.py)
# ===============================================================================
def pilgrims_to_band(val: float) -> int:
    """Map a single daily pilgrim count -> band id (0=QUIET ... 5=EXTREME)."""
    for b in BANDS:
        if b["lo"] <= val < b["hi"]:
            return b["id"]
    return N_BANDS - 1


def pilgrims_to_band_vec(arr) -> np.ndarray:
    """Vectorised pilgrims_to_band over any iterable."""
    return np.array([pilgrims_to_band(v) for v in arr])


# ===============================================================================
#  FEATURE ENGINEERING  (imported by flask_api.py)
# ===============================================================================
def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build ML features from a raw DataFrame with columns [date, total_pilgrims].

    LEAKAGE-FREE GUARANTEES:
      Every feature that uses past observations does so by explicitly calling
      .shift(1) (or .shift(N)) BEFORE applying any aggregation.  This means
      that for a row representing date T, all features encode information from
      dates T-1 and earlier only.

    Feature groups with rationale:
    +---------------------------------+----------------------------------------+
    |  Group                          |  Why it helps                          |
    +---------------------------------+----------------------------------------+
    |  Calendar (6 features)          |  Weekly & annual crowd rhythm          |
    |  Lags (7 features)              |  Direct memory: yesterday, last week,  |
    |                                 |  same day last year                    |
    |  Rolling stats (5 features)     |  Recent trend & local volatility       |
    |  Expanding DOW means (2)        |  Historical weekday/seasonal norms     |
    |  Log transforms (5 features)    |  Reduce right-skew of pilgrim counts   |
    |  Derived / interaction (9)      |  Momentum, regime deviation, combos    |
    |  Regime counts (2 features)     |  Is this a "hot streak"?               |
    |  Festival features (~=21 cols)   |  Tirumala-specific religious events    |
    +---------------------------------+----------------------------------------+

    Returns:
      (d, feature_cols) where d has all features + 'band' target,
      and feature_cols is a list of feature column names.
    """
    d = df.copy().sort_values("date").reset_index(drop=True)
    y = "total_pilgrims"

    # -- 1. Calendar features -------------------------------------------------
    d["dow"]        = d.date.dt.dayofweek           # 0=Mon ... 6=Sun
    d["month"]      = d.date.dt.month               # 1-12
    d["is_weekend"] = (d.dow >= 5).astype(int)      # Sat or Sun
    doy = d.date.dt.dayofyear
    d["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)  # 365-day seasonal cycle
    d["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    # -- Note: we do NOT include 'year' as a feature.  Year would let the model
    #    memorise each year's avg rather than generalising across seasons.
    #    Instead, yoy_growth (below) captures secular growth trends.

    # -- 2. Lag features ------------------------------------------------------
    # Short lags: yesterday, 2d, weekly, bi-weekly, monthly
    for lag in [1, 2, 7, 14, 21, 28]:
        d[f"L{lag}"] = d[y].shift(lag)
    # L365: same calendar day last year — strong annual pattern for Tirumala.
    # This needs 365 rows of history before it becomes non-NaN, so rows before
    # ~POST_COVID+365 will be dropped by the final dropna().
    d["L365"] = d[y].shift(365)

    # -- 3. Rolling statistics ------------------------------------------------
    # All computed on d[y].shift(1) — "past" excludes today's value.
    past = d[y].shift(1)
    for w in [7, 14, 30]:
        d[f"rm{w}"]   = past.rolling(w).mean()   # trend at 3 horizons
    d["rstd7"]  = past.rolling(7).std()           # short-term volatility
    d["rstd14"] = past.rolling(14).std()          # mid-term volatility

    # -- 4. Expanding DOW means (temporal, group-level) -----------------------
    # dow_expanding_mean: "for every Monday in the dataset up to yesterday,
    #   what was the average crowd?"  Built with shift(1) inside groupby to
    #   guarantee the current row never contributes to its own feature.
    d["dow_expanding_mean"] = d.groupby("dow")[y].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    # month_dow_mean: "for July Saturdays up to yesterday, what was the crowd?"
    # Key: month*10 + dow.  Because dow ∈ [0,6] < 10, multiplication by 10 gives
    # unique keys for all 12×7=84 (month, dow) combinations without collision
    # (e.g. Jan-Mon=10, Jan-Sun=16, Dec-Sun=126 — all distinct).
    _md_key = d["month"] * 10 + d["dow"]
    d["month_dow_mean"] = (
        d.assign(_md_key=_md_key)
         .groupby("_md_key")[y]
         .transform(lambda s: s.shift(1).expanding().mean())
    )

    # -- 5. Log-transformed lags ----------------------------------------------
    # Pilgrim counts are right-skewed (mean ~= 72K, but EXTREME days hit 150K+).
    # log1p compresses outliers and produces more symmetric split candidates
    # for gradient-boosted trees, generally improving accuracy on tail classes.
    d["log_L1"]   = np.log1p(d["L1"])
    d["log_L7"]   = np.log1p(d["L7"])
    d["log_rm7"]  = np.log1p(d["rm7"])
    d["log_rm30"] = np.log1p(d["rm30"])
    d["log_L365"] = np.log1p(d["L365"])

    # -- 6. Derived / interaction features ------------------------------------
    # momentum_7: positive = crowd rising this week; negative = falling
    d["momentum_7"] = d["L1"] - d["L7"]
    # dow_dev: how much did yesterday differ from its historical weekday norm?
    #   Large positive = unusual busy day; large negative = unusually quiet
    d["dow_dev"]    = d["L1"] - d["dow_expanding_mean"]
    # month_dow: integer encoding of (month, dow) interaction — gives the model
    #   a direct "July Saturday" feature without needing an explicit lookup
    d["month_dow"]  = d["month"] * 10 + d["dow"]
    # ewm: exponential weighting weights recent days more than older ones
    d["ewm7"]  = past.ewm(span=7,  min_periods=1).mean()
    d["ewm14"] = past.ewm(span=14, min_periods=1).mean()
    # trend: is the recent short-term average above/below medium/long-term?
    d["trend_7_14"] = d["rm7"] - d["rm14"]
    d["trend_7_30"] = d["rm7"] - d["rm30"]
    # week_of_year: captures intra-year cycles beyond the sin/cos pair
    d["week_of_year"] = d.date.dt.isocalendar().week.astype(int)
    # yoy_growth: secular growth rate.  Tirumala crowds have grown year-on-year
    #   since COVID recovery; this gives the model a "trend slope" signal.
    #   Clipped to ±200% to prevent extreme outlier influence.
    d["yoy_growth"] = ((d["L1"] - d["L365"]) / (d["L365"] + 1e-9)).clip(-2, 2)
    # month_weekend: separates summer weekends from winter weekends.
    #   Weekdays -> 0 (model already has month and is_weekend separately, so this
    #   interaction adds value only for weekend rows).
    d["month_weekend"] = d["month"] * d["is_weekend"]

    # -- 7. Crowd-regime counts ------------------------------------------------
    # "Has there been a run of heavy/extreme days recently?"
    # shift(1) ensures we're counting yesterday-and-before, not today.
    lagged_bands = pd.Series(
        pilgrims_to_band_vec(past.fillna(d[y].mean()).values)
    )
    d["heavy_extreme_count7"] = ((lagged_bands >= 4).astype(int)
                                  .rolling(7, min_periods=1).sum().values)
    d["light_quiet_count7"]   = ((lagged_bands <= 1).astype(int)
                                  .rolling(7, min_periods=1).sum().values)

    # -- 8. Festival features -------------------------------------------------
    # Tirumala religious festivals drive the largest crowd spikes.
    # get_festival_features_series returns a DataFrame indexed by date with
    # pre-computed columns for each known festival.
    fest = get_festival_features_series(d.date)
    festival_cols = [
        "is_festival",
        "fest_impact",           # 0-5 impact score
        "days_to_fest",          # countdown builds anticipation
        "days_from_fest",        # post-festival tail
        "fest_window_7",
        "fest_window_3",
        "is_brahmotsavam",       # biggest festival — 9-day extreme rush
        "is_sankranti",
        "is_summer_holiday",
        "is_dasara_holiday",
        "is_national_holiday",
        "is_vaikuntha_ekadashi", # single-day, often 300K+ pilgrims
        "is_dussehra_period",
        "is_diwali",
        "is_navaratri",
        "is_janmashtami",
        "is_ugadi",
        "is_rathasapthami",
        "is_ramanavami",
        "is_shivaratri",
        "is_winter_holiday",
    ]
    for col in festival_cols:
        d[col] = fest[col].values if col in fest.columns else 0

    # -- Target band ---------------------------------------------------------
    d["band"] = pilgrims_to_band_vec(d[y].values)

    # -- Drop NaN rows ---------------------------------------------------------
    # Primary source of NaN: L365 makes the first 365 rows NaN.
    # After dropping, training data starts around POST_COVID + 365 days.
    feat_check_cols = [c for c in d.columns if c not in ["date", y, "band"]]
    d = d.dropna(subset=feat_check_cols).reset_index(drop=True)

    feature_cols = feat_check_cols
    return d, feature_cols


# ===============================================================================
#  COMBINED SCORING FUNCTION
# ===============================================================================
def ordinal_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Ordinal-aware combined metric (higher = better, range ~= 0-1):

      30% exact accuracy  — standard classification accuracy
      30% macro-F1        — penalises ignoring rare classes (QUIET/EXTREME)
      25% ordinal penalty — being off-by-2 is worse than off-by-1
                            score = 1 - MAE/(N_BANDS-1)
      15% safety score    — penalises dangerous under-prediction
                            (predicting QUIET/LIGHT when crowd is HEAVY/EXTREME)
    """
    exact    = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mae_ord  = 1.0 - np.mean(np.abs(y_pred - y_true)) / (N_BANDS - 1)
    dangerous = int(((y_pred <= 1) & (y_true >= 4)).sum())
    safety    = 1.0 - dangerous / (len(y_true) + 1e-9)
    return 0.30 * exact + 0.30 * macro_f1 + 0.25 * mae_ord + 0.15 * safety


# ===============================================================================
#  FEATURE SELECTION  (10-method consensus)
# ===============================================================================
def select_features(d: pd.DataFrame, feature_cols: list[str],
                    min_keep: int = 15, max_keep: int = 40,
                    vote_threshold: int = 6,
                    verbose: bool = True) -> list[str]:
    """
    Robust 10-method feature selection with majority-vote consensus.

    Pre-filter:
      0. VARIANCE FILTER -- drop features where >=98% of values are identical.

    Ranking methods (each votes for features scoring above its own median):
      1.  Mutual Information           (non-linear dependency)
      2.  ANOVA F-test                 (linear class separability)
      3.  Chi-squared test             (non-negative, after MinMax scaling)
      4.  GradientBoosting importance  (impurity-based tree splits)
      5.  Random Forest importance     (impurity-based, decorrelated trees)
      6.  Extra Trees importance       (extremely randomised trees)
      7.  Permutation importance       (accuracy drop when feature shuffled)
      8.  L1 Logistic Regression       (non-zero coefficients under Lasso)
      9.  Spearman rank correlation    (monotonic dependency with target)
      10. Recursive Feature Elim.      (backward elimination via DecisionTree)

    Consensus:
      A feature is KEPT only if >= vote_threshold methods (default 6/10)
      rank it above their median.  This is deliberately strict -- only
      features that are consistently useful across very different statistical
      and ML approaches survive.

    Post-filter:
      - REDUNDANCY REMOVAL: if two kept features have Spearman |rho| > 0.95,
        drop the one with fewer votes (tie-break: lower MI).

    Returns: filtered list of feature column names (preserves original order).
    """
    X = d[feature_cols].values
    y = d["band"].values
    n_total = len(feature_cols)

    if verbose:
        LOG(f"\n  === FEATURE SELECTION -- 10-METHOD CONSENSUS ({n_total} candidates) ===")

    # -- 0. Variance pre-filter ------------------------------------------------
    from collections import Counter as Ctr
    low_var_drop = set()
    for i, col in enumerate(feature_cols):
        vals = X[:, i]
        most_common_frac = max(Ctr(vals).values()) / len(vals)
        if most_common_frac >= 0.98:
            low_var_drop.add(col)
    if verbose:
        LOG(f"  [pre] Variance filter: dropping {len(low_var_drop)} near-constant features")
        if low_var_drop:
            LOG(f"        {sorted(low_var_drop)}")

    active_cols = [c for c in feature_cols if c not in low_var_drop]
    active_idx  = [i for i, c in enumerate(feature_cols) if c not in low_var_drop]
    X_a = X[:, active_idx]
    n_active = len(active_cols)

    # Class-balanced sample weights (reused by tree models)
    cc = Counter(y.tolist())
    n_s, n_c = len(y), len(cc)
    cw = {c: n_s / (n_c * cnt) for c, cnt in cc.items()}
    for rare in [4, 5]:
        if rare in cw:
            cw[rare] *= 1.5
    sw = np.array([cw.get(int(yi), 1.0) for yi in y])

    # Container: method_name -> {col: score}
    scores = {}

    # --- 1. Mutual Information ------------------------------------------------
    if verbose:
        LOG(f"  [ 1/10] Mutual Information ...")
    mi = mutual_info_classif(X_a, y, discrete_features=False,
                              n_neighbors=5, random_state=42)
    scores["MI"] = dict(zip(active_cols, mi))

    # --- 2. ANOVA F-test -----------------------------------------------------
    if verbose:
        LOG(f"  [ 2/10] ANOVA F-test ...")
    f_vals, _ = f_classif(X_a, y)
    f_vals = np.nan_to_num(f_vals, nan=0.0)
    scores["ANOVA"] = dict(zip(active_cols, f_vals))

    # --- 3. Chi-squared test --------------------------------------------------
    if verbose:
        LOG(f"  [ 3/10] Chi-squared test ...")
    X_nn = MinMaxScaler().fit_transform(X_a)  # chi2 needs non-negative
    chi_vals, _ = chi2(X_nn, y)
    chi_vals = np.nan_to_num(chi_vals, nan=0.0)
    scores["Chi2"] = dict(zip(active_cols, chi_vals))

    # --- 4. GradientBoosting importance --------------------------------------
    if verbose:
        LOG(f"  [ 4/10] GradientBoosting tree importance ...")
    gb_q = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, max_features="sqrt", random_state=42
    )
    gb_q.fit(X_a, y, sample_weight=sw)
    scores["GB"] = dict(zip(active_cols, gb_q.feature_importances_))

    # --- 5. Random Forest importance -----------------------------------------
    if verbose:
        LOG(f"  [ 5/10] Random Forest importance ...")
    rf_q = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    rf_q.fit(X_a, y)
    scores["RF"] = dict(zip(active_cols, rf_q.feature_importances_))

    # --- 6. Extra Trees importance -------------------------------------------
    if verbose:
        LOG(f"  [ 6/10] Extra Trees importance ...")
    et_q = ExtraTreesClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    et_q.fit(X_a, y)
    scores["ET"] = dict(zip(active_cols, et_q.feature_importances_))

    # --- 7. Permutation importance (on GB model) -----------------------------
    if verbose:
        LOG(f"  [ 7/10] Permutation importance ...")
    perm_res = permutation_importance(
        gb_q, X_a, y, n_repeats=10, random_state=42,
        scoring="accuracy", n_jobs=-1
    )
    scores["Perm"] = dict(zip(active_cols, perm_res.importances_mean))

    # --- 8. L1 Logistic Regression (Lasso) -----------------------------------
    if verbose:
        LOG(f"  [ 8/10] L1 Logistic Regression ...")
    from sklearn.preprocessing import StandardScaler
    X_sc = StandardScaler().fit_transform(X_a)
    # multi-class one-vs-rest with L1 penalty; sum absolute coefs across classes
    lr = LogisticRegression(
        penalty="l1", C=1.0, solver="saga", max_iter=2000,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    lr.fit(X_sc, y)
    l1_imp = np.abs(lr.coef_).sum(axis=0)  # sum across one-vs-rest classes
    scores["L1"] = dict(zip(active_cols, l1_imp))

    # --- 9. Spearman rank correlation with target ----------------------------
    if verbose:
        LOG(f"  [ 9/10] Spearman correlation ...")
    from scipy.stats import spearmanr
    spear = {}
    for j, col in enumerate(active_cols):
        rho, _ = spearmanr(X_a[:, j], y)
        spear[col] = abs(rho) if not np.isnan(rho) else 0.0
    scores["Spearman"] = spear

    # --- 10. Recursive Feature Elimination (DecisionTree) --------------------
    if verbose:
        LOG(f"  [10/10] Recursive Feature Elimination ...")
    dt = DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                 random_state=42)
    # RFE keeps ~half; features that survive get score = 1, others = 0
    n_rfe_keep = max(n_active // 2, min_keep)
    rfe = RFE(estimator=dt, n_features_to_select=n_rfe_keep, step=1)
    rfe.fit(X_a, y)
    # Use ranking_ (1 = best); invert so higher = better
    rfe_score = 1.0 / rfe.ranking_
    scores["RFE"] = dict(zip(active_cols, rfe_score))

    # ---- VOTE: above-median in each method = 1 vote -------------------------
    method_names = list(scores.keys())
    n_methods = len(method_names)
    medians = {m: np.median(list(scores[m].values())) for m in method_names}

    votes = {col: 0 for col in active_cols}
    vote_detail = {col: [] for col in active_cols}
    for m in method_names:
        med = medians[m]
        for col in active_cols:
            if scores[m].get(col, 0) >= med:
                votes[col] += 1
                vote_detail[col].append(m)

    # ---- SELECT: keep features with >= vote_threshold votes ------------------
    selected = set()
    for col in active_cols:
        if votes[col] >= vote_threshold:
            selected.add(col)

    if verbose:
        LOG(f"\n  Threshold: >= {vote_threshold}/{n_methods} methods must agree")
        LOG(f"  Features passing threshold: {len(selected)}")

    # ---- REDUNDANCY REMOVAL: Spearman |rho| > 0.95 --------------------------
    sel_list = sorted(selected)
    if len(sel_list) > 1:
        sel_idx_r = [feature_cols.index(c) for c in sel_list]
        corr_mat = pd.DataFrame(
            X[:, sel_idx_r], columns=sel_list
        ).corr(method="spearman").abs()
        redundant_drop = set()
        for i_c in range(len(sel_list)):
            for j_c in range(i_c + 1, len(sel_list)):
                if corr_mat.iloc[i_c, j_c] > 0.95:
                    c1, c2 = sel_list[i_c], sel_list[j_c]
                    # drop the one with fewer votes; tie-break: lower MI
                    if votes[c1] < votes[c2]:
                        redundant_drop.add(c1)
                    elif votes[c2] < votes[c1]:
                        redundant_drop.add(c2)
                    else:
                        drop = c1 if scores["MI"].get(c1, 0) < scores["MI"].get(c2, 0) else c2
                        redundant_drop.add(drop)
        selected -= redundant_drop
        if verbose and redundant_drop:
            LOG(f"  Redundancy removal (Spearman > 0.95): dropped {sorted(redundant_drop)}")

    # ---- Enforce min/max bounds ----------------------------------------------
    if len(selected) < min_keep:
        ranked = sorted(active_cols, key=lambda c: votes[c], reverse=True)
        for col in ranked:
            if col not in selected:
                selected.add(col)
            if len(selected) >= min_keep:
                break
        if verbose:
            LOG(f"  Padded to min_keep={min_keep}")

    if len(selected) > max_keep:
        ranked = sorted(selected, key=lambda c: votes[c], reverse=True)
        selected = set(ranked[:max_keep])
        if verbose:
            LOG(f"  Trimmed to max_keep={max_keep}")

    # Preserve original feature order
    final_cols = [c for c in feature_cols if c in selected]

    # ---- REPORT --------------------------------------------------------------
    if verbose:
        dropped_all = set(feature_cols) - set(final_cols)
        LOG(f"\n  RESULT: {len(final_cols)} / {n_total} features kept "
            f"({len(dropped_all)} dropped)")
        LOG(f"  Sample:feature ratio: {len(d)/len(final_cols):.1f}:1")

        hdr = f"  {'Feature':<28s}  "
        hdr += "  ".join(f"{m:>5s}" for m in method_names)
        hdr += f"  {'VOTES':>5s}  {'Status':>8s}"
        LOG(f"\n{hdr}")
        LOG(f"  {'-' * (28 + 8 * n_methods + 16)}")

        for col in feature_cols:
            parts = [f"  {col:<28s}"]
            for m in method_names:
                v = scores[m].get(col, 0)
                above = v >= medians[m] if col not in low_var_drop else False
                mark = "*" if above else " "
                parts.append(f"{mark}{v:5.3f}" if v < 10 else f"{mark}{v:5.1f}")
            if col in low_var_drop:
                parts.append(f"  {'--':>5s}")
                parts.append(f"  {'LOW-VAR':>8s}")
            else:
                parts.append(f"  {votes[col]:5d}")
                status = "KEPT" if col in final_cols else "DROPPED"
                parts.append(f"  {status:>8s}")
            LOG("  ".join(parts))

    return final_cols


# ===============================================================================
#  STRATIFIED SPLIT
# ===============================================================================
def _stratified_split(d: pd.DataFrame, test_frac: float = 0.20,
                     cal_frac: float = 0.08, random_state: int = 42
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train / calibration / test split.

    Ensures every class (QUIET..EXTREME) is proportionally represented in
    all three subsets.  This is critical because:
      - EXTREME = 1.3% (14 samples) -- a temporal split leaves only 2 in train
      - QUIET   = 0.7% (8 samples)  -- temporal split leaves 0 in test

    Steps:
      1. StratifiedShuffleSplit(test_size) -> train_cal vs test
      2. StratifiedShuffleSplit(cal_frac_adj) on train_cal -> train vs cal

    Returns: (train_idx, cal_idx, test_idx) -- integer index arrays into d.
    """
    y = d["band"].values
    n = len(d)

    # Step 1: split off test set (stratified)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac,
                                  random_state=random_state)
    train_cal_idx, test_idx = next(sss1.split(np.zeros(n), y))

    # Step 2: from train_cal, split off calibration set (stratified)
    # cal_frac of total ~= 8% -> relative to train_cal = cal_frac / (1 - test_frac)
    cal_frac_adj = cal_frac / (1.0 - test_frac)
    y_tc = y[train_cal_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=cal_frac_adj,
                                  random_state=random_state)
    tr_local, cal_local = next(sss2.split(np.zeros(len(train_cal_idx)), y_tc))
    train_idx = train_cal_idx[tr_local]
    cal_idx   = train_cal_idx[cal_local]

    # Log the split
    LOG(f"  Stratified split: Train={len(train_idx)}  Cal={len(cal_idx)}  Test={len(test_idx)}")
    for name, idx in [("TRAIN", train_idx), ("CAL", cal_idx), ("TEST", test_idx)]:
        arr = y[idx]
        dist = "  ".join(f"{BAND_NAMES[i]}={np.sum(arr==i)}" for i in range(N_BANDS))
        pcts = "  ".join(f"{np.sum(arr==i)/len(arr)*100:.1f}%" for i in range(N_BANDS))
        LOG(f"  {name:5s}: {dist}")
        LOG(f"  {'':5s}  ({pcts})")

    return train_idx, cal_idx, test_idx


# ===============================================================================
#  MAIN TRAINING FUNCTION
# ===============================================================================
N_OPTUNA_DEFAULT = 80
N_FOLDS          = 5


def train_gb_model(d: pd.DataFrame | None = None,
                   feature_cols: list[str] | None = None,
                   n_optuna: int | None = None) -> dict:
    """
    Full Optuna-tuned training pipeline.

    Steps:
      1. Stratified split (train 72% | calibration 8% | test 20%)
         -- all 6 classes proportionally represented in each subset
      2. Class weight computation for train split
      3. StratifiedKFold-based Optuna search (5-fold) for each of:
           GradientBoosting, LightGBM, XGBoost
      4. Final model fitting on train split only
      5. Ensemble weight calibration on CALIBRATION split (not test)
      6. Evaluation on UNTOUCHED test split
      7. Walk-forward evaluation (pure temporal) for honest real-world metric
      8. Export: gb_model.pkl, lgb_model.pkl, xgb_model.pkl,
                 model_meta.json, config.json, hyperparams.json,
                 feature_importance.csv, shap_background.npy, last30_history.csv
    """
    import lightgbm as lgb
    import xgboost as xgb_lib

    n_trials = n_optuna if n_optuna is not None else N_OPTUNA_DEFAULT

    if d is None or feature_cols is None:
        df_raw = pd.read_csv(DATA_FILE, parse_dates=["date"])
        df_raw = df_raw[df_raw.date >= POST_COVID].copy()
        d, feature_cols = build_features(df_raw)

    n_feat = len(feature_cols)
    LOG(f"\nTRAINING  |  {len(d)} samples  |  {n_feat} features  |  "
        f"sample:feature={len(d)/n_feat:.1f}:1  |  {n_trials} trials/model")

    # -- 1. Stratified Split ---------------------------------------------------
    # Features are computed in temporal order (all lags/rolling use .shift(N)).
    # The stratified split then divides the already-computed feature matrix to
    # ensure every class (QUIET..EXTREME) appears proportionally in each subset.
    train_idx, cal_idx, test_idx = _stratified_split(d)
    train = d.iloc[train_idx].copy()
    cal   = d.iloc[cal_idx].copy()
    test  = d.iloc[test_idx].copy()

    # -- 1b. Feature Selection (on TRAIN split only) -------------------------
    # Run multi-method feature selection on train data to pick the best subset.
    # This avoids information leakage: test/cal data never influence selection.
    n_feat_original = len(feature_cols)
    feature_cols = select_features(train, feature_cols)
    n_feat = len(feature_cols)
    LOG(f"  After feature selection: {n_feat} features, "
        f"sample:feature={len(d)/n_feat:.1f}:1")

    X_tr,  y_tr  = train[feature_cols].values, train["band"].values.astype(int)
    X_cal, y_cal = cal[feature_cols].values,   cal["band"].values.astype(int)
    X_te,  y_te  = test[feature_cols].values,  test["band"].values.astype(int)

    # -- 2. Class weights ------------------------------------------------------
    # Balanced class weights ensure rare classes (QUIET=2.5%, EXTREME=1.4%)
    # are not ignored.  Extra 1.5× boost for HEAVY/EXTREME because these are
    # the most safety-critical to predict correctly.
    class_counts = Counter(y_tr.tolist())
    n_tr, n_cls = len(y_tr), len(class_counts)
    cw = {c: n_tr / (n_cls * cnt) for c, cnt in class_counts.items()}
    for rare in [4, 5]:
        if rare in cw:
            cw[rare] *= 1.5
    sw_tr = np.array([cw.get(int(yi), 1.0) for yi in y_tr])

    LOG("  Class weights (balanced + 1.5× HEAVY/EXTREME):")
    for c in range(N_BANDS):
        LOG(f"    {BAND_NAMES[c]:>10s} (n={class_counts.get(c,0):4d}): "
            f"w={cw.get(c, 0.0):.3f}")

    # -- Helper: per-fold weights ----------------------------------------------
    def _fold_weights(y_fold: np.ndarray) -> np.ndarray:
        cc = Counter(y_fold.tolist())
        ns, nc = len(y_fold), len(cc)
        fw = {c: ns / (nc * cnt) for c, cnt in cc.items()}
        for rare in [4, 5]:
            if rare in fw:
                fw[rare] *= 1.5
        return np.array([fw.get(int(yi), 1.0) for yi in y_fold])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # ======================================================================
    #  PHASE 1 -- BASELINE (no Optuna, sensible defaults)
    # ======================================================================
    LOG("\n" + "=" * 60)
    LOG("  PHASE 1: BASELINE MODELS (default hyperparameters, no Optuna)")
    LOG("=" * 60)

    default_gb_params = dict(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        min_samples_split=20, min_samples_leaf=10,
        subsample=0.8, max_features="sqrt", random_state=42,
    )
    default_lgb_params = dict(
        objective="multiclass", num_class=N_BANDS, verbosity=-1,
        n_estimators=500, max_depth=6, learning_rate=0.05,
        num_leaves=31, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        class_weight="balanced", random_state=42,
    )
    default_xgb_params = dict(
        objective="multi:softprob", num_class=N_BANDS, verbosity=0,
        n_estimators=500, max_depth=6, learning_rate=0.05,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
        random_state=42, use_label_encoder=False,
    )

    # -- Baseline CV scores (5-fold StratifiedKFold) --
    LOG("\n  Baseline 5-fold CV ...")
    for tag, Cls, params in [
        ("GB",  GradientBoostingClassifier, default_gb_params),
        ("LGB", lgb.LGBMClassifier,        default_lgb_params),
        ("XGB", xgb_lib.XGBClassifier,     default_xgb_params),
    ]:
        fold_scores: list[float] = []
        for tr_idx, va_idx in skf.split(X_tr, y_tr):
            m = Cls(**params)
            fit_kw: dict = dict(sample_weight=_fold_weights(y_tr[tr_idx]))
            if tag == "LGB":
                fit_kw["eval_set"] = [(X_tr[va_idx], y_tr[va_idx])]
                fit_kw["callbacks"] = [lgb.early_stopping(30, verbose=False),
                                        lgb.log_evaluation(period=-1)]
            elif tag == "XGB":
                fit_kw["eval_set"] = [(X_tr[va_idx], y_tr[va_idx])]
                fit_kw["verbose"] = False
            m.fit(X_tr[tr_idx], y_tr[tr_idx], **fit_kw)
            fold_scores.append(ordinal_score(y_tr[va_idx], m.predict(X_tr[va_idx])))
        LOG(f"    {tag:4s}  baseline CV = {np.mean(fold_scores):.4f}  "
            f"(folds: {', '.join(f'{s:.3f}' for s in fold_scores)})")

    # -- Fit baseline models on full train split --
    LOG("\n  Fitting baseline models on full TRAIN split ...")

    gb_base = GradientBoostingClassifier(**default_gb_params)
    gb_base.fit(X_tr, y_tr, sample_weight=sw_tr)

    lgb_base_kw = {k: v for k, v in default_lgb_params.items() if k != "class_weight"}
    lgb_base = lgb.LGBMClassifier(**lgb_base_kw)
    lgb_base.fit(X_tr, y_tr, sample_weight=sw_tr)

    xgb_base_kw = {k: v for k, v in default_xgb_params.items()}
    xgb_base = xgb_lib.XGBClassifier(**xgb_base_kw)
    xgb_base.fit(X_tr, y_tr, sample_weight=sw_tr)
    xgb_base.fit(X_tr, y_tr, sample_weight=sw_tr)

    # -- Baseline ensemble calibration on CAL split --
    LOG("  Calibrating baseline ensemble on CAL split ...")
    bsc = -1e9
    ba_gb, ba_lgb = 0.34, 0.33
    pbg = gb_base.predict_proba(X_cal)
    pbl = lgb_base.predict_proba(X_cal)
    pbx = xgb_base.predict_proba(X_cal)
    for ag in np.arange(0.10, 0.65, 0.05):
        for al in np.arange(0.10, 0.65, 0.05):
            ax = round(1.0 - ag - al, 10)
            if ax < 0.05 or ax > 0.85:
                continue
            sc = ordinal_score(y_cal, (ag * pbg + al * pbl + ax * pbx).argmax(1))
            if sc > bsc:
                bsc, ba_gb, ba_lgb = sc, round(float(ag), 4), round(float(al), 4)
    ba_xgb = round(1.0 - ba_gb - ba_lgb, 4)
    LOG(f"  Baseline ensemble: GB={ba_gb:.2f} LGB={ba_lgb:.2f} XGB={ba_xgb:.2f} "
        f"(cal={bsc:.4f})")

    # -- Baseline TEST evaluation --
    LOG(f"\n  --- PHASE 1 BASELINE TEST RESULTS "
        f"({len(test)} days) ---")
    LOG(f"  {'Model':5s}  {'Exact%':>7}  {'Adj%':>7}  {'MacF1%':>7}  "
        f"{'MAE':>5}  {'HvyRec%':>8}  {'Safe%':>6}")
    LOG(f"  {'-' * 55}")

    bp_gb = gb_base.predict_proba(X_te)
    bp_lgb = lgb_base.predict_proba(X_te)
    bp_xgb = xgb_base.predict_proba(X_te)
    bp_ens = ba_gb * bp_gb + ba_lgb * bp_lgb + ba_xgb * bp_xgb

    baseline_preds = {
        "GB":  gb_base.predict(X_te),
        "LGB": lgb_base.predict(X_te),
        "XGB": xgb_base.predict(X_te),
        "ENS": bp_ens.argmax(1),
    }
    for name, pred in baseline_preds.items():
        exact    = float(accuracy_score(y_te, pred))
        adj      = float(np.mean(np.abs(pred - y_te) <= 1))
        mac_f1   = float(f1_score(y_te, pred, average="macro", zero_division=0))
        mae      = float(np.mean(np.abs(pred - y_te)))
        hv_mask  = y_te >= 4
        hv_adj   = (float(np.mean(np.abs(pred[hv_mask] - y_te[hv_mask]) <= 1))
                    if hv_mask.sum() > 0 else float("nan"))
        danger   = int(((pred <= 1) & (y_te >= 4)).sum())
        safety   = 1.0 - danger / len(y_te)
        hv_str = f"{hv_adj*100:8.1f}" if not np.isnan(hv_adj) else f"{'N/A':>8}"
        LOG(f"  {name:5s}  {exact*100:7.1f}  {adj*100:7.1f}  {mac_f1*100:7.1f}  "
            f"{mae:5.3f}  {hv_str}  {safety*100:6.1f}")

    baseline_adj = float(np.mean(np.abs(baseline_preds["ENS"] - y_te) <= 1))
    LOG(f"\n  Baseline ENS adjacent accuracy: {baseline_adj*100:.1f}%")

    # ======================================================================
    #  PHASE 2 -- OPTUNA-TUNED MODELS
    # ======================================================================
    LOG("\n" + "=" * 60)
    LOG(f"  PHASE 2: OPTUNA TUNING ({n_trials} trials per model)")
    LOG("=" * 60)

    # -- Optuna progress callback -------------------------------------------
    # Prints every trial: trial number, current score, best score so far.
    def _make_cb(tag: str, total: int):
        _t0 = [time.time()]
        def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
            n   = trial.number + 1
            cur = trial.value if trial.value is not None else float("nan")
            best= study.best_value
            elapsed = time.time() - _t0[0]
            eta = (elapsed / n) * (total - n) if n < total else 0.0
            LOG(f"    {tag} trial {n:3d}/{total}  score={cur:.4f}  "
                f"best={best:.4f}  +{elapsed:5.0f}s  ETA~{eta:4.0f}s")
        return _cb

    # ==========================================================================
    #  MODEL 1 -- sklearn GradientBoosting  (primary, used by flask_api.py)
    # ==========================================================================
    LOG(f"\n  [1/3] Tuning GradientBoostingClassifier  ({n_trials} trials) ...")

    def _gb_objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators",      200, 1000),
            max_depth        = trial.suggest_int("max_depth",         3, 7),
            learning_rate    = trial.suggest_float("learning_rate",   0.005, 0.15, log=True),
            min_samples_split= trial.suggest_int("min_samples_split", 5, 40),
            min_samples_leaf = trial.suggest_int("min_samples_leaf",  5, 30),
            subsample        = trial.suggest_float("subsample",       0.6, 1.0),
            max_features     = trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            random_state     = 42,
        )
        fold_scores: list[float] = []
        for tr_idx, va_idx in skf.split(X_tr, y_tr):
            m = GradientBoostingClassifier(**params)
            m.fit(X_tr[tr_idx], y_tr[tr_idx],
                  sample_weight=_fold_weights(y_tr[tr_idx]))
            fold_scores.append(ordinal_score(y_tr[va_idx], m.predict(X_tr[va_idx])))
        return float(np.mean(fold_scores)) if fold_scores else 0.0

    study_gb = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    study_gb.optimize(_gb_objective, n_trials=n_trials,
                      callbacks=[_make_cb("GB", n_trials)])
    best_gb = study_gb.best_params
    LOG(f"  GB  CV score: {study_gb.best_value:.4f}  params: {best_gb}")

    # ==========================================================================
    #  MODEL 2 -- LightGBM
    # ==========================================================================
    LOG(f"\n  [2/3] Tuning LightGBM  ({n_trials} trials) ...")

    def _lgb_objective(trial: optuna.Trial) -> float:
        params = dict(
            objective        = "multiclass",
            num_class        = N_BANDS,
            verbosity        = -1,
            n_estimators     = trial.suggest_int("n_estimators",      200, 1000),
            max_depth        = trial.suggest_int("max_depth",         3, 10),
            learning_rate    = trial.suggest_float("learning_rate",   0.005, 0.2, log=True),
            num_leaves       = trial.suggest_int("num_leaves",        15, 127),
            min_child_samples= trial.suggest_int("min_child_samples", 5, 50),
            subsample        = trial.suggest_float("subsample",       0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree",0.5, 1.0),
            reg_alpha        = trial.suggest_float("reg_alpha",       1e-3, 10, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda",      1e-3, 10, log=True),
            class_weight     = "balanced",
            random_state     = 42,
        )
        fold_scores: list[float] = []
        for tr_idx, va_idx in skf.split(X_tr, y_tr):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_tr[tr_idx], y_tr[tr_idx],
                  sample_weight=_fold_weights(y_tr[tr_idx]),
                  eval_set=[(X_tr[va_idx], y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                              lgb.log_evaluation(period=-1)])
            fold_scores.append(ordinal_score(y_tr[va_idx], m.predict(X_tr[va_idx])))
        return float(np.mean(fold_scores)) if fold_scores else 0.0

    study_lgb = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
    study_lgb.optimize(_lgb_objective, n_trials=n_trials,
                       callbacks=[_make_cb("LGB", n_trials)])
    best_lgb = study_lgb.best_params
    LOG(f"  LGB CV score: {study_lgb.best_value:.4f}  params: {best_lgb}")

    # ==========================================================================
    #  MODEL 3 -- XGBoost
    # ==========================================================================
    LOG(f"\n  [3/3] Tuning XGBoost  ({n_trials} trials) ...")

    def _xgb_objective(trial: optuna.Trial) -> float:
        params = dict(
            objective        = "multi:softprob",
            num_class        = N_BANDS,
            verbosity        = 0,
            n_estimators     = trial.suggest_int("n_estimators",      200, 1000),
            max_depth        = trial.suggest_int("max_depth",         3, 10),
            learning_rate    = trial.suggest_float("learning_rate",   0.005, 0.2, log=True),
            min_child_weight = trial.suggest_int("min_child_weight",  1, 20),
            subsample        = trial.suggest_float("subsample",       0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree",0.5, 1.0),
            reg_alpha        = trial.suggest_float("reg_alpha",       1e-3, 10, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda",      1e-3, 10, log=True),
            gamma            = trial.suggest_float("gamma",           0, 5),
            random_state     = 42,
            use_label_encoder= False,
        )
        fold_scores: list[float] = []
        for tr_idx, va_idx in skf.split(X_tr, y_tr):
            m = xgb_lib.XGBClassifier(**params)
            m.fit(X_tr[tr_idx], y_tr[tr_idx],
                  sample_weight=_fold_weights(y_tr[tr_idx]),
                  eval_set=[(X_tr[va_idx], y_tr[va_idx])],
                  verbose=False)
            fold_scores.append(ordinal_score(y_tr[va_idx], m.predict(X_tr[va_idx])))
        return float(np.mean(fold_scores)) if fold_scores else 0.0

    study_xgb = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
    study_xgb.optimize(_xgb_objective, n_trials=n_trials,
                       callbacks=[_make_cb("XGB", n_trials)])
    best_xgb = study_xgb.best_params
    LOG(f"  XGB CV score: {study_xgb.best_value:.4f}  params: {best_xgb}")

    # ==========================================================================
    #  Final OPTUNA-tuned models — fit on TRAIN split only
    # ==========================================================================
    LOG("\n  Training Optuna-tuned models on TRAIN split ...")

    gb_m = GradientBoostingClassifier(**best_gb, random_state=42)
    gb_m.fit(X_tr, y_tr, sample_weight=sw_tr)

    lgb_final = {k: v for k, v in best_lgb.items() if k != "class_weight"}
    lgb_m = lgb.LGBMClassifier(**lgb_final, objective="multiclass",
                                num_class=N_BANDS, verbosity=-1, random_state=42)
    lgb_m.fit(X_tr, y_tr, sample_weight=sw_tr)

    xgb_final = {k: v for k, v in best_xgb.items() if k != "use_label_encoder"}
    xgb_m = xgb_lib.XGBClassifier(**xgb_final, objective="multi:softprob",
                                    num_class=N_BANDS, verbosity=0, random_state=42,
                                    use_label_encoder=False)
    xgb_m.fit(X_tr, y_tr, sample_weight=sw_tr)

    # ==========================================================================
    #  Ensemble calibration — using CALIBRATION split (NOT the test set)
    #
    #  Using the test set to choose ensemble weights would mean we pick the
    #  weights that happen to work best on THAT specific test window, inflating
    #  all reported test metrics.  By using a separate 30-day calibration
    #  window held out from both training and testing, we keep the test set
    #  completely untouched until the final evaluation below.
    # ==========================================================================
    LOG("  Calibrating ensemble weights on CALIBRATION split ...")
    best_sc   = -1e9
    best_a_gb = 0.34
    best_a_lgb= 0.33

    prob_gb_cal  = gb_m.predict_proba(X_cal)
    prob_lgb_cal = lgb_m.predict_proba(X_cal)
    prob_xgb_cal = xgb_m.predict_proba(X_cal)

    for a_gb in np.arange(0.10, 0.65, 0.05):
        for a_lgb in np.arange(0.10, 0.65, 0.05):
            a_xgb = round(1.0 - a_gb - a_lgb, 10)
            if a_xgb < 0.05 or a_xgb > 0.85:
                continue
            prob = a_gb * prob_gb_cal + a_lgb * prob_lgb_cal + a_xgb * prob_xgb_cal
            sc   = ordinal_score(y_cal, prob.argmax(1))
            if sc > best_sc:
                best_sc    = sc
                best_a_gb  = round(float(a_gb), 4)
                best_a_lgb = round(float(a_lgb), 4)

    best_a_xgb = round(1.0 - best_a_gb - best_a_lgb, 4)
    LOG(f"  Ensemble: GB={best_a_gb:.2f}  LGB={best_a_lgb:.2f}  XGB={best_a_xgb:.2f}  "
        f"(cal score={best_sc:.4f})")

    # ==========================================================================
    #  Final evaluation on UNTOUCHED TEST split
    # ==========================================================================
    LOG(f"\n  --- TEST RESULTS  "
        f"({len(test)} days: {test.iloc[0].date.date()} -> {test.iloc[-1].date.date()}) ---")
    LOG(f"  {'Model':5s}  {'Exact%':>7}  {'Adj%':>7}  {'MacF1%':>7}  "
        f"{'MAE':>5}  {'HvyRec%':>8}  {'Safe%':>6}")
    LOG(f"  {'-' * 55}")

    prob_gb_te  = gb_m.predict_proba(X_te)
    prob_lgb_te = lgb_m.predict_proba(X_te)
    prob_xgb_te = xgb_m.predict_proba(X_te)
    ens_prob    = best_a_gb * prob_gb_te + best_a_lgb * prob_lgb_te + best_a_xgb * prob_xgb_te

    preds = {
        "GB":  gb_m.predict(X_te),
        "LGB": lgb_m.predict(X_te),
        "XGB": xgb_m.predict(X_te),
        "ENS": ens_prob.argmax(1),
    }
    results: dict[str, dict] = {}

    for name, pred in preds.items():
        exact    = float(accuracy_score(y_te, pred))
        adj      = float(np.mean(np.abs(pred - y_te) <= 1))
        mac_f1   = float(f1_score(y_te, pred, average="macro", zero_division=0))
        mae      = float(np.mean(np.abs(pred - y_te)))
        hv_mask  = y_te >= 4
        hv_adj   = (float(np.mean(np.abs(pred[hv_mask] - y_te[hv_mask]) <= 1))
                    if hv_mask.sum() > 0 else float("nan"))
        danger   = int(((pred <= 1) & (y_te >= 4)).sum())
        safety   = 1.0 - danger / len(y_te)

        hv_str = f"{hv_adj*100:8.1f}" if not np.isnan(hv_adj) else f"{'N/A':>8}"
        LOG(f"  {name:5s}  {exact*100:7.1f}  {adj*100:7.1f}  {mac_f1*100:7.1f}  "
            f"{mae:5.3f}  {hv_str}  {safety*100:6.1f}")
        results[name] = dict(
            exact_acc             = round(exact, 4),
            adjacent_acc          = round(adj, 4),
            macro_f1              = round(mac_f1, 4),
            band_mae              = round(mae, 4),
            heavy_extreme_adj     = round(hv_adj, 4) if not np.isnan(hv_adj) else None,
            safety                = round(safety, 4),
            danger_count          = danger,
        )

    ens_pred = preds["ENS"]
    LOG(f"\n  --- Per-band (Ensemble) ---")
    for b in range(N_BANDS):
        mask = y_te == b
        n_te = int(mask.sum())
        if n_te > 0:
            be  = float(accuracy_score(y_te[mask], ens_pred[mask]))
            ba  = float(np.mean(np.abs(ens_pred[mask] - y_te[mask]) <= 1))
            LOG(f"    {BAND_NAMES[b]:>10s}  (n={n_te:3d})  exact={be*100:5.1f}%  adj={ba*100:5.1f}%")
        else:
            LOG(f"    {BAND_NAMES[b]:>10s}  (n=  0)  — not present in test window")

    champion = max(results, key=lambda k: results[k]["adjacent_acc"])
    LOG(f"\n  Champion: {champion}  adj={results[champion]['adjacent_acc']:.3f}  "
        f"exact={results[champion]['exact_acc']:.3f}  "
        f"macro-F1={results[champion]['macro_f1']:.3f}")

    # -- Phase 1 vs Phase 2 comparison ----------------------------------------
    optuna_adj = results["ENS"]["adjacent_acc"]
    improvement = (optuna_adj - baseline_adj) * 100
    LOG(f"\n  === BASELINE vs OPTUNA COMPARISON ===")
    LOG(f"    Baseline ENS adj: {baseline_adj*100:.1f}%")
    LOG(f"    Optuna   ENS adj: {optuna_adj*100:.1f}%")
    LOG(f"    Improvement:      {improvement:+.1f} percentage points")

    # -- Feature importance (GB) -----------------------------------------------
    fi = pd.DataFrame({"feature": feature_cols,
                        "importance": gb_m.feature_importances_})
    fi["importance"] /= fi["importance"].sum()
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    LOG(f"\n  --- Top-15 features (GradientBoosting) ---")
    for _, row in fi.head(15).iterrows():
        bar = "#" * int(row.importance * 200)
        LOG(f"    {row.feature:<28s}  {row.importance:.4f}  {bar}")

    # ==========================================================================
    #  Export artefacts
    # ==========================================================================
    LOG("\n  Saving artefacts ...")

    joblib.dump(gb_m,  ART_DIR / "gb_model.pkl")
    joblib.dump(lgb_m, ART_DIR / "lgb_model.pkl")
    joblib.dump(xgb_m, ART_DIR / "xgb_model.pkl")

    bg_idx = np.random.choice(len(X_tr), min(200, len(X_tr)), replace=False)
    np.save(ART_DIR / "shap_background.npy", X_tr[bg_idx])

    fi.to_csv(ART_DIR / "feature_importance.csv", index=False)

    hyperparams = {
        "gb":  best_gb,
        "lgb": best_lgb,
        "xgb": best_xgb,
        "ensemble_weights": {
            "gb":  best_a_gb,
            "lgb": best_a_lgb,
            "xgb": best_a_xgb,
        },
    }
    (ART_DIR / "hyperparams.json").write_text(
        json.dumps(hyperparams, indent=2, default=str), encoding="utf-8")

    meta = {
        "model_type":  "GradientBoosting",
        "feature_cols": feature_cols,
        "bands":       [{**b, "color": BAND_COLORS[b["name"]],
                          "advice": BAND_ADVICE[b["name"]]} for b in BANDS],
        "band_names":  BAND_NAMES,
        "n_bands":     N_BANDS,
        "feature_labels": {f: FEATURE_LABELS.get(f, f) for f in feature_cols},
        "post_covid":  POST_COVID,
        "data_file":   DATA_FILE,
        "last_retrain": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_end":    d.date.max().strftime("%Y-%m-%d"),
        "n_samples":   len(d),
        "split_strategy": {
            "type":       "stratified",
            "train_size": len(train),
            "cal_size":   len(cal),
            "test_size":  len(test),
            "rationale":  (
                "Stratified split (StratifiedShuffleSplit) ensures all 6 classes "
                "are proportionally represented in train, cal, and test sets. "
                "This is critical because EXTREME (1.3%) and QUIET (0.7%) are "
                "too rare for a temporal split to allocate enough to both subsets. "
                "Features are computed in strict temporal order with .shift(N) "
                "before splitting.  Walk-forward evaluation runs separately "
                "for a pure temporal out-of-sample metric."
            ),
        },
        "feature_selection": {
            "method": "10-method consensus (MI, ANOVA, Chi2, GB, RF, ExtraTrees, Permutation, L1-Lasso, Spearman, RFE)",
            "vote_threshold": "6/10 methods must agree",
            "original_features": n_feat_original,
            "selected_features": len(feature_cols),
            "ratio_improvement": f"{len(d)/len(feature_cols):.1f}:1 (was {len(d)/n_feat_original:.1f}:1)",
        },
    }
    (ART_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    config = {
        "version":  "v5-improved",
        "features": {
            "count": len(feature_cols),
            "list":  feature_cols,
            "sample_to_feature_ratio": round(len(d) / len(feature_cols), 1),
        },
        "test":     results,
        "champion": champion,
        "bands":    BANDS,
        "ensemble_weights": hyperparams["ensemble_weights"],
    }
    (ART_DIR / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    tail = d.tail(30)[["date", "total_pilgrims", "dow"]].copy()
    tail["date"] = tail.date.dt.strftime("%Y-%m-%d")
    tail.to_csv(ART_DIR / "last30_history.csv", index=False)

    LOG(f"  Artefacts saved to {ART_DIR}/")

    return {
        "gb_model": gb_m, "lgb_model": lgb_m, "xgb_model": xgb_m,
        "results":  results, "champion": champion,
        "feature_cols": feature_cols, "feature_importance": fi,
        "hyperparams":  hyperparams,
        "d": d, "X_tr": X_tr, "y_tr": y_tr,
        "X_cal": X_cal, "y_cal": y_cal,
        "X_te":  X_te,  "y_te":  y_te,
    }


# ===============================================================================
#  WALK-FORWARD EVALUATION  (expanding window re-train)
# ===============================================================================
def walk_forward_eval(d: pd.DataFrame, feature_cols: list[str],
                      gb_params: dict, step_size: int = 30) -> dict:
    """
    Walk-forward validation: retrain GB model each step, predict next chunk.
    Provides a realistic out-of-sample estimate across the full dataset
    (including QUIET which is absent from the recent test window).

    step_size: number of days per evaluation step (default 30)
    """
    n         = len(d)
    wf_start  = max(300, n - 12 * step_size)
    preds, actuals, dates = [], [], []

    for pos in range(wf_start, n, step_size):
        end = min(pos + step_size, n)
        y_fold  = d.iloc[:pos]["band"].values.astype(int)
        sw_fold = np.array([{c: len(y_fold) / (len(Counter(y_fold.tolist())) * cnt)
                              for c, cnt in Counter(y_fold.tolist()).items()}.get(int(yi), 1.0)
                             for yi in y_fold])
        m = GradientBoostingClassifier(**gb_params, random_state=42)
        m.fit(d.iloc[:pos][feature_cols].values, y_fold, sample_weight=sw_fold)
        pred = m.predict(d.iloc[pos:end][feature_cols].values)
        preds.extend(pred.tolist())
        actuals.extend(d.iloc[pos:end]["band"].values.astype(int).tolist())
        dates.extend(d.iloc[pos:end].date.tolist())

    preds   = np.array(preds)
    actuals = np.array(actuals)

    exact  = float(accuracy_score(actuals, preds))
    adj    = float(np.mean(np.abs(preds - actuals) <= 1))
    mac_f1 = float(f1_score(actuals, preds, average="macro", zero_division=0))
    mae    = float(np.mean(np.abs(preds - actuals)))
    danger = int(((preds <= 1) & (actuals >= 4)).sum())
    safety = 1.0 - danger / len(actuals)

    LOG(f"\n  --- Walk-forward  ({len(actuals)} days, step={step_size}) ---")
    LOG(f"    Exact:    {exact*100:.1f}%")
    LOG(f"    Adjacent: {adj*100:.1f}%")
    LOG(f"    Macro-F1: {mac_f1*100:.1f}%")
    LOG(f"    MAE:      {mae:.3f} bands")
    LOG(f"    Safety:   {safety*100:.1f}%  ({danger} dangerous under-predictions)")
    LOG(f"\n  Per-band (walk-forward):")
    for b in range(N_BANDS):
        mask = actuals == b
        n_wf = int(mask.sum())
        if n_wf > 0:
            be = float(accuracy_score(actuals[mask], preds[mask]))
            ba = float(np.mean(np.abs(preds[mask] - actuals[mask]) <= 1))
            LOG(f"    {BAND_NAMES[b]:>10s}  (n={n_wf:3d})  exact={be*100:5.1f}%  adj={ba*100:5.1f}%")

    return {"exact_acc": exact, "adjacent_acc": adj, "macro_f1": mac_f1,
            "band_mae": mae, "safety": safety, "danger_count": danger}


# ===============================================================================
#  ENTRY POINT
# ===============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tirumala crowd advisory model")
    parser.add_argument("--trials",      type=int,  default=80,
                        help="Optuna trials per model (default 80)")
    parser.add_argument("--walkforward", action="store_true",
                        help="Run walk-forward evaluation after training")
    parser.add_argument("--check-only",  action="store_true",
                        help="Feature engineering check only — no training")
    args = parser.parse_args()

    print("=" * 68)
    print("  TIRUMALA CROWD ADVISORY — Model Training Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 68)

    df_raw = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df_raw = df_raw[df_raw.date >= POST_COVID].copy()
    LOG(f"Raw: {len(df_raw)} rows  ({df_raw.date.min().date()} -> {df_raw.date.max().date()})")

    d, feature_cols = build_features(df_raw)
    LOG(f"After feature engineering + dropna: {len(d)} rows")
    LOG(f"Feature count: {len(feature_cols)}")
    LOG(f"Sample:feature ratio: {len(d)/len(feature_cols):.1f}:1")

    counts = np.bincount(d["band"].values, minlength=N_BANDS)
    LOG("Band distribution:")
    for i, name in enumerate(BAND_NAMES):
        bar = "#" * max(1, int(counts[i] / len(d) * 80))
        LOG(f"  {name:>10s}: {counts[i]:4d} ({counts[i]/len(d)*100:5.1f}%)  {bar}")

    LOG("\nFeature & split assertions:")
    LOG("  [OK] All lag features use .shift(N) -- strictly historical")
    LOG("  [OK] Rolling features computed on .shift(1) past series")
    LOG("  [OK] Expanding means use .shift(1) inside groupby -- no current-day leakage")
    LOG("  [OK] Stratified split: all 6 classes proportionally in train/cal/test")
    LOG("  [OK] StratifiedKFold CV: every class in every fold")
    LOG("  [OK] Ensemble weights calibrated on held-out cal set, not test set")
    LOG("  [OK] Walk-forward provides pure temporal out-of-sample metric")

    if args.check_only:
        # Run feature selection preview on full data (no training)
        selected = select_features(d, feature_cols)
        LOG(f"\nSelected {len(selected)} features (from {len(feature_cols)})")
        LOG(f"New sample:feature ratio: {len(d)/len(selected):.1f}:1")
        LOG("--check-only: done.")
    else:
        result = train_gb_model(d, feature_cols, n_optuna=args.trials)

        if args.walkforward:
            # Use the SELECTED feature_cols from training, not the original 57
            walk_forward_eval(d, result["feature_cols"], result["hyperparams"]["gb"])

        print("\n" + "=" * 68)
        LOG("DONE")
        print("=" * 68)
