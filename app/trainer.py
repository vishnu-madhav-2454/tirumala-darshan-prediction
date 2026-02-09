"""
Online-learning retrainer.

Re-trains LGB-GOSS and Bi-GRU on the latest data, re-optimises blend
weights, and saves updated artefacts to disk.

Usage:
    python -m app.trainer            # retrain if new data exists
    python -m app.trainer --force    # force retrain regardless
"""
import json
import os
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.feature_selection import mutual_info_regression
import xgboost as xgb

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from app.config import (
    DATA_CSV, SEED, TEST_N, SEQ_LEN, N_SEEDS,
    BIGRU_HIDDEN, BIGRU_LAYERS, BIGRU_DROPOUT,
    LGB_GOSS_PARAMS,
    LGB_MODEL_PATH, BIGRU_MODEL_DIR, SCALER_PATH,
    TGT_SCALER_PATH, EXOG_SCALER_PATH,
    FEATURES_PATH, BLEND_WEIGHTS_PATH,
    METRICS_LOG_PATH, RETRAIN_LOG_PATH,
    DEFAULT_BLEND, ARTEFACTS_DIR,
)
from app.features import make_features, get_dl_features


DEVICE = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"


# ── Helpers ─────────────────────────────────────────────────────────
def _ev(yt, yp):
    return {
        "MAE": round(mean_absolute_error(yt, yp), 1),
        "RMSE": round(np.sqrt(mean_squared_error(yt, yp)), 1),
        "MAPE%": round(mean_absolute_percentage_error(yt, yp) * 100, 2),
        "R2": round(r2_score(yt, yp), 4),
    }


def _select_features(X, y, feat_cols):
    """MI + XGB importance union → selected column names."""
    mi = mutual_info_regression(X, y, random_state=SEED, n_neighbors=5)
    mi_df = pd.DataFrame({"f": feat_cols, "v": mi}).sort_values("v", ascending=False)

    xg = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        random_state=SEED, verbosity=0,
    )
    xg.fit(X, y)
    xi_df = pd.DataFrame({"f": feat_cols, "v": xg.feature_importances_}).sort_values(
        "v", ascending=False
    )
    sel = sorted(list(set(mi_df.head(50)["f"]) | set(xi_df.head(50)["f"])))
    return sel


# ── Bi-GRU model ───────────────────────────────────────────────────
if HAS_TORCH:
    class _TSDs(Dataset):
        def __init__(self, X, y, noise=0.0):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
            self.noise = noise
            self.train_mode = True
        def __len__(self):
            return len(self.X)
        def __getitem__(self, i):
            x = self.X[i]
            if self.train_mode and self.noise > 0:
                x = x + torch.randn_like(x) * self.noise
            return x, self.y[i]

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


# ── Public retraining function ─────────────────────────────────────
def retrain(force: bool = False) -> dict:
    """Retrain both models on the current CSV.

    Returns dict with metrics and timestamps.
    """
    t0 = time.time()
    print("=" * 60)
    print("  ONLINE RETRAINING")
    print("=" * 60)

    # 1. Load data
    raw = pd.read_csv(DATA_CSV, parse_dates=["date"])
    raw = raw[["date", "total_pilgrims"]].sort_values("date").dropna().reset_index(drop=True)
    print(f"  Data: {len(raw)} rows  ({raw.date.min().date()} → {raw.date.max().date()})")

    # Check if retrain needed (compare to last retrain data size)
    last_size = 0
    if os.path.exists(RETRAIN_LOG_PATH):
        log = pd.read_csv(RETRAIN_LOG_PATH)
        if len(log):
            last_size = int(log.iloc[-1]["data_rows"])
    if len(raw) == last_size and not force:
        print("  No new data since last retrain. Use --force to override.")
        return {"status": "skipped", "reason": "no_new_data"}

    # 2. Feature engineering
    print("  Engineering features ...")
    df = make_features(raw)
    df = df.dropna().reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in ["date", "total_pilgrims"]]
    X_all = df[feat_cols].values
    y_all = df["total_pilgrims"].values

    # 3. Feature selection
    print("  Selecting features ...")
    sel = _select_features(X_all, y_all, feat_cols)
    print(f"  Selected {len(sel)} features")

    X = df[sel].values
    y = y_all
    test_n = min(TEST_N, len(X) // 5)   # ensure enough train data

    Xtr, Xte = X[:-test_n], X[-test_n:]
    ytr, yte = y[:-test_n], y[-test_n:]

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    # 4. Train LGB-GOSS
    print("  Training LGB-GOSS ...")
    tscv = TimeSeriesSplit(n_splits=5)
    lgb_model = lgb.LGBMRegressor(**LGB_GOSS_PARAMS, random_state=SEED)
    lgb_model.fit(
        Xtr, ytr,
        eval_set=[(Xte, yte)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    lgb_pred = lgb_model.predict(Xte)
    lgb_m = _ev(yte, lgb_pred)
    print(f"    LGB-GOSS  MAE={lgb_m['MAE']:.0f}  R²={lgb_m['R2']:.4f}")

    # 5. Train Bi-GRU (multi-seed)
    bigru_pred = None
    bigru_m = {}
    if HAS_TORCH:
        print(f"  Training Bi-GRU ({N_SEEDS} seeds) on {DEVICE} ...")
        dl_df = get_dl_features(raw)
        dl_feat_cols = [c for c in dl_df.columns if c not in ["date", "total_pilgrims"]]

        tgt_vals = dl_df["total_pilgrims"].values.reshape(-1, 1)
        exog_vals = dl_df[dl_feat_cols].values

        tgt_sc = MinMaxScaler()
        tgt_scaled = tgt_sc.fit_transform(tgt_vals)
        exog_sc = MinMaxScaler()
        exog_scaled = exog_sc.fit_transform(exog_vals)
        combined = np.hstack([tgt_scaled, exog_scaled])
        n_feat = combined.shape[1]

        # Make sequences
        Xseq, yseq = [], []
        for i in range(SEQ_LEN, len(combined)):
            Xseq.append(combined[i - SEQ_LEN : i])
            yseq.append(tgt_scaled[i, 0])
        Xseq, yseq = np.array(Xseq), np.array(yseq)

        n_test_dl = test_n
        n_val_dl = min(45, len(Xseq) // 10)
        n_train_dl = len(Xseq) - n_test_dl - n_val_dl

        Xtr_dl = Xseq[:n_train_dl]
        ytr_dl = yseq[:n_train_dl]
        Xva_dl = Xseq[n_train_dl : n_train_dl + n_val_dl]
        yva_dl = yseq[n_train_dl : n_train_dl + n_val_dl]
        Xte_dl = Xseq[-n_test_dl:]
        yte_dl = yseq[-n_test_dl:]
        dl_test_actuals = dl_df.total_pilgrims.values[-n_test_dl:]

        BS = 32
        tr_ds = _TSDs(Xtr_dl, ytr_dl, noise=0.02)
        va_ds = _TSDs(Xva_dl, yva_dl); va_ds.train_mode = False
        te_ds = _TSDs(Xte_dl, yte_dl); te_ds.train_mode = False

        tr_loader = DataLoader(tr_ds, batch_size=BS, shuffle=True, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=BS, shuffle=False)
        te_loader = DataLoader(te_ds, batch_size=BS, shuffle=False)

        os.makedirs(BIGRU_MODEL_DIR, exist_ok=True)
        seed_preds = []
        for s in range(N_SEEDS):
            seed = SEED + s * 111
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = BiGRU(n_feat, BIGRU_HIDDEN, BIGRU_LAYERS, BIGRU_DROPOUT).to(DEVICE)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=3e-3)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=10)
            criterion = nn.SmoothL1Loss()
            best_val, best_state, wait = float("inf"), None, 0

            for ep in range(300):
                model.train(); tr_ds.train_mode = True
                for xb, yb in tr_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                model.eval(); tr_ds.train_mode = False
                vl = []
                with torch.no_grad():
                    for xb, yb in va_loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        vl.append(criterion(model(xb), yb).item())
                avg_vl = np.mean(vl)
                sched.step(avg_vl)
                if avg_vl < best_val:
                    best_val = avg_vl
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                if wait >= 40:
                    break

            model.load_state_dict(best_state)
            model = model.to(DEVICE).eval()
            preds = []
            with torch.no_grad():
                for xb, _ in te_loader:
                    preds.append(model(xb.to(DEVICE)).cpu().numpy())
            scaled = np.concatenate(preds)
            inv = tgt_sc.inverse_transform(scaled.reshape(-1, 1)).flatten()
            seed_preds.append(inv)
            # Save each seed model
            torch.save(best_state, os.path.join(BIGRU_MODEL_DIR, f"seed_{s}.pt"))
            print(f"    Seed {s+1}/{N_SEEDS}  MAE={mean_absolute_error(dl_test_actuals, inv):.0f}")

        bigru_pred = np.mean(seed_preds, axis=0)
        bigru_m = _ev(dl_test_actuals, bigru_pred)
        print(f"    Bi-GRU (ensemble)  MAE={bigru_m['MAE']:.0f}  R²={bigru_m['R2']:.4f}")

        # Save DL scalers
        joblib.dump(tgt_sc, TGT_SCALER_PATH)
        joblib.dump(exog_sc, EXOG_SCALER_PATH)
        # Save n_feat for inference
        with open(os.path.join(BIGRU_MODEL_DIR, "meta.json"), "w") as f:
            json.dump({"n_feat": n_feat, "seq_len": SEQ_LEN,
                       "dl_feat_cols": dl_feat_cols}, f)
    else:
        print("  ⚠ PyTorch not available — skipping Bi-GRU")

    # 6. Optimise blend weights
    print("  Optimising blend weights ...")
    if bigru_pred is not None:
        best_a, best_mae = 0.55, 1e18
        for a in np.arange(0.0, 1.01, 0.01):
            bp = a * bigru_pred + (1 - a) * lgb_pred[:len(bigru_pred)]
            bm = mean_absolute_error(yte[:len(bigru_pred)], bp)
            if bm < best_mae:
                best_mae = bm
                best_a = a
        blend_w = {"BiGRU": round(float(best_a), 3),
                   "LGB-GOSS": round(1 - float(best_a), 3)}
        blend_pred = best_a * bigru_pred + (1 - best_a) * lgb_pred[:len(bigru_pred)]
        blend_m = _ev(yte[:len(bigru_pred)], blend_pred)
    else:
        blend_w = {"BiGRU": 0.0, "LGB-GOSS": 1.0}
        blend_m = lgb_m

    print(f"    Blend weights: {blend_w}")
    print(f"    Blend  MAE={blend_m['MAE']:.0f}  R²={blend_m['R2']:.4f}")

    # 7. Save artefacts
    print("  Saving artefacts ...")
    joblib.dump(lgb_model, LGB_MODEL_PATH)
    joblib.dump(sc, SCALER_PATH)
    with open(FEATURES_PATH, "w") as f:
        json.dump(sel, f, indent=2)
    with open(BLEND_WEIGHTS_PATH, "w") as f:
        json.dump(blend_w, f, indent=2)

    # Metrics log (append)
    row = {
        "timestamp": datetime.now().isoformat(),
        "data_rows": len(raw),
        "latest_date": str(raw.date.max().date()),
        "lgb_mae": lgb_m["MAE"],
        "lgb_r2": lgb_m["R2"],
        "bigru_mae": bigru_m.get("MAE", ""),
        "bigru_r2": bigru_m.get("R2", ""),
        "blend_mae": blend_m["MAE"],
        "blend_r2": blend_m["R2"],
        "blend_w_bigru": blend_w["BiGRU"],
        "blend_w_lgb": blend_w["LGB-GOSS"],
    }
    if os.path.exists(METRICS_LOG_PATH):
        mlog = pd.read_csv(METRICS_LOG_PATH)
    else:
        mlog = pd.DataFrame()
    mlog = pd.concat([mlog, pd.DataFrame([row])], ignore_index=True)
    mlog.to_csv(METRICS_LOG_PATH, index=False)

    # Retrain log
    rrow = {"timestamp": datetime.now().isoformat(), "data_rows": len(raw)}
    if os.path.exists(RETRAIN_LOG_PATH):
        rlog = pd.read_csv(RETRAIN_LOG_PATH)
    else:
        rlog = pd.DataFrame()
    rlog = pd.concat([rlog, pd.DataFrame([rrow])], ignore_index=True)
    rlog.to_csv(RETRAIN_LOG_PATH, index=False)

    elapsed = time.time() - t0
    print(f"\n  ✅ Retraining complete in {elapsed:.0f}s")
    print("=" * 60)

    return {
        "status": "retrained",
        "data_rows": len(raw),
        "lgb": lgb_m,
        "bigru": bigru_m,
        "blend": blend_m,
        "blend_weights": blend_w,
        "elapsed_s": round(elapsed, 1),
    }


# ── CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain TTD models")
    parser.add_argument("--force", action="store_true", help="Force retrain")
    args = parser.parse_args()
    retrain(force=args.force)
