"""
Shared configuration for the Tirumala Darshan Prediction pipeline.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Data ──
DATA_CSV = os.path.join(BASE_DIR, "tirumala_darshan_data_clean.csv")

# ── Model artefacts directory ──
ARTEFACTS_DIR = os.path.join(BASE_DIR, "artefacts")
os.makedirs(ARTEFACTS_DIR, exist_ok=True)

LGB_MODEL_PATH   = os.path.join(ARTEFACTS_DIR, "lgb_goss.pkl")
BIGRU_MODEL_DIR   = os.path.join(ARTEFACTS_DIR, "bigru")
SCALER_PATH       = os.path.join(ARTEFACTS_DIR, "scaler.pkl")
TGT_SCALER_PATH   = os.path.join(ARTEFACTS_DIR, "tgt_scaler.pkl")
EXOG_SCALER_PATH   = os.path.join(ARTEFACTS_DIR, "exog_scaler.pkl")
FEATURES_PATH     = os.path.join(ARTEFACTS_DIR, "selected_features.json")
BLEND_WEIGHTS_PATH = os.path.join(ARTEFACTS_DIR, "blend_weights.json")
METRICS_LOG_PATH   = os.path.join(ARTEFACTS_DIR, "metrics_log.csv")
RETRAIN_LOG_PATH   = os.path.join(ARTEFACTS_DIR, "retrain_log.csv")

# ── Scraper ──
SCRAPE_URL = "https://news.tirumala.org/category/darshan/page/{page}/"
SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
SCRAPE_DELAY = 1.0  # seconds between requests

# ── Model hyper-params (champion blend) ──
SEED = 42
TEST_N = 30           # hold-out window for re-evaluation
SEQ_LEN = 30          # Bi-GRU lookback
N_SEEDS = 5           # Bi-GRU seed count
BIGRU_HIDDEN = 48
BIGRU_LAYERS = 2
BIGRU_DROPOUT = 0.35

# LGB-GOSS best params from grid search
LGB_GOSS_PARAMS = dict(
    n_estimators=3000,
    boosting_type="goss",
    max_depth=6,
    learning_rate=0.05,
    num_leaves=63,
    colsample_bytree=0.7,
    top_rate=0.1,
    other_rate=0.1,
    min_child_samples=10,
    reg_alpha=0.5,
    reg_lambda=2.0,
    verbosity=-1,
)

# Default blend weights  (Bi-GRU × 0.55,  LGB-GOSS × 0.45)
DEFAULT_BLEND = {"BiGRU": 0.55, "LGB-GOSS": 0.45}

# ── Server ──
API_HOST = "0.0.0.0"
API_PORT = 8000
