"""
Shared configuration for the Tirumala Darshan Prediction pipeline.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Data ──
DATA_CSV = os.path.join(BASE_DIR, "tirumala_darshan_data_CLEAN_NO_OUTLIERS.csv")

# ── Model artefacts directory ──
ARTEFACTS_DIR = os.path.join(BASE_DIR, "artefacts")
os.makedirs(ARTEFACTS_DIR, exist_ok=True)

PROD_DIR = os.path.join(ARTEFACTS_DIR, "production")
os.makedirs(PROD_DIR, exist_ok=True)

# ── Scraper ──
SCRAPE_URL = "https://news.tirumala.org/category/darshan/page/{page}/"
SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
SCRAPE_DELAY = 1.0  # seconds between requests

# ── Model hyper-params ──
SEED = 42
TEST_N = 30           # hold-out window for re-evaluation
SEQ_LEN = 30          # DL lookback
N_SEEDS = 5           # DL seed count

# ── COVID period (removed from training data) ──
COVID_START = "2020-03-19"
COVID_END = "2022-01-31"

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

# ── Champion Blend (Top5-Blend, MAE=2354, R²=0.7504) ──
BLEND_WEIGHTS = {
    "Chronos-T5": 0.587,
    "Tuned-XGB": 0.211,
    "N-HiTS": 0.170,
    "N-BEATS": 0.017,
    "LGB-GOSS": 0.015,
}

# ── Server ──
API_HOST = "127.0.0.1"
API_PORT = 5000
