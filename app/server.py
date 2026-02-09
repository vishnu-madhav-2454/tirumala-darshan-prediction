"""
FastAPI prediction server for Tirumala Darshan forecasting.

Endpoints:
    GET  /health          — health check
    GET  /predict?days=7  — forecast next N days
    POST /scrape          — trigger incremental scrape
    POST /retrain         — trigger model retraining
    GET  /metrics         — get training metrics history
    GET  /data/latest     — latest data summary
"""
import os
import json
import threading
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, Query, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import (
    DATA_CSV, METRICS_LOG_PATH, RETRAIN_LOG_PATH,
    BLEND_WEIGHTS_PATH, API_HOST, API_PORT,
)
from app.scraper import scrape_incremental
from app.trainer import retrain
from app.predictor import predict_next_days

app = FastAPI(
    title="Tirumala Darshan Prediction API",
    description="Automated ML pipeline — scrape, train, predict pilgrim counts",
    version="1.0.0",
)

# Allow CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple lock to prevent concurrent retraining
_retrain_lock = threading.Lock()
_scrape_lock = threading.Lock()


# ── Health ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    has_model = os.path.exists(
        os.path.join(os.path.dirname(__file__), "..", "artefacts", "lgb_goss.pkl")
    )
    has_data = os.path.exists(DATA_CSV)
    return {
        "status": "ok",
        "model_ready": has_model,
        "data_available": has_data,
        "timestamp": datetime.now().isoformat(),
    }


# ── Predict ─────────────────────────────────────────────────────────
@app.get("/predict")
def predict(days: int = Query(default=7, ge=1, le=90)):
    """Forecast pilgrim count for the next N days."""
    try:
        df = predict_next_days(days)
        records = df.to_dict(orient="records")
        # Convert dates to strings
        for r in records:
            r["date"] = r["date"].strftime("%Y-%m-%d")
        return {"forecast": records, "days": days}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Scrape ──────────────────────────────────────────────────────────
def _bg_scrape(pages: int):
    if _scrape_lock.locked():
        return
    with _scrape_lock:
        scrape_incremental(max_pages=pages)

@app.post("/scrape")
def trigger_scrape(background_tasks: BackgroundTasks, pages: int = Query(default=5, ge=1, le=20)):
    """Trigger incremental scrape in background."""
    if _scrape_lock.locked():
        return {"status": "already_running"}
    background_tasks.add_task(_bg_scrape, pages)
    return {"status": "started", "pages": pages}


# ── Retrain ─────────────────────────────────────────────────────────
def _bg_retrain(force: bool):
    if _retrain_lock.locked():
        return
    with _retrain_lock:
        retrain(force=force)

@app.post("/retrain")
def trigger_retrain(background_tasks: BackgroundTasks, force: bool = Query(default=False)):
    """Trigger model retraining in background."""
    if _retrain_lock.locked():
        return {"status": "already_running"}
    background_tasks.add_task(_bg_retrain, force)
    return {"status": "started", "force": force}


# ── Metrics ─────────────────────────────────────────────────────────
@app.get("/metrics")
def get_metrics():
    """Return training metrics history."""
    if not os.path.exists(METRICS_LOG_PATH):
        return {"metrics": []}
    df = pd.read_csv(METRICS_LOG_PATH)
    return {"metrics": df.to_dict(orient="records")}


# ── Data summary ────────────────────────────────────────────────────
@app.get("/data/latest")
def data_summary():
    """Return summary of the current dataset."""
    if not os.path.exists(DATA_CSV):
        raise HTTPException(status_code=404, detail="No data file found")
    df = pd.read_csv(DATA_CSV, parse_dates=["date"])
    tp = df["total_pilgrims"].dropna()
    return {
        "total_records": len(df),
        "date_range": {
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
        },
        "pilgrim_stats": {
            "mean": round(float(tp.mean())),
            "median": round(float(tp.median())),
            "min": round(float(tp.min())),
            "max": round(float(tp.max())),
            "std": round(float(tp.std())),
        },
        "last_5": df.tail(5)[["date", "total_pilgrims"]].assign(
            date=lambda x: x["date"].dt.strftime("%Y-%m-%d")
        ).to_dict(orient="records"),
    }


# ── Blend weights ──────────────────────────────────────────────────
@app.get("/blend")
def get_blend_weights():
    if os.path.exists(BLEND_WEIGHTS_PATH):
        with open(BLEND_WEIGHTS_PATH) as f:
            return json.load(f)
    return {"BiGRU": 0.55, "LGB-GOSS": 0.45}


# ── Run server ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host=API_HOST, port=API_PORT, reload=True)
