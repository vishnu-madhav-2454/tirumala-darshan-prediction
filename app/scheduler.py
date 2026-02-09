"""
Scheduler / Orchestrator — ties scraping + retraining + serving together.

Three modes:
  1. python -m app.scheduler --once    (run scrape+retrain once, then exit)
  2. python -m app.scheduler --cron    (run daily at 10:00 AM IST)
  3. python -m app.scheduler --service (run scraper + trainer + API server)
"""
import argparse
import threading
import time
from datetime import datetime

import schedule


def _daily_pipeline():
    """Scrape new data → retrain if needed."""
    print(f"\n{'='*60}")
    print(f"  ⏰ Scheduled pipeline run — {datetime.now()}")
    print(f"{'='*60}")

    from app.scraper import scrape_incremental
    from app.trainer import retrain

    # Step 1: Scrape
    print("\n📥 Step 1: Incremental scrape")
    added = scrape_incremental(max_pages=5)

    # Step 2: Retrain (only if new data)
    if added > 0:
        print("\n🧠 Step 2: Retraining models (new data found)")
        retrain(force=False)
    else:
        print("\n⏭️  Step 2: Skipping retrain (no new data)")

    print(f"\n✅ Pipeline complete at {datetime.now()}")


def run_once():
    """Run the pipeline once and exit."""
    _daily_pipeline()


def run_cron():
    """Schedule the pipeline to run daily at 10:00 AM."""
    schedule.every().day.at("10:00").do(_daily_pipeline)
    print("📅 Scheduler started — runs daily at 10:00 AM")
    print("   Press Ctrl+C to stop\n")

    # Also run once immediately
    _daily_pipeline()

    while True:
        schedule.run_pending()
        time.sleep(60)


def run_service():
    """Start everything: scheduler thread + API server."""
    # Start scheduler in background thread
    def _sched_thread():
        schedule.every().day.at("10:00").do(_daily_pipeline)
        print("📅 Background scheduler: daily at 10:00 AM")
        while True:
            schedule.run_pending()
            time.sleep(60)

    t = threading.Thread(target=_sched_thread, daemon=True)
    t.start()

    # Start API server (blocking)
    import uvicorn
    from app.config import API_HOST, API_PORT
    print(f"\n🚀 Starting API server on {API_HOST}:{API_PORT}")
    uvicorn.run("app.server:app", host=API_HOST, port=API_PORT, reload=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tirumala Pipeline Scheduler")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--once", action="store_true",
                       help="Run scrape+retrain once, then exit")
    group.add_argument("--cron", action="store_true",
                       help="Run daily at 10:00 AM (stays alive)")
    group.add_argument("--service", action="store_true",
                       help="Run scheduler + API server together")
    args = parser.parse_args()

    if args.once:
        run_once()
    elif args.cron:
        run_cron()
    elif args.service:
        run_service()
