"""
═══════════════════════════════════════════════════════════════════
  Tirumala Darshan — Production Deployment Server
═══════════════════════════════════════════════════════════════════
  Uses Waitress (production-grade WSGI server) to serve both the
  Flask API and the React frontend build.

  Usage:
    python deploy.py                   # default: 0.0.0.0:5000
    python deploy.py --port 8080       # custom port
    python deploy.py --host 127.0.0.1  # localhost only
═══════════════════════════════════════════════════════════════════
"""

import os
import sys
import argparse

# HuggingFace settings — respect env vars (Render sets HF_HUB_OFFLINE=0)
if "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"          # default: offline (use cache)
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser(description="Deploy Tirumala Darshan API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)), help="Port to bind (default: $PORT or 5000)")
    parser.add_argument("--threads", type=int, default=4, help="Number of worker threads (default: 4)")
    args = parser.parse_args()

    # Import flask app (this triggers model loading)
    from flask_api import app

    # Check frontend build
    build_dir = os.path.join(os.path.dirname(__file__), "client", "dist")
    has_build = os.path.exists(os.path.join(build_dir, "index.html"))

    if not has_build:
        print("⚠️  Frontend build not found! Run: cd client && npm run build")
        sys.exit(1)

    print()
    print("═" * 60)
    print("  🚀 PRODUCTION SERVER")
    print(f"  Server: Waitress (WSGI)")
    print(f"  Threads: {args.threads}")
    print(f"  URL: http://{args.host}:{args.port}")
    print(f"  Frontend: ✅ Serving from client/dist")
    print("═" * 60)
    print()

    from waitress import serve
    serve(app, host=args.host, port=args.port, threads=args.threads)


if __name__ == "__main__":
    main()
