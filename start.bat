@echo off
:: ═══════════════════════════════════════════════════════════════
::  Tirumala Darshan — One-Click Deploy Script (Windows)
:: ═══════════════════════════════════════════════════════════════
echo.
echo ========================================
echo   Tirumala Darshan - Deploying...
echo ========================================
echo.

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Set environment for offline Chronos model
set HF_HUB_OFFLINE=1
set TOKENIZERS_PARALLELISM=false

:: Build frontend if needed
if not exist "client\dist\index.html" (
    echo Building frontend...
    cd client
    call npm run build
    cd ..
)

:: Start production server
echo Starting production server on http://localhost:5000
echo Press Ctrl+C to stop
python deploy.py --port 5000 --threads 4
