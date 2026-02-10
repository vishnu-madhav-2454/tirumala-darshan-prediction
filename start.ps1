# ═══════════════════════════════════════════════════════════════
#  Tirumala Darshan — One-Click Deploy Script (PowerShell)
# ═══════════════════════════════════════════════════════════════

Write-Host ""
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  Tirumala Darshan - Deploying..." -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host ""

# Navigate to project root
Set-Location $PSScriptRoot

# Activate virtual environment
& .\.venv\Scripts\Activate.ps1

# Set environment for offline Chronos model
$env:HF_HUB_OFFLINE = "1"
$env:TOKENIZERS_PARALLELISM = "false"

# Build frontend if needed
if (-not (Test-Path "client\dist\index.html")) {
    Write-Host "Building frontend..." -ForegroundColor Cyan
    Push-Location client
    npm run build
    Pop-Location
}

# Start production server
Write-Host ""
Write-Host "Starting production server on http://localhost:5000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host ""

python deploy.py --port 5000 --threads 4
