# ═══════════════════════════════════════════════════════════════
#  Dockerfile — Hugging Face Spaces (Docker SDK)
#  శ్రీవారి సేవ — Tirumala Darshan Prediction
# ═══════════════════════════════════════════════════════════════

FROM python:3.11-slim

# Install Node.js for React build
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (CPU-only torch + Chronos) ──
COPY requirements-hf.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-hf.txt

# ── Copy project files ──
COPY . .

# ── Build React frontend ──
RUN cd client && npm install && npm run build && rm -rf node_modules

# ── HF Spaces uses port 7860 ──
ENV PORT=7860
ENV HF_HUB_OFFLINE=0
ENV TOKENIZERS_PARALLELISM=false
EXPOSE 7860

# ── Start production server ──
CMD ["python", "deploy.py", "--port", "7860", "--threads", "4"]
