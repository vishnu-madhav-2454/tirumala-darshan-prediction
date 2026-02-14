FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install Node.js for building React frontend
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Build React frontend
WORKDIR /app/client
RUN npm ci && npm run build
WORKDIR /app

# Build vector database for RAG chatbot (needs sentence-transformers)
RUN python build_vectordb.py || echo "VectorDB build skipped (will rebuild on first request)"

# Expose port 7860 (HuggingFace default)
EXPOSE 7860

# Run with gunicorn — increased timeout for LLM calls
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "180", "flask_api:app"]
