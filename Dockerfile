# ============================================================
# Dockerfile
# ============================================================
# Multi-stage-aware single-stage Docker build for GridSense AI.
#
# Base image: python:3.11-slim
#   - slim = Debian without development tools (smaller image)
#   - 3.11 = minimum required for pydantic-settings v2
#
# Build and run locally:
#   docker build -t gridsense-ai .
#   docker run -p 8000:8000 gridsense-ai
# ============================================================

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# ── System dependencies ───────────────────────────────────────
# build-essential: needed to compile C extensions (some numpy deps)
# curl: used in the HEALTHCHECK below
# We clean apt cache after install to keep the image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────
# Copy requirements first so Docker can cache this layer.
# If requirements.txt hasn't changed, Docker reuses the cached
# pip install layer even if source code has changed — faster builds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source code ───────────────────────────────────
COPY . .

# Create data directories (models saved here at runtime)
RUN mkdir -p data/models data/mlruns

# ── Environment defaults ──────────────────────────────────────
# These can be overridden via docker run -e or docker-compose env_file
ENV APP_ENV=production
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000
ENV LOG_LEVEL=INFO

# Expose the port the app listens on
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────
# Docker will mark the container unhealthy after 3 failures.
# --start-period=120s gives time for model loading at startup.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start command ─────────────────────────────────────────────
# --workers 2: two Uvicorn worker processes (tune based on CPU)
# For free-tier Render (0.5 CPU), keep at 1 worker
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
