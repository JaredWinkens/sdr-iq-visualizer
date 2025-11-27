# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies (libiio for SDR, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libiio0 libiio-dev libusb-1.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for better layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu \
       torch==2.9.0 torchaudio==2.9.0 torchvision==0.24.0 \
    && pip install -r requirements.txt

# Copy application source
COPY . .

EXPOSE 8050

# Healthcheck: Dash root should return 200 once running
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8050/ || exit 1

# Default environment variables (can be overridden by compose/.env)
ENV DASH_HOST=0.0.0.0 \
    DASH_PORT=8050 \
    DASH_DEBUG=false

ENTRYPOINT ["python", "run_dashboard.py"]
