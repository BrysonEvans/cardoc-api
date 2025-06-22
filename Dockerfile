# ---------- base image ----------
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files & buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install ffmpeg & curl
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- build-time model URLs ----------
ARG STAGE1_URL="https://cardoc-models-us-east-2.s3.amazonaws.com/models/stage1_engine_detector.pth"
ARG STAGE2_URL="https://cardoc-models-us-east-2.s3.amazonaws.com/models/panns_cnn14_checklist_best_aug.pth"

# ---------- fetch models from S3 ----------
RUN mkdir -p models && \
    echo "Downloading Stage-1 checkpoint..." && \
    curl -fSL "$STAGE1_URL" -o models/stage1_engine_detector.pth && \
    echo "Downloading Stage-2 checkpoint..." && \
    curl -fSL "$STAGE2_URL" -o models/panns_cnn14_checklist_best_aug.pth

# ---------- Python deps ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- app code ----------
COPY . .

# Ensure your app sees /app on its import path
ENV PYTHONPATH=/app

# ---------- launch ----------
CMD ["gunicorn", "-k", "gevent", "-w", "4", "-b", "0.0.0.0:$PORT", "app:app"]
