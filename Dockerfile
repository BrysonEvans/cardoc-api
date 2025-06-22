# ────────── base image ──────────
FROM python:3.11-slim

# ────────── metadata ──────────
LABEL maintainer="you@example.com"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ────────── system deps ──────────
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# ────────── workdir ──────────
WORKDIR /app

# ────────── python deps ──────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ────────── model fetch ──────────
# You must set these two build‐args (or env vars) to your presigned URLs
ARG STAGE1_URL
ARG STAGE2_URL

RUN mkdir -p models \
 && curl -sSL "$STAGE1_URL" -o models/stage1_engine_detector.pth \
 && curl -sSL "$STAGE2_URL" -o models/panns_cnn14_checklist_best_aug.pth

# ────────── copy source ──────────
COPY . .

# ────────── runtime ──────────
# Render (and many PaaSes) will set $PORT for you
EXPOSE 5050
CMD ["gunicorn", "-k", "gevent", "-w", "4", "-b", "0.0.0.0:$PORT", "app:app"]
