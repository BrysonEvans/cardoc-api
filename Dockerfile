# ---------- base image ----------
FROM python:3.11-slim AS base

# ─ Prevent Python from writing .pyc files & buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ─ Working directory
WORKDIR /app

# ─ Install ffmpeg & curl for audio transcoding and downloads
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ─ Expose the port your Flask/Gunicorn server listens on
ENV PORT=5050
EXPOSE 5050

# ─ Build-time args for your S3 model URLs
ARG STAGE1_URL="https://cardoc-models-us-east-2.s3.amazonaws.com/models/stage1_engine_detector.pth"
ARG STAGE2_URL="https://cardoc-models-us-east-2.s3.amazonaws.com/models/panns_cnn14_checklist_best_aug.pth"

# ─ Fetch Stage-1 and Stage-2 checkpoints from S3
RUN mkdir -p models && \
    echo "→ Downloading Stage-1 model…" && \
    curl -fSL "$STAGE1_URL"  -o models/stage1_engine_detector.pth && \
    echo "→ Downloading Stage-2 model…" && \
    curl -fSL "$STAGE2_URL"  -o models/panns_cnn14_checklist_best_aug.pth

# ─ Copy & install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─ Copy application code
COPY . .

# ─ Ensure your app can do absolute imports from /app
ENV PYTHONPATH=/app

# ─ Launch with Gunicorn + Gevent
CMD ["gunicorn", "-k", "gevent", "-w", "4", "-b", "0.0.0.0:$PORT", "app:app"]
