# Dockerfile — CarDoc AI
# pin to the exact slim image you tested locally
FROM python:3.11.5-slim

# don’t write .pyc, buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 0) Install system deps (curl for downloads, ffmpeg for audio)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl \
      ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 1) Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Pull your two models from S3 via presigned URL args
ARG STAGE1_URL
ARG STAGE2_URL
RUN test -n "$STAGE1_URL" || (echo "❌ STAGE1_URL unset" && exit 1) && \
    test -n "$STAGE2_URL" || (echo "❌ STAGE2_URL unset" && exit 1) && \
    mkdir -p models && \
    curl -fsSL "$STAGE1_URL" -o models/stage1_engine_detector.pth && \
    curl -fsSL "$STAGE2_URL" -o models/panns_cnn14_checklist_best_aug.pth

# 3) Copy your app code
COPY . .

# 4) Expose the port your app listens on
EXPOSE 5050

# 5) Launch under Gunicorn + Gevent
CMD ["gunicorn", "-k", "gevent", "-w", "4", "-b", "0.0.0.0:5050", "app:app"]
