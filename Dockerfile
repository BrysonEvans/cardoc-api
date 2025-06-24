# Dockerfile — CarDoc AI
FROM python:3.11.5-slim

# Don’t write .pyc files; unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1) Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ffmpeg \
      curl && \
    rm -rf /var/lib/apt/lists/*

# 2) Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy app code
COPY . .

# 4) Download your two model weights via Render’s injected env vars
ARG STAGE1_URL
ARG STAGE2_URL

RUN test -n "$STAGE1_URL" \
    && curl -fsSL "$STAGE1_URL" -o models/stage1_engine_detector.pth

RUN test -n "$STAGE2_URL" \
    && curl -fsSL "$STAGE2_URL" -o models/panns_cnn14_checklist_best_aug.pth

# 5) Expose & run
EXPOSE 5050
CMD ["gunicorn", "-k", "gevent", "-w", "4", "-b", "0.0.0.0:5050", "app:app"]
