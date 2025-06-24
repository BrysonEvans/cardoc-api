FROM python:3.11.5-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system deps for ta-lib, ffmpeg, curl, matplotlib, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libta-lib-dev \
      python3-dev \
      libatlas-base-dev \
      ffmpeg \
      curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Fetch the models via the presigned URLs
ARG STAGE1_URL
ARG STAGE2_URL
RUN test -n "$STAGE1_URL" && curl -fsSL "$STAGE1_URL" -o models/stage1_engine_detector.pth
RUN test -n "$STAGE2_URL" && curl -fsSL "$STAGE2_URL" -o models/panns_cnn14_checklist_best_aug.pth

EXPOSE 5050
CMD ["gunicorn", "-k", "gevent", "-w", "4", "-b", "0.0.0.0:5050", "app:app"]
