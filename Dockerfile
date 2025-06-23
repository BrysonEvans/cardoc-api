ARG STAGE1_URL
ARG STAGE2_URL

FROM python:3.11-slim AS base
WORKDIR /app

# 1) system deps + Python deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) fetch models at build time
RUN mkdir -p models && \
    curl -fsSL "$STAGE1_URL" -o models/stage1_engine_detector.pth && \
    curl -fsSL "$STAGE2_URL" -o models/panns_cnn14_checklist_best_aug.pth

# 3) copy code
COPY . .

# 4) runtime
CMD ["gunicorn","-k","gevent","-w","4","-b","0.0.0.0:$PORT","app:app"]
