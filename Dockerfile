# Dockerfile — CarDoc AI

FROM python:3.11.5-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1) Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Pull both models from S3 via presigned URLs
ARG STAGE1_URL
ARG STAGE2_URL

RUN mkdir -p models \
 && if [ -z "$STAGE1_URL" ]; then echo "❌ STAGE1_URL is empty" && exit 1; fi \
 && curl -sSL "$STAGE1_URL" -o models/stage1_engine_detector.pth \
 && if [ -z "$STAGE2_URL" ]; then echo "❌ STAGE2_URL is empty" && exit 1; fi \
 && curl -sSL "$STAGE2_URL" -o models/panns_cnn14_checklist_best_aug.pth

# 3) Copy your app code
COPY . .

# 4) Tell Docker which port your Flask/Gunicorn listens on
EXPOSE 5050

# 5) Launch under Gunicorn + Gevent
CMD ["gunicorn", "-k", "gevent", "-w", "4", "-b", "0.0.0.0:5050", "app:app"]
