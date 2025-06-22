# ---------- base image ----------
FROM python:3.11-slim AS base

# don’t write .pyc files, buffer logs, and skip pip version warnings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app

# this URL will be provided via Render’s env vars (or default to S3)
ARG STAGE2_URL=https://cardoc-models-us-east-2.s3.us-east-2.amazonaws.com/models/panns_cnn14_checklist_best_aug.pth

# ---------- dependencies ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pull down the stage-2 model into models/ at build time
RUN mkdir -p models \
 && curl -fSL "$STAGE2_URL" -o models/panns_cnn14_checklist_best_aug.pth

# ---------- project files ----------
COPY . .

# ---------- launch command ----------
CMD ["gunicorn", "-k", "gevent", "-w", "4", "-b", "0.0.0.0:$PORT", "app:app"]
