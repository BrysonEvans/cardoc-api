# Use the exact 3.11.5 slim image so it matches your local env
FROM python:3.11.5-slim

# Don’t write .pyc files, buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# At build time, fetch both models from your presigned URLs
# (Render will inject STAGE1_URL and STAGE2_URL from your Env settings)
RUN mkdir -p models && \
    curl -fSL "$STAGE1_URL" -o models/stage1_engine_detector.pth && \
    curl -fSL "$STAGE2_URL" -o models/panns_cnn14_checklist_best_aug.pth

# Launch under Gunicorn + gevent on $PORT
CMD ["gunicorn", "-k", "gevent", "-w", "4", "-b", "0.0.0.0:$PORT", "app:app"]
