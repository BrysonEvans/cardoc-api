# ---------- base image ----------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /app

# ---------- dependencies ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- project ----------
COPY . .

# ---------- launch ----------
CMD ["gunicorn", "-k", "gevent", "-w", "4", "-b", "0.0.0.0:$PORT", "app:app"]
