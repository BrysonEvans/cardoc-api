# Dockerfile — CarDoc AI
FROM python:3.11-slim

# Don’t write .pyc, buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1) Install system deps (including gcc, make, TA-Lib headers, ffmpeg, curl)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libta-lib-dev \
      ffmpeg \
      curl && \
    rm -rf /var/lib/apt/lists/*

# 2) Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy app code
COPY . .

# 4) Expose Flask port
EXPOSE 5050

# 5) Runtime entrypoint
CMD ["gunicorn", "-k", "gevent", "-w", "4", "-b", "0.0.0.0:$PORT", "app:app"]
