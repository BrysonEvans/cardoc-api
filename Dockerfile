# Dockerfile — CarDoc AI API
FROM python:3.11-slim

# Don’t write .pyc files; unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1) Install system deps for ta-lib, ffmpeg, curl
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      libta-lib0-dev \
      ffmpeg \
      curl \
 && rm -rf /var/lib/apt/lists/*

# 2) Copy & install Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) Copy your application code
COPY . .

# 4) Download-helper + startup script
COPY download_models.py start.sh /app/
RUN chmod +x /app/start.sh

# 5) Expose port & set entrypoint
EXPOSE 10000
ENTRYPOINT ["./start.sh"]
