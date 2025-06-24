# Use the exact base to match your local env
FROM python:3.11.5-slim

# Don’t write .pyc files, buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1) Install only what we need at runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg \
      curl \
 && rm -rf /var/lib/apt/lists/*

# 2) Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy downloader and startup script
COPY download_models.py start.sh ./
RUN chmod +x download_models.py start.sh

# 4) Ensure the weights directory exists
RUN mkdir -p /app/weights

# 5) Copy the rest of your app
COPY . .

# 6) Expose your app port and launch
EXPOSE 5050
ENTRYPOINT ["./start.sh"]
