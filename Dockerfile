# Use the exact Python slim image to match your local environment
FROM python:3.11-slim

# Don’t write .pyc files & buffer logs to stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Install only the system packages you need at runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      curl && \
    rm -rf /var/lib/apt/lists/*

# 2) Copy & install your Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy your entire app code
COPY . .

# 4) Copy & make your startup script executable
COPY start.sh /start.sh
RUN chmod +x /start.sh

# 5) At runtime, start.sh will download the models and launch Gunicorn
ENTRYPOINT ["/start.sh"]
