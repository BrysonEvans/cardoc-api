# Use the exact same image you develop against
FROM python:3.11.5-slim

# 1) Install system deps: build tools + ffmpeg + curl
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      ffmpeg \
      curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Copy & install Python deps (including matplotlib)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy your full backend
COPY . .

# 4) Make start script executable
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Render (and Heroku, etc.) will bind your app to the $PORT env var
EXPOSE $PORT

# 5) At container start, fetch models and launch Gunicorn
CMD ["/start.sh"]
