# Use slim so the image stays small
FROM python:3.11-slim

# Don’t write .pyc files, buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1) Copy & install Python deps
COPY requirements.txt .
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg curl \
 && rm -rf /var/lib/apt/lists/* \
 && pip install .

# 2) Copy the rest of your code + helper
COPY . .

# 3) Make start script executable
COPY start.sh /start.sh
RUN chmod +x /start.sh

# 4) Bind to the PORT Render gives us
EXPOSE 10000

ENTRYPOINT ["/start.sh"]
