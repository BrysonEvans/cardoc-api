# ---------- 1) base image ----------
FROM python:3.11.5-slim AS build

WORKDIR /app

# ---------- 2) system & python deps ----------
# - ffmpeg for audio transcoding
# - curl to fetch the model weights at runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- 3) copy source & entrypoint ----------
COPY . .
COPY start.sh /start.sh     # <── explicit copy
RUN chmod +x /start.sh

# ---------- 4) container start command ----------
ENTRYPOINT ["/start.sh"]
