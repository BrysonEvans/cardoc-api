FROM python:3.11.5-slim

WORKDIR /app

# 1) System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential python3-dev ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# 2) Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy code + start script
COPY . .
COPY start.sh /start.sh
RUN chmod +x /start.sh

# 4) Entry point
CMD ["/start.sh"]
