# 1) Base
FROM python:3.11-slim

# 2) Prevent .pyc, buffer logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 3) System deps (audio libs, build tools, curl)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
       build-essential \
       python3-dev \
       libsndfile1-dev \
       ffmpeg \
       curl \
 && rm -rf /var/lib/apt/lists/*

# 4) Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5) App code + launch script
COPY . .
RUN chmod +x start.sh

# 6) Entrypoint
CMD ["./start.sh"]
