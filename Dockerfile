# Dockerfile — CarDoc AI
FROM python:3.11.5-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 0) Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 1) Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy everything (app + entrypoint)
COPY . .

# 3) Expose (optional—Render detects $PORT)
EXPOSE 5050

# 4) Run our startup script
ENTRYPOINT ["./start.sh"]
