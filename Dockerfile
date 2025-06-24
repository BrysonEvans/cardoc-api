# Dockerfile — CarDoc AI
FROM python:3.11.5-slim

#  Avoid .pyc, buffer logs, skip pip version check
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1) System deps for ffmpeg (audio) + curl (optional)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# 2) Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy your entire app (models/, weights will be created at runtime)
COPY . .

# 4) Expose your port (Render will still use $PORT)
EXPOSE 5050

# 5) Use our entrypoint to fetch weights then start Gunicorn
ENTRYPOINT ["./start.sh"]
