# Dockerfile — CarDoc AI
FROM python:3.11.5-slim

# 1) Never prompt & no .pyc, no pip version checks
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 2) System deps: gcc, make, TA-Lib headers, ffmpeg, curl
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libta-lib-dev \
      ffmpeg \
      curl && \
    rm -rf /var/lib/apt/lists/*

# 3) Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy app + helper script
COPY . .
COPY start.sh .
RUN chmod +x start.sh

# 5) Expose & run
EXPOSE 5050
CMD ["./start.sh"]
