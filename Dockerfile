# Dockerfile
FROM python:3.11-slim

ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Install system deps (gcc, make, curl, ffmpeg, plus autoconf tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      ffmpeg \
      wget \
      libbz2-dev \
      liblzma-dev \
      libssl-dev \
      libffi-dev \
      autoconf \
      automake \
      libtool && \
    rm -rf /var/lib/apt/lists/*

# 2) Build TA-Lib from source
RUN wget https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -O /tmp/ta-lib.tar.gz && \
    mkdir -p /tmp/ta-lib && \
    tar -xzf /tmp/ta-lib.tar.gz -C /tmp/ta-lib --strip-components=1 && \
    cd /tmp/ta-lib && \
      ./configure --prefix=/usr && \
      make && \
      make install && \
    cd / && \
    rm -rf /tmp/ta-lib /tmp/ta-lib.tar.gz

# 3) Copy & install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy your code
COPY . .

# 5) Run your app
CMD ["python", "main.py"]
