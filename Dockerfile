# Dockerfile — CarDoc AI API
FROM python:3.11-slim

# avoid .pyc, unbuffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1) System deps: build TA-Lib C lib, ffmpeg, curl, python headers, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      wget \
      ca-certificates \
      ffmpeg \
      curl \
      python3-dev \
      libatlas-base-dev && \
    rm -rf /var/lib/apt/lists/*

# 2) Download & compile TA-Lib (0.4.0)
RUN wget -qO- https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    | tar xz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && make install && \
    cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# 3) Copy & install Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# 4) Copy app + helper scripts
COPY . .

# 5) Downloader + entrypoint
COPY download_models.py start.sh /app/
RUN chmod +x /app/start.sh

# 6) Expose & run
EXPOSE 10000
ENTRYPOINT ["./start.sh"]
