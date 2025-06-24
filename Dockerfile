# ------------ 1) Base image ----------------------------------------------------
FROM python:3.11.5-slim

# ------------ 2) System packages ----------------------------------------------
# ffmpeg   : we resample / mono-convert uploads
# curl     : start.sh pulls model weights at runtime
# libsndfile1 : required by python-soundfile (librosa backend)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# ------------ 3) Copy & install Python deps ------------------------------------
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------ 4) Copy application code -----------------------------------------
COPY . .
RUN chmod +x /start.sh          # make the entry script executable

# ------------ 5) Runtime configuration -----------------------------------------
ENV PYTHONUNBUFFERED=1
CMD ["/start.sh"]
