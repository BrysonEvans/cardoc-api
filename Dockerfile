# ───────── Base image ─────────
FROM python:3.11.5-slim

# ───────── Workdir ─────────
WORKDIR /app

# ───────── OS-level deps ─────────
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg \
      curl \
 && rm -rf /var/lib/apt/lists/*

# ───────── Python deps ─────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ───────── App source ─────────
COPY . .

# copy entry script to PATH root & make executable
COPY start.sh /start.sh
RUN chmod +x /start.sh

# ───────── Runtime env ─────────
ENV PYTHONUNBUFFERED=1 \
    PORT=10000           # (Render will still inject its own $PORT)

EXPOSE 10000

# ───────── Entrypoint ─────────
ENTRYPOINT ["/start.sh"]
