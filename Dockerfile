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

# ───────── Python deps (layer cached) ─────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ───────── Application source ─────────
COPY . .

# copy entry script separately & make it executable
COPY start.sh /start.sh
RUN chmod +x /start.sh

# ───────── Runtime env ─────────
ENV PYTHONUNBUFFERED=1 \
    PORT=10000               # Render will override with $PORT

EXPOSE 10000

# ───────── Entrypoint ─────────
ENTRYPOINT ["/start.sh"]
