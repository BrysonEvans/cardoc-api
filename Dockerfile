# ── Step 0: pick a reproducible base ─────────────────────────────
FROM python:3.11.5-slim

# ── Step 1: env tweaks & workdir ────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ── Step 2: install system deps ─────────────────────────────────
#   build-essential + python3-dev for any wheels, libta-lib-dev for TA-Lib,
#   ffmpeg for audio, curl so we can download at runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      libta-lib-dev \
      libatlas-base-dev \
      ffmpeg \
      curl \
 && rm -rf /var/lib/apt/lists/*

# ── Step 3: install Python deps ─────────────────────────────────
COPY requirements.txt .
RUN pip install -r requirements.txt

# ── Step 4: copy your entire app ────────────────────────────────
COPY . .

# ── Step 5: add startup script ──────────────────────────────────
COPY start.sh /start.sh
RUN chmod +x /start.sh

# ── Step 6: tell Docker (and Render) which port we serve on ─────
EXPOSE 5050

# ── Final: hand off to our script ───────────────────────────────
CMD ["/start.sh"]
