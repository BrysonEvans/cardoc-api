#!/usr/bin/env bash
set -euo pipefail            # safer bash

# ────────── 0. sanity-check env ──────────
: "${STAGE1_URL:?❌ STAGE1_URL is empty}"
: "${STAGE2_URL:?❌ STAGE2_URL is empty}"
echo "STAGE1_URL=$STAGE1_URL"
echo "STAGE2_URL=$STAGE2_URL"

# ────────── 1. download weights ──────────
mkdir -p weights

echo "⏬  Downloading STAGE1 model…"
curl -fSL "$STAGE1_URL" -o weights/stage1_engine_detector.pth

echo "⏬  Downloading STAGE2 model…"
curl -fSL "$STAGE2_URL" -o weights/panns_cnn14_checklist_best_aug.pth

# ────────── 2. launch Gunicorn ──────────
echo "🚀  Starting Gunicorn…"
exec gunicorn -k gevent -w 4 -b "0.0.0.0:${PORT:-10000}" app:app
