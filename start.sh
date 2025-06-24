#!/usr/bin/env bash
set -euo pipefail

# remove stray new-lines / spaces from the two URLs
STAGE1_URL="$(printf %s "$STAGE1_URL" | tr -d '\r\n[:space:]')"
STAGE2_URL="$(printf %s "$STAGE2_URL" | tr -d '\r\n[:space:]')"

mkdir -p weights

echo "⏬  Downloading STAGE1 model…"
curl -fSL "$STAGE1_URL" -o weights/stage1_engine_detector.pth

echo "⏬  Downloading STAGE2 model…"
curl -fSL "$STAGE2_URL" -o weights/panns_cnn14_checklist_best_aug.pth

echo "🚀  Starting Gunicorn…"
exec gunicorn -k gevent -w 4 -b "0.0.0.0:${PORT:-10000}" app:app
