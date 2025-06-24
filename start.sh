#!/usr/bin/env bash
set -euo pipefail

echo "📁 Ensuring weights directory…"
mkdir -p weights

echo "⏬ Downloading STAGE1 model…"
curl -fSL --retry 3 "$STAGE1_URL" -o weights/stage1_engine_detector.pth

echo "⏬ Downloading STAGE2 model…"
curl -fSL --retry 3 "$STAGE2_URL" -o weights/panns_cnn14_checklist_best_aug.pth

echo "🚀 Starting Gunicorn…"
exec gunicorn -k gevent -w 4 -b 0.0.0.0:"${PORT}" app:app
