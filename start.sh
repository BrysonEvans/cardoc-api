#!/usr/bin/env bash
set -e

echo "📁 Ensuring models directory…"
mkdir -p models

echo "⏬ Downloading STAGE1 model…"
curl -fSL -o models/stage1_engine_detector.pth "${STAGE1_URL}"

echo "⏬ Downloading STAGE2 model…"
curl -fSL -o models/panns_cnn14_checklist_best_aug.pth "${STAGE2_URL}"

echo "🚀 Starting Gunicorn…"
exec gunicorn -k gevent -w 4 -b 0.0.0.0:"${PORT:-5050}" app:app
