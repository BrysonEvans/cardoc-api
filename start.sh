#!/usr/bin/env bash
set -euo pipefail

echo "📁 Ensuring models directory…"
mkdir -p models

# both must be set in your Render env
: "${STAGE1_URL:?STAGE1_URL must be set}"
: "${STAGE2_URL:?STAGE2_URL must be set}"

echo "⏬ Downloading STAGE1 model…"
python download_models.py --url "$STAGE1_URL" \
    --out models/stage1_engine_detector.pth

echo "⏬ Downloading STAGE2 model…"
python download_models.py --url "$STAGE2_URL" \
    --out models/panns_cnn14_checklist_best_aug.pth

echo "🚀 Starting Gunicorn…"
exec gunicorn -k gevent -w 4 \
    -b "0.0.0.0:${PORT:-10000}" \
    app:app
