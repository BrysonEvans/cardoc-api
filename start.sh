#!/usr/bin/env sh
set -e

echo "📁 Ensuring models directory…"
mkdir -p models

# Dump the raw URL for sanity
echo "Raw STAGE1_URL = |$STAGE1_URL|"

# Download Stage1
echo "⏬ Fetching Stage1 model…"
curl -fSL -o models/stage1_engine_detector.pth "$STAGE1_URL" \
  || { echo "❌ Stage1 download failed"; exit 1; }
echo "✅ Stage1 downloaded"

# Dump Stage2 URL
echo "Raw STAGE2_URL = |$STAGE2_URL|"

# Download Stage2
echo "⏬ Fetching Stage2 model…"
curl -fSL -o models/panns_cnn14_checklist_best_aug.pth "$STAGE2_URL" \
  || { echo "❌ Stage2 download failed"; exit 1; }
echo "✅ Stage2 downloaded"

echo "🚀 Starting Gunicorn…"
exec gunicorn app:app \
  --worker-class gevent \
  --workers 4 \
  --bind 0.0.0.0:"$PORT"
