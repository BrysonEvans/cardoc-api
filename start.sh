#!/usr/bin/env sh
set -e

echo "📁 Ensuring models directory…"
mkdir -p models

# Clean and show Stage1 URL
URL1=$(printf '%s' "$STAGE1_URL" | tr -d '\r\n')
echo "Raw Stage1 URL (trimmed) = |$URL1|"

echo "⏬ Fetching Stage1 model…"
curl -fSL -o models/stage1_engine_detector.pth "$URL1" \
  || { echo "❌ Stage1 download failed"; exit 1; }
echo "✅ Stage1 downloaded"

# Clean and show Stage2 URL
URL2=$(printf '%s' "$STAGE2_URL" | tr -d '\r\n')
echo "Raw Stage2 URL (trimmed) = |$URL2|"

echo "⏬ Fetching Stage2 model…"
curl -fSL -o models/panns_cnn14_checklist_best_aug.pth "$URL2" \
  || { echo "❌ Stage2 download failed"; exit 1; }
echo "✅ Stage2 downloaded"

echo "🚀 Starting Gunicorn…"
exec gunicorn app:app \
  --worker-class gevent \
  --workers 4 \
  --bind 0.0.0.0:"$PORT"
