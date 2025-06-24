#!/usr/bin/env sh
set -ex

echo "📁 Ensuring models directory…"
mkdir -p models

# Dump out exactly what Curl will see
echo "Raw STAGE1_URL = |$STAGE1_URL|"

# Attempt Stage1 download
echo "⏬ Curl command:"
echo "  curl -fSL -o models/stage1_engine_detector.pth \"$STAGE1_URL\""
curl -fSL -o models/stage1_engine_detector.pth "$STAGE1_URL"
echo "✅ Stage1 model downloaded"

# Dump Stage2 too
echo "Raw STAGE2_URL = |$STAGE2_URL|"
echo "⏬ Curl command:"
echo "  curl -fSL -o models/panns_cnn14_checklist_best_aug.pth \"$STAGE2_URL\""
curl -fSL -o models/panns_cnn14_checklist_best_aug.pth "$STAGE2_URL"
echo "✅ Stage2 model downloaded"

echo "🚀 Starting Gunicorn…"
exec gunicorn app:app \
  --worker-class gevent \
  --workers 4 \
  --bind 0.0.0.0:"$PORT"
