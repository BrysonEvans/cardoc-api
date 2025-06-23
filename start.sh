#!/usr/bin/env sh
set -e

echo "📁 Ensuring models directory…"
mkdir -p models

# ========== Stage1 ==========
echo "⏬ Downloading Stage1 model from:"
echo "   $STAGE1_URL"
if curl -fSL -o models/stage1_engine_detector.pth "$STAGE1_URL"; then
  echo "✅ Stage1 model downloaded"
else
  echo "❌ ERROR: Stage1 download failed"
  exit 1
fi

# ========== Stage2 ==========
echo "⏬ Downloading Stage2 model from:"
echo "   $STAGE2_URL"
if curl -fSL -o models/panns_cnn14_checklist_best_aug.pth "$STAGE2_URL"; then
  echo "✅ Stage2 model downloaded"
else
  echo "❌ ERROR: Stage2 download failed"
  exit 1
fi

echo "🚀 Starting Gunicorn…"
exec gunicorn app:app \
  --worker-class gevent \
  --workers 4 \
  --bind 0.0.0.0:"$PORT"
