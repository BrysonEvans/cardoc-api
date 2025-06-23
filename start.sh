#!/usr/bin/env sh
set -e

echo "📁 Ensuring models directory…"
mkdir -p models

# ========== Stage1 ==========
echo "⏬ Downloading Stage1 model from:"
echo "   $STAGE1_URL"

if ! curl -fSL -- "$STAGE1_URL" -o "models/stage1_engine_detector.pth"; then
  echo "❌ ERROR: Stage1 download failed (curl exit $?)"
  exit 1
else
  echo "✅ Stage1 model downloaded"
fi

# ========== Stage2 ==========
echo "⏬ Downloading Stage2 model from:"
echo "   $STAGE2_URL"

if ! curl -fSL -- "$STAGE2_URL" -o "models/panns_cnn14_checklist_best_aug.pth"; then
  echo "❌ ERROR: Stage2 download failed (curl exit $?)"
  exit 1
else
  echo "✅ Stage2 model downloaded"
fi

echo "🚀 Starting Gunicorn…"
exec gunicorn app:app \
  --worker-class gevent \
  --workers 4 \
  --bind 0.0.0.0:$PORT
