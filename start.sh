#!/usr/bin/env sh
set -e

echo "📁 Ensuring models directory…"
mkdir -p models

# Download Stage1 model if it's not already present
if [ ! -f models/stage1_engine_detector.pth ]; then
  echo "⏬ Downloading Stage1 model from $STAGE1_URL"
  curl -fsSL "$STAGE1_URL" -o models/stage1_engine_detector.pth
else
  echo "✅ Stage1 model already present"
fi

# Download Stage2 model if it's not already present
if [ ! -f models/panns_cnn14_checklist_best_aug.pth ]; then
  echo "⏬ Downloading Stage2 model from $STAGE2_URL"
  curl -fsSL "$STAGE2_URL" -o models/panns_cnn14_checklist_best_aug.pth
else
  echo "✅ Stage2 model already present"
fi

echo "🚀 Starting Gunicorn…"
exec gunicorn app:app \
  --worker-class gevent \
  --workers 4 \
  --bind 0.0.0.0:$PORT
