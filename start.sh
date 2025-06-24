#!/usr/bin/env bash
set -euo pipefail

echo "→ STAGE1_URL is: [$STAGE1_URL]"
echo "→ STAGE2_URL is: [$STAGE2_URL]"

mkdir -p /app/weights

# the critical change is the quotes around "$STAGE?_URL"
curl -fSL "$STAGE1_URL" -o /app/weights/stage1_engine_detector.pth \
  || { echo "❌ giving up on $STAGE1_URL"; exit 1; }
echo "✅ stage1_engine_detector.pth saved"

curl -fSL "$STAGE2_URL" -o /app/weights/panns_cnn14_checklist_best_aug.pth \
  || { echo "❌ giving up on $STAGE2_URL"; exit 1; }
echo "✅ panns_cnn14_checklist_best_aug.pth saved"

exec gunicorn -k gevent -w 4 -b 0.0.0.0:"${PORT:-5050}" app:app
