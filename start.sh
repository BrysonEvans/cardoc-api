#!/usr/bin/env bash
set -euo pipefail

echo "📁 Ensuring models directory…"
mkdir -p models

if [[ -z "${STAGE1_URL:-}" ]]; then
  echo "❌ STAGE1_URL unset" >&2
  exit 1
fi
echo "⏬ Downloading STAGE1 model…"
curl -fsSL "$STAGE1_URL" -o models/stage1_engine_detector.pth

if [[ -z "${STAGE2_URL:-}" ]]; then
  echo "❌ STAGE2_URL unset" >&2
  exit 1
fi
echo "⏬ Downloading STAGE2 model…"
curl -fsSL "$STAGE2_URL" -o models/panns_cnn14_checklist_best_aug.pth

echo "🚀 Starting Gunicorn…"
exec gunicorn -k gevent -w 4 -b "0.0.0.0:${PORT:-5050}" app:app
