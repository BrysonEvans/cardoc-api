#!/usr/bin/env bash
set -euo pipefail

echo "📁  Ensuring weights directory…"
mkdir -p /app/weights

download() {
  local url="$1"
  local dest="$2"

  if [[ -z "$url" ]]; then
    echo "❌  Environment variable for $(basename "$dest") not set"; exit 1
  fi

  echo "⏬  Downloading $(basename "$dest")…"
  curl -fL --retry 3 --retry-delay 2 "$url" -o "$dest"
}

download "${STAGE1_URL:-}" "/app/weights/stage1_engine_detector.pth"
download "${STAGE2_URL:-}" "/app/weights/panns_cnn14_checklist_best_aug.pth"

echo "🚀  Starting Gunicorn…"
exec gunicorn -k gevent -w 4 -b 0.0.0.0:"${PORT}" app:app
