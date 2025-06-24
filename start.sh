#!/usr/bin/env bash
set -euo pipefail

# ───────── ensure weight directory ─────────
mkdir -p weights

download() {
  local name="$1" url="$2" dst="$3"
  echo "⏬  Downloading ${name} …"
  for i in {1..3}; do
      if curl -fL --progress-bar "${url}" -o "${dst}"; then
          echo "✅  ${name} saved"
          return 0
      fi
      echo "⚠️   attempt ${i} failed for ${url}"
      sleep 2
  done
  echo "❌  giving up on ${url}"
  exit 1
}

# URLs must be supplied via Render “Environment Variables”
download "STAGE1 model" "${STAGE1_URL}" "weights/stage1_engine_detector.pth"
download "STAGE2 model" "${STAGE2_URL}" "weights/panns_cnn14_checklist_best_aug.pth"

# ───────── launch server ─────────
echo "🚀  Starting Gunicorn…"
exec gunicorn -k gevent -w 4 -b 0.0.0.0:"${PORT:-10000}" app:app
