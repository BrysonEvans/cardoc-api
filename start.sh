#!/usr/bin/env bash
set -euo pipefail

# 1) sanity check
if [ -z "${STAGE1_URL-}" ] || [ -z "${STAGE2_URL-}" ]; then
  echo "❌ You must set both STAGE1_URL and STAGE2_URL env vars"
  exit 1
fi

echo "→ STAGE1_URL=$STAGE1_URL"
echo "→ STAGE2_URL=$STAGE2_URL"

# 2) download into weights/
mkdir -p weights

# a tiny wrapper to retry on failure
download() {
  local url=$1 dest=$2
  for i in 1 2 3; do
    if curl --fail --location --retry 3 --retry-delay 2 \
            --output "$dest" \
            "$url"; then
      echo "✅ downloaded $dest"
      return 0
    else
      echo "⚠️  attempt $i failed for $url"
      sleep 1
    fi
  done
  echo "❌ giving up on $url"
  exit 1
}

download "$STAGE1_URL" weights/stage1_engine_detector.pth
download "$STAGE2_URL" weights/panns_cnn14_checklist_best_aug.pth

# 3) launch your app
exec gunicorn -k gevent -w 4 -b "0.0.0.0:${PORT:-5050}" app:app
