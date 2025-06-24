#!/usr/bin/env bash
set -e

# ensure weights dir
mkdir -p weights

# download only if missing
if [ ! -f weights/stage1_engine_detector.pth ]; then
  curl -fsSL "$STAGE1_URL" -o weights/stage1_engine_detector.pth
fi
if [ ! -f weights/panns_cnn14_checklist_best_aug.pth ]; then
  curl -fsSL "$STAGE2_URL" -o weights/panns_cnn14_checklist_best_aug.pth
fi

# launch under Gunicorn + Gevent on $PORT
exec gunicorn -k gevent -w 4 -b 0.0.0.0:$PORT app:app
