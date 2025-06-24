#!/usr/bin/env sh
set -e

echo "📁 Ensuring models directory & downloading via Python…"
./download_models.py

echo "🚀 Starting Gunicorn…"
exec gunicorn app:app \
  --worker-class gevent \
  --workers 4 \
  --bind 0.0.0.0:"$PORT"
