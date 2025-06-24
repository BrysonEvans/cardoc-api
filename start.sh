#!/usr/bin/env bash
set -euo pipefail

echo "📁 Ensuring models directory & downloading via Python…"
mkdir -p models

python3 <<'PYCODE'
import os, urllib.request
for name,url,fn in [
    ("STAGE1", os.getenv("STAGE1_URL"), "models/stage1_engine_detector.pth"),
    ("STAGE2", os.getenv("STAGE2_URL"), "models/panns_cnn14_checklist_best_aug.pth")
]:
    if not url:
        echo "❌ $name URL missing" >&2
        exit(1)
    print(f"⏬ Downloading {name} model…")
    urllib.request.urlretrieve(url, fn)
    print("✅ saved", fn)
PYCODE

echo "🚀 Starting Gunicorn…"
exec gunicorn -k gevent -w 4 -b 0.0.0.0:5050 app:app
