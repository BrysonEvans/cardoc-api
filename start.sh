#!/bin/sh
set -e

# 1) Create weights folder
mkdir -p /app/weights

# 2) Download the .pth files into /app/weights via Python
python - <<'PYCODE'
import os, sys
from urllib import request

targets = [
  ("stage1_engine_detector.pth",  os.environ.get("STAGE1_URL")),
  ("panns_cnn14_checklist_best_aug.pth", os.environ.get("STAGE2_URL")),
]
for fname, url in targets:
    if not url:
        sys.exit(f"❌ {fname}: missing URL in environment")
    out = f"/app/weights/{fname}"
    print("Downloading", fname, "…")
    request.urlretrieve(url, out)
    print("✅", fname, "saved")
PYCODE

# 3) Launch Gunicorn on $PORT (fallback to 5050)
PORT=${PORT:-5050}
exec gunicorn -k gevent -w 4 -b 0.0.0.0:"$PORT" app:app
