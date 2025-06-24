#!/bin/sh
set -e

# 1) Make sure models folder exists
mkdir -p /app/models

# 2) Download via Python (no shell‐quoting issues)
python - <<'PYCODE'
import os, sys
from urllib import request

for fname, url in [
    ("stage1_engine_detector.pth", os.environ.get("STAGE1_URL")),
    ("panns_cnn14_checklist_best_aug.pth", os.environ.get("STAGE2_URL"))
]:
    if not url:
        sys.exit(f"❌ {fname}: missing URL in env")
    out = f"/app/models/{fname}"
    print("Downloading", fname, "…")
    request.urlretrieve(url, out)
    print("✅", fname, "saved")
PYCODE

# 3) Launch Gunicorn on the port Render provides (fallback to 5050)
PORT=${PORT:-5050}
exec gunicorn -k gevent -w 4 -b 0.0.0.0:"$PORT" app:app
