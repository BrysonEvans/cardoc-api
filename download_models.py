#!/usr/bin/env python3
import os, sys
from urllib.request import urlopen, Request

MODELS = [
    ("STAGE1_URL", "models/stage1_engine_detector.pth"),
    ("STAGE2_URL", "models/panns_cnn14_checklist_best_aug.pth")
]

os.makedirs("models", exist_ok=True)

for env_var, out_path in MODELS:
    url = os.environ.get(env_var)
    if not url:
        print(f"❌ Missing environment variable {env_var}", file=sys.stderr)
        sys.exit(1)
    print(f"⏬ Downloading {env_var} to {out_path}…")
    try:
        req = Request(url, headers={"User-Agent":"python-urllib"})
        with urlopen(req) as resp, open(out_path, "wb") as f:
            f.write(resp.read())
    except Exception as e:
        print(f"❌ Failed to download {env_var}: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"✅ {env_var} downloaded")
