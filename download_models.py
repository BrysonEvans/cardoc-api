#!/usr/bin/env python3
"""
download_models.py – pulls model weights to /var/models/weights

• Downloads each file only if it’s missing
• Uses permanent GitHub-Release URLs (no expiry)
• Works in both build- and runtime containers
"""
from pathlib import Path
from urllib.request import urlopen, Request
import sys, shutil

# ── destination directory on the persistent Render disk ──
W_DIR = Path("/var/models/weights")          # <── keep in-sync with render.yaml

FILES = {
    "stage1_engine_detector.pth":
        "https://github.com/BrysonEvans/cardoc-api/releases/download/weights-v1/stage1_engine_detector.pth",
    "panns_cnn14_checklist_best_aug.pth":
        "https://github.com/BrysonEvans/cardoc-api/releases/download/weights-v1/panns_cnn14_checklist_best_aug.pth",
}

def fetch(url: str, dst: Path) -> None:
    req = Request(url, headers={"User-Agent": "curl/8.0"})
    try:
        with urlopen(req) as resp, open(dst, "wb") as f:
            shutil.copyfileobj(resp, f)
    except Exception as exc:
        sys.stderr.write(f"❌ download failed for {dst.name}: {exc}\n")
        sys.exit(1)

def main() -> None:
    W_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dst = W_DIR / name
        if dst.exists():
            print(f"✓ {name} already present – skipping")
            continue
        print(f"⬇️  downloading {name} …", flush=True)
        fetch(url, dst)
        print(f"✅  {name} ready ({dst.stat().st_size/1_048_576:.1f} MB)")

if __name__ == "__main__":
    main()
