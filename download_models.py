#!/usr/bin/env python3
"""
download_models.py  –  runs at container start

• Downloads each weight file only if it’s missing
• Uses permanent GitHub-Release URLs (no expiry, 2 GB/file limit)
• Writes to /var/models/weights so the persistent disk caches them
"""
from pathlib import Path
from urllib.request import urlopen, Request
import sys

# Destination directory (created on the attached Render disk)
W_DIR = Path("/var/models/weights")

# Permanent asset URLs from the weights-v1 GitHub release
FILES = {
    "stage1_engine_detector.pth":
        "https://github.com/BrysonEvans/cardoc-api/releases/download/weights-v1/stage1_engine_detector.pth",
    "panns_cnn14_checklist_best_aug.pth":
        "https://github.com/BrysonEvans/cardoc-api/releases/download/weights-v1/panns_cnn14_checklist_best_aug.pth",
}

def fetch(url: str, dst: Path) -> None:
    req = Request(url, headers={"User-Agent": "curl/7.68.0"})
    try:
        with urlopen(req) as resp, open(dst, "wb") as f:
            f.write(resp.read())
    except Exception as e:
        sys.stderr.write(f"❌ download failed for {dst.name}: {e}\n")
        sys.exit(1)

def main() -> None:
    for name, url in FILES.items():
        dst = W_DIR / name
        if dst.exists():
            print(f"✓ {name} already present – skipping")
            continue

        W_DIR.mkdir(parents=True, exist_ok=True)
        print(f"⬇️  downloading {name} …", flush=True)
        fetch(url, dst)
        print(f"✅  {name} ready ({dst.stat().st_size/1_048_576:.1f} MB)")

if __name__ == "__main__":
    main()
