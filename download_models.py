#!/usr/bin/env python3
"""
download_models.py  – executed at each Render deploy

▸ If the two weight files aren't yet on the persistent disk
  (/var/models/weights) they’re fetched from the *GitHub-release*
  you published (unlimited lifetime, 2 GiB per asset).

▸ If the disk already contains the files they’re skipped, so
  subsequent deploys are fast and bandwidth-free.
"""
from pathlib import Path
from urllib.request import urlopen, Request
import sys, shutil

# ── target directory on the mounted Render disk ───────────────────────
W_DIR = Path("/var/models/weights")

# ── permanent GitHub-release asset URLs (public) ──────────────────────
FILES = {
    "stage1_engine_detector.pth":
        "https://github.com/BrysonEvans/cardoc-api/releases/download/weights-v1/stage1_engine_detector.pth",
    "panns_cnn14_checklist_best_aug.pth":
        "https://github.com/BrysonEvans/cardoc-api/releases/download/weights-v1/panns_cnn14_checklist_best_aug.pth",
}

# helpers --------------------------------------------------------------
def download(url: str, dst: Path) -> None:
    """stream-download *url* → *dst* with a curl-like UA string"""
    req = Request(url, headers={"User-Agent": "curl/7.68.0"})
    try:
        with urlopen(req) as resp, open(dst, "wb") as f:
            shutil.copyfileobj(resp, f, length=1 << 20)  # 1 MiB chunks
    except Exception as e:
        sys.exit(f"❌ download failed for {dst.name}: {e}")

def main() -> None:
    W_DIR.mkdir(parents=True, exist_ok=True)

    for fname, url in FILES.items():
        path = W_DIR / fname
        if path.exists():
            print(f"✓ {fname} already present – skipping")
            continue

        print(f"⬇️  downloading {fname} …", flush=True)
        download(url, path)
        size_mb = path.stat().st_size / 1_048_576
        print(f"✅  {fname} ready ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
