#!/usr/bin/env python3
import argparse
import sys
from urllib.request import urlopen, Request

def main():
    p = argparse.ArgumentParser(
        description="Fetch a presigned URL into a local file"
    )
    p.add_argument("--url",  required=True, help="Presigned S3 URL")
    p.add_argument("--out",  required=True, help="Local output path")
    args = p.parse_args()

    req = Request(args.url, headers={"User-Agent": "curl/7.68.0"})
    try:
        with urlopen(req) as resp, open(args.out, "wb") as f:
            f.write(resp.read())
    except Exception as e:
        sys.stderr.write(f"❌ Download failed: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
