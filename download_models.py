import argparse
import subprocess
import sys

def download(url: str, out: str):
    subprocess.check_call(["curl", "-fSL", url, "-o", out])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="presigned S3 URL")
    p.add_argument("--out", required=True, help="output file path")
    args = p.parse_args()
    download(args.url, args.out)
