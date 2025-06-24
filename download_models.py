import argparse, subprocess, sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url",  required=True, help="S3 presigned URL")
    p.add_argument("--out",  required=True, help="where to save the file")
    args = p.parse_args()

    try:
        subprocess.check_call(["curl", "-fSL", args.url, "-o", args.out])
    except Exception as e:
        print(f"❌ Failed to download {args.url}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

