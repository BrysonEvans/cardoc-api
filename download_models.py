import argparse, sys, subprocess

def download(url: str, out: str):
    # -f: fail on HTTP errors; -S: show error; -L: follow redirects
    subprocess.check_call(["curl", "-fSL", url, "-o", out])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    download(args.url, args.out)
