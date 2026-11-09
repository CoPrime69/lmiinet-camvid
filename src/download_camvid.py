"""Download the 11-class CamVid split (367/101/233, 360x480) used by LMIINet.

Source: https://github.com/alexgkendall/SegNet-Tutorial (folder CamVid/).
Tries a sparse git clone first (only the CamVid folder), then a zip download.

    python -m src.download_camvid            # -> data/CamVid
    python -m src.download_camvid --out D:/datasets/CamVid
"""
import argparse
import io
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

REPO = "https://github.com/alexgkendall/SegNet-Tutorial.git"
ZIP = "https://github.com/alexgkendall/SegNet-Tutorial/archive/refs/heads/master.zip"
SPLITS = ["train", "trainannot", "val", "valannot", "test", "testannot"]
EXPECTED = {"train": 367, "val": 101, "test": 233}


def ok(root):
    return all(os.path.isdir(os.path.join(root, s)) for s in SPLITS)


def via_git(out):
    if shutil.which("git") is None:
        return False
    tmp = tempfile.mkdtemp()
    try:
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, tmp], check=True)
        subprocess.run(["git", "-C", tmp, "sparse-checkout", "set", "CamVid"], check=True)
        src = os.path.join(tmp, "CamVid")
        for s in SPLITS:
            shutil.copytree(os.path.join(src, s), os.path.join(out, s), dirs_exist_ok=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"git route failed ({e}); falling back to zip")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def via_zip(out):
    print(f"downloading {ZIP} (this is the whole tutorial repo, a few hundred MB)...")
    with urllib.request.urlopen(ZIP) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for member in z.namelist():
            parts = member.split("/")
            if len(parts) >= 4 and parts[1] == "CamVid" and parts[2] in SPLITS and parts[3]:
                dst = os.path.join(out, parts[2], parts[3])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with z.open(member) as fsrc, open(dst, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)
    return ok(out)


def ensure_camvid(out="data/CamVid", log=print):
    os.makedirs(out, exist_ok=True)
    if ok(out):
        log(f"CamVid already present at {out}")
    elif not (via_git(out) or via_zip(out)):
        raise SystemExit("Download failed. Get CamVid/ from the SegNet-Tutorial repo manually and place "
                         f"{', '.join(SPLITS)} under {out}")
    for s, n in EXPECTED.items():
        got = len([f for f in os.listdir(os.path.join(out, s)) if f.endswith(".png")])
        log(f"  {s:<5s}: {got} images (expected {n})" + ("" if got == n else "  MISMATCH"))
        if got != n:
            raise SystemExit("CamVid split sizes do not match the SegNet split")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/CamVid")
    ensure_camvid(p.parse_args().out)


if __name__ == "__main__":
    main()
