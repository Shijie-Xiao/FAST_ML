#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch and deploy the complete prepared training data (820 storms, 2003-2024).

The archive is hosted on Google Drive as sequential split tar.gz parts
(~75 GB total). Each part has its own manifest entry (group
``training_full``) with size and SHA-256, so a truncated or corrupted part is
caught before the merge. After all parts verify, they are concatenated,
extracted to ``training/training_data/`` and removed.

Usage::

    python scripts/download_training_data.py              # fetch + deploy
    python scripts/download_training_data.py --verify     # what is present
    python scripts/download_training_data.py --keep-parts # keep the tar.gz

Requires ``gdown`` for Google Drive downloads (same as download_data.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"
GROUP = "training_full"
PARTS_DIR = REPO_ROOT / "data" / "training_archive"
DEPLOY_DIR = REPO_ROOT / "training" / "training_data"
_CHUNK = 1 << 20


def _human(n):
    v = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if v < 1024 or unit == "GB":
            return f"{v:.1f} {unit}"
        v /= 1024
    return f"{v:.1f} TB"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(_CHUNK), b""):
            h.update(blk)
    return h.hexdigest()


def load_entries():
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    entries = sorted((e for e in manifest["files"] if e.get("group") == GROUP),
                     key=lambda e: e["target"])
    if not entries:
        sys.exit(f"No '{GROUP}' entries in {MANIFEST_PATH}")
    return entries


def check_part(entry, verbose=True):
    """ok / missing / wrong_size / wrong_hash for one split part."""
    path = PARTS_DIR / Path(entry["target"]).name
    if not path.exists():
        if verbose:
            print(f"  missing    {path.name}  ({_human(entry.get('size', 0))})")
        return "missing"
    if entry.get("size") and path.stat().st_size != entry["size"]:
        if verbose:
            print(f"  wrong size {path.name}  ({_human(path.stat().st_size)} "
                  f"vs {_human(entry['size'])})")
        return "wrong_size"
    if entry.get("sha256") and _sha256(path) != entry["sha256"]:
        if verbose:
            print(f"  CORRUPT    {path.name}  (SHA-256 mismatch)")
        return "wrong_hash"
    if verbose:
        print(f"  ok         {path.name}  ({_human(path.stat().st_size)})")
    return "ok"


def download_part(entry):
    path = PARTS_DIR / Path(entry["target"]).name
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    drive_id = entry.get("drive_id")
    url = entry.get("url")
    if drive_id:
        try:
            import gdown
        except ImportError:
            sys.exit(f"{path.name}: hosted on Google Drive; pip install gdown")
        gdown.download(id=drive_id, output=str(path), quiet=False)
    elif url:
        print(f"  fetching {url}")
        subprocess.run(["curl", "-fL", "-o", str(path), url], check=True)
    else:
        sys.exit(f"{path.name}: no drive_id/url in the manifest yet "
                 f"(upload pending; see the Drive folder link in the README)")
    status = check_part(entry, verbose=False)
    if status != "ok":
        path.unlink(missing_ok=True)
        sys.exit(f"{path.name}: {status} after download; removed")
    print(f"  done       {path.name}")


def merge_and_extract(entries, keep_parts=False):
    parts = [PARTS_DIR / Path(e["target"]).name for e in entries]
    merged = PARTS_DIR / (parts[0].name.rsplit(".tar.gz.", 1)[0] + ".tar.gz")
    print(f"==> merging {len(parts)} parts -> {merged.name}")
    with open(merged, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out, _CHUNK)

    print(f"==> extracting -> {DEPLOY_DIR}")
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(merged) as tar:
        tar.extractall(DEPLOY_DIR, filter="data")

    years = sorted(d.name for d in DEPLOY_DIR.iterdir()
                   if d.is_dir() and d.name.isdigit())
    n_pkls = sum(1 for _ in DEPLOY_DIR.glob("*/*/*_dataset.pkl"))
    print(f"    {len(years)} years ({years[0]}..{years[-1]}), "
          f"{n_pkls} storm dataset.pkls")

    if keep_parts:
        print(f"    parts + tar kept in {PARTS_DIR}")
    else:
        for p in parts:
            p.unlink()
        merged.unlink()
        print(f"    removed parts + tar ({PARTS_DIR} cleaned)")
    print("==> deploy complete: training/SciML_Fast_TwoStream.py is ready to run")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="check parts present, download nothing")
    ap.add_argument("--keep-parts", action="store_true",
                    help="keep the merged tar.gz after extraction")
    args = ap.parse_args()

    entries = load_entries()
    total = sum(e.get("size", 0) for e in entries)
    print(f"Complete training data: {len(entries)} parts, "
          f"~{_human(total)} compressed, ~104 GB unpacked\n")

    if args.verify:
        for e in entries:
            check_part(e)
        return 0

    for e in entries:
        if check_part(e, verbose=False) != "ok":
            download_part(e)
    merge_and_extract(entries, keep_parts=args.keep_parts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
