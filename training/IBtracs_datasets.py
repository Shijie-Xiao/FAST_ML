#!/usr/bin/env python3
"""Download IBTrACS by basin and save raw 6h tracks."""

from pathlib import Path
import argparse
import shutil
import urllib.request

import pandas as pd

BASE_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs"
)
URL_TMPL = BASE_URL + "/{version}/access/csv/ibtracs.{basin}.list.{version}.csv"
DEFAULT_VERSION = "v04r01"
VALID_BASINS = ("NA", "EP")


def load_config(path: Path):
    cfg = {
        "basins": "ALL",
        "year_start": "2003",
        "year_end": "2024",
        "output_dir": "data",
        "clear_first": "false",
        "ibtracs_version": DEFAULT_VERSION,
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = [x.strip() for x in line.split("=", 1)]
        cfg[k.lower()] = v

    basins = [b.strip().upper() for b in cfg["basins"].split(",") if b.strip()]
    basins = list(VALID_BASINS) if "ALL" in basins else list(dict.fromkeys(basins))
    bad = [b for b in basins if b not in VALID_BASINS]
    if bad:
        raise ValueError(f"unsupported basin(s): {bad}")

    return {
        "basins": basins,
        "year_start": int(cfg["year_start"]),
        "year_end": int(cfg["year_end"]),
        "output_dir": cfg["output_dir"],
        "clear_first": cfg["clear_first"].lower() in {"1", "true", "yes", "y"},
        "ibtracs_version": cfg["ibtracs_version"] or DEFAULT_VERSION,
    }


def download_basin_csv(out_root: Path, basin: str, version: str):
    cache = out_root / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / f"ibtracs.{basin}.list.{version}.csv"
    req = urllib.request.Request(
        URL_TMPL.format(version=version, basin=basin),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        dst.write_bytes(r.read())
    return dst


def to_num(s):
    return pd.to_numeric(s.astype(str).str.strip().replace("", pd.NA), errors="coerce")


def clean_track(grp):
    out = pd.DataFrame()
    out["time"] = pd.to_datetime(grp["ISO_TIME"], errors="coerce")
    out["lat"] = to_num(grp["USA_LAT"])
    out["lon"] = to_num(grp["USA_LON"])
    out["vmax"] = to_num(grp["USA_WIND"])
    out["mslp"] = to_num(grp["USA_PRES"])
    out = out.dropna(subset=["time", "lat", "lon"]).sort_values("time").drop_duplicates("time")
    return out.reset_index(drop=True) if len(out) >= 2 else None


def safe_name(name):
    n = str(name).strip().replace(" ", "_")
    return "UNNAMED" if (not n or n.upper() in {"NAN", "UNNAMED"}) else n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.txt")
    args = p.parse_args()
    c = load_config(Path(args.config))

    out_root = Path(c["output_dir"])
    if c["clear_first"] and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    years = set(range(c["year_start"], c["year_end"] + 1))
    v = c["ibtracs_version"]
    for basin in c["basins"]:
        csv_path = download_basin_csv(out_root, basin, v)
        df = pd.read_csv(csv_path, skiprows=[1], low_memory=False)
        df["year"] = pd.to_datetime(df["ISO_TIME"], errors="coerce").dt.year
        df = df[df["year"].isin(years)]
        n_saved = 0
        for sid, grp in df.groupby("SID", sort=False):
            tr = clean_track(grp)
            if tr is None:
                continue
            y = int(grp["year"].iloc[0])
            name = safe_name(grp["NAME"].iloc[0] if "NAME" in grp else sid)
            d = out_root / basin / str(y) / f"{sid}_{name}"
            d.mkdir(parents=True, exist_ok=True)
            tr.to_csv(d / "track_intensity_6h.csv", index=False)
            n_saved += 1
        print(f"[Done] {basin}: saved {n_saved} storms")

if __name__ == "__main__":
    main()
