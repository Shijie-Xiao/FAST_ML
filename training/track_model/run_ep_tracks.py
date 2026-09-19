#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-1 ensemble synthetic tracks for East-Pacific (EP) storms.

The stock pipeline (config.py) auto-discovers only NA storms. This driver runs
the 4 stages for a user-given list of EP storms (data/EP/{year}/<IBID>_<NAME>),
against an ISOLATED processed dir per run, so the ~47 already-processed NA storms
are never touched (their per-dir loops would otherwise re-run all of them).

Usage:
  python run_ep_tracks.py --year 2024 --names GILMA JOHN KRISTY ...
  python run_ep_tracks.py --year 2024 --all        # all named EP storms w/ best track

Output: ensemble_tracks/{year}/{storm_lower}/synthetic_tracks_1000members.nc
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import config
import read_files
import build_pairs
import train_markov
import sample_tracks

PROJECT_ROOT = config.PROJECT_ROOT


def _name_from_dir(dirname):
    import re
    m = re.match(r"^\d{4}\d+[NS]\d+_(.+)$", dirname)
    if not m:
        return None
    nm = m.group(1).strip().upper()
    if nm.startswith("UNNAMED") or nm.startswith("STORM"):
        return None
    return nm


def _genesis(csv):
    with open(csv) as f:
        f.readline()
        first = f.readline().strip().split(",")[0]
    return datetime.strptime(first, "%Y-%m-%d %H:%M:%S")


def discover_ep(year, names=None):
    """Build storm_cfg dicts from data/EP/{year}. names=None -> all named."""
    ep_dir = PROJECT_ROOT / "data" / "EP" / str(year)
    out = []
    for sdir in sorted(ep_dir.iterdir()):
        if not sdir.is_dir():
            continue
        nm = _name_from_dir(sdir.name)
        if not nm:
            continue
        if names and nm not in names:
            continue
        csv = sdir / "track_intensity_6h.csv"
        if not csv.exists():
            continue
        out.append({
            "storm_name": nm,
            "ibtracs_id": sdir.name,
            "storm_dir": str(sdir),
            "year": year,
            "basin": "EP",
            "genesis": _genesis(csv),
        })
    return out


def run_one(cfg):
    """Run the 4 stages for a single storm in an isolated processed dir."""
    storm = cfg["storm_name"]
    year = cfg["year"]
    proc = HERE / f"processed_ep_{storm.lower()}"
    proc.mkdir(parents=True, exist_ok=True)
    synth = PROJECT_ROOT / "ensemble_tracks" / str(year)
    synth.mkdir(parents=True, exist_ok=True)
    for mod in (read_files, build_pairs, train_markov, sample_tracks):
        if hasattr(mod, "PROCESSED_TRACKS_DIR"):
            mod.PROCESSED_TRACKS_DIR = proc
    sample_tracks.SYNTH_TRACKS_DIR = synth

    print(f"\n===== {storm} ({year}) genesis {cfg['genesis']:%Y-%m-%d %Hz} =====", flush=True)
    got = read_files.run_read_tracks(storms=[cfg])
    if not got:
        print(f"  [SKIP] {storm}: no ECMWF ensemble", flush=True)
        return None
    build_pairs.run_build_pairs()
    train_markov.run_train_markov()
    sample_tracks.run_sample_tracks()
    ncs = sorted(synth.glob(f"{storm.lower()}/synthetic_tracks_*members.nc"))
    return str(ncs[0]) if ncs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    names = None if args.all else ([n.upper() for n in args.names] if args.names else None)
    storms = discover_ep(args.year, names)
    print(f"=== EP track pipeline: {len(storms)} storms ({args.year}) ===", flush=True)
    done, skipped = [], []
    for cfg in storms:
        nc = run_one(cfg)
        (done if nc else skipped).append(cfg["storm_name"])
    print(f"\n=== DONE: {len(done)} tracks: {done}", flush=True)
    if skipped:
        print(f"=== SKIPPED: {skipped}", flush=True)


if __name__ == "__main__":
    main()
