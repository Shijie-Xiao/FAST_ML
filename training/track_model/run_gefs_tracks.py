#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-1 driver: GEFS-vortex tracks -> Markov sampling -> 1000 synthetic tracks.

Pipeline (identical math to the ECMWF path, only the track SOURCE differs):
  1. gefs_tracks.extract_gefs_tracks: 31 GEFS members -> raw.pkl
     (same dict schema as read_files.read_storm_ensemble)
  2. build_pairs.run_build_pairs      (redirected PROCESSED_TRACKS_DIR)
  3. train_markov.run_train_markov    (per-step conditional Gaussians)
  4. sample_tracks.run_sample_tracks  (Monte Carlo, 1000 members)
       -> ensemble_tracks_gefs/{year}/{storm}/synthetic_tracks_1000members.nc
          with 'parent_member' per track (FHLO member-paired inheritance)

Usage (single storm):
  python run_gefs_tracks.py --case 2026_ELIDA_EP --year 2026
  python run_gefs_tracks.py --case 2026_ELIDA_EP --year 2026 --dry_run
Batch:
  python run_gefs_tracks.py --all            # all GEFS cases 2024-2026
"""
import argparse
import json
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

import config                                    # noqa: E402
import build_pairs                               # noqa: E402
import train_markov                              # noqa: E402
import sample_tracks                             # noqa: E402
from gefs_tracks import extract_gefs_tracks, save_raw  # noqa: E402

import os
GFS_ROOT = Path(os.environ.get("GEFS_GRIB_ROOT", "/global/cfs/cdirs/m5011/Jay/ERA5/GFS"))
SYNTH_ROOT = config.PROJECT_ROOT / "ensemble_tracks_gefs"


def _bt_csv_for(case, year, name, basin):
    """Best-track CSV for a GEFS case: data/{basin}/{year}/{SID}_{NAME}/..."""
    for sub in (config.BEST_TRACK_DIR.parent / basin / str(year)).iterdir():
        if not sub.is_dir() or name.upper() not in sub.name.upper():
            continue
        csv = sub / "track_intensity_6h.csv"
        if csv.exists():
            return csv
    return None


def _parse_case(case):
    """'2026_ELIDA_EP' -> (2026, 'ELIDA', 'EP')"""
    parts = case.split("_")
    if len(parts) != 3:
        return None
    y, nm, b = int(parts[0]), parts[1], parts[2]
    return y, nm, b


def run_one(case, dry_run=False, fhour_max=None):
    case_dir = GFS_ROOT / case
    meta = _parse_case(case)
    if meta is None or not (case_dir / "manifest.json").exists():
        print(f"[SKIP] {case}: bad case name or missing manifest")
        return None
    year, name, basin = meta
    csv = _bt_csv_for(case, year, name, basin)
    if csv is None:
        print(f"[SKIP] {case}: no best-track CSV")
        return None

    print(f"\n===== GEFS tracks: {name} ({year}) =====", flush=True)
    try:
        result, info = extract_gefs_tracks(case_dir, csv, fhour_max=fhour_max)
    except Exception as e:
        print(f"[FAIL] {case}: {e}")
        return None
    print(f"  vortex tracking: {info['n_ok']} members ok, failed={info['failed']}")
    if dry_run:
        for t in result["tracks"]:
            print(f"    {t['parent_member']}: {len(t['lon'])} pts "
                  f"{t['lon'][0]:.1f},{t['lat'][0]:.1f} -> "
                  f"{t['lon'][-1]:.1f},{t['lat'][-1]:.1f}")
        return None

    # isolated processed dir per storm (same pattern as run_ep_tracks)
    proc = THIS / f"processed_gefs_{name.lower()}"
    proc.mkdir(parents=True, exist_ok=True)
    for mod in (build_pairs, train_markov, sample_tracks):
        mod.PROCESSED_TRACKS_DIR = proc
    synth_dir = SYNTH_ROOT / str(year)
    synth_dir.mkdir(parents=True, exist_ok=True)
    sample_tracks.SYNTH_TRACKS_DIR = synth_dir

    raw = save_raw(result, proc / name.lower())
    print(f"  raw.pkl -> {raw}")

    build_pairs.run_build_pairs()
    train_markov.run_train_markov()
    sample_tracks.run_sample_tracks()

    ncs = sorted((synth_dir / name.lower()).glob("synthetic_tracks_*members.nc"))
    if not ncs:
        print(f"[FAIL] {name}: sampler produced no NC")
        return None
    print(f"  DONE -> {ncs[-1]}")
    return str(ncs[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, help="GEFS case, e.g. 2026_ELIDA_EP")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--fhour_max", type=int, default=None)
    a = ap.parse_args()

    if a.all:
        cases = sorted(d.name for d in GFS_ROOT.iterdir()
                       if d.is_dir() and _parse_case(d.name))
    elif a.case:
        cases = [a.case]
    else:
        ap.error("need --case or --all")

    done, fail = [], []
    for c in cases:
        nc = run_one(c, dry_run=a.dry_run, fhour_max=a.fhour_max)
        (done if nc else fail).append(c)
    print(f"\n=== GEFS tracks: {len(done)} done, {len(fail)} failed ===")
    if fail:
        print(f"    failed: {fail}")


if __name__ == "__main__":
    main()
