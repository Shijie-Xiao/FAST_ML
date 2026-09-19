#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract 31-member storm tracks from GEFS pgrb2a GRIB2 fields (vortex tracking).

Produces raw.pkl in the EXACT same structure as read_files.read_storm_ensemble
(ECMWF TIGGE XML path), so the downstream build_pairs -> train_markov ->
sample_tracks Markov stages run unchanged.

Tracking method (validated in _proto_vort.py on ELIDA 2026, c00, all 41 steps):
  * 850 hPa relative vorticity from U/V (central diff, sigma=4-grid gaussian
    smoothing ~2deg), tropics only (|lat|<=60)
  * fh=0: search +-10deg around best-track genesis
  * later: search +-8deg around previous position (continuity lock)
  * peak must exceed VORT_MIN=1e-5 /s; 3 consecutive misses terminate member

raw.pkl tracks carry "parent_member" (e.g. 'c00','p07') so the sampler can
implement FHLO-style member-paired inheritance: synthetic tracks seeded from
member j ALWAYS sample their environment from member j (not round-robin).
"""
import json
import pickle
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

from config import Earth_Radius, MIN_MEMBERS  # noqa: E402

import os
GFS_ROOT = Path(os.environ.get("GEFS_GRIB_ROOT", "/global/cfs/cdirs/m5011/Jay/ERA5/GFS"))
VORT_MIN = 1e-5
SEARCH_INIT = 10.0
SEARCH_CONT = 8.0
MAX_MISS = 3
VORT_LEVEL = 850


def load_vort(fp):
    """850 hPa relative vorticity from U/V, gaussian-smoothed ~2deg."""
    ds = xr.open_dataset(
        fp, engine="cfgrib", backend_kwargs={"indexpath": ""},
        filter_by_keys={"typeOfLevel": "isobaricInhPa", "level": VORT_LEVEL},
    )
    u = np.squeeze(ds["u"].values)
    v = np.squeeze(ds["v"].values)
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    ds.close()
    coslat = np.cos(np.deg2rad(lats))[:, None]
    dlon = np.deg2rad(0.5)
    dlat = np.deg2rad(0.5)
    dvdx = (np.roll(v, -1, 1) - np.roll(v, 1, 1)) / (2 * Earth_Radius * dlon * coslat)
    dudy = (np.roll(u, 1, 0) - np.roll(u, -1, 0)) / (2 * Earth_Radius * dlat)
    vort = gaussian_filter(dvdx - dudy, 4.0)
    vort[np.abs(lats) > 60] = np.nan
    return lons, lats, vort


def find_peak(lons, lats, field, clon, clat, half):
    """Peak (max) of field within +-half deg box around (clon, clat)."""
    lon_rel = ((lons - clon + 180) % 360) - 180
    mlon = np.abs(lon_rel) <= half
    mlat = np.abs(lats - clat) <= half
    if not mlon.any() or not mlat.any():
        return None
    sub = field[np.ix_(mlat, mlon)]
    if sub.size == 0 or np.all(np.isnan(sub)):
        return None
    k = int(np.nanargmax(sub))
    jj, ii = np.unravel_index(k, sub.shape)
    lon = ((float(lons[mlon][ii]) + 180) % 360) - 180
    return lon, float(lats[mlat][jj]), float(sub[jj, ii])


def _member_files(case_dir, member):
    """pgrb2a files for one member across all forecast hours."""
    d = case_dir / "grib2" / "pgrb2a"
    if not d.is_dir():
        return []
    pat = re.compile(rf"^(ge{'c00' if member == 'c00' else member})\.t(\d{{2}})z\.pgrb2a\.0p50\.f(\d{{3}})\.grib2$")
    out = []
    for fp in sorted(d.glob(f"ge{'c00' if member == 'c00' else member}.*.grib2")):
        m = pat.match(fp.name)
        if m:
            out.append((int(m.group(3)), fp))
    return sorted(out)


def _seed_from_bt(bt_csv, init):
    """Genesis position at init time from best-track CSV."""
    with open(bt_csv) as f:
        f.readline()  # header comment
        for line in f:
            p = line.strip().split(",")
            if len(p) < 3:
                continue
            try:
                t = datetime.strptime(p[0], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if t == init:
                return float(p[2]), float(p[1])  # lon, lat
    # no exact hit: nearest line
    best, bd = None, 1e9
    with open(bt_csv) as f:
        f.readline()
        for line in f:
            p = line.strip().split(",")
            if len(p) < 3:
                continue
            try:
                t = datetime.strptime(p[0], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            d = abs((t - init).total_seconds())
            if d < bd:
                bd, best = d, (float(p[2]), float(p[1]))
    return best


def extract_gefs_tracks(case_dir, bt_csv, min_members=MIN_MEMBERS,
                         fhour_max=None, verbose=False):
    """Track all 31 GEFS members. Returns read_storm_ensemble-style dict."""
    case_dir = Path(case_dir)
    man = json.load(open(case_dir / "manifest.json"))
    init = datetime.fromisoformat(man["init_utc"])
    members = list(man["members"])
    fhours = [h for h in man["forecast_hours"] if h % 6 == 0]
    if fhour_max is not None:
        fhours = [h for h in fhours if h <= fhour_max]
    seed = _seed_from_bt(bt_csv, init)
    if seed is None:
        raise RuntimeError(f"no best-track point at {init} in {bt_csv}")
    if verbose:
        print(f"  init={init} seed={seed} members={len(members)} fh={fhours[0]}..{fhours[-1]}")

    tracks, failed = [], []
    for member in members:
        files = {fh: fp for fh, fp in _member_files(case_dir, member)}
        fhs = [h for h in fhours if h in files]
        if not fhs:
            failed.append((member, "no files"))
            continue
        lon_l, lat_l, dt_l = [], [], []
        pos, miss = None, 0
        for fh in fhs:
            try:
                lons, lats, vort = load_vort(files[fh])
            except Exception as e:
                miss += 1
                if miss >= MAX_MISS:
                    break
                continue
            if pos is None:
                r = find_peak(lons, lats, vort, seed[0], seed[1], SEARCH_INIT)
            else:
                r = find_peak(lons, lats, vort, pos[0], pos[1], SEARCH_CONT)
            if r is None or r[2] < VORT_MIN:
                miss += 1
                if miss >= MAX_MISS:
                    break
                continue
            miss = 0
            pos = (r[0], r[1])
            lon_l.append(r[0])
            lat_l.append(r[1])
            dt_l.append(init + timedelta(hours=fh))
        if len(lon_l) >= 4:  # >= 24h of track -> keep
            tracks.append({
                "ensemble_system": "gefs",
                "member_id": 0 if member == "c00" else int(member[1:]),
                "parent_member": member,
                "storm_name": man["storm"]["name"],
                "basin": man["storm"]["basin"],
                "init_time": init,
                "lon": np.array(lon_l),
                "lat": np.array(lat_l),
                "datetime": dt_l,
            })
        else:
            failed.append((member, f"len={len(lon_l)}"))
        if verbose:
            print(f"  {member}: {len(lon_l):3d} pts"
                  + ("" if len(lon_l) >= 4 else f"  DROP ({failed[-1][1]})"))

    if len(tracks) < min_members:
        raise RuntimeError(
            f"only {len(tracks)}/{len(members)} members tracked "
            f"(<{min_members}); failed: {failed}")

    t0 = min(min(t["datetime"]) for t in tracks)
    t1 = max(max(t["datetime"]) for t in tracks)
    return {
        "storm_config": {
            "storm_name": man["storm"]["name"],
            "ibtracs_id": man["storm"].get("sid", case_dir.name),
            "storm_dir": str(Path(bt_csv).parent),
            "year": init.year,
            "basin": man["storm"]["basin"],
            "genesis": t0,
            "ensemble_source": "gefs",
        },
        "tracks": tracks,
        "n_tracks": len(tracks),
        "time_range": {"init_time": t0, "end_time": t1},
        "forced_init_time": init,
        "ensemble_systems": ["gefs"],
    }, {"failed": failed, "n_ok": len(tracks)}


def save_raw(result, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    init = result["forced_init_time"]
    fp = out_dir / f"{result['storm_config']['storm_name'].lower()}_{init:%Y%m%dT%H%M%S}_raw.pkl"
    with open(fp, "wb") as f:
        pickle.dump(result, f)
    return fp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_dir", required=True)
    ap.add_argument("--bt_csv", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--fhour_max", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    res, meta = extract_gefs_tracks(a.case_dir, a.bt_csv, verbose=a.verbose,
                                    fhour_max=a.fhour_max)
    print(f"tracked {meta['n_ok']} members; failed: {meta['failed']}")
    if a.out_dir:
        print("saved:", save_raw(res, a.out_dir))
