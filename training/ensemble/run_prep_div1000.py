#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STAGE 1 of the 1000-member fully-divergent IRMA pipeline.

Prepare *_dataset.pkl + *_spatial_1000km.pkl for N fully-divergent synthetic
members (init_time = reference_time = 2017-08-30 00:00, so NO best-track
prepend -> members diverge from h=0).

Uses prepare_complete_training_data.process_track_data (ERA5 _first_glob picks
the largest/30-level/828893 file = the complete version matching BT training).

Parallelized with ProcessPoolExecutor; each worker keeps its own ERA5 cache and
processes a contiguous chunk of members (good cache reuse).

Env vars:
  PREP_WORKERS  (default 32)
  PREP_START    (default 0)     first member index
  PREP_END      (default 1000)  stop index (exclusive)
"""
import os, sys, csv, pickle
from pathlib import Path
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
sys.path.insert(0, str(_THIS.parent))  # training/ root: prepare_ensemble_storm + prep modules

# reuse the exact track-building logic from prepare_ensemble_storm
from prepare_ensemble_storm import _load_best_track, _load_synthetic, _build_member_track

# All key inputs are env-var driven (defaults = IRMA div1000 example workflow;
# see training/README.md for a full example)
_TRAIN_ROOT = Path(__file__).resolve().parent.parent   # .../FAST_ML/training
SYNTH_NC   = os.environ.get('PREP_SYNTH_NC',
    str(_TRAIN_ROOT / 'ensemble_runs/irma/synthetic_tracks_1000members.nc'))
BEST_TRACK = os.environ.get('PREP_BEST_TRACK',
    str(_TRAIN_ROOT / 'training_data/2017/AL112017_north_atlantic_IRMA/AL112017_north_atlantic_IRMA_dataset.pkl'))
HID        = os.environ.get('PREP_HID', 'AL112017_north_atlantic_IRMA')
OUT_ROOT   = Path(os.environ.get('PREP_OUT_ROOT',
    str(_TRAIN_ROOT / 'ensemble_runs/irma_div1000')))
YEAR       = int(os.environ.get('PREP_YEAR', '2017'))
REF_TIME   = pd.Timestamp(os.environ.get('PREP_REF_TIME', '2017-08-30 00:00'))
DURATION_H = int(os.environ.get('PREP_DURATION_H', '264'))
N_MEMBERS  = int(os.environ.get('PREP_N_MEMBERS', '1000'))
SEED       = int(os.environ.get('PREP_SEED', '42'))
VORTEX     = os.environ.get('PREP_VORTEX', 'annulus')

N_WORKERS = int(os.environ.get('PREP_WORKERS', '32'))
START     = int(os.environ.get('PREP_START', '0'))
END       = int(os.environ.get('PREP_END', str(N_MEMBERS)))

# ---- shared (per-process) inputs, lazily loaded in each worker ----
_G = {}


def _ensure_inputs():
    if _G:
        return _G
    _G['bt'] = _load_best_track(Path(BEST_TRACK))
    init_time, dt_hours, lons, lats, t_sec, picked = _load_synthetic(
        Path(SYNTH_NC), N_MEMBERS, SEED)
    _G['init_time'] = init_time
    _G['lons'] = lons; _G['lats'] = lats; _G['t_sec'] = t_sec; _G['picked'] = picked
    _G['sfc_cache'] = {}; _G['pl_cache'] = {}
    return _G


def _prep_member(mi):
    # PINN new-version single-pkl prep: process_one_storm -> {name}_dataset.pkl
    # (dataset already carries 72x72 spatial_3d; no separate _spatial_1000km.pkl)
    from prepare_complete_training_data import process_one_storm
    g = _ensure_inputs()
    name = f'{HID}_M{mi:03d}'
    sdir = OUT_ROOT / str(YEAR) / name
    sdir.mkdir(parents=True, exist_ok=True)
    ds_pkl = sdir / f'{name}_dataset.pkl'
    if ds_pkl.exists():
        return (mi, 'skip', '')
    try:
        # reference_time == init_time  ->  empty [ref,init) prepend  ->  fully divergent from h=0
        track = _build_member_track(
            g['bt'], g['lons'][mi], g['lats'][mi], g['t_sec'], g['init_time'],
            g['init_time'], DURATION_H)
        csv_path = sdir / f'{name}_track.csv'
        track.to_csv(csv_path, index=False)
        _era5 = os.environ.get('PREP_ERA5_ROOT') or None
        result = process_one_storm(str(csv_path), era5_root_override=_era5)
        if result is None:
            return (mi, 'fail', 'process_one_storm None')
        result['hurricane'] = name
        with open(ds_pkl, 'wb') as f:
            pickle.dump(result, f, protocol=4)
        T = result['spatial_3d'].shape[1]
        return (mi, 'ok', f'T={T}')
    except Exception as e:
        return (mi, 'fail', repr(e)[:200])


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / str(YEAR)).mkdir(parents=True, exist_ok=True)
    # write member index once (member 0..N-1 -> picked synthetic track id)
    g = _ensure_inputs()
    idx_csv = OUT_ROOT / str(YEAR) / f'{HID}_ensemble_index.csv'
    with open(idx_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['member', 'storm_dir', 'synthetic_track_id'])
        for mi, tid in enumerate(g['picked']):
            w.writerow([mi, f'{HID}_M{mi:03d}', int(tid)])
    members = list(range(START, min(END, N_MEMBERS)))
    print(f'[prep] {len(members)} members [{START},{END}) workers={N_WORKERS} '
          f'synth={Path(SYNTH_NC).name} init={g["init_time"]}', flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    ok = fail = skip = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_prep_member, mi): mi for mi in members}
        done = 0
        for fut in as_completed(futs):
            mi, status, msg = fut.result()
            done += 1
            if status == 'ok': ok += 1
            elif status == 'skip': skip += 1
            else:
                fail += 1
                print(f'  [M{mi:03d}] FAIL {msg}', flush=True)
            if done % 25 == 0 or done == len(members):
                print(f'  progress {done}/{len(members)}  ok={ok} skip={skip} fail={fail}', flush=True)
    print(f'[prep] done ok={ok} skip={skip} fail={fail} out={OUT_ROOT/str(YEAR)}', flush=True)


if __name__ == '__main__':
    main()
