#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STAGE 2 (lean) of the 1000-member fully-divergent IRMA pipeline.

ONLY predicts/extracts chi & s for every member -- NO ODE here.
For each member we save:
  * FAST-ML:  ml_chi, ml_s   (TwoStream model forward, use_run_fast_physics=False)
  * FAST:     fast_chi (calibrated chi_ref), fast_s (s_ref)  -- the ERA5 physics
plus lat/lon/v_obz_kts/vp_kts per step.

The parameterized ODE is run *separately afterwards* with the Reproduce code,
consuming these chi/s time series. We multiply chi*s by a coefficient there.

Members are loaded in CHUNKS (storm_include) to cap RAM (each spatial pkl ~200MB).

Usage:
  python predict_chi_s_div1000.py --ckpt twostream_final_d2.pth \
      --ensemble_dir ensemble_data/irma_div1000 --year 2017 \
      --hurricane_id AL112017_north_atlantic_IRMA \
      --train_data_dir training_data --train_years 2003-2021 \
      --reference_time '2017-08-30 00:00' --chunk_size 100 \
      --out_nc ensemble_data/irma_div1000/chi_s_div1000.nc
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
sys.path.insert(0, str(_THIS.parent))  # training/ root: shared model + eval modules

from SciML_Fast_TwoStream import (   # noqa: E402
    TwoStreamFASTModel, load_1km_storms, compute_spatial_stats, load_ckpt,
)
from eval_twostream_by_year import _predict_chi_s_one   # noqa: E402


def _parse_years(s):
    if not s:
        return []
    out = []
    for tok in str(s).split(','):
        tok = tok.strip()
        if not tok:
            continue
        if '-' in tok:
            a, b = tok.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


def main():
    p = argparse.ArgumentParser(description='Predict chi/s for ensemble members (no ODE)')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--ensemble_dir', required=True)
    p.add_argument('--year', type=int, required=True)
    p.add_argument('--hurricane_id', required=True)
    p.add_argument('--train_data_dir', default='training_data')
    p.add_argument('--train_years', default='2003-2021')
    p.add_argument('--min_vmax_kts', type=float, default=0.0)
    p.add_argument('--min_duration_h', type=int, default=0,
                   help='Min forecast-segment duration (h) for ENSEMBLE members. '
                        'Default 0 = keep all members (short/recurving tracks that '
                        'lose late steps to land/ET NaN-chi would otherwise be dropped). '
                        'Training-stats load keeps 72h for consistency.')
    p.add_argument('--reference_time', default=None)
    p.add_argument('--chunk_size', type=int, default=100)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--stats_pkl', default=None,
                   help='Cache path for spatial normalization stats. If it exists '
                        'it is loaded (skips the slow training-data scan); else '
                        'stats are computed from --train_years and saved here.')
    p.add_argument('--out_nc', required=True)
    args = p.parse_args()

    ensemble_dir = Path(args.ensemble_dir).resolve()
    year_dir = ensemble_dir / str(args.year)
    if not year_dir.is_dir():
        raise FileNotFoundError(f'No such year dir: {year_dir}')
    member_dirs = sorted(
        d for d in year_dir.iterdir()
        if d.is_dir() and d.name.startswith(args.hurricane_id + '_M'))
    if not member_dirs:
        raise FileNotFoundError(f'No member dirs under {year_dir}')
    if args.limit is not None:
        member_dirs = member_dirs[:args.limit]
    print(f'[chi-s] {len(member_dirs)} member dirs in {year_dir}', flush=True)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[chi-s] device={dev}', flush=True)
    model = TwoStreamFASTModel().to(dev)
    load_ckpt(model, args.ckpt, dev)
    model.eval()

    import pickle
    stats = None
    if args.stats_pkl and Path(args.stats_pkl).exists():
        with open(args.stats_pkl, 'rb') as f:
            stats = pickle.load(f)
        print(f'[chi-s] loaded cached stats from {args.stats_pkl}', flush=True)
    else:
        train_years = _parse_years(args.train_years)
        tr, _, _ = load_1km_storms(
            args.train_data_dir, 480, args.min_vmax_kts, args.min_vmax_kts, 72,
            train_years, [], [], None, None)
        if not tr:
            raise RuntimeError('No training storms loaded - cannot compute stats')
        stats = compute_spatial_stats(tr)
        del tr
        if args.stats_pkl:
            with open(args.stats_pkl, 'wb') as f:
                pickle.dump(stats, f, protocol=4)
            print(f'[chi-s] saved stats cache to {args.stats_pkl}', flush=True)
    print(f'[chi-s] spatial stats ready', flush=True)

    ml_chi_l, ml_s_l, fa_chi_l, fa_s_l = [], [], [], []
    obz_l, vp_l, lat_l, lon_l, names = [], [], [], [], []
    vgt_l, utran_l, vtran_l, scal_l, ew_l = [], [], [], [], []
    seqlen_l, t0_l = [], []
    Tmax = 0
    n_chunks = (len(member_dirs) + args.chunk_size - 1) // args.chunk_size
    for ci in range(n_chunks):
        chunk = member_dirs[ci * args.chunk_size:(ci + 1) * args.chunk_size]
        inc = {f'{args.year}/{md.name}' for md in chunk}
        ens_tr, _, _ = load_1km_storms(
            str(ensemble_dir), 480, args.min_vmax_kts, args.min_vmax_kts,
            args.min_duration_h,
            [args.year], [], [], inc, None, keep_weak=True)
        by_id = {str(d.get('hurricane', '')): d for d in ens_tr}
        print(f'[chi-s] chunk {ci+1}/{n_chunks}: requested {len(chunk)} loaded {len(ens_tr)}',
              flush=True)
        for md in chunk:
            ds = by_id.get(md.name)
            if ds is None:
                print(f'  [warn] no dataset for {md.name}, skip', flush=True)
                continue
            try:
                r = _predict_chi_s_one(model, ds, stats, dev)
            except Exception as e:
                print(f'  [warn] predict failed {md.name}: {e}', flush=True)
                continue
            ml_chi_l.append(r['ml_chi']); ml_s_l.append(r['ml_s'])
            fa_chi_l.append(r['fast_chi']); fa_s_l.append(r['fast_s'])
            obz_l.append(r['v_obz_kts']); vp_l.append(r['vp_kts'])
            lat_l.append(r['lats']); lon_l.append(r['lons'])
            vgt_l.append(r['v_gt_ms']); utran_l.append(r['utran']); vtran_l.append(r['vtran'])
            scal_l.append(r['scalars']); ew_l.append(r['env_wnds'])
            seqlen_l.append(int(r['seq_len'])); t0_l.append(int(r['t0']))
            names.append(md.name)
            Tmax = max(Tmax, int(r['T']))
        del ens_tr, by_id
        print(f'  [chi-s] cumulative {len(names)}/{len(member_dirs)}', flush=True)

    if not names:
        raise RuntimeError('No members predicted successfully')

    def _pad(a, T):
        a = np.asarray(a, dtype=np.float32)
        if a.size < T:
            a = np.concatenate([a, np.full(T - a.size, np.nan, dtype=np.float32)])
        return a[:T]

    def _stack(lst):
        return np.stack([_pad(x, Tmax) for x in lst], axis=0).astype(np.float32)

    def _pad2(a, T):
        a = np.asarray(a, dtype=np.float32)
        if a.shape[0] < T:
            a = np.concatenate([a, np.full((T - a.shape[0], a.shape[1]), np.nan, np.float32)], 0)
        return a[:T]

    def _stack2(lst):
        return np.stack([_pad2(x, Tmax) for x in lst], axis=0).astype(np.float32)

    out = xr.Dataset(
        data_vars={
            'ml_chi':   (('member', 'step'), _stack(ml_chi_l)),
            'ml_s':     (('member', 'step'), _stack(ml_s_l)),
            'fast_chi': (('member', 'step'), _stack(fa_chi_l)),
            'fast_s':   (('member', 'step'), _stack(fa_s_l)),
            'v_obz_kts': (('member', 'step'), _stack(obz_l)),
            'vp_kts':   (('member', 'step'), _stack(vp_l)),
            'lat':      (('member', 'step'), _stack(lat_l)),
            'lon':      (('member', 'step'), _stack(lon_l)),
            # ── ODE inputs (trimmed, aligned with chi/s) ──
            'v_gt_ms':  (('member', 'step'), _stack(vgt_l)),
            'utran':    (('member', 'step'), _stack(utran_l)),
            'vtran':    (('member', 'step'), _stack(vtran_l)),
            'scalars':  (('member', 'step', 'ns'), _stack2(scal_l)),
            'env_wnds': (('member', 'step', 'ne'), _stack2(ew_l)),
            'seq_len':  (('member',), np.array(seqlen_l, dtype=np.int32)),
            't0':       (('member',), np.array(t0_l, dtype=np.int32)),
        },
        coords={
            'member': np.array(names),
            'step': np.arange(Tmax, dtype=np.int32),
            'ns': np.array(['alpha', 'beta', 'gamma', 'vp']),
            'ne': np.arange(4, dtype=np.int32),
        },
        attrs={
            'reference_time': args.reference_time or '',
            'hurricane_id': args.hurricane_id,
            'year': int(args.year),
            'ckpt': str(args.ckpt),
            'n_members': int(len(names)),
            'note': 'chi/s only (no ODE). ml_*=TwoStream, fast_*=ERA5 physics '
                    '(fast_chi=calibrated chi_ref). vent=chi*s; apply coeff at ODE stage.',
        },
    )
    out_nc = Path(args.out_nc)
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_nc)
    print(f'[chi-s] wrote {out_nc}  ({len(names)} members x {Tmax} steps)', flush=True)


if __name__ == '__main__':
    main()
