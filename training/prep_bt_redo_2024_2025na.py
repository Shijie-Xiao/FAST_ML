#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate best-track *_dataset.pkl for 2024 (NA+EP) and 2025 NA storms.

Fixes the 7-level vp incident (see ensemble/INCIDENT_na2025_vp_7level.md).
Storm list is taken from scripts/gefs_storms_2024_2026.csv (authoritative),
matched to data/{basin}/{year}/{sid}_{name}/ dirs.

Archives:
  2024  -> /global/cfs/cdirs/m5011/Jay/ERA5/{NA|EP}/  (30-level, through 2024-12)
  2025  -> /global/cfs/cdirs/m5011/Jay/ERA5/2025/     (30-level flat archive)
  2026  -> no archive, skipped.

Polluted/old pkl is moved to *.vp7level.bak before recompute.
Validation: vp peak must bound observed vmax peak (>= 0.9x), sounding >= 25 levs.

Env: WORKERS (default 4), ONLY=name-substring filter.
"""
import os, sys, shutil, pickle
from pathlib import Path
import pandas as pd

_TRAIN_ROOT = Path(__file__).resolve().parent          # .../FAST_ML/training
sys.path.insert(0, str(_TRAIN_ROOT))
PINN = _TRAIN_ROOT
CSV = _TRAIN_ROOT / 'gefs' / 'gefs_storms_2024_2026.csv'
ERA5 = {
    # NA 2024: basin archive has complete 30-level batch (828891/2/3)
    2024: {'NA': '/global/cfs/cdirs/m5011/Jay/ERA5'},
    # 2025 flat archive: global 30-level (covers both NA and EP)
    2025: {'NA': '/global/cfs/cdirs/m5011/Jay/ERA5/2025',
           'EP': '/global/cfs/cdirs/m5011/Jay/ERA5/2025'},
}
# EP 2024 basin archive is 7-level regional only -> cannot be fixed here;
# excluded (needs full-level download first).


def storm_list():
    df = pd.read_csv(CSV, keep_default_na=False)
    t = df[(df['basin'].isin(['NA', 'EP'])) & (df['name'] != 'UNNAMED') & (df['n_pts'] >= 50)]
    out = []
    for _, r in t.iterrows():
        b, y = r['basin'], int(r['season'])
        if y not in ERA5 or b not in ERA5[y]:
            continue  # 2026: no archive
        d = PINN / 'data' / b / str(y) / f"{r['sid']}_{r['name']}"
        if not (d / 'track_intensity_6h.csv').exists():
            continue
        # 2025 EP pkls are the healthy Jul-6 recompute (2025 flat archive) -> keep
        if y == 2025 and b == 'EP' and (d / f'{d.name}_dataset.pkl').exists():
            continue
        out.append((b, y, d))
    return out


def redo(entry):
    basin, year, sdir = entry
    name = sdir.name.split('_')[-1]
    pkl = sdir / f'{sdir.name}_dataset.pkl'
    csv = sdir / 'track_intensity_6h.csv'
    try:
        if pkl.exists():
            bak = pkl.with_name(pkl.name + '.vp7level.bak')
            if not bak.exists():
                shutil.move(str(pkl), str(bak))
        from prepare_complete_training_data import process_one_storm
        ds = process_one_storm(str(csv), era5_root_override=ERA5[year][basin])
        if ds is None:
            return (name, 'FAIL', 'process_one_storm None')
        with open(pkl, 'wb') as f:
            pickle.dump(ds, f, protocol=4)
        import numpy as np
        vp = np.asarray(ds['scalars'])[0, :, 3]
        vg = np.asarray(ds['v_gt']).reshape(len(vp), -1)[:, 0]
        n_lev = np.asarray(ds['spatial_3d']).shape[3]
        ok = (np.nanmax(vp) >= 0.9 * np.nanmax(vg)) and n_lev >= 25
        return (name, 'OK' if ok else 'WARN',
                f'vp_med={np.nanmedian(vp):5.1f} vp_max={np.nanmax(vp):5.1f} '
                f'obs_max={np.nanmax(vg):5.1f} sond_levs={n_lev}')
    except Exception as e:
        return (name, 'FAIL', repr(e)[:150])


if __name__ == '__main__':
    W = int(os.environ.get('WORKERS', '4'))
    only = os.environ.get('ONLY', '')
    storms = [(b, y, d) for b, y, d in storm_list() if only.lower() in d.name.lower()]
    print(f'[bt-redo] {len(storms)} storms (2024 NA+EP, 2025 NA), workers={W}', flush=True)
    if W <= 1:
        for s in storms:
            print('  ', redo(s), flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=W) as ex:
            futs = {ex.submit(redo, s): s for s in storms}
            for i, fut in enumerate(as_completed(futs)):
                print(f'  [{i+1}/{len(storms)}]', fut.result(), flush=True)
