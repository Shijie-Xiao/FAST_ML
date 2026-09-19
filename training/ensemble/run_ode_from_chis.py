#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STAGE 5: per-member ODE with the Reproduce FAST model (NO forcing, free run).

Reads ONLY the chi/s NetCDF from predict_chi_s_div1000.py and runs the Reproduce
`Fast` ODE (solve_ivp RK45) for each ensemble member. chi & S are INJECTED; the
remaining environment (v_pot, h_m, t_strat, bathymetry) is served by a
PrecomputedEnvProvider:
  * v_pot:                 from the per-track scalars saved during data prep
  * h_m, t_strat, bathy:   climatology lookup (precalc_data/*.nc) by (lat,lon,month)
So NO per-storm regional ERA5 is needed -- everything comes from the already
prepared per-track data + climatology. This is the no-forcing, no-nudging,
fully-divergent free run (real intensity spread).

  * FAST     : chi = fast_chi (= clip(chi_ref*5,4)),  S = fast_s
  * FAST-ML  : chi = ml_chi * vent_scale,             S = ml_s

Usage:
  python run_ode_from_chis.py --in_nc ...chi_s.nc --out_nc ...ode.nc \
      --vent_scale 1.0 --workers 32
"""
import argparse
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Self-contained layout inside FAST_ML: everything resolves relative to this file.
# training/               <- repo copy of the PINN training pipeline
# training/fast_ode/      <- vendored copy of the Reproduce FAST ODE (Fast.py etc.)
TRAIN_ROOT = Path(__file__).resolve().parent.parent      # .../FAST_ML/training
PINN = TRAIN_ROOT                                       # prep modules live here now
REPRO = TRAIN_ROOT / 'fast_ode'                          # vendored ODE
PRECALC = PINN / 'precalc_data'
sys.path.insert(0, str(TRAIN_ROOT))
MS = 0.514444
T_MAX_H = 264.0
DT_EVAL_H = 1.0
C_K = 0.0015
# prep (process_one_storm) stores tcpyPI with V_reduc=0.8 (gradient->10m surface
# reduction). The FAST model's v_pot ceiling is calibrated on the *gradient-wind*
# PI (as the old Reproduce ERA5EnvProvider supplied). So un-reduce: v_pot = vp/0.8.
VP_REDUC = 0.8


# ── axi -> vmax conversion (identical to the validated ensemble pipeline) ──
def _axi_to_max_single(v_axisym, s, env, ut, vt, lat):
    v_axisym = float(v_axisym); s = float(np.nan_to_num(s, nan=0.0))
    env = np.asarray(env, float).reshape(-1); ut = float(ut); vt = float(vt); lat = float(lat)
    G = min(1.0, 0.8 + 0.35 * (1.0 + np.tanh((abs(lat) - 35.0) / 10.0)))
    u_shr = env[0] - env[2]; v_shr = env[1] - env[3]
    shear_mag = np.sqrt(u_shr**2 + v_shr**2 + 1e-12)
    has_env = not (np.isnan(u_shr) or np.isnan(v_shr) or shear_mag < 1e-6)
    u_dir = (u_shr / shear_mag) if has_env else 0.0
    v_dir = (v_shr / shear_mag) if has_env else 0.0
    shear_coeff = 0.1 * s * v_axisym / 15.0
    U_inc = G * ut + shear_coeff * u_dir; V_inc = G * vt + shear_coeff * v_dir
    mag_inc = np.sqrt(U_inc**2 + V_inc**2 + 1e-12)
    mag_fac = min(1.0, (v_axisym * 0.5) / mag_inc) if mag_inc > 1e-12 else 0.0
    theta_opt = np.arctan2(-U_inc, V_inc)
    ug = v_axisym * (-np.sin(theta_opt)) + U_inc * mag_fac
    vg = v_axisym * np.cos(theta_opt) + V_inc * mag_fac
    return float(np.sqrt(ug**2 + vg**2 + 1e-12))


def _invert_axi_to_max(target_ms, s0, env0, ut0, vt0, lat0):
    """Solve v_axi such that _axi_to_max_single(v_axi,...) == target_ms (bisection;
    the conversion is monotonic in v_axisym). Obs vmax already contains the
    asymmetric (translation/shear) component, so the ODE must start from the
    inverted axisymmetric wind, not from obs vmax directly."""
    target_ms = float(target_ms)
    if not np.isfinite(target_ms) or target_ms <= 0:
        return target_ms
    f = lambda v: _axi_to_max_single(v, s0, env0, ut0, vt0, lat0)
    lo, hi = 0.1, max(target_ms, 1.0)
    if f(hi) <= target_ms:          # conversion did not exceed target: keep obs value
        return target_ms
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) < target_ms:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _axi_to_max_vec(V_ms, s_arr, env_arr, ut_arr, vt_arr, lat_arr):
    n = len(V_ms); out = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(V_ms[i]) or V_ms[i] <= 0:
            out[i] = V_ms[i]; continue
        out[i] = _axi_to_max_single(V_ms[i], s_arr[i], env_arr[i], ut_arr[i], vt_arr[i], lat_arr[i])
    return out


_DAT = {}
_CLIM = {}
_VENT_SCALE = 1.0
_VP_SCALE = 1.0
_KL_PERTURB = True      # FHLO KL(n=10) initial-intensity perturbation
_INIT_MODE = 'fhlo'     # 'fhlo' (48h replay+KL history) | 'free' (t=0 KL only)
_BT_HISTORY = None      # hourly best-track V(t) ending at t=0 (m/s), len>=49
_ERA_ENV = None         # ERA5 replay env (vp, chi_eff, s) hourly, len 49, or None
_REF_TIME = None
_ENV_MODE = 'precomputed'   # 'precomputed' (self-contained) | 'era5' (live ERA5EnvProvider)
_ERA5_DIR = None
_USE_FORCING = True     # apply forecast-phase F_init forcing (False = free physics)
_REPLAY_UNTIL = None    # >0: force V to obs over [0, this hour] of the FORECAST,
                        # then release with F_init*exp(-2((t-h_rel)/24h)^2). The
                        # classic FHLO replay (pre-ref 48h) still runs first.


def _init_worker(in_nc, vent_scale, ref_time_str, env_mode, era5_dir, vp_scale=1.0,
                 kl_perturb=True, init_mode='fhlo', bt_pkl=None, use_forcing=True,
                 replay_until_h=None):
    global _DAT, _CLIM, _VENT_SCALE, _VP_SCALE, _KL_PERTURB, _REF_TIME, _ENV_MODE, _ERA5_DIR
    global _INIT_MODE, _BT_HISTORY, _ERA_ENV, _USE_FORCING, _REPLAY_UNTIL
    _USE_FORCING = bool(use_forcing)
    _REPLAY_UNTIL = float(replay_until_h) if replay_until_h is not None else None
    ds = xr.open_dataset(in_nc)
    keys = ['ml_chi', 'ml_s', 'fast_chi', 'fast_s', 'v_gt_ms', 'utran', 'vtran',
            'lat', 'lon', 'env_wnds', 'seq_len', 'scalars']
    _DAT = {k: ds[k].values for k in keys if k in ds}
    ds.close()
    _VENT_SCALE = float(vent_scale)
    _VP_SCALE = float(vp_scale)
    _KL_PERTURB = bool(kl_perturb)
    _INIT_MODE = str(init_mode)
    _REF_TIME = pd.Timestamp(ref_time_str) if ref_time_str else pd.Timestamp('2017-08-30')
    hist = _load_bt_history(bt_pkl, _REF_TIME) if _INIT_MODE == 'fhlo' else None
    if isinstance(hist, tuple):
        _BT_HISTORY, _ERA_ENV = hist
    else:
        _BT_HISTORY, _ERA_ENV = hist, None
    _ENV_MODE = str(env_mode)
    _ERA5_DIR = era5_dir
    # climatology (loaded once per worker)
    import xarray as _xr
    _CLIM['mld'] = _xr.open_dataset(PRECALC / 'mld_climatology.nc') if (PRECALC / 'mld_climatology.nc').exists() else None
    _CLIM['strat'] = _xr.open_dataset(PRECALC / 'strat_climatology.nc') if (PRECALC / 'strat_climatology.nc').exists() else None
    _CLIM['bathy'] = _xr.open_dataset(PRECALC / 'bathymetry.nc') if (PRECALC / 'bathymetry.nc').exists() else None
    _CLIM['land'] = _xr.open_dataset(PRECALC / 'land.nc') if (PRECALC / 'land.nc').exists() else None


def _get_era5():
    """Lazy per-worker ERA5EnvProvider (live PI/h_m/t_strat, exact-reproduction mode)."""
    global _ERA5
    if _ERA5 is None:
        sys.path.insert(0, str(REPRO)); os.chdir(REPRO)
        from env import ERA5EnvProvider
        _ERA5 = ERA5EnvProvider(era5_dir=Path(_ERA5_DIR),
                                data_dir=REPRO / 'Intensity' / 'data',
                                init_time=_REF_TIME.to_pydatetime(), basin='NA')
    return _ERA5


def _clim_lookup(lat, lon, month_idx):
    """h_m, t_strat, bathy from climatology (reuse prep logic)."""
    sys.path.insert(0, str(PINN))
    from prepare_complete_training_data import _get_mld_strat_bathy
    hm, strat, bathy, land = _get_mld_strat_bathy(
        _CLIM['mld'], _CLIM['strat'], _CLIM['bathy'], _CLIM['land'], lat, lon, month_idx)
    hm = hm if np.isfinite(hm) else 50.0
    strat = strat if np.isfinite(strat) else 0.2
    bathy = bathy if np.isfinite(bathy) else -5000.0
    return hm, strat, bathy


# ── FHLO appendix b: KL perturbation of the observed intensity history ──
def _sigma2_y(v_ms):
    """Piecewise observation-error variance (m^2 s^-2), Lin et al. 2020 appendix:
    sigma^2(y) = 5 if y < 32 m/s, 10 if y >= 33 m/s (approximation of the
    intensity-uncertainty distributions in Landsea & Franklin 2013)."""
    return 10.0 if v_ms >= 33.0 else 5.0


def _kl_intensity_history(v_obs_hist, n_dim=10, T_days=1.0, rng=None):
    """FHLO appendix b, strict version.

    Model V(t) as a Gaussian process about the observed history with kernel
        C_V(t1,t2) = sigma^2 * exp(-|t1 - t2| / T),  T = 1 day
    on the 6-hourly observation grid, then draw one realization from its
    Karhunen-Loeve expansion
        V(t) = mu_V(t) + sum_i sqrt(lambda_i) c_i u_i,  u_i ~ N(0,1), n = 10
    (truncation keeps only fluctuations on observation time scales, removing
    high-frequency variability). sigma^2 is the PIECEWISE variance
    _sigma2_y(y) evaluated per observation y (the paper's 'piecewise constant
    function' case for a time-varying observation grid).

    v_obs_hist : 6-hourly observed intensity (m/s), past window ending at t=0.
    Returns the perturbed history (same length).
    """
    v = np.asarray(v_obs_hist, float).copy()
    n = len(v)
    if n < 2:
        return v
    rng = rng or np.random.default_rng()
    dt = 1.0 / 4.0                                   # days (6-h observation grid)
    lag = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) * dt
    sig2 = np.array([_sigma2_y(y) for y in v])       # piecewise sigma^2(y)
    C = np.sqrt(sig2[:, None] * sig2[None, :]) * np.exp(-lag / T_days)
    eigval, eigvec = np.linalg.eigh(C)               # ascending
    order = np.argsort(eigval)[::-1][:n_dim]         # top-n eigenpairs
    u = rng.standard_normal(n_dim)
    pert = np.zeros(n)
    for k, i in enumerate(order):
        pert += np.sqrt(max(eigval[i], 0.0)) * eigvec[:, i] * u[k]
    return np.clip(v + pert, 0.0, None)


def _load_bt_history(bt_pkl, ref_time, hours=48):
    """Hourly best-track V(t) over [ref-hours, ref] (m/s), len = hours+1.

    Also returns the ERA5 environment (vp, chi_eff, s) along the best track for
    the same window, from the same pkl ('scalars'/'chi_ref'/'s_ref'): ERA5
    analysis fields driving the FHLO initialization period, per Lin et al.
    2020 §3e ("environmental parameters ... from the analysis fields").
    chi_eff = clip(chi_ref * 5, 0, 4), identical to the FAST calibration.
    Hours before the pkl start are clamped to the first available value.
    Returns (v_hist, era_env) where era_env is None if the pkl lacks scalars.
    """
    if not bt_pkl:
        return None
    try:
        with open(bt_pkl, 'rb') as f:
            ds = pickle.load(f)
        times = pd.to_datetime(np.asarray(ds['times']).ravel())
        vgt = np.asarray(ds['v_gt']).reshape(-1).astype(float)   # m/s
        n = min(len(times), len(vgt))
        s = pd.Series(vgt[:n], index=times[:n]).sort_index()
        s = s[~s.index.duplicated()]
        hourly = s.asfreq('h').interpolate('linear').ffill().bfill()
        ref = pd.Timestamp(ref_time)
        end = hourly.index.searchsorted(ref)
        end = min(end + 1, len(hourly))            # include ref itself
        start = max(0, end - (hours + 1))
        hist = hourly.iloc[start:end].to_numpy(float)
        if len(hist) < hours + 1:
            pre = np.full(hours + 1 - len(hist), hist[0] if len(hist) else 0.0)
            hist = np.concatenate([pre, hist])
        # ERA5 env series aligned to hist (relative hours -hours..0)
        era_env = None
        try:
            sc = np.asarray(ds['scalars']).reshape(ds['scalars'].shape[0], -1, 4)[0]
            chi_r = np.asarray(ds['chi_ref']).reshape(-1)
            s_r = np.asarray(ds['s_ref']).reshape(-1)
            n2 = min(len(times), len(sc))
            evp = pd.Series(sc[:n2, 3], index=times[:n2]).sort_index()
            evp = evp[~evp.index.duplicated()].asfreq('h').interpolate('linear')
            ech = pd.Series(chi_r[:n2], index=times[:n2]).sort_index()
            ech = ech[~ech.index.duplicated()].asfreq('h').interpolate('linear')
            esh = pd.Series(s_r[:n2], index=times[:n2]).sort_index()
            esh = esh[~esh.index.duplicated()].asfreq('h').interpolate('linear')
            win = evp.index[(evp.index > ref - pd.Timedelta(hours=hours)) &
                            (evp.index <= ref)]
            if len(win) >= 2:
                n_hist = len(hist)                 # hours+1
                vp_a = evp.reindex(win).to_numpy(float)
                chi_a = ech.reindex(win).to_numpy(float)
                s_a = esh.reindex(win).to_numpy(float)
                chi_a = np.clip(np.nan_to_num(chi_a, nan=1e-10) * 5.0, 0.0, 4.0)
                if len(win) < n_hist:              # clamp earliest hours
                    pad = n_hist - len(win)
                    vp_a = np.concatenate([np.full(pad, vp_a[0]), vp_a])
                    chi_a = np.concatenate([np.full(pad, chi_a[0]), chi_a])
                    s_a = np.concatenate([np.full(pad, s_a[0]), s_a])
                else:
                    vp_a = vp_a[-n_hist:]; chi_a = chi_a[-n_hist:]; s_a = s_a[-n_hist:]
                # Sanity gate (INCIDENT_na2025_vp_7level). Two fingerprint
                # checks for 7-level-sounding pollution, where tcpyPI is fed
                # a truncated profile and vp collapses to a narrow ~50-59 m/s
                # band regardless of storm strength (verified on 13 NA-2025
                # + CARLOTTA-2024 pkls), instead of tracking SST:
                #   (a) absolute ceiling: healthy 30-level PI exceeds 60 m/s
                #       over a 48h window in the tropics in all cases seen;
                #   (b) physical bound: PI < 0.9 * observed peak is impossible.
                # Polluted ERA5 replay env is rejected outright; the caller
                # then uses the GEFS f000 analysis env instead.
                v_win = hist[-len(vp_a):]
                if np.nanmax(vp_a) < 60.0 or np.nanmax(vp_a) < 0.9 * np.nanmax(v_win):
                    print('  [warn] bt-pkl ERA5 vp fails health gate '
                          f'(vp_max={np.nanmax(vp_a):.1f} < 60 or < 0.9*obs'
                          f'={0.9 * np.nanmax(v_win):.1f}); 7-level pollution? '
                          'replay uses GEFS t0 env', flush=True)
                    return hist, None
                vp_a = vp_a * _VP_SCALE            # same scaling as forecast env
                era_env = (vp_a, chi_a, s_a)
        except Exception as e:
            print(f'  [warn] ERA5 env history unavailable ({e}); replay uses GEFS t0', flush=True)
        return hist, era_env
    except Exception as e:
        print(f'  [warn] bt history load failed ({e}); init falls back to free mode', flush=True)
        return None


def _run_member(task):
    midx, mode = task
    sys.path.insert(0, str(REPRO)); os.chdir(REPRO)
    from Fast import Fast
    from track import Track

    d = _DAT
    n_pt = int(d['seq_len'][midx])
    if n_pt < 2:
        return (midx, mode, None)
    sl = slice(0, n_pt)
    lat = d['lat'][midx, sl].astype(float)
    lon = d['lon'][midx, sl].astype(float); lon = np.where(lon > 180, lon - 360, lon)
    ew = d['env_wnds'][midx, sl, :].astype(float)
    ut = d['utran'][midx, sl].astype(float)
    vt = d['vtran'][midx, sl].astype(float)
    v_gt = d['v_gt_ms'][midx, sl].astype(float)
    vp = d['scalars'][midx, sl, 3].astype(float)   # v_pot per step
    vp = vp * _VP_SCALE  # apply optional v_pot scaling (e.g. 1.1)
    ts = np.arange(n_pt, dtype=float) * 3600.0
    th = np.arange(n_pt, dtype=float)

    if mode == 'fast':
        chi_inj = d['fast_chi'][midx, sl].astype(float); s_inj = d['fast_s'][midx, sl].astype(float)
    else:
        chi_inj = d['ml_chi'][midx, sl].astype(float) * _VENT_SCALE; s_inj = d['ml_s'][midx, sl].astype(float)
    conv_s = s_inj

    class BTTrack(Track):
        def __init__(s): s.lon, s.lat, s.ts, s.u, s.v = lon, lat, ts, None, None
        def _vel(s):
            u = np.zeros_like(s.lon); v = np.zeros_like(s.lat)
            for i in range(1, len(s.lon)):
                dlon = s.lon[i] - s.lon[i - 1]
                if dlon > 180: dlon -= 360
                elif dlon < -180: dlon += 360
                dt = s.ts[i] - s.ts[i - 1]; latm = np.deg2rad((s.lat[i] + s.lat[i - 1]) / 2)
                u[i] = dlon * 111000 * np.cos(latm) / dt; v[i] = (s.lat[i] - s.lat[i - 1]) * 111000 / dt
            u[0] = u[1]; v[0] = v[1]; s.u, s.v = u, v
        def get_velocity(s, t, lon_, lat_):
            if s.u is None: s._vel()
            if t <= s.ts[0]: return np.array([s.u[0], s.v[0]], float)
            if t >= s.ts[-1]: return np.array([s.u[-1], s.v[-1]], float)
            return np.array([np.interp(t, s.ts, s.u), np.interp(t, s.ts, s.v)], float)
        def get_position(s, t):
            if t <= s.ts[0]: return (s.lon[0], s.lat[0])
            if t >= s.ts[-1]: return (s.lon[-1], s.lat[-1])
            return (np.interp(t, s.ts, s.lon), np.interp(t, s.ts, s.lat))

    class PrecomputedEnv:
        """chi/S injected. env_mode='precomputed': v_pot from saved scalars (/0.8),
        h_m/t_strat/bathy from climatology (self-contained, for storms w/o regional
        ERA5). env_mode='era5': v_pot/h_m/t_strat/bathy from live ERA5EnvProvider
        (exact reproduction of the old Reproduce pipeline; needs regional ERA5)."""
        def get_env(s, t, lon_, lat_):
            hh = float(t) / 3600.0
            if hh < 0:                                # 48h init replay: analysis fields
                hh = 0.0
            chi = float(np.interp(hh, th, chi_inj)) if hh < th[-1] else float(chi_inj[-1])
            ss = float(np.interp(hh, th, s_inj)) if hh < th[-1] else float(s_inj[-1])
            if not np.isfinite(chi) or chi <= 0: chi = 0.5
            if not np.isfinite(ss) or ss < 0: ss = 0.0
            if _ENV_MODE == 'era5':
                base = _get_era5().get_env(float(t), float(lon_), float(lat_))
                vpot = float(base.get('v_pot', 0.0)); hm = float(base.get('h_m', 50.0))
                strat = float(base.get('t_strat', 0.2)); bathy = float(base.get('bathymetry', -5000.0))
                is_land = bool(base.get('is_land', bathy >= 0))
            else:
                vpot = float(np.interp(hh, th, vp)) if hh < th[-1] else float(vp[-1])
                # NOTE: v_pot used as stored (tcpyPI). No /0.8 un-reduction (parked per user).
                month_idx = (_REF_TIME + timedelta(hours=hh)).month - 1
                hm, strat, bathy = _clim_lookup(float(lat_), float(lon_), month_idx)
                is_land = bathy >= 0
            if not np.isfinite(vpot) or vpot < 0: vpot = 0.0
            return {
                'v_pot': 0.0 if is_land else vpot, 'h_m': hm, 't_strat': strat,
                'chi': chi, 'C_k': C_K, 'env_wind_profile': (0.0, 0.0, ss, 0.0),
                'bathymetry': bathy, 'is_land': is_land, 'rh_mid': None,
            }

    fast = Fast(env_provider=PrecomputedEnv(), track_provider=BTTrack(), h_bl=1000.0)
    t_max_s = (n_pt - 1) * 3600.0
    t_eval = np.arange(n_pt, dtype=float) * 3600.0

    if _INIT_MODE == 'fhlo' and _BT_HISTORY is not None:
        # ---- FHLO §2c/§3e initialization: 48h replay + KL history perturbation ----
        # Mirrors run_fast_reference.py run_fast_with_init(): over the 48 h
        # pre-forecast window V is forced to track the (KL-perturbed) observed
        # target; the physics residual F = observed accel - physics rhs is
        # accumulated and its last-12-h mean is carried into the forecast,
        # decaying as exp(-2 (t/24h)^2) (same constant as run_fast_reference;
        # the FHLO paper writes exp[-(t/t0)^2], t0 = 1 day).
        from utils import compute_alpha, compute_beta, compute_gamma, compute_vent
        from constants import Epsilon as _EPS, Kappa as _KAP
        beta_c = compute_beta(_EPS, _KAP)
        STEP_H = 0.25                                  # 4 sub-steps per hour

        bt48 = np.asarray(_BT_HISTORY, float)          # hourly V_obs, len 49, ends t=0
        rng = np.random.default_rng(100000 + midx)     # deterministic per member
        if _KL_PERTURB:
            # KL(n=10) perturbation of the past-24h observed history on the
            # 6-h observation grid; piecewise sigma^2(y) per observation.
            v24 = bt48[-25::6]
            v24_p = _kl_intensity_history(v24, n_dim=10, T_days=1.0, rng=rng)
            pert_line = np.interp(np.arange(25), np.arange(0, 25, 6), v24_p - v24)
            bt48 = bt48.copy()
            bt48[-25:] += pert_line
            bt48 = np.clip(bt48, 0.0, None)
        # observed vmax -> axisymmetric target (t=0 shear/translation analysis;
        # replay-period env is the t=0 analysis, as PrecomputedEnv clamps hh<0)
        Vtar = np.array([
            _invert_axi_to_max(bt48[k], conv_s[0], ew[0], ut[0], vt[0], lat[0])
            for k in range(len(bt48))])
        ok_v = np.isfinite(Vtar) & (Vtar > 0)
        if np.any(ok_v):
            idx_ok = np.where(ok_v)[0]
            Vtar = np.interp(np.arange(len(Vtar)), idx_ok, Vtar[idx_ok])
            # Replay-period environment: ERA5 analysis along the best track
            # (Lin et al. 2020 §3e) when available; else t=0 GEFS analysis
            # (PrecomputedEnv clamps hh<0 to 0).
            n_rep = len(bt48)
            if _ERA_ENV is not None:
                vp_r, chi_r, s_r = _ERA_ENV
                vent_r = chi_r * s_r
            else:
                env0 = fast.env_provider.get_env(0.0, lon[0], lat[0])
                vp_r = np.full(n_rep, float(env0['v_pot']))
                vent_r = np.full(n_rep, float(env0['chi']) *
                                 float(env0['env_wind_profile'][2]))
            # alpha via ocean-coupling parameterization at the ERA5/bt0 position
            env0 = fast.env_provider.get_env(0.0, lon[0], lat[0])
            uT0 = float(np.hypot(ut[0], vt[0]))
            alpha_r = np.empty(n_rep)
            for k in range(n_rep):
                alpha_r[k] = float(np.clip(compute_alpha(
                    Vtar[k], env0['h_m'], vp_r[k], (uT0, 0.0), env0['bathymetry'],
                    env0['t_strat']), 0.0, 1.0))
            gamma_r = np.array([compute_gamma(a, _EPS, _KAP) for a in alpha_r])
            coeff0 = 0.5 * C_K / 1000.0 * 3600.0       # per-hour rate, h_bl=1000
            # m0 from local tendency (calculate_m0 in run_fast_reference)
            v0 = float(Vtar[-1])
            dvdt0 = float(Vtar[-1] - Vtar[-2])
            a_end, g_end, vp_end = alpha_r[-1], gamma_r[-1], vp_r[-1]
            num = dvdt0 / (coeff0 + 1e-12) + v0**2
            den = a_end * beta_c * vp_end**2 + g_end * v0**2
            m = float(np.clip((np.clip(num / (den + 1e-8), 0, None)) ** (1.0 / 3.0),
                              0.01, 1.0))
            F_hist = []
            for k in range(len(bt48)):
                Vt = float(Vtar[k])
                Vt_next = float(Vtar[min(k + 1, len(bt48) - 1)])
                with np.errstate(invalid='ignore'):
                    prhs = coeff0 * (alpha_r[k] * beta_c * vp_r[k]**2 * m**3
                                     - (1.0 - gamma_r[k] * m**3) * Vt**2)
                F_k = (Vt_next - Vt) - prhs
                if k < len(bt48) - 1:
                    F_hist.append(F_k)
                # V tracks target; m integrates free physics (4 sub-steps/h)
                for _ in range(4):
                    dm = coeff0 * ((1.0 - m) * Vt_next - vent_r[k] * m)
                    m = float(np.clip(m + dm * STEP_H, 0.01, 1.0))
            F_init = float(np.mean(F_hist[-12:])) if F_hist else 0.0  # m/s per hour

            y0 = np.array([lon[0], lat[0], v0, m], float)

            if _USE_FORCING:
                class ForcedFast(Fast):
                    """FAST + FHLO forecast-phase forcing: dV/dt += F_init *
                    exp(-2 (t/24h)^2) with F_init converted to per-second."""
                    def dydt(self, t, y):
                        d = super().dydt(t, y)
                        lead_h = float(t) / 3600.0
                        d[2] += (F_init / 3600.0) * np.exp(-2.0 * (lead_h / 24.0) ** 2)
                        return d
                solver = ForcedFast
            else:
                # free-physics forecast from the replayed state (F_init kept as
                # diagnostic only; no forecast-phase forcing applied)
                solver = Fast
            ffast = solver(env_provider=PrecomputedEnv(), track_provider=BTTrack(),
                           h_bl=1000.0)
            sol = ffast.run(t_span=(0, t_max_s), y0=y0, t_eval=t_eval,
                            method='RK45', max_step=3600, rtol=1e-4, atol=1e-6)
            f_diag = dict(v0_kts=float(v0 * 1.94384), m0=m, F=F_init,
                          v0_pert_kts=float(bt48[-1] * 1.94384))
        else:
            f_diag = None
            v_obs0 = float(v_gt[0]) if np.isfinite(v_gt[0]) and v_gt[0] > 0 else 15.0
            y0 = np.array([lon[0], lat[0],
                           _invert_axi_to_max(v_obs0, conv_s[0], ew[0], ut[0], vt[0], lat[0]),
                           0.4], float)
            sol = fast.run(t_span=(0, t_max_s), y0=y0, t_eval=t_eval,
                           method='RK45', max_step=3600, rtol=1e-4, atol=1e-6)
    else:
        # ---- free mode: t=0 KL perturbation only, no forcing ----
        f_diag = None
        v_obs0 = float(v_gt[0]) if np.isfinite(v_gt[0]) and v_gt[0] > 0 else 15.0
        if _KL_PERTURB:
            rng = np.random.default_rng(100000 + midx)
            v_obs0 = float(_kl_intensity_history(np.array([v_obs0]), rng=rng)[0])
        y0 = np.array([lon[0], lat[0],
                       _invert_axi_to_max(v_obs0, conv_s[0], ew[0], ut[0], vt[0], lat[0]),
                       0.4], float)
        sol = fast.run(t_span=(0, t_max_s), y0=y0, t_eval=t_eval,
                       method='RK45', max_step=3600, rtol=1e-4, atol=1e-6)

    hidx = np.clip(np.round(sol.t / 3600.0).astype(int), 0, n_pt - 1)
    V_ms = np.asarray(sol.y[2], float)

    # ---- optional forecast-phase replay: V glued to obs over [0, replay_until],
    # then released with F_rel * exp(-2((t-h_rel)/24h)^2) forcing (run_fast_ref). ----
    if (_REPLAY_UNTIL is not None and _REPLAY_UNTIL > 0
            and _REPLAY_UNTIL < n_pt - 1 and np.isfinite(v_gt[0])):
        from utils import compute_alpha, compute_beta, compute_gamma
        from constants import Epsilon as _EPS2, Kappa as _KAP2
        beta_c2 = compute_beta(_EPS2, _KAP2)
        h_rel = int(round(_REPLAY_UNTIL))
        # hourly observed target through the glue window (v_gt is hourly per step)
        n_glue = h_rel + 1
        vt_seq = np.asarray(v_gt[:n_glue], float)
        vt_seq = np.where(np.isfinite(vt_seq), vt_seq, max(float(v_gt[0]), 10.0))
        vent_seq = chi_inj[:n_glue] * s_inj[:n_glue]
        vp_seq = vp[:n_glue]
        # axisymmetric target conversion per hour (uses that hour's env)
        Vtar2 = np.array([_invert_axi_to_max(vt_seq[k], conv_s[k], ew[k], ut[k], vt[k], lat[k])
                          for k in range(n_glue)])
        ok2 = np.isfinite(Vtar2) & (Vtar2 > 0)
        if np.any(ok2):
            ix = np.where(ok2)[0]
            Vtar2 = np.interp(np.arange(n_glue), ix, Vtar2[ix])
            Vr = float(Vtar2[0])
            mr = float(np.clip(y0[3] if np.ndim(y0) else 0.4, 0.01, 1.0))
            coeff0 = 0.5 * C_K / 1000.0 * 3600.0
            # ocean coupling alpha at the member's own hourly positions
            alpha_seq = np.empty(n_glue); gamma_seq = np.empty(n_glue)
            for k in range(n_glue):
                uT = float(np.hypot(ut[k], vt[k]))
                alpha_seq[k] = float(np.clip(compute_alpha(
                    Vtar2[k], 100.0, vp_seq[k], (uT, 0.0), -4000.0, 26.0), 0.0, 1.0))
                gamma_seq[k] = compute_gamma(alpha_seq[k], _EPS2, _KAP2)
            F_hist2 = []
            for k in range(n_glue):
                with np.errstate(invalid='ignore'):
                    prhs = coeff0 * (alpha_seq[k] * beta_c2 * vp_seq[k]**2 * mr**3
                                     - (1.0 - gamma_seq[k] * mr**3) * Vtar2[k]**2)
                if k + 1 < n_glue:
                    F_hist2.append((Vtar2[k + 1] - Vtar2[k]) - prhs)
                    Vr = float(Vtar2[k + 1])
                for _ in range(4):     # m keeps integrating free physics
                    dm = coeff0 * ((1.0 - mr) * Vr - vent_seq[k] * mr)
                    mr = float(np.clip(mr + dm * 0.25, 0.01, 1.0))
            F_rel = float(np.mean(F_hist2[-12:])) if F_hist2 else 0.0

            class ReleasedFast(Fast):
                """Forecast from the glued state with F_rel decaying as
                exp(-2 ((t-h_rel)/24h)^2) — same FHLO shape, clock restarted."""
                def dydt(self, t, y):
                    d = super().dydt(t, y)
                    lead_h = (float(t) / 3600.0) - h_rel
                    d[2] += (F_rel / 3600.0) * np.exp(-2.0 * (lead_h / 24.0) ** 2)
                    return d
            rfast = ReleasedFast(env_provider=PrecomputedEnv(), track_provider=BTTrack(),
                                 h_bl=1000.0)
            y0r = np.array([lon[h_rel], lat[h_rel], Vr, mr], float)
            sol2 = rfast.run(t_span=(h_rel * 3600.0, t_max_s), y0=y0r,
                             t_eval=np.arange(h_rel, n_pt) * 3600.0,
                             method='RK45', max_step=3600, rtol=1e-4, atol=1e-6)
            # splice: glued hours 0..h_rel come from Vtar2 (exact obs), release
            # hours h_rel.. from sol2
            t_splice = np.concatenate([np.arange(0, h_rel + 1),
                                       np.asarray(sol2.t, float) / 3600.0])
            V_splice = np.concatenate([Vtar2[:h_rel + 1], np.asarray(sol2.y[2], float)])
            sol_t = t_splice
            V_ms = V_splice
            hidx = np.clip(np.round(sol_t).astype(int), 0, n_pt - 1)
            class _S:  # minimal shim so downstream code keeps working
                t = None
                y = None
                success = True
            _S.t = sol_t * 3600.0
            _S.y = np.vstack([np.full_like(sol_t, lon[0]), np.full_like(sol_t, lat[0]),
                              V_ms, np.full_like(sol_t, mr)])
            sol = _S
    cs = np.asarray(conv_s, float)[hidx]
    vmax_ms = _axi_to_max_vec(V_ms, cs, ew[hidx], ut[hidx], vt[hidx], lat[hidx])
    out = dict(
        t_h=sol.t / 3600.0, rawV_kts=V_ms / MS, vmax_kts=np.asarray(vmax_ms) / MS,
        success=bool(sol.success), peak=float(np.nanmax(vmax_ms) / MS))
    if f_diag is not None:
        out.update(f_diag)
    return (midx, mode, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_nc', required=True)
    p.add_argument('--out_nc', required=True)
    p.add_argument('--vent_scale', type=float, default=1.0)
    p.add_argument('--vp_scale', type=float, default=1.0,
                   help='Multiply v_pot (PI ceiling) by this factor before feeding ODE. '
                        'Default 1.0 = no scaling. Use e.g. 1.1 to compensate for '
                        'systematic PI low-bias in GEFS-derived vp.')
    p.add_argument('--no_kl_perturb', action='store_true',
                   help='Disable FHLO Karhunen-Loeve initial-intensity perturbation.')
    p.add_argument('--no_forcing', action='store_true',
                   help='init_mode=fhlo keeps 48h replay + KL init but drops the '
                        'forecast-phase F_init*exp(-2(t/24)^2) forcing term '
                        '(free-physics forecast from the replayed initial state).')
    p.add_argument('--replay_until_h', type=float, default=None,
                   help='Glue V to obs over the FIRST N hours of the forecast window '
                        '(e.g. hours from orig init to the 60kt node), accumulate the '
                        'physics residual, then release with F*exp(-2((t-N)/24h)^2).')
    p.add_argument('--init_mode', choices=['fhlo', 'free'], default='fhlo',
                   help="'fhlo' = 48h replay + KL history perturbation + forcing decay "
                        "(strict FHLO); 'free' = old behavior (t=0 KL only, no forcing).")
    p.add_argument('--bt_pkl', default=None,
                   help='best-track *_dataset.pkl providing the past-48h observed '
                        'intensity history (required for init_mode=fhlo).')
    p.add_argument('--modes', default='fast,ml',
                   help='逗号分隔, 跑哪些模式 (vent_scale 扫描时可设 ml 省一半算力)')
    p.add_argument('--workers', type=int, default=int(os.environ.get('ODE_WORKERS', '32')))
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--env_mode', choices=['precomputed', 'era5'], default='precomputed',
                   help="'precomputed' = self-contained (saved vp/0.8 + climatology); "
                        "'era5' = live ERA5EnvProvider (exact old-pipeline reproduction)")
    p.add_argument('--era5_dir', default=None, help='regional ERA5 dir (required for env_mode=era5)')
    args = p.parse_args()

    ds = xr.open_dataset(args.in_nc)
    names = list(ds['member'].values)
    ref_time_str = str(ds.attrs.get('reference_time', ''))
    n = len(names) if args.limit is None else min(args.limit, len(names))
    obs = ds['v_obz_kts'].values if 'v_obz_kts' in ds else None
    lat_all = ds['lat'].values; lon_all = ds['lon'].values
    ds.close()

    modes = tuple(x.strip() for x in args.modes.split(',') if x.strip())
    tasks = [(m, mode) for m in range(n) for mode in modes]
    print(f'[ode] {n} members x {modes} = {len(tasks)} sims, init_mode={args.init_mode} '
          f'(KL={not args.no_kl_perturb}), vent_scale={args.vent_scale}, '
          f'env_mode={args.env_mode}, workers={args.workers}', flush=True)
    if args.init_mode == 'fhlo' and not args.bt_pkl:
        print('[ode] WARNING: init_mode=fhlo without --bt_pkl; falling back to free mode', flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    res = {}; done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(args.in_nc, args.vent_scale, ref_time_str,
                                       args.env_mode, args.era5_dir, args.vp_scale,
                                       (not args.no_kl_perturb), args.init_mode,
                                       args.bt_pkl, (not args.no_forcing),
                                       args.replay_until_h)) as ex:
        futs = {ex.submit(_run_member, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                midx, mode, r = fut.result()
                if r is not None: res[(midx, mode)] = r
            except Exception as e:
                print(f'  [warn] task {futs[fut]} failed: {e}', flush=True)
            done += 1
            if done % 100 == 0 or done == len(tasks):
                print(f'  [ode] {done}/{len(tasks)}', flush=True)
                try: _write_nc(res, names, n, obs, lat_all, lon_all, args, ref_time_str)
                except Exception as e: print(f'    [ckpt warn] {e}', flush=True)
    _write_nc(res, names, n, obs, lat_all, lon_all, args, ref_time_str)
    print(f'[ode] wrote {args.out_nc}', flush=True)
    for mode in ('fast', 'ml'):
        pk = [res[(m, mode)]['peak'] for m in range(n) if (m, mode) in res]
        if pk: print(f'  {mode:4} n={len(pk)} mean_peak={np.mean(pk):.1f} median={np.median(pk):.1f}', flush=True)


def _write_nc(res, names, n, obs, lat_all, lon_all, args, ref_time_str=''):
    if not res: return
    ts_len = min(len(v['t_h']) for v in res.values())
    out = xr.Dataset(coords=dict(member=np.array(names[:n]), step=np.arange(ts_len)))
    out['time_hours'] = ('step', res[next(iter(res))]['t_h'][:ts_len].astype(np.float32))
    for mode in ('fast', 'ml'):
        vm = np.full((n, ts_len), np.nan, np.float32); rv = np.full((n, ts_len), np.nan, np.float32)
        for m in range(n):
            if (m, mode) in res:
                vm[m] = res[(m, mode)]['vmax_kts'][:ts_len]; rv[m] = res[(m, mode)]['rawV_kts'][:ts_len]
        out[f'{mode}_vmax_kts'] = (('member', 'step'), vm)
        out[f'{mode}_rawV_kts'] = (('member', 'step'), rv)
    if obs is not None:
        out['v_obz_kts'] = (('member', 'step'), obs[:n, :ts_len].astype(np.float32))
    out['lat'] = (('member', 'step'), lat_all[:n, :ts_len].astype(np.float32))
    out['lon'] = (('member', 'step'), lon_all[:n, :ts_len].astype(np.float32))
    # FHLO init diagnostics (present when init_mode=fhlo produced them)
    has_diag = any('F' in v for v in res.values())
    if has_diag:
        for key in ('init_v0_kts', 'init_m0', 'init_F', 'init_v0_pert_kts'):
            arr = np.full(n, np.nan, np.float32)
            for m in range(n):
                r = res.get((m, 'ml')) or res.get((m, 'fast'))
                src = key.replace('init_', '')
                if r and src in r:
                    arr[m] = r[src]
            out[key] = (('member',), arr)
    out.attrs['note'] = ('FAST vs FAST-ML (Reproduce Fast RK45). '
                         'env: v_pot from prepared scalars, h_m/t_strat/bathy from '
                         'climatology, chi/S injected. init_mode=fhlo: 48h replay + '
                         'KL(n=10, piecewise sigma^2_y) history perturbation + '
                         'forcing decay exp(-2(t/24h)^2); free: no forcing.')
    out.attrs['vent_scale'] = float(args.vent_scale)
    out.attrs['vp_scale'] = float(args.vp_scale)
    out.attrs['kl_perturb'] = int(bool(not args.no_kl_perturb))
    out.attrs['init_mode'] = str(args.init_mode)
    out.attrs['use_forcing'] = str(not args.no_forcing)
    if ref_time_str:
        out.attrs['reference_time'] = str(ref_time_str)
    op = Path(args.out_nc); op.parent.mkdir(parents=True, exist_ok=True)
    tmp = op.with_suffix('.nc.tmp'); out.to_netcdf(tmp); os.replace(tmp, op)


if __name__ == '__main__':
    main()
