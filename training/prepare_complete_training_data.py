#!/usr/bin/env python3
"""
Training data prep for FAST_ML.
Input: track_intensity_6h.csv -> interpolate to 1h -> ERA5 extraction -> pkl.
Output: {STORM}_dataset.pkl with 72x72 spatial fields, scalars, chi, shear, env winds.
"""
import os
import sys
import glob
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.interpolate import CubicSpline, PchipInterpolator

warnings.filterwarnings('ignore')

THIS_DIR = Path(__file__).resolve().parent
PRECALC_DIR = THIS_DIR / 'precalc_data'

# All dependencies are local to FAST_ML (self-contained for GitHub).
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR / 'vortex_inversion'))

import namelist
if not hasattr(namelist, 'select_thermo'):
    namelist.select_thermo = 1
if not hasattr(namelist, 'p_midlevel'):
    namelist.p_midlevel = 60000
sys.modules.setdefault('namelist', namelist)

from thermo import thermo
from vortex_lib import VORTEX_LIB

# ---------- Constants ----------
R_EARTH_M = 6.3781e6
R_EARTH_KM = 6.3781e3
K_TO_C = 273.15
PA_TO_HPA = 100.0
KNOTS_TO_MS = 0.514444
MS_TO_KNOTS = 1.94384

SPATIAL_SIZE = 72
PL_LEVELS = [1000, 850, 700, 600, 500, 250, 200]
P_MIDLEVEL_PA = float(getattr(namelist, 'p_midlevel', 60000))
P_MIDLEVEL_HPA = P_MIDLEVEL_PA / 100.0

EPSILON_FAST, KAPPA_FAST = 0.33, 0.1
BETA_FAST = 1.0 - EPSILON_FAST - KAPPA_FAST
Rd, Rv, Lv = 287.04, 461.5, 2.5e6
EPS_H2O = Rd / Rv
CHI_D = 4.0
CHI_PARAMS = {
    'NA': {'percentile': 90, 'renv_km': 1000},
    'EP': {'percentile': 50, 'renv_km': 900},
}

ERA5_ROOT = os.environ.get('FASTML_ERA5_ROOT', '/global/cfs/cdirs/m5011/Jay/ERA5')
SUPPORTED_BASINS = ('NA', 'EP')
TIME_TOLERANCE = pd.Timedelta(minutes=30)

# PL variable code mapping: var name -> ECMWF param code in filename
PL_VAR_CODE = {'T': '130_t', 'Q': '133_q', 'U': '131_u', 'V': '132_v', 'Z': '129_z'}
# SFC variable code mapping
SFC_VAR_CODE = {'SSTK': '034_sstk', 'MSL': '151_msl'}


# ---------- Config ----------
def load_config(path: Path):
    cfg = {'basins': 'ALL', 'year_start': '2003', 'year_end': '2024',
           'output_dir': 'data', 'era5_root': ERA5_ROOT}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            s = line.strip()
            if not s or s.startswith('#') or '=' not in s:
                continue
            k, v = [x.strip() for x in s.split('=', 1)]
            cfg[k.lower()] = v
    return cfg


def parse_basins(s):
    vals = [x.strip().upper() for x in str(s).split(',') if x.strip()]
    return list(SUPPORTED_BASINS) if ('ALL' in vals or not vals) else vals


# ---------- Basin / ERA5 config ----------
def infer_basin(track_csv_path=None, hurricane_name=None):
    path_s = str(track_csv_path or '').upper()
    name_s = str(hurricane_name or '').upper()
    if '/EP/' in path_s or name_s.startswith('EP') or '_EAST_PACIFIC_' in name_s:
        return 'EP'
    return 'NA'


def get_era5_config(basin='NA', era5_root=None):
    root = era5_root or ERA5_ROOT
    basin_upper = (basin or 'NA').upper()
    basin_root = os.path.join(root, basin_upper)
    # Support both layouts: old (ERA5/{NA|EP}/{VAR}/) and new (ERA5/Test/PL|SFC/)
    if os.path.isdir(os.path.join(root, 'PL')):
        return {
            'basin': basin_upper,
            'pl_root': os.path.join(root, 'PL'),
            'sfc_root': os.path.join(root, 'SFC'),
        }
    return {
        'basin': basin_upper,
        'pl_root': basin_root,
        'sfc_root': basin_root,
    }


# ---------- Lon / range helpers ----------
def _norm_lon(lon, coords=None):
    if coords is None or len(coords) == 0:
        return lon
    mn, mx = coords.min(), coords.max()
    return (lon + 360 if lon < 0 else lon) if 0 <= mn <= mx <= 360 else lon




def _sanitize_lon_continuity(lons, max_deg=25):
    lons = np.asarray(lons, dtype=np.float64)
    if len(lons) < 2:
        return lons
    ln = np.where(lons < 0, lons + 360, lons)
    out = ln.copy()
    for i in range(1, len(out)):
        d = abs(out[i] - out[i - 1])
        if min(d, 360 - d) > max_deg:
            out[i] = out[i - 1]
    return out


# ---------- ERA5 I/O ----------

# 同一天同一变量可能存在多个下载批次（如 NA/2024 T 同时有 828421 的 7 层
# 截断版和 828892 的 30 层完整版，字母序 828421 < 828892 导致旧逻辑永远
# 选中 7 层版 —— 即 NA-2025 vp 7 层事故的根源）。所有候选全部打开比层数，
# 层数最多者胜出；层数并列时取最后一个（通常批次号更大 = 更新下载）。
def _best_level_file(pattern):
    import xarray as _xr
    cands = sorted(glob.glob(pattern))
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    best_p, best_n = cands[0], -1
    for p in cands:
        try:
            with _xr.open_dataset(p, decode_times=False) as _ds:
                lk = _level_key(_ds[list(_ds.data_vars)[0]])
                n = int(_ds.sizes[lk]) if lk else -1
        except Exception:
            continue
        if n > best_n:            # strictly greater: tie -> keep later (larger batch id)
            best_p, best_n = p, n
    return best_p


def _first_glob(p):
    m = sorted(glob.glob(p))
    return m[0] if m else None


def _open_sfc(ts, cache, cfg):
    ym = ts.strftime('%Y%m')
    k = f"sfc:{cfg['basin']}:{ym}"
    if k in cache:
        return cache[k]
    sfc_root = cfg['sfc_root']
    # New layout: SFC/{YYYYMM}/*.nc
    sfc_dir = os.path.join(sfc_root, ym)
    flat = False
    if os.path.isdir(sfc_dir):
        sst_p = _first_glob(os.path.join(sfc_dir, f'*{SFC_VAR_CODE["SSTK"]}*{ym}*.nc'))
        msl_p = _first_glob(os.path.join(sfc_dir, f'*{SFC_VAR_CODE["MSL"]}*{ym}*.nc'))
    elif os.path.isdir(os.path.join(sfc_root, 'SSTK')):
        # Old layout: ERA5/{BASIN}/SSTK/*.nc and ERA5/{BASIN}/MSL/*.nc
        sst_p = _first_glob(os.path.join(sfc_root, 'SSTK', f'*SSTK*{ym}*.nc')) or \
                _first_glob(os.path.join(sfc_root, 'SSTK', f'*sstk*{ym}*.nc'))
        msl_p = _first_glob(os.path.join(sfc_root, 'MSL', f'*MSL*{ym}*.nc')) or \
                _first_glob(os.path.join(sfc_root, 'MSL', f'*msl*{ym}*.nc'))
    else:
        # Flat layout: monthly files directly in SFC/ (e.g. ERA5/2025/SFC/*.nc)
        flat = True
        sst_p = _first_glob(os.path.join(sfc_root, f'*{SFC_VAR_CODE["SSTK"]}*{ym}*.nc'))
        msl_p = _first_glob(os.path.join(sfc_root, f'*{SFC_VAR_CODE["MSL"]}*{ym}*.nc'))
    if not sst_p or not msl_p:
        cache[k] = None
        return None
    datasets = [xr.open_dataset(sst_p), xr.open_dataset(msl_p)]
    # Try loading BLH and SP if available
    if os.path.isdir(sfc_dir):
        blh_p = _first_glob(os.path.join(sfc_dir, f'*159_blh*{ym}*.nc'))
    elif flat:
        blh_p = _first_glob(os.path.join(sfc_root, f'*159_blh*{ym}*.nc'))
    else:
        blh_p = _first_glob(os.path.join(sfc_root, 'BLH', f'*BLH*{ym}*.nc'))
    if blh_p:
        datasets.append(xr.open_dataset(blh_p))
    ds = xr.merge(datasets, compat='override')
    cache[k] = {'ds': ds}
    return cache[k]


def _sel_time(ds, t, tol=TIME_TOLERANCE):
    if ds is None:
        return None
    tk = 'time' if 'time' in ds.coords else 'valid_time'
    sel = ds.sel({tk: t}, method='nearest')
    tt = pd.Timestamp(sel[tk].values)
    if abs(tt - t) > tol:
        return None
    return sel


def _sel_point(da, lat, lon):
    lk = 'latitude' if 'latitude' in da.coords else 'lat'
    nk = 'longitude' if 'longitude' in da.coords else 'lon'
    ln = _norm_lon(lon, da[nk].values)
    return float(da.sel({lk: lat, nk: ln}, method='nearest').values)


def _level_key(da):
    if da is None:
        return None
    for k in ['level', 'pressure_level']:
        if k in da.dims:
            return k
    for d in da.dims:
        if d not in ['latitude', 'longitude', 'lat', 'lon', 'time', 'valid_time']:
            return d
    return None


def _find_pl(var, date_str, cfg):
    ym = date_str[:6]
    # New layout: PL/{YYYYMM}/*.128_{code}*.nc
    pl_dir = os.path.join(cfg['pl_root'], ym)
    if os.path.isdir(pl_dir):
        code = PL_VAR_CODE.get(var)
        if code:
            p = _best_level_file(os.path.join(pl_dir, f'*128_{code}*{date_str}00_{date_str}23.nc'))
            if p:
                return p
    # Old layout: ERA5/{BASIN}/{VAR}/*.{VAR}.*.nc
    var_dir = os.path.join(cfg['pl_root'], var)
    if os.path.isdir(var_dir):
        p = _best_level_file(os.path.join(var_dir, f'*.{var}.*.{date_str}00_{date_str}23.nc'))
        if not p:
            p = _best_level_file(os.path.join(var_dir, f'*.{var.lower()}.*.{date_str}00_{date_str}23.nc'))
        return p
    # Flat layout: daily files directly in PL/ (e.g. ERA5/2025/PL/*.nc)
    code = PL_VAR_CODE.get(var)
    if code:
        p = _best_level_file(os.path.join(cfg['pl_root'], f'*128_{code}*{date_str}00_{date_str}23.nc'))
        if p:
            return p
    return None


def _open_pl(date_str, cache, cfg):
    k = f"pl:{date_str}"
    if k in cache:
        return cache[k]
    out = {}
    for v in ['T', 'Q', 'U', 'V', 'Z']:
        p = _find_pl(v, date_str, cfg)
        out[v] = xr.open_dataset(p) if p else None
    cache[k] = out
    return cache[k]


def _get_pl_slices(t, pl_cache, cfg):
    entry = _open_pl(t.strftime('%Y%m%d'), pl_cache, cfg)
    slices = {}
    for v, ds in entry.items():
        if ds is None or v not in ds:
            slices[v] = None
        else:
            sel = _sel_time(ds, t)
            slices[v] = sel[v].squeeze() if sel is not None else None
    return slices


def close_caches(sfc_cache, pl_cache):
    for e in sfc_cache.values():
        if e and e.get('ds'):
            try: e['ds'].close()
            except Exception: pass
    for ds_dict in pl_cache.values():
        if isinstance(ds_dict, dict):
            for d in ds_dict.values():
                if d is not None:
                    try: d.close()
                    except Exception: pass


# ---------- Distance masks ----------
def _haversine_grid(ds, clat, clon):
    lk = 'latitude' if 'latitude' in ds.coords else 'lat'
    nk = 'longitude' if 'longitude' in ds.coords else 'lon'
    lats, lons = ds[lk].values, ds[nk].values
    cln = _norm_lon(clon, lons)
    phi1 = np.deg2rad(clat)
    dphi = np.deg2rad(lats[:, np.newaxis] - clat)
    dlam = np.deg2rad(lons[np.newaxis, :] - cln)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(np.deg2rad(lats[:, np.newaxis])) * np.sin(dlam / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R_EARTH_KM * c
    return d, lk, nk, lats, lons


def _annulus_mask(ds, clat, clon, rmin=200, rmax=800):
    d, lk, nk, lats, lons = _haversine_grid(ds, clat, clon)
    return xr.DataArray((d >= rmin) & (d <= rmax), coords={lk: lats, nk: lons}, dims=(lk, nk))


def _disk_mask(ds, clat, clon, rmax_km):
    d, lk, nk, lats, lons = _haversine_grid(ds, clat, clon)
    return xr.DataArray(d <= rmax_km, coords={lk: lats, nk: lons}, dims=(lk, nk))


# ---------- Env winds ----------
class BoundaryError(Exception):
    """Raised when vortex surgery hits ERA5 data boundary."""
    pass


def apply_vortex_surgery(u_2d, v_2d, lon_1d, lat_1d, storm_lon, storm_lat):
    ln = storm_lon + 360 if storm_lon < 0 else storm_lon
    u_np = np.asarray(u_2d, dtype=np.float64)
    v_np = np.asarray(v_2d, dtype=np.float64)
    lon_np = np.where(np.asarray(lon_1d) < 0, np.asarray(lon_1d) + 360, lon_1d)
    lat_np = np.asarray(lat_1d, dtype=np.float64)
    if u_np.shape != (len(lat_np), len(lon_np)):
        u_np, v_np = (u_np.T, v_np.T) if u_np.shape == (len(lon_np), len(lat_np)) else (u_np, v_np)
    try:
        vl = VORTEX_LIB(lon_np, lat_np, num_inv=3, xres=1)
        _, _, uf, vf, _, _ = vl.vortex_surgery(u_np, v_np, ln, storm_lat)
    except (IndexError, ValueError) as e:
        raise BoundaryError(
            f"Vortex surgery boundary error at lat={storm_lat:.2f} lon={ln:.2f}, "
            f"data lat=[{lat_np.min():.1f},{lat_np.max():.1f}]: {e}"
        ) from e
    li = np.argmin(np.abs(lat_np - storm_lat))
    lj = np.argmin(np.abs(lon_np - ln))
    return float(np.nan_to_num(uf[li, lj], nan=0.0)), float(np.nan_to_num(vf[li, lj], nan=0.0))


def get_env_wnds_vortex(u_d, v_d, lat, lon, lk, latk, lonk, box=25):
    if u_d is None or v_d is None or lk not in u_d.coords:
        return np.nan, np.nan, np.nan, np.nan
    lonv = np.where(np.asarray(u_d[lonk].values) < 0, np.asarray(u_d[lonk].values) + 360, u_d[lonk].values)
    latv = np.asarray(u_d[latk].values).flatten()
    ln = lon + 360 if lon < 0 else lon
    loi = np.where((latv >= lat - box) & (latv <= lat + box))[0]
    loi2 = np.where((lonv >= ln - box) & (lonv <= ln + box))[0]
    if len(loi) < 5 or len(loi2) < 5:
        return np.nan, np.nan, np.nan, np.nan
    u250_box = u_d.sel({lk: 250}, method='nearest').squeeze().isel({latk: loi, lonk: loi2}).values
    u850_box = u_d.sel({lk: 850}, method='nearest').squeeze().isel({latk: loi, lonk: loi2}).values
    v250_box = v_d.sel({lk: 250}, method='nearest').squeeze().isel({latk: loi, lonk: loi2}).values
    v850_box = v_d.sel({lk: 850}, method='nearest').squeeze().isel({latk: loi, lonk: loi2}).values
    lat_1d, lon_1d = latv[loi], lonv[loi2]
    ue250, ve250 = apply_vortex_surgery(u250_box, v250_box, lon_1d, lat_1d, ln, lat)
    ue850, ve850 = apply_vortex_surgery(u850_box, v850_box, lon_1d, lat_1d, ln, lat)
    return ue250, ve250, ue850, ve850


def get_env_wnds_annulus(u_d, v_d, lat, lon, lk, latk, lonk, rmin=200, rmax=800):
    """历史方法（与 ODE/训练数据一致）：环形掩膜(200-800km)平均 250/850 hPa 风，
    不做 vortex surgery。模型(ml_s)就是按此约定训练的，FAST 路径须与之一致。"""
    if u_d is None or v_d is None or lk is None or lk not in u_d.dims:
        return np.nan, np.nan, np.nan, np.nan
    mask = _annulus_mask(u_d, lat, lon, rmin=rmin, rmax=rmax)
    u250 = u_d.sel({lk: 250}, method='nearest').where(mask).mean(dim=[latk, lonk])
    u850 = u_d.sel({lk: 850}, method='nearest').where(mask).mean(dim=[latk, lonk])
    v250 = v_d.sel({lk: 250}, method='nearest').where(mask).mean(dim=[latk, lonk])
    v850 = v_d.sel({lk: 850}, method='nearest').where(mask).mean(dim=[latk, lonk])
    u250 = float(u250.item()) if not np.isnan(u250.item()) else np.nan
    u850 = float(u850.item()) if not np.isnan(u850.item()) else np.nan
    v250 = float(v250.item()) if not np.isnan(v250.item()) else np.nan
    v850 = float(v850.item()) if not np.isnan(v850.item()) else np.nan
    return u250, v250, u850, v850


def get_env_wnds(u_d, v_d, lat, lon, lk, latk, lonk):
    """Env winds via annulus ring-average (matches ODE/training convention).
    NOTE: switched from vortex_surgery to annulus -- surgery gave a different
    (noisier/inflated) shear convention than the annulus-trained model expects."""
    return get_env_wnds_annulus(u_d, v_d, lat, lon, lk, latk, lonk)


# ---------- Translational speed ----------
def calc_utran_vtran(lons, lats, times, window=18):
    try:
        import sphere as vsphere
    except ImportError:
        from util import sphere as vsphere
    lons_360 = np.where(np.asarray(lons) < 0, np.asarray(lons) + 360, lons)
    lats_np = np.asarray(lats)
    ts = pd.to_datetime(times)
    dt = np.diff(ts.values).astype('timedelta64[s]').astype(float)
    dt_s = np.median(dt[dt > 0]) if np.any(dt > 0) else 21600.0
    ut, vt = vsphere.calc_translational_speed(lons_360, lats_np, dt_s)
    ut = pd.Series(np.asarray(ut).flatten()).rolling(window, min_periods=1, center=True).mean().values
    vt = pd.Series(np.asarray(vt).flatten()).rolling(window, min_periods=1, center=True).mean().values
    return ut, vt


def calc_translation_speed(lats, lons, times):
    p1, p2 = np.deg2rad(lats[:-1]), np.deg2rad(lats[1:])
    dp = np.deg2rad(lats[1:] - lats[:-1])
    dl = np.deg2rad(lons[1:] - lons[:-1])
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dt = np.diff(pd.to_datetime(times).values).astype('timedelta64[s]').astype(float)
    dt[dt == 0] = 1.0
    sp = np.append(R_EARTH_M * c / dt, R_EARTH_M * c[-1] / dt[-1])
    return pd.Series(sp).rolling(18, min_periods=1, center=True).mean().values


# ---------- Physics ----------
def _sat_thermo(T_K):
    T_c = T_K - K_TO_C
    es_hpa = 6.1094 * np.exp((17.625 * T_c) / (T_c + 243.04))
    return es_hpa * PA_TO_HPA


def _calc_fast_params_dynamic(Ts_K, To_K, Ps_Pa, alpha_dyn):
    es = _sat_thermo(Ts_K)
    rs = EPS_H2O * es / (Ps_Pa - es + 1e-8)
    qs_star = rs / (1.0 + rs)
    epsilon = 0.0 if To_K >= Ts_K else (Ts_K - To_K) / Ts_K
    Ck, Cd = 1.2e-3, 1.2e-3
    kappa = (epsilon / 2.0) * (Ck / Cd) * ((Lv * qs_star) / (Rd * Ts_K))
    beta = 1.0 - epsilon - kappa
    gamma = epsilon + alpha_dyn * kappa
    return qs_star, epsilon, kappa, beta, gamma


def calc_chi(sst, psl, T_mid, p_mid, r_mid):
    """Chi (饱和熵亏) from thermo.sat_deficit -- EXACT port of ODE/training prep.
    chi = (sps-sp)/(spss-sps); 分母为海表不平衡 (spss-sps)。
    分母失效(冷水/陆上, chi<=0 或 NaN) 时设为物理上限 CHI_D=4.0, 绝不设 0。
    输入 T_mid/r_mid 为 annulus(200-800km) 环平均中层值 (与 s_ref 同口径)。"""
    if np.isnan(sst):
        sst = 273.15
    if np.isnan(psl) or np.isnan(T_mid) or np.isnan(p_mid) or np.isnan(r_mid):
        return np.nan
    sst = np.atleast_1d(sst); psl = np.atleast_1d(psl)
    T_mid = np.atleast_1d(T_mid); p_mid = np.atleast_1d(p_mid); r_mid = np.atleast_1d(r_mid)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        chi = thermo.sat_deficit(sst, psl, T_mid, p_mid, r_mid)
    chi = chi[0] if np.size(chi) > 0 else np.nan
    if np.isnan(chi) or chi <= 0 or not np.isfinite(chi):
        return float(CHI_D)
    return float(np.clip(chi, 0, CHI_D))


def calc_chi_spatial(sst_k, msl_pa, t_data, q_data, lat, lon, p_midlevel_hpa, basin='NA'):
    """Compute chi following the paper: spatial percentile within renv, median denominator within 200km.
    Returns (chi_effective, chi_raw_median) where chi_effective is the N-th percentile value."""
    params = CHI_PARAMS.get(basin, CHI_PARAMS['NA'])
    pct = params['percentile']
    renv = params['renv_km']

    if t_data is None or q_data is None:
        return np.nan, np.nan

    lkw = 'latitude' if 'latitude' in t_data.coords else 'lat'
    nkw = 'longitude' if 'longitude' in t_data.coords else 'lon'
    lk_dim = _level_key(t_data)
    pm_idx_val = None
    p_lev = t_data[lk_dim].values if lk_dim else None
    if p_lev is not None:
        if p_lev[0] < p_lev[-1]:
            p_lev = np.flip(p_lev)
        pm_idx_val = np.argmin(np.abs(p_lev - p_midlevel_hpa))
        mid_lev = p_lev[pm_idx_val]
    else:
        return np.nan, np.nan

    t_mid_2d = t_data.sel({lk_dim: mid_lev}, method='nearest').squeeze()
    q_mid_2d = q_data.sel({lk_dim: mid_lev}, method='nearest').squeeze()

    T_arr = np.asarray(t_mid_2d.values, dtype=np.float64)
    Q_arr = np.asarray(q_mid_2d.values, dtype=np.float64)
    Q_arr = np.where(np.isfinite(Q_arr) & (Q_arr < 1.0), Q_arr, np.nan)
    R_arr = Q_arr / (1.0 - Q_arr)
    P_mid_pa = float(mid_lev) * 100.0

    sst_arr = np.full_like(T_arr, sst_k)
    msl_arr = np.full_like(T_arr, msl_pa)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        chi_grid = thermo.sat_deficit(sst_arr, msl_arr, T_arr, np.full_like(T_arr, P_mid_pa), R_arr)

    chi_grid = np.where(np.isfinite(chi_grid), chi_grid, np.nan)
    chi_grid = np.where(chi_grid > 0, chi_grid, np.nan)

    renv_mask = _disk_mask(t_mid_2d, lat, lon, renv)
    renv_vals = chi_grid[renv_mask.values]
    renv_vals = renv_vals[np.isfinite(renv_vals)]

    if len(renv_vals) == 0:
        return float(CHI_D), float(CHI_D)

    chi_eff = float(np.percentile(renv_vals, pct))
    chi_eff = float(np.clip(chi_eff, 0.0, CHI_D))

    inner_mask = _disk_mask(t_mid_2d, lat, lon, 200)
    inner_vals = chi_grid[inner_mask.values]
    inner_vals = inner_vals[np.isfinite(inner_vals)]
    chi_median = float(np.median(inner_vals)) if len(inner_vals) > 0 else chi_eff

    return chi_eff, chi_median


def calc_alpha_z(v, vp, ut, hm, strat, bathy):
    if np.isfinite(bathy) and bathy >= 0:
        return 1.0
    hm, strat = float(hm) if np.isfinite(hm) else np.nan, float(strat) if np.isfinite(strat) else np.nan
    if not np.isfinite(hm) or hm <= 0 or -hm <= bathy or not (np.isfinite(strat) and strat > 0):
        return 1.0
    v_s, u_s = max(float(v), 1.0), max(float(ut), 0.5)
    with np.errstate(invalid='ignore', divide='ignore'):
        z = np.clip(0.01 * (strat ** -0.4) * hm * u_s * vp / v_s, 0, 100)
        alpha = 1.0 - 0.87 * np.exp(-z)
    return float(alpha) if np.isfinite(alpha) else 1.0


def _get_mld_strat_bathy(ds_mld, ds_strat, ds_bathy, ds_land, lat, lon, month_idx):
    def _iv(ds, var, method='linear', mi=None):
        if ds is None or var not in ds:
            return np.nan
        try:
            lc = ds.coords.get('lon', ds.coords.get('longitude')).values
            ln = _norm_lon(lon, lc)
            md = 'month' if 'month' in ds[var].dims else ('time' if 'time' in ds[var].dims else None)
            da = ds[var].isel({md: mi}) if md and mi is not None else ds[var]
            v = float(da.interp(lat=lat, lon=ln, method=method).item())
            return v if np.isfinite(v) else np.nan
        except Exception:
            return np.nan
    hm = _iv(ds_mld, 'mixed_layer', mi=month_idx)
    strat = _iv(ds_strat, 'strat', mi=month_idx)
    bathy = _iv(ds_bathy, 'bathymetry', mi=None) if ds_bathy and 'bathymetry' in ds_bathy else np.nan
    land = _iv(ds_land, 'land', method='nearest') if ds_land and 'land' in ds_land else np.nan
    return hm, strat, bathy, land


# ---------- Spatial crop (72x72) ----------
def _crop_patch(da, lat_c, lon_c, latk, lonk, size=SPATIAL_SIZE, use_360=True):
    lat_arr = np.asarray(da[latk].values).flatten()
    lon_arr = np.asarray(da[lonk].values).flatten()
    lon_n = np.where(lon_arr < 0, lon_arr + 360, lon_arr) if use_360 else lon_arr.copy()
    ln_c = lon_c + 360 if (use_360 and lon_c < 0) else float(lon_c)
    i_lat = int(np.argmin(np.abs(lat_arr - lat_c)))
    i_lon = int(np.argmin(np.abs(lon_n - ln_c)))
    half = size // 2
    n_lat, n_lon = len(lat_arr), len(lon_arr)
    i0 = max(0, min(i_lat - half, n_lat - size))
    i1 = min(n_lat, i0 + size)
    j0 = max(0, min(i_lon - half, n_lon - size))
    j1 = min(n_lon, j0 + size)
    out = np.asarray(da.isel({latk: slice(i0, i1), lonk: slice(j0, j1)}).values, dtype=np.float32)
    while out.ndim > 2:
        out = out.squeeze(axis=0)
    out = np.where(np.isfinite(out), out, 0.0)
    h, w = out.shape[-2], out.shape[-1]
    if h < size or w < size:
        out = np.pad(out, ((0, max(0, size - h)), (0, max(0, size - w))), mode='edge')
    return out


def create_spatial_fields(sfc_sel, pl_slices, lat_c, lon_c, use_360=True):
    out_3d = np.zeros((5, len(PL_LEVELS), SPATIAL_SIZE, SPATIAL_SIZE), dtype=np.float32)
    out_2d = np.zeros((2, SPATIAL_SIZE, SPATIAL_SIZE), dtype=np.float32)
    if sfc_sel is not None:
        lk_sfc = 'longitude' if 'longitude' in sfc_sel.coords else 'lon'
        ak_sfc = 'latitude' if 'latitude' in sfc_sel.coords else 'lat'
        sst_v = next((v for v in sfc_sel.data_vars if 'sst' in v.lower()), None)
        msl_v = next((v for v in sfc_sel.data_vars if 'msl' in v.lower() or 'sp' in v.lower()), None)
        if sst_v:
            g = _crop_patch(sfc_sel[sst_v], lat_c, lon_c, ak_sfc, lk_sfc, SPATIAL_SIZE, use_360)
            if np.nanmax(g) < 200:
                g = g + K_TO_C
            out_2d[0] = g
        if msl_v:
            g = _crop_patch(sfc_sel[msl_v], lat_c, lon_c, ak_sfc, lk_sfc, SPATIAL_SIZE, use_360)
            if np.nanmax(g) > 2000:
                g = g / 100.0
            out_2d[1] = g
    for vname, vidx in [('T', 0), ('Q', 1), ('U', 2), ('V', 3), ('Z', 4)]:
        da = pl_slices.get(vname) if pl_slices else None
        if da is None:
            continue
        lk = 'latitude' if 'latitude' in da.coords else 'lat'
        nk = 'longitude' if 'longitude' in da.coords else 'lon'
        lk_dim = _level_key(da)
        for li, lev in enumerate(PL_LEVELS):
            try:
                d = da.sel({lk_dim: lev}, method='nearest').squeeze() if lk_dim else da
                out_3d[vidx, li] = _crop_patch(d, lat_c, lon_c, lk, nk, SPATIAL_SIZE, use_360)
            except Exception:
                pass
    return out_3d, out_2d


# ---------- Track I/O + interpolation ----------
def read_track_6h(csv_path):
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    for c in ['lat', 'lon', 'vmax']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['time', 'lat', 'lon', 'vmax']).sort_values('time').drop_duplicates('time')
    df['vmax_ms'] = df['vmax'] * KNOTS_TO_MS
    return df.reset_index(drop=True)


def interpolate_to_1h(df6):
    df6 = df6.copy()
    df6['time'] = df6['time'].dt.round('1h')
    df6 = df6.sort_values('time').drop_duplicates('time').reset_index(drop=True)
    if len(df6) < 4:
        return df6, df6.index.values
    t0 = df6['time'].iloc[0]
    t_hours = (df6['time'] - t0).dt.total_seconds() / 3600.0
    t_max = t_hours.iloc[-1]
    t_target = np.arange(0, t_max + 0.001, 1.0)
    lon_rad = np.deg2rad(df6['lon'].values)
    lon_unwrap = np.unwrap(lon_rad)
    cs_lat = CubicSpline(t_hours, df6['lat'].values, bc_type='natural')
    cs_lon = CubicSpline(t_hours, lon_unwrap, bc_type='natural')
    lat_1h = cs_lat(t_target)
    lon_1h = (np.rad2deg(cs_lon(t_target)) + 180) % 360 - 180
    mask_v = ~np.isnan(df6['vmax_ms'].values)
    if mask_v.sum() > 2:
        f_v = PchipInterpolator(t_hours[mask_v], df6['vmax_ms'].values[mask_v])
        vmax_1h = f_v(t_target)
    else:
        vmax_1h = np.interp(t_target, t_hours, df6['vmax_ms'].values)
    df1 = pd.DataFrame({
        'time': t0 + pd.to_timedelta(t_target, unit='h'),
        'lat': lat_1h,
        'lon': lon_1h,
        'vmax_ms': vmax_1h,
    })
    idx6 = np.array([int(np.round(h)) for h in t_hours.values], dtype=np.int32)
    idx6 = idx6[idx6 < len(df1)]
    return df1, idx6


# ---------- Discover tracks ----------
def discover_tracks(data_root, basins, y0, y1):
    out = []
    for basin in basins:
        for y in range(y0, y1 + 1):
            ydir = Path(data_root) / basin / str(y)
            if not ydir.is_dir():
                continue
            for sd in sorted([x for x in ydir.iterdir() if x.is_dir()]):
                f6 = sd / 'track_intensity_6h.csv'
                f1 = sd / 'track_intensity_1h.csv'
                if f6.exists():
                    out.append((basin, y, f6))
                elif f1.exists():
                    out.append((basin, y, f1))
    return out


# ---------- Main processing ----------
class _Abort(Exception):
    pass


def process_one_storm(track_csv, era5_root_override=None):
    track_csv = Path(track_csv)
    track_name = track_csv.parent.name
    basin = infer_basin(track_csv_path=str(track_csv), hurricane_name=track_name)
    era5_cfg = get_era5_config(basin, era5_root=era5_root_override)
    df6 = read_track_6h(track_csv)
    if len(df6) < 2:
        return None
    df, idx_6h = interpolate_to_1h(df6)
    utran_all, vtran_all = calc_utran_vtran(df['lon'].values, df['lat'].values, df['time'].values)
    u_trans = calc_translation_speed(df['lat'].values, df['lon'].values, df['time'].values)
    use_360 = True
    sfc_cache, pl_cache = {}, {}
    ds_mld = xr.open_dataset(PRECALC_DIR / 'mld_climatology.nc') if (PRECALC_DIR / 'mld_climatology.nc').exists() else None
    ds_strat = xr.open_dataset(PRECALC_DIR / 'strat_climatology.nc') if (PRECALC_DIR / 'strat_climatology.nc').exists() else None
    ds_bathy = xr.open_dataset(PRECALC_DIR / 'bathymetry.nc') if (PRECALC_DIR / 'bathymetry.nc').exists() else None
    ds_land = xr.open_dataset(PRECALC_DIR / 'land.nc') if (PRECALC_DIR / 'land.nc').exists() else None

    # Load Cd interpolator from precalc (same as ODE geo.read_drag)
    from scipy.interpolate import RectBivariateSpline
    _f_Cd = None
    _Cd_const = 1.2e-3
    if (PRECALC_DIR / 'Cd.nc').exists():
        ds_cd = xr.open_dataset(PRECALC_DIR / 'Cd.nc')
        cd_lat = ds_cd['latitude'].values
        cd_lon = ds_cd['longitude'].values
        cd_vals = ds_cd['Cd'].values
        _f_Cd = RectBivariateSpline(cd_lon, cd_lat, cd_vals.T)
        ds_cd.close()

    def _get_cd_at(lon, lat):
        if _f_Cd is None:
            return _Cd_const
        lon = float(lon) if np.isfinite(lon) else 0.0
        lat = float(lat) if np.isfinite(lat) else 0.0
        lon = lon + 360 if lon < 0 else lon
        try:
            return float(_f_Cd.ev(lon, lat).flatten()[0])
        except Exception:
            return _Cd_const

    out = {k: [] for k in ['spatial_3d', 'spatial_2d', 'scalars', 'chi_ref', 's_ref', 'xs_ref',
                            'v_init', 'v_gt', 'times', 'lats', 'lons', 'env_wnds', 'utran', 'vtran',
                            'cd_ref', 'blh_ref']}
    v_init_val = None

    try:
        from tqdm import tqdm
        it = tqdm(df.iterrows(), total=len(df), desc=track_name[:30], ncols=80, leave=False)
    except ImportError:
        it = df.iterrows()

    for loop_idx, (_, row) in enumerate(it):
        qlat, qlon = row['lat'], row['lon']
        qlon_n = qlon + 360 if (use_360 and qlon < 0) else qlon
        t_curr = pd.Timestamp(row['time'])
        v_ms = row['vmax_ms']

        try:
            # SST (24h lag)
            t_lag = t_curr - pd.Timedelta(hours=24)
            sfc_lag = _open_sfc(t_lag, sfc_cache, era5_cfg)
            sfc_curr = _open_sfc(t_curr, sfc_cache, era5_cfg)
            if not sfc_lag or not sfc_lag.get('ds'):
                raise _Abort("SST missing")
            sst_s = _sel_time(sfc_lag['ds'], t_lag)
            if sst_s is None:
                raise _Abort("SST time mismatch")
            sst_var = next((v for v in sst_s.data_vars if 'sst' in v.lower()), None)
            if not sst_var:
                raise _Abort("no SST var")
            lk = 'latitude' if 'latitude' in sst_s.coords else 'lat'
            nk = 'longitude' if 'longitude' in sst_s.coords else 'lon'
            sst_raw = float(sst_s[sst_var].sel({lk: qlat, nk: _norm_lon(qlon, sst_s[nk].values)}, method='nearest').values)
            sst_k = K_TO_C if np.isnan(sst_raw) else (float(sst_raw) + K_TO_C if sst_raw < 200 else float(sst_raw))

            # MSL
            if not sfc_curr or not sfc_curr.get('ds'):
                raise _Abort("MSL missing")
            msl_s = _sel_time(sfc_curr['ds'], t_curr)
            if msl_s is None:
                raise _Abort("MSL time mismatch")
            msl_var = next((v for v in msl_s.data_vars if 'msl' in v.lower() or 'sp' in v.lower()), None)
            if not msl_var:
                raise _Abort("no MSL var")
            msl_pa = float(msl_s[msl_var].sel({lk: qlat, nk: _norm_lon(qlon, msl_s[nk].values)}, method='nearest').values)
            if msl_pa < 2000:
                msl_pa *= 100.0

            # PL slices
            pl_slices = _get_pl_slices(t_curr, pl_cache, era5_cfg)
            t_data, q_data = pl_slices.get('T'), pl_slices.get('Q')
            u_data, v_data = pl_slices.get('U'), pl_slices.get('V')
            if t_data is None or q_data is None:
                raise _Abort("T/Q missing")
            lkw = 'latitude' if 'latitude' in t_data.coords else 'lat'
            nkw = 'longitude' if 'longitude' in t_data.coords else 'lon'

            # Env winds + shear
            shear_S, env_wnds_curr = np.nan, None
            ln_pl = _norm_lon(qlon, u_data[nkw].values) if u_data is not None else None
            if u_data is not None and v_data is not None:
                lk_pl = _level_key(u_data)
                if lk_pl:
                    u250, v250, u850, v850 = get_env_wnds(u_data, v_data, qlat, qlon_n, lk_pl, lkw, nkw)
                    if not np.any(np.isnan([u250, v250, u850, v850])):
                        shear_S = np.sqrt((u250 - u850) ** 2 + (v250 - v850) ** 2)
                        env_wnds_curr = (float(u250), float(v250), float(u850), float(v850))
                    elif np.isnan(sst_raw) and ln_pl is not None:
                        u250 = float(u_data.sel({lk_pl: 250, lkw: qlat, nkw: ln_pl}, method='nearest').values)
                        v250 = float(v_data.sel({lk_pl: 250, lkw: qlat, nkw: ln_pl}, method='nearest').values)
                        u850 = float(u_data.sel({lk_pl: 850, lkw: qlat, nkw: ln_pl}, method='nearest').values)
                        v850 = float(v_data.sel({lk_pl: 850, lkw: qlat, nkw: ln_pl}, method='nearest').values)
                        if not np.any(np.isnan([u250, v250, u850, v850])):
                            shear_S = np.sqrt((u250 - u850) ** 2 + (v250 - v850) ** 2)
                            env_wnds_curr = (u250, v250, u850, v850)

            # T/Q profiles for chi + tcpyPI
            mask = _annulus_mask(t_data, qlat, qlon)
            p_lev = t_data[_level_key(t_data)].values
            t_prof = t_data.where(mask).mean(dim=[lkw, nkw]).values
            q_prof = q_data.where(mask).mean(dim=[lkw, nkw]).values
            if np.isnan(sst_raw):
                ln = _norm_lon(qlon, t_data[nkw].values)
                t_prof = np.asarray(t_data.sel({lkw: qlat, nkw: ln}, method='nearest').values).flatten()
                q_prof = np.asarray(q_data.sel({lkw: qlat, nkw: ln}, method='nearest').values).flatten()
            if p_lev[0] < p_lev[-1]:
                p_lev, t_prof, q_prof = np.flip(p_lev), np.flip(t_prof), np.flip(q_prof)

            # tcpyPI
            vmax_pi, to_val, pi_ok = np.nan, np.nan, False
            try:
                from tcpyPI import pi as calc_pi
                r_prof = (q_prof / (1 - q_prof)) * 1000.0
                res = calc_pi(sst_k - K_TO_C, msl_pa / PA_TO_HPA, p_lev,
                              t_prof - K_TO_C, r_prof, CKCD=0.9, ascent_flag=0, diss_flag=1, V_reduc=0.8)
                vmax_pi, to_val = res[0], res[3]
                pi_ok = (res[2] == 1) and np.isfinite(to_val) and (180 <= to_val <= 260)
            except Exception:
                pass

            pm_idx = np.argmin(np.abs(p_lev - P_MIDLEVEL_HPA))
            T_mid = t_prof[pm_idx]
            p_mid = float(p_lev[pm_idx]) * 100.0
            q_mid = q_prof[pm_idx]
            r_mid = q_mid / (1 - q_mid) if not np.isnan(q_mid) and q_mid < 1.0 else np.nan

            # Ocean/land
            hm, strat, bathy, land_val = _get_mld_strat_bathy(ds_mld, ds_strat, ds_bathy, ds_land, qlat, qlon_n, t_curr.month - 1)
            is_land = (land_val == 1) if np.isfinite(land_val) else (np.isfinite(bathy) and bathy >= 0)
            if is_land:
                vmax_pi = 0.0

            # chi via ODE/training method: annulus(200-800km) mid-level T_mid/r_mid
            # -> thermo.sat_deficit (single value). Matches s_ref convention + the
            # annulus-trained model. (Replaces calc_chi_spatial 90th-percentile-in-disk,
            # which spiked to CHI_D=4 and oscillated in mid-latitudes.)
            chi_val = calc_chi(sst_k, msl_pa, T_mid, p_mid, r_mid)
            if np.isnan(chi_val):
                # KEEP the step (do NOT skip) -- skipping compresses/shifts the time
                # axis (Google/IBTrACS misalignment). NaN T_mid/r_mid (no thermo data,
                # land/edge) -> treat as hostile chi_d ceiling, consistent with calc_chi.
                chi_val = float(CHI_D)

            alpha = calc_alpha_z(v_ms, vmax_pi, u_trans[loop_idx] if loop_idx < len(u_trans) else 0.5,
                                 hm, strat, bathy) if vmax_pi > 0 and v_ms > 0 else 1.0
            alpha = float(alpha) if np.isfinite(alpha) else 1.0

            if pi_ok and np.isfinite(sst_k) and np.isfinite(msl_pa) and msl_pa > 0:
                _, _, _, bet, gam = _calc_fast_params_dynamic(sst_k, to_val, msl_pa, alpha)
            else:
                bet = BETA_FAST
                gam = EPSILON_FAST + alpha * KAPPA_FAST

            # Spatial fields
            sfc_ds = sfc_curr['ds'] if sfc_curr else None
            sfc_sel = _sel_time(sfc_ds, t_curr) if sfc_ds else None
            sp_3d, sp_2d = create_spatial_fields(sfc_sel, pl_slices, qlat, qlon_n, use_360)

            out['spatial_3d'].append(sp_3d); out['spatial_2d'].append(sp_2d)
            out['env_wnds'].append(env_wnds_curr)
            out['utran'].append(utran_all[loop_idx] if loop_idx < len(utran_all) else 0.0)
            out['vtran'].append(vtran_all[loop_idx] if loop_idx < len(vtran_all) else 0.0)
            vp_out = float(vmax_pi) if np.isfinite(vmax_pi) else 0.0
            out['scalars'].append([alpha, bet, gam, vp_out])
            out['chi_ref'].append(chi_val)
            out['s_ref'].append(shear_S)
            out['xs_ref'].append(shear_S * chi_val)
            out['cd_ref'].append(_get_cd_at(qlon_n, qlat))
            blh_val = np.nan
            if sfc_curr and sfc_curr.get('ds'):
                blh_ds = _sel_time(sfc_curr['ds'], t_curr)
                if blh_ds is not None:
                    blh_v = next((v for v in blh_ds.data_vars if 'blh' in v.lower()), None)
                    if blh_v:
                        blh_val = _sel_point(blh_ds[blh_v], qlat, qlon)
            out['blh_ref'].append(float(blh_val) if np.isfinite(blh_val) else 1400.0)
            out['v_gt'].append(v_ms)
            out['times'].append(t_curr)
            out['lats'].append(qlat); out['lons'].append(qlon_n)
            v_init_val = v_ms if v_init_val is None else v_init_val
            out['v_init'].append(v_init_val)

        except BoundaryError as e:
            print(f"  [BOUNDARY] {track_name} step {loop_idx}: {e}")
            print(f"  Skipping entire storm due to ERA5 boundary limitation.")
            close_caches(sfc_cache, pl_cache)
            return None
        except _Abort as e:
            if loop_idx < 3:
                print(f"  Abort step {loop_idx}: {e}")
            continue
        except Exception as e:
            if loop_idx < 3:
                import traceback
                traceback.print_exc()
            continue

    close_caches(sfc_cache, pl_cache)
    if len(out['spatial_3d']) == 0:
        print(f"  Warning: no valid steps for {track_name}")
        return None

    env_arr = np.full((len(out['env_wnds']), 4), np.nan, dtype=np.float32)
    for i, ew in enumerate(out['env_wnds']):
        if ew and len(ew) == 4:
            env_arr[i] = ew

    return {
        'hurricane': track_name,
        'spatial_3d': np.array(out['spatial_3d'], dtype=np.float32)[np.newaxis],
        'spatial_2d': np.array(out['spatial_2d'], dtype=np.float32)[np.newaxis],
        'scalars': np.array(out['scalars'], dtype=np.float32)[np.newaxis],
        'chi_ref': np.array(out['chi_ref'], dtype=np.float32)[np.newaxis, :, np.newaxis],
        's_ref': np.array(out['s_ref'], dtype=np.float32)[np.newaxis, :, np.newaxis],
        'xs_ref': np.array(out['xs_ref'], dtype=np.float32)[np.newaxis, :, np.newaxis],
        'v_init': np.array(out['v_init'], dtype=np.float32)[np.newaxis, :, np.newaxis],
        'v_gt': np.array(out['v_gt'], dtype=np.float32)[np.newaxis, :, np.newaxis],
        'times': np.array(out['times']),
        'lats': np.array(out['lats']),
        'lons': _sanitize_lon_continuity(np.array(out['lons'])),
        'env_wnds': env_arr[np.newaxis],
        'utran': np.array(out['utran'], dtype=np.float32)[np.newaxis, :, np.newaxis],
        'vtran': np.array(out['vtran'], dtype=np.float32)[np.newaxis, :, np.newaxis],
        'cd_ref': np.array(out['cd_ref'], dtype=np.float32)[np.newaxis, :, np.newaxis],
        'blh_ref': np.array(out['blh_ref'], dtype=np.float32)[np.newaxis, :, np.newaxis],
        'idx_6h_in_1h': np.array(idx_6h, dtype=np.int32),
    }


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description='FAST_ML data prep: 6h track -> 1h ERA5 -> pkl')
    p.add_argument('--config', default='config.txt')
    p.add_argument('--data_root', default='')
    p.add_argument('--era5_root', default='')
    p.add_argument('--basins', default='')
    p.add_argument('--year_start', type=int, default=0)
    p.add_argument('--year_end', type=int, default=0)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()

    cfg = load_config(Path(args.config))
    data_root = args.data_root or cfg['output_dir']
    era5 = args.era5_root or cfg.get('era5_root', ERA5_ROOT)
    basins = parse_basins(args.basins or cfg['basins'])
    y0 = args.year_start if args.year_start > 0 else int(cfg['year_start'])
    y1 = args.year_end if args.year_end > 0 else int(cfg['year_end'])

    tracks = discover_tracks(data_root, basins, y0, y1)
    if args.limit > 0:
        tracks = tracks[:args.limit]
    print(f"Found {len(tracks)} storms for basins={basins} years={y0}-{y1}")

    ok, fail = 0, 0
    for basin, year, csv_path in tracks:
        storm_dir = csv_path.parent
        out_pkl = storm_dir / f"{storm_dir.name}_dataset.pkl"
        if out_pkl.exists() and not args.overwrite:
            print(f"[Skip] {basin} {year} {storm_dir.name}")
            continue
        try:
            ds = process_one_storm(csv_path, era5_root_override=era5)
            if ds is None:
                fail += 1
                continue
            with open(out_pkl, 'wb') as f:
                pickle.dump(ds, f, protocol=4)
            T = ds['spatial_3d'].shape[1]
            print(f"[OK] {basin} {year} {storm_dir.name} T={T}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {basin} {year} {storm_dir.name}: {e}")
            fail += 1
    print(f"Done: ok={ok}, fail={fail}")


if __name__ == '__main__':
    main()
