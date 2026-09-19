"""Sample synthetic ensemble tracks using Markov model.

Adapted for 2023-2025 NA hurricanes.
Generates NC files with synthetic tracks that can be fed into prepare_ensemble_storm.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
from datetime import datetime
from config import (
    PROCESSED_TRACKS_DIR, SYNTH_TRACKS_DIR, Earth_Radius,
    N_TRACKS, DURATION_DAYS,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


def _load_init_cloud(storm_dir: Path):
    """Load initial lon/lat and velocities from raw file."""
    raw_file = max(storm_dir.glob("*_*_raw.pkl"),
                   key=lambda p: p.stat().st_mtime, default=None)
    if not raw_file:
        return None
    try:
        with open(raw_file, "rb") as f:
            raw = pickle.load(f)
        tracks = raw.get("tracks", [])
        lon_list, lat_list, u_list, v_list = [], [], [], []
        parent_list = []
        for tr in tracks:
            lon, lat, time_arr = (tr.get("lon"), tr.get("lat"),
                                  tr.get("datetime"))
            if (lon is None or lat is None or len(lon) < 2
                    or not time_arr or len(time_arr) < 2):
                continue
            dt_sec = (time_arr[1] - time_arr[0]).total_seconds()
            if dt_sec <= 0:
                continue
            lon_list.append(float(lon[0]))
            lat_list.append(float(lat[0]))
            dx = (np.deg2rad(float(lon[1]) - float(lon[0]))
                  * np.cos(np.deg2rad(float(lat[0]))))
            dy = np.deg2rad(float(lat[1]) - float(lat[0]))
            u_list.append(dx * Earth_Radius / dt_sec)
            v_list.append(dy * Earth_Radius / dt_sec)
            # parent attribution: GEFS raw.pkl carries 'parent_member'
            # ('c00'/'p07'...); ECMWF TIGGE raw.pkl carries numeric member_id
            # (0=control, 1..50=perturbed) -> synthesize 'e00'/'e07'
            pm = tr.get("parent_member")
            if pm is None and tr.get("member_id") is not None:
                pm = f"e{int(tr['member_id']):02d}"
            parent_list.append(pm)
        if not lon_list:
            return None
        return {
            "lon": np.array(lon_list),
            "lat": np.array(lat_list),
            "u": np.array(u_list) if u_list else None,
            "v": np.array(v_list) if v_list else None,
            "parent": parent_list,
        }
    except Exception:
        return None


def _plot_tracks(lon, lat, storm_name, out_file, max_tracks=200):
    """Plot ensemble tracks."""
    sel = np.arange(min(max_tracks, lon.shape[0]))
    fig = plt.figure(figsize=(10, 6))
    if HAS_CARTOPY:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
        transform = ccrs.PlateCarree()
    else:
        ax = fig.add_subplot(111)
        transform = None

    for i in sel:
        ax.plot(lon[i, :], lat[i, :], linewidth=0.4, alpha=0.5,
                color="#555555", transform=transform)

    ax.set_title(f"{storm_name} – synthetic tracks ({len(sel)}/{lon.shape[0]})")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def sample_for_storm(storm_dir: Path, storm_name: str,
                     n_tracks: int = 1000, duration_days: float = 5.0):
    """Sample synthetic tracks for one storm."""
    params_file = max(storm_dir.glob("*_*_markov_params_6h.pkl"),
                      key=lambda p: p.stat().st_mtime, default=None)
    if not params_file:
        print(f"  [SKIP] {storm_name}: no markov params found")
        return None

    with open(params_file, "rb") as f:
        data = pickle.load(f)

    mp = data.get("markov_params", {})
    cfg = data.get("storm_config", {})
    init_time = data.get("reference_time")
    storm_name = cfg.get("storm_name", storm_name) if cfg else storm_name
    dt_hours = float(mp.get("dt_hours", 6.0))

    if "step_params" not in mp or not mp["step_params"]:
        print(f"  [SKIP] {storm_name}: step_params missing")
        return None

    step_params = {
        int(k): {
            "mu_old": np.array(v["mu_old"]),
            "mu_new": np.array(v["mu_new"]),
            "A": np.array(v["A"]),
            "Sigma_cond": np.array(v["Sigma_cond"]),
        }
        for k, v in mp["step_params"].items()
    }

    init_cloud = _load_init_cloud(storm_dir)
    mu_old = np.array(mp["mu_old"])
    Sigma_oo = np.array(mp["Sigma_oo"])
    rng = np.random.default_rng(42)

    lon_init_arr = np.full(n_tracks, 0.0, float)
    lat_init_arr = np.full(n_tracks, 0.0, float)
    u_init_arr = v_init_arr = None
    parent_track = np.full(n_tracks, -1, int)

    if init_cloud:
        parents = init_cloud.get("parent") or []
        have_parents = len(parents) == len(init_cloud["lon"]) and \
            all(p is not None for p in parents)
        if have_parents:
            # FHLO-style member-paired bootstrap: each parent member seeds an
            # equal block of synthetic tracks (n_tracks split evenly), sampled
            # WITH replacement within the block. Track i in block j inherits
            # parent j's initial position AND its environment assignment.
            n_par = len(parents)
            base = n_tracks // n_par
            rem = n_tracks - base * n_par
            blocks, sizes = [], []
            for k in range(n_par):
                size = base + (1 if k < rem else 0)
                sizes.append(size)
                if size:
                    blocks.append(rng.integers(k, k + 1, size=size))
            idx = np.concatenate(blocks)
            parent_track = np.concatenate([
                np.full(sz, k, int) for k, sz in enumerate(sizes)
            ])
        else:
            idx = rng.integers(0, len(init_cloud["lon"]), size=n_tracks)
        lon_init_arr = init_cloud["lon"][idx]
        lat_init_arr = init_cloud["lat"][idx]
        u_seeds = init_cloud.get("u")
        v_seeds = init_cloud.get("v")
        if (u_seeds is not None and v_seeds is not None
                and len(u_seeds) > 0):
            u_init_arr = u_seeds[idx]
            v_init_arr = v_seeds[idx]

    # Cap requested length to the trained reliable horizon (no extrapolation,
    # which would cause runaway random-walk to non-physical lat/lon).
    max_reliable_step = int(mp.get("max_reliable_step", max(step_params.keys())))
    max_reliable_step = min(max_reliable_step, max(step_params.keys()))
    n_steps_req = int(duration_days * 24 / dt_hours) + 1
    n_steps = min(n_steps_req, max_reliable_step + 1)
    dt_seconds = dt_hours * 3600.0
    lon = np.zeros((n_tracks, n_steps))
    lat = np.zeros((n_tracks, n_steps))
    u = np.zeros((n_tracks, n_steps))
    v = np.zeros((n_tracks, n_steps))
    lon[:, 0] = lon_init_arr
    lat[:, 0] = lat_init_arr
    if u_init_arr is not None:
        u[:, 0] = u_init_arr
        v[:, 0] = v_init_arr
    else:
        uv0 = rng.multivariate_normal(mu_old, Sigma_oo, size=n_tracks)
        u[:, 0] = uv0[:, 0]
        v[:, 0] = uv0[:, 1]

    for i in range(1, n_steps):
        sp = step_params.get(i)
        if not sp:
            # within reliable horizon a step may still be missing -> nearest lower
            avail = [k for k in step_params if k <= i]
            sp = step_params[max(avail)] if avail else step_params[min(step_params)]
        prev_vec = np.stack([u[:, i - 1], v[:, i - 1]], axis=1)
        mean_cond = sp["mu_new"] + (sp["A"] @ (prev_vec - sp["mu_old"]).T).T
        curr = mean_cond + rng.multivariate_normal(
            [0, 0], sp["Sigma_cond"], size=n_tracks
        )
        u[:, i], v[:, i] = curr[:, 0], curr[:, 1]

        dlon_rad = ((u[:, i] * dt_seconds)
                    / (Earth_Radius * np.cos(np.deg2rad(lat[:, i - 1]))))
        dlat_rad = (v[:, i] * dt_seconds) / Earth_Radius
        lon[:, i] = ((lon[:, i - 1] + np.rad2deg(dlon_rad) + 180) % 360) - 180
        lat[:, i] = lat[:, i - 1] + np.rad2deg(dlat_rad)

    # Output to the separate PINN dataset path: ensemble_tracks/{name}/
    out_storm_dir = SYNTH_TRACKS_DIR / storm_name.lower()
    out_storm_dir.mkdir(parents=True, exist_ok=True)

    # Plot
    out_png = out_storm_dir / f"{storm_name.lower()}_synthetic_tracks.png"
    _plot_tracks(lon, lat, storm_name, out_png)

    # Save as NetCDF (canonical name consumed by prepare_ensemble_storm.py)
    nc_path = out_storm_dir / f"synthetic_tracks_{n_tracks}members.nc"
    if HAS_XARRAY:
        try:
            data_vars = {
                "lon": (["track", "time"], lon),
                "lat": (["track", "time"], lat),
                "u": (["track", "time"], u),
                "v": (["track", "time"], v),
            }
            attrs = {
                "storm": storm_name,
                "dt_hours": dt_hours,
                "init_time": str(init_time),
                "n_tracks": n_tracks,
                "duration_days": duration_days,
            }
            # member-paired inheritance (GEFS path): track i inherits seed AND
            # environment from parent ensemble member parent_track[i]
            if init_cloud and np.any(parent_track >= 0):
                parents = init_cloud.get("parent") or []
                codes = [p if p is not None else "" for p in parents]
                data_vars["parent_track"] = (["track"], parent_track)
                attrs["parent_members"] = ",".join(codes)
                attrs["assignment"] = "member_paired"
            else:
                attrs["assignment"] = "pooled"
            xr.Dataset(
                data_vars,
                coords={
                    "track": np.arange(n_tracks),
                    "time": np.arange(n_steps) * dt_hours * 3600.0,
                },
                attrs=attrs,
            ).to_netcdf(nc_path, format="NETCDF4")
            print(f"  {storm_name}: saved {n_tracks} tracks -> {nc_path}")
        except Exception as e:
            print(f"  [WARN] {storm_name}: NC save failed: {e}")

    return nc_path if nc_path.exists() else None


def run_sample_tracks(n_tracks=N_TRACKS, duration_days=DURATION_DAYS):
    """Sample synthetic tracks for all storms."""
    if not PROCESSED_TRACKS_DIR.exists():
        print("[ERROR] Processed tracks dir not found:", PROCESSED_TRACKS_DIR)
        return

    for storm_dir in sorted(PROCESSED_TRACKS_DIR.iterdir()):
        if not storm_dir.is_dir():
            continue
        sample_for_storm(storm_dir, storm_dir.name,
                         n_tracks=n_tracks, duration_days=duration_days)


if __name__ == "__main__":
    run_sample_tracks()
