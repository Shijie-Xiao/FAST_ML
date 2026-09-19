"""Build 6h velocity pairs with 75% coverage requirement.

Adapted from the Reproduce/track_model pipeline (build_pairs.py).
for 2023-2025 NA hurricanes with new ECMWF data paths.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any
import pickle
import numpy as np
from config import PROCESSED_TRACKS_DIR, Earth_Radius, ALL_STORMS
from datetime import datetime


def time_to_step_index(time_val, reference_time, dt_hours=6.0,
                       tolerance_hours=3.0):
    """Convert absolute time to discrete forecast step index."""
    if reference_time is None or time_val is None:
        return None
    delta_hours = (time_val - reference_time).total_seconds() / 3600.0
    step_float = delta_hours / dt_hours
    step_idx = int(round(step_float))
    step_hours = step_idx * dt_hours
    if abs(delta_hours - step_hours) > tolerance_hours:
        return None
    return step_idx if step_idx >= 0 else None


def _compute_6h_velocities(track: Dict[str, Any]) -> Dict[str, Any]:
    """Compute 6h velocities from original track data."""
    lon, lat, time_arr = track.get("lon"), track.get("lat"), track.get("datetime")
    if lon is None or lat is None or time_arr is None:
        return {}
    if min(len(lon), len(lat), len(time_arr)) < 2:
        return {}

    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    time_np = np.array(time_arr, "datetime64[s]")
    n, dt_sec = len(lon), 6.0 * 3600.0
    u, v = np.zeros(n, float), np.zeros(n, float)
    if n >= 2:
        lon_rad, lat_rad = np.deg2rad(lon), np.deg2rad(lat)
        for i in range(1, n):
            dx = (lon_rad[i] - lon_rad[i - 1]) * np.cos(lat_rad[i - 1]) * Earth_Radius
            dy = (lat_rad[i] - lat_rad[i - 1]) * Earth_Radius
            u[i], v[i] = dx / dt_sec, dy / dt_sec
        u[0], v[0] = u[1], v[1]
    return {"lon": lon, "lat": lat, "u": u, "v": v, "time": time_np}


def _build_pairs(u, v, step_times, ref_time):
    """Build velocity pairs (u_{t-1}, v_{t-1}, u_t, v_t)."""
    if len(u) < 2:
        return np.zeros((0, 4)), np.array([], int)
    pairs = np.stack([u[:-1], v[:-1], u[1:], v[1:]], axis=1)
    steps = []
    for t in step_times[1:]:
        s = time_to_step_index(t, ref_time, 6.0, 3.0)
        if s and s > 0:
            steps.append(int(s))
    if len(steps) != len(pairs):
        steps = list(range(1, len(pairs) + 1))
    return pairs, np.array(steps, int)


def run_build_pairs():
    """Build 6h velocity pairs for all storms in processed directory."""
    if not PROCESSED_TRACKS_DIR.exists():
        print("[ERROR] Processed tracks dir not found:", PROCESSED_TRACKS_DIR)
        return

    for storm_dir in sorted(PROCESSED_TRACKS_DIR.iterdir()):
        if not storm_dir.is_dir():
            continue

        raw_file = max(storm_dir.glob("*_*_raw.pkl"),
                       key=lambda p: p.stat().st_mtime, default=None)
        if not raw_file:
            continue

        with open(raw_file, "rb") as f:
            raw = pickle.load(f)

        cfg = raw["storm_config"]
        forced_init = raw.get("forced_init_time")
        if not forced_init:
            continue

        storm_name = cfg.get("storm_name", storm_dir.name)
        processed = [(tr, r) for tr in raw["tracks"]
                     if (r := _compute_6h_velocities(tr))]
        if not processed:
            continue

        max_len = max(len(r["lon"]) for _, r in processed)
        min_req = int(np.ceil(0.75 * len(processed)))
        max_valid_step = next(
            (i - 1 for i in range(max_len)
             if sum(1 for _, r in processed if i < len(r["lon"])) < min_req),
            max_len - 1
        )

        keep_threshold = int(0.75 * max_len)
        kept_pairs, kept_steps, kept_members = [], [], []
        for tr, r in processed:
            if len(r["lon"]) < keep_threshold:
                continue
            truncate_len = min(len(r["lon"]), max_valid_step + 1)
            if truncate_len < 2:
                continue

            pairs, steps = _build_pairs(
                r["u"][:truncate_len], r["v"][:truncate_len],
                r["time"][:truncate_len], forced_init
            )
            valid = [(pairs[i], steps[i])
                     for i, s in enumerate(steps) if 0 < s <= max_valid_step]
            if valid:
                kept_pairs.append(np.array([v[0] for v in valid]))
                kept_steps.append(np.array([v[1] for v in valid]))
                kept_members.append(int(tr["member_id"]))

        if not kept_pairs:
            continue

        all_pairs = np.vstack(kept_pairs)
        all_steps = np.concatenate(kept_steps)
        time_length_hours = (max_valid_step + 1) * 6.0
        time_length_days = time_length_hours / 24.0
        filename = (f"{storm_name.lower()}_"
                    f"{forced_init.strftime('%Y%m%dT%H%M%S')}_pairs_6h.pkl")
        with open(storm_dir / filename, "wb") as f:
            pickle.dump({
                "storm_config": cfg,
                "forced_init_time": forced_init,
                "members": kept_members,
                "velocity_pairs": all_pairs,
                "step_indices": all_steps,
                "dt_hours": 6.0,
                "n_pairs": len(all_pairs),
                "n_tracks": len(kept_pairs),
            }, f)
        print(f"  {storm_name}: {len(kept_pairs)} tracks, "
              f"{len(all_pairs)} pairs, {time_length_days:.1f} days")


if __name__ == "__main__":
    run_build_pairs()
