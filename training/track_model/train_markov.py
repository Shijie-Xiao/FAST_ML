"""Train Markov model on 6h velocity pairs.

Adapted for 2023-2025 NA hurricanes with new ECMWF data paths.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
from config import PROCESSED_TRACKS_DIR


def _fit_gaussian(velocity_pairs: np.ndarray):
    """Fit Gaussian to joint (u_{t-1}, v_{t-1}, u_t, v_t) and compute conditional parameters."""
    mu_joint = np.mean(velocity_pairs, axis=0)
    Sigma_joint = np.cov(velocity_pairs, rowvar=False)
    mu_old, mu_new = mu_joint[:2], mu_joint[2:]
    Sigma_oo, Sigma_on = Sigma_joint[:2, :2], Sigma_joint[:2, 2:]
    Sigma_no, Sigma_nn = Sigma_joint[2:, :2], Sigma_joint[2:, 2:]

    try:
        L_oo = np.linalg.cholesky(Sigma_oo + np.eye(2) * 1e-8)
        Sigma_oo_inv = np.linalg.solve(L_oo.T, np.linalg.solve(L_oo, np.eye(2)))
    except np.linalg.LinAlgError:
        Sigma_oo_inv = np.linalg.pinv(Sigma_oo)

    A = Sigma_no @ Sigma_oo_inv
    Sigma_cond = Sigma_nn - A @ Sigma_on
    Sigma_cond = (Sigma_cond + Sigma_cond.T) / 2.0
    eig = np.linalg.eigvals(Sigma_cond)
    if np.any(eig <= 0):
        Sigma_cond += np.eye(2) * max(1e-6, float(-np.min(eig) + 1e-6))

    return {
        "mu_old": mu_old,
        "mu_new": mu_new,
        "Sigma_oo": Sigma_oo,
        "A": A,
        "Sigma_cond": Sigma_cond,
    }


def _fit_per_step(step_indices: np.ndarray, velocity_pairs: np.ndarray,
                  min_samples: int = 5):
    """Fit per-step Gaussian for each unique step index."""
    step_params = {}
    for s in np.unique(step_indices):
        pairs_s = velocity_pairs[step_indices == s]
        if len(pairs_s) >= min_samples:
            step_params[int(s)] = _fit_gaussian(pairs_s)
    return step_params


def run_train_markov():
    """Train Markov model on 6h velocity pairs for all storms."""
    if not PROCESSED_TRACKS_DIR.exists():
        print("[ERROR] Processed tracks dir not found:", PROCESSED_TRACKS_DIR)
        return

    for storm_dir in sorted(PROCESSED_TRACKS_DIR.iterdir()):
        if not storm_dir.is_dir():
            continue

        pairs_file = max(storm_dir.glob("*_*_pairs_6h.pkl"),
                         key=lambda p: p.stat().st_mtime, default=None)
        if not pairs_file:
            continue

        with open(pairs_file, "rb") as f:
            payload = pickle.load(f)

        velocity_pairs = payload["velocity_pairs"]
        step_indices = payload.get("step_indices")
        cfg = payload.get("storm_config", {})
        ref_time = payload.get("forced_init_time")
        dt_hours = payload.get("dt_hours", 6.0)

        if step_indices is None or len(step_indices) == 0:
            continue

        storm_name = cfg.get("storm_name", storm_dir.name)
        step_params = _fit_per_step(step_indices, velocity_pairs)
        if not step_params:
            continue

        first_step = min(step_params.keys())
        params = {
            "mu_old": step_params[first_step]["mu_old"],
            "Sigma_oo": step_params[first_step]["Sigma_oo"],
            "dt_hours": dt_hours,
            "reference_time": ref_time,
            "step_params": step_params,
            "max_reliable_step": int(step_indices.max()),
        }

        filename = (f"{storm_name.lower()}_"
                    f"{ref_time.strftime('%Y%m%dT%H%M%S')}_markov_params_6h.pkl"
                    if ref_time else "markov_params_6h.pkl")
        with open(storm_dir / filename, "wb") as f:
            pickle.dump({
                "storm_config": cfg,
                "markov_params": params,
                "velocity_pairs": velocity_pairs,
                "step_indices": step_indices,
                "dt_hours": dt_hours,
                "reference_time": ref_time,
            }, f)
        print(f"  {storm_name}: {len(velocity_pairs)} pairs, "
              f"{len(step_params)} steps")


if __name__ == "__main__":
    run_train_markov()
