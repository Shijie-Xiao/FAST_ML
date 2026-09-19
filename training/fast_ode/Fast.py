"""
Fast.py
Author: Shijie Xiao
Email: sxiao73@gatech.edu
Date: 2025-11-15
"""

import math
import numpy as np
from typing import Optional
from scipy.integrate import solve_ivp

from utils import (compute_vent, compute_dm_dt, compute_dv_dt,
                   compute_alpha, compute_beta, compute_gamma)
from constants import Earth_Radius, Epsilon, Kappa

from track import Track, RandomTrack
from env import BaseEnvProvider, ConstantEnvProvider

class Fast:
    """
    FAST model class for tropical cyclone intensity and track prediction
    """
    def __init__(self, env_provider: BaseEnvProvider, track_provider: Track, h_bl: float=1000.0):
        """
        Initialize FAST model
        
        Args:
            env_provider: Environment data provider (e.g., ConstantEnvProvider)
            track_provider: Track velocity provider (e.g., RandomTrack)
            h_bl: Boundary layer depth in meters (default: 1000.0)
        """
        self.env_provider = env_provider
        self.track_provider = track_provider
        self.h_bl = h_bl
        self.beta = compute_beta(Epsilon, Kappa)
    
    def dydt(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Time derivative of state vector y = [lon, lat, v, m]
        
        Args:
            t: Time in seconds
            y: State vector [longitude (deg), latitude (deg), wind speed (m/s), moisture (dimensionless)]
        
        Returns:
            Derivative vector [dLon/dt, dLat/dt, dv/dt, dm/dt]
        """
        lon, lat, v, m = y
        # Clamp to physical ranges to avoid numerical blow-ups.
        v = max(v, 0.0)
        m = min(max(m, 0.0), 1.0)

        # Get translational speed from track provider
        translational_speed = self.track_provider.get_velocity(t, lon, lat)
        u_T, v_T = translational_speed
        
        # Get environment parameters from environment provider
        env = self.env_provider.get_env(t, lon, lat)

        v_pot = env["v_pot"]
        h_m = env["h_m"]
        t_strat = env["t_strat"]
        chi = env["chi"]
        C_k = env["C_k"]
        env_wind_profile = env["env_wind_profile"]
        bathymetry = env["bathymetry"]

        # Ensure all environment parameters are finite
        if not np.isfinite(v_pot):
            v_pot = 0.0
        if not np.isfinite(h_m):
            h_m = 50.0
        if not np.isfinite(t_strat):
            t_strat = 0.2
        if not np.isfinite(chi):
            chi = 0.5
        if not np.isfinite(bathymetry):
            bathymetry = -5000.0

        # Compute FAST physics parameters
        alpha = compute_alpha(v, h_m, v_pot, translational_speed, bathymetry, t_strat)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        beta = self.beta
        gamma = compute_gamma(alpha, Epsilon, Kappa)
        vent = compute_vent(env_wind_profile, chi)
        
        # Ensure vent is finite
        if not np.isfinite(vent):
            vent = 0.0

        # Compute intensity and moisture evolution (equations 1 and 2)
        dv_dt = compute_dv_dt(v, m, v_pot, alpha, beta, gamma, C_k, self.h_bl)
        dm_dt = compute_dm_dt(v, m, vent, C_k, self.h_bl)

        if not np.isfinite(dv_dt) or not np.isfinite(dm_dt):
            print(f"[Fast] Non-finite derivative: t={t:.1f}s "
                  f"(lon={lon:.2f}, lat={lat:.2f}, v={v:.2f}, m={m:.3f}) "
                  f"alph={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}, "
                  f"v_pot={v_pot:.2f}, vent={vent:.2f}, C_k={C_k:.6f}")
            dv_dt = 0.0 if not np.isfinite(dv_dt) else dv_dt
            dm_dt = 0.0 if not np.isfinite(dm_dt) else dm_dt

        #Convert translational speed to degrees per second
        coslat = math.cos(math.radians(lat))
        coslat = coslat if abs(coslat) > 1e-6 else 1e-6

        dLondt = (u_T/(Earth_Radius*coslat))*180/math.pi
        dLatdt = (v_T/Earth_Radius)*180/math.pi
 
        return np.array([dLondt, dLatdt, dv_dt, dm_dt], dtype=float)
    def run(self,t_span,y0,t_eval:Optional[np.ndarray] = None, **kwargs):
        return solve_ivp(self.dydt, t_span, y0, t_eval=t_eval, **kwargs)








