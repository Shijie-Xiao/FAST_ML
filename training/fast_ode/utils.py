# sxiao73@gatech.edu
# @title Import libraries and constants
import math
import numpy as np
from constants import Epsilon, Kappa

def calculate_z(v, h_m, v_pot, u_T, t_strat):
    # Prevent division by zero when v is very small.
    v_eff = max(abs(v), 5.0)
    return 0.01 * (t_strat**-0.4) * h_m * u_T * (v_pot / v_eff)

def compute_epsilon():
    return Epsilon

def compute_kappa():
    return Kappa

def compute_beta(epsilon, kappa):
    return 1-epsilon-kappa

def compute_gamma(alpha, Epsilon = Epsilon, Kappa = Kappa):
    return Epsilon + alpha*Kappa

def compute_alpha(v, h_m, v_pot, translational_speed, bathymetry, t_strat):
    u_T = math.hypot(translational_speed[0], translational_speed[1])

    if bathymetry >= 0 or (-h_m <= bathymetry) or t_strat == 0:
        return 1.0
    else:
        z = calculate_z(v, h_m, v_pot, u_T, t_strat)
        z = max(0.0, min(z, 100.0))
        alpha = 1.0 - 0.87 * math.exp(-z)
        return alpha

def compute_S(env_wind_profile):
    u250, v250, u850, v850 = env_wind_profile
    shear_vector_x = u850 - u250
    shear_vector_y = v850 - v250
    S = math.sqrt(shear_vector_x**2 + shear_vector_y**2)
    return S

def compute_dv_dt(v, m, v_pot_current, alpha, beta, gamma, C_k, h):
    dv_dt = 0.5 * C_k / h * (alpha * beta * (v_pot_current**2) * (m**3)-(1-gamma*m**3)*(v**2))
    if math.isnan(dv_dt):
        return 0.0
    else:
        return dv_dt        

def compute_vent(env_wind_profile, chi):
    S = compute_S(env_wind_profile)
    vent = chi * S
    # Ensure vent is finite
    if not np.isfinite(vent):
        return 0.0
    return vent

def compute_dm_dt(v, m, vent, C_k, h):
    return 0.5 * C_k / h * ((1-m)*v - vent*m)



    





