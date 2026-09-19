"""
thermo_simple.py
Simplified thermodynamic calculations for potential intensity and chi.
Based on Emanuel (1995) and simplified approximations.
"""

import numpy as np

try:
    import tc_thermo
    HAS_FULL_THERMO = True
except ImportError:
    HAS_FULL_THERMO = False

# Constants
EPS = 0.622  # Ratio of molecular weights (water/air)
RD = 287.0   # Gas constant for dry air (J/kg/K)
RV = 461.5   # Gas constant for water vapor (J/kg/K)
CP = 1005.0  # Specific heat at constant pressure (J/kg/K)
LV = 2.5e6   # Latent heat of vaporization (J/kg)
G = 9.81     # Gravitational acceleration (m/s^2)

# FAST / Lin (2023) specific constants
SW = 0.80         # Surface-wind reduction factor (Table A1)
Ck_Cd_DEFAULT = 0.75  # Default Ck/Cd ratio
CHI_SIGMA = 0.5   # χσ parameter (Table A1)
CHI_A = 1.3       # χa parameter (Table A1)

def sat_vapor_pressure(T):
    """
    Calculate saturation vapor pressure using Clausius-Clapeyron equation
    
    Args:
        T: Temperature in Kelvin
    
    Returns:
        Saturation vapor pressure in Pa
    """
    # Simplified formula: e_s = 611 * exp(17.67 * (T - 273.15) / (T - 29.65))
    # More accurate: e_s = 611.2 * exp(17.67 * (T - 273.15) / (T - 35.86))
    T_C = T - 273.15
    e_s = 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))
    return e_s

def mixing_ratio_from_specific_humidity(q):
    """
    Convert specific humidity to mixing ratio
    
    Args:
        q: Specific humidity (kg/kg)
    
    Returns:
        Mixing ratio (kg/kg)
    """
    return q / (1 - q)

def relative_humidity_from_mixing_ratio(r, r_s):
    """
    Calculate relative humidity from mixing ratios
    
    Args:
        r: Mixing ratio (kg/kg)
        r_s: Saturation mixing ratio (kg/kg)
    
    Returns:
        Relative humidity (0-1)
    """
    return (r / r_s) * (1 + r_s / EPS) / (1 + r / EPS)

def compute_vpot_simple(sst, p_surf, T_env, q_env, p_env,
                        Sw: float = SW, Ck_Cd: float = Ck_Cd_DEFAULT):
    """
    Potential intensity using the full CAPE algorithm with entropy table lookup.
    REQUIRES entropy_table.npz to be available - no fallback to proxy method.
    
    This function uses CAPE_PI_vectorized which requires:
    - entropy_table.npz (for select_thermo=1, Pseudoadiabatic)
    - entropy_table_reversible.npz (for select_thermo=2, Reversible)
    
    Configure using: tc_thermo.set_thermo_config(tables_dir=Path(...))
    """
    if not HAS_FULL_THERMO:
        raise RuntimeError(
            "tc_thermo module not available. Cannot compute PI without entropy table. "
            "Please ensure tc_thermo is installed and entropy_table.npz is accessible."
        )
    
    sst_scalar = float(np.asarray(sst).squeeze())
    p_surf_scalar = float(np.asarray(p_surf).squeeze())
    T_profile = np.asarray(T_env).squeeze()
    q_profile = np.asarray(q_env).squeeze()
    p_profile = np.asarray(p_env).squeeze()
    
    if T_profile.ndim != 1 or q_profile.ndim != 1 or p_profile.ndim != 1:
        raise ValueError("Profiles must be one-dimensional for CAPE_PI_vectorized.")
    if T_profile.shape[0] != q_profile.shape[0]:
        raise ValueError("Temperature and humidity profiles must have same length.")
    
    r_profile = q_profile / np.clip(1.0 - q_profile, 1e-6, None)
    T_env_3d = T_profile.reshape(-1, 1, 1)
    r_env_3d = r_profile.reshape(-1, 1, 1)
    sst_field = np.array([[sst_scalar]], dtype=float)
    p_surf_field = np.array([[p_surf_scalar]], dtype=float)
    
    # Use entropy table lookup (no fallback)
    pi_field = tc_thermo.CAPE_PI_vectorized(
        sst_field,
        p_surf_field,
        p_profile,
        T_env_3d,
        r_env_3d
    )
    
    pi_val = float(pi_field[0, 0])
    if not np.isfinite(pi_val) or pi_val <= 0:
        raise ValueError(
            f"Invalid PI value computed: {pi_val}. "
            "Check entropy table configuration and input data quality."
        )
    
    return np.clip(pi_val, 0, 120)


# Removed _compute_vpot_proxy() - now using entropy table lookup exclusively
# No fallback method available. entropy_table.npz is REQUIRED.

def saturation_mixing_ratio(T, p):
    """Saturation mixing ratio (kg/kg) given temperature T (K) and pressure p (Pa)."""
    e_s = sat_vapor_pressure(T)
    return EPS * e_s / (p - e_s)


def moist_entropy_pseudo(T, p, q, RH):
    """
    Pseudoadiabatic moist entropy approximation (Bryan 2008).
    Adequate for computing entropy differences used in χ.
    """
    T = np.asarray(T, dtype=float)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    RH = np.asarray(RH, dtype=float)
    e = sat_vapor_pressure(T) * np.clip(RH, 0.0, 1.0)
    p_d = np.clip(p - e, 1.0, None)
    T0 = 300.0
    p0 = 1.0e5
    term_dry = CP * np.log(T / T0) - RD * np.log(p_d / p0)
    term_latent = LV * q / T
    term_humidity = -RV * q * np.log(np.clip(RH, 1e-6, 1.0))
    return term_dry + term_latent + term_humidity

def compute_chi_simple(sst, p_surf, T_mid, p_mid, q_mid):
    """
    Compute χ following Lin et al. (2023):
        χ_grid = (s* - s_m) / (s0* - s*)
        χ      = exp(log χ_grid + χσ) + χa
    where the entropies use a pseudoadiabatic approximation.
    """
    # Mid-level saturation/actual mixing ratios
    e_s_mid = sat_vapor_pressure(T_mid)
    r_s_mid = EPS * e_s_mid / (p_mid - e_s_mid)
    r_mid = mixing_ratio_from_specific_humidity(q_mid)
    RH_mid = relative_humidity_from_mixing_ratio(r_mid, r_s_mid)
    # Entropies
    s_m = moist_entropy_pseudo(T_mid, p_mid, q_mid, RH_mid)
    q_s_mid = r_s_mid / (1.0 + r_s_mid)
    s_star = moist_entropy_pseudo(T_mid, p_mid, q_s_mid, np.ones_like(RH_mid))
    e_s_surf = sat_vapor_pressure(sst)
    r_s_surf = EPS * e_s_surf / (p_surf - e_s_surf)
    q_s_surf = r_s_surf / (1.0 + r_s_surf)
    s0_star = moist_entropy_pseudo(sst, p_surf, q_s_surf, np.ones_like(sst))
    numerator = s_star - s_m
    denom = np.clip(s0_star - s_star, 1e-6, None)
    chi_grid = numerator / denom
    chi_grid = np.clip(chi_grid, 1e-6, None)
    chi = np.exp(np.log(chi_grid) + CHI_SIGMA) + CHI_A
    return np.clip(chi, 0.0, 10.0)

