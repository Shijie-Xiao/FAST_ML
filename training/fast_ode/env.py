"""
env.py
Author: Shijie Xiao
Email: sxiao73@gatech.edu
Date: 2025-11-15
"""

from dataclasses import dataclass
from typing import Dict, Any, Sequence, Optional
from pathlib import Path
from datetime import datetime, timedelta
import importlib
import numpy as np
import xarray as xr
from scipy.interpolate import RectBivariateSpline

# Try to import vortex inversion library
try:
    from vortex_inversion_main.vortex_lib import VORTEX_LIB
    HAS_VORTEX_LIB = True
except ImportError:
    HAS_VORTEX_LIB = False
    VORTEX_LIB = None
try:
    from vortex_inversion_main import VORTEX_LIB
    HAS_VORTEX_INV = True
except Exception:
    HAS_VORTEX_INV = False

class BaseEnvProvider:
    """
    base env provider class
    """
    def get_env(self, t: float, lon: float, lat: float) -> Dict[str, Any]:
        """
        get the environment at time t, lon, lat
        """
        raise NotImplementedError("Subclasses must implement this method")
    
class ConstantEnvProvider(BaseEnvProvider):
    """
    constant env provider class
    """
    v_pot: float
    h_m: float
    t_strat: float
    chi: float
    C_k: float
    env_wind_profile: Sequence[float]
    bathymetry: float

    def __init__(self, v_pot: float = 80.0, h_m: float = 50.0, t_strat: float = 0.2, chi: float = 0.5, C_k: float = 0.0015, env_wind_profile: Sequence[float] = (5.0, 0.0, 0.0, 0.0), bathymetry: float = -5000.0):
        self.v_pot = float(v_pot)
        self.h_m = float(h_m)
        self.t_strat = float(t_strat)
        self.chi = float(chi)
        self.C_k = float(C_k)
        self.env_wind_profile = env_wind_profile
        self.bathymetry = float(bathymetry)
    def get_env(self, t: float, lon: float, lat: float) -> Dict[str, Any]:
        """
        get the environment at time t, lon, lat
        """
        return {
            "v_pot": float(self.v_pot),
            "h_m": float(self.h_m),
            "t_strat": float(self.t_strat),
            "chi": float(self.chi),
            "C_k": float(self.C_k),
            "env_wind_profile": self.env_wind_profile,
            "bathymetry": float(self.bathymetry),
            "rh_mid": None,
            "is_land": self.bathymetry >= 0.0
        }


class ERA5EnvProvider(BaseEnvProvider):
    """
    Environment provider using ERA5 6-hourly data and climatological fields.
    Handles interpolation from monthly climatology to 6-hourly time steps.
    """
    
    def __init__(self, 
                 era5_dir: Path,
                 data_dir: Path,
                 init_time: datetime,
                 basin: str = 'NA',
                 enable_vortex_inversion: bool = False):
        """
        Initialize ERA5 environment provider
        
        Args:
            era5_dir: Directory containing ERA5 data (ERA5/irma/)
            data_dir: Directory containing static/climatological data (Intensity/data/)
            init_time: Initialization time for the simulation
            basin: Basin code ('NA', 'EP', etc.) for coordinate transformation
        """
        self.era5_dir = Path(era5_dir)
        self.data_dir = Path(data_dir)
        self.init_time = init_time
        self.basin = basin
        self.vortex_removal_radius_km = 400.0  # r* per Lin et al. (2020)
        # Allow caller to disable vortex inversion (default False as requested)
        self.enable_vortex_inversion = HAS_VORTEX_LIB and bool(enable_vortex_inversion)
        self._vortex_lib_cache = {}
        self._vortex_removal_count = 0
        self._vortex_fallback_count = 0
        # Optional access to TC-risk namelist (for scaling PI and m_init)
        try:
            self.tcr_namelist = importlib.import_module("tropical_cyclone_risk.namelist")
        except ModuleNotFoundError:
            self.tcr_namelist = None
        
        # Initialize vortex library (lazy initialization)
        self.vortex_lib = None
        self._vortex_removal_count = 0
        self._vortex_fallback_count = 0
        
        # Load ERA5 6-hourly data
        print("[ERA5Env] Loading ERA5 6-hourly data...")
        self._load_era5_data()
        
        # Load climatological ocean data (monthly)
        print("[ERA5Env] Loading climatological ocean data...")
        self._load_climatology()
        
        # Load static fields
        print("[ERA5Env] Loading static fields...")
        self._load_static_fields()

        # Load precomputed thermo fields if available
        print("[ERA5Env] Loading thermodynamic fields...")
        self._load_thermo_fields()
        
        # Print vortex removal status
        if HAS_VORTEX_LIB:
            print("[ERA5Env] ✓ Vortex removal enabled (r*=400 km, following Lin et al. 2020)")
        else:
            print("[ERA5Env] ⚠ Vortex removal disabled (pyamg/pyshtools not available)")
        
        # Import thermo calculation
        try:
            from thermo_simple import compute_vpot_simple, compute_chi_simple
            self.compute_vpot = compute_vpot_simple
            self.compute_chi = compute_chi_simple
        except ImportError:
            print("[ERA5Env] Warning: thermo_simple not found, using simplified approximations")
            self.compute_vpot = self._vpot_approx
            self.compute_chi = self._chi_approx
    
    def _find_time_coord(self, ds):
        """Find time coordinate name in dataset"""
        # Common time coordinate names
        time_names = ['time', 'valid_time', 't', 'datetime']
        for name in time_names:
            if name in ds.coords:
                return name
        # If not found, check for coordinates with 'time' in the name
        for coord in ds.coords:
            if 'time' in coord.lower():
                return coord
        return None
    
    def _find_lon_lat_coords(self, ds):
        """Find longitude and latitude coordinate names in dataset"""
        lon_names = ['lon', 'longitude', 'x']
        lat_names = ['lat', 'latitude', 'y']
        
        lon_coord = None
        lat_coord = None
        
        for name in lon_names:
            if name in ds.coords:
                lon_coord = name
                break
        
        for name in lat_names:
            if name in ds.coords:
                lat_coord = name
                break
        
        return lon_coord, lat_coord
    
    def _find_level_coord(self, ds):
        """Find pressure level coordinate name in dataset"""
        level_names = ['level', 'pressure_level', 'plev', 'pressure']
        for name in level_names:
            if name in ds.coords:
                return name
        # If not found, check for coordinates with 'level' or 'pressure' in the name
        for coord in ds.coords:
            if 'level' in coord.lower() or 'pressure' in coord.lower():
                return coord
        return None
    
    def _load_era5_data(self):
        """Load ERA5 6-hourly data"""
        single_dir = self.era5_dir / "single"
        press_dir = self.era5_dir / "press"
        
        # Load SST
        sst_files = sorted(single_dir.glob("era5_sst_6h_*.nc"))
        if sst_files:
            self.ds_sst = xr.open_mfdataset(sst_files, combine='by_coords')
            # Get variable name
            sst_vars = [v for v in self.ds_sst.data_vars if 'sst' in v.lower() or 'temperature' in v.lower()]
            self.sst_var = sst_vars[0] if sst_vars else list(self.ds_sst.data_vars)[0]
            # Find time and spatial coordinates
            self.time_coord = self._find_time_coord(self.ds_sst)
            if self.time_coord is None:
                raise ValueError("Could not find time coordinate in SST dataset")
            self.lon_coord, self.lat_coord = self._find_lon_lat_coords(self.ds_sst)
            if self.lon_coord is None or self.lat_coord is None:
                raise ValueError("Could not find lon/lat coordinates in SST dataset")
        else:
            raise FileNotFoundError(f"No SST files found in {single_dir}")
        
        # Load surface pressure
        sp_files = sorted(single_dir.glob("era5_sp_6h_*.nc"))
        if sp_files:
            self.ds_sp = xr.open_mfdataset(sp_files, combine='by_coords')
            sp_vars = [v for v in self.ds_sp.data_vars if 'sp' in v.lower() or 'pressure' in v.lower()]
            self.sp_var = sp_vars[0] if sp_vars else list(self.ds_sp.data_vars)[0]
        else:
            raise FileNotFoundError(f"No surface pressure files found in {single_dir}")
        
        # Load temperature
        t_files = sorted(press_dir.glob("era5_t_6h_*.nc"))
        if t_files:
            self.ds_t = xr.open_mfdataset(t_files, combine='by_coords')
            t_vars = [v for v in self.ds_t.data_vars if 't' in v.lower() or 'temperature' in v.lower()]
            self.t_var = t_vars[0] if t_vars else list(self.ds_t.data_vars)[0]
            # Find pressure level coordinate
            self.level_coord = self._find_level_coord(self.ds_t)
            if self.level_coord is None:
                raise ValueError("Could not find pressure level coordinate in temperature dataset")
        else:
            raise FileNotFoundError(f"No temperature files found in {press_dir}")
        
        # Load specific humidity
        q_files = sorted(press_dir.glob("era5_q_6h_*.nc"))
        if q_files:
            self.ds_q = xr.open_mfdataset(q_files, combine='by_coords')
            q_vars = [v for v in self.ds_q.data_vars if 'q' in v.lower() or 'humidity' in v.lower()]
            self.q_var = q_vars[0] if q_vars else list(self.ds_q.data_vars)[0]
        else:
            raise FileNotFoundError(f"No specific humidity files found in {press_dir}")
        
        # Load wind components
        u_files = sorted(press_dir.glob("era5_u_6h_*.nc"))
        v_files = sorted(press_dir.glob("era5_v_6h_*.nc"))
        if u_files and v_files:
            self.ds_u = xr.open_mfdataset(u_files, combine='by_coords')
            self.ds_v = xr.open_mfdataset(v_files, combine='by_coords')
            u_vars = [v for v in self.ds_u.data_vars if 'u' in v.lower() or 'wind' in v.lower()]
            v_vars = [v for v in self.ds_v.data_vars if 'v' in v.lower() or 'wind' in v.lower()]
            self.u_var = u_vars[0] if u_vars else list(self.ds_u.data_vars)[0]
            self.v_var = v_vars[0] if v_vars else list(self.ds_v.data_vars)[0]
        else:
            raise FileNotFoundError(f"No wind files found in {press_dir}")
        
        # Get time dimension length
        time_len = len(self.ds_sst[self.time_coord])
        print(f"[ERA5Env] Loaded ERA5 data: {time_len} time steps (time coord: '{self.time_coord}')")
    
    def _load_climatology(self):
        """Load monthly climatological data (mld, strat)"""
        # Load mixed layer depth
        mld_file = self.data_dir / "mld_climatology.nc"
        if mld_file.exists():
            ds_mld = xr.open_dataset(mld_file)
            # Create interpolator for monthly to 6-hourly
            self.mld_data = ds_mld
            # Store months for interpolation
            self.mld_months = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])  # 13 wraps to Jan
        else:
            print(f"[ERA5Env] Warning: {mld_file} not found, using default MLD")
            self.mld_data = None
        
        # Load stratification
        strat_file = self.data_dir / "strat_climatology.nc"
        if strat_file.exists():
            ds_strat = xr.open_dataset(strat_file)
            self.strat_data = ds_strat
        else:
            print(f"[ERA5Env] Warning: {strat_file} not found, using default stratification")
            self.strat_data = None
    
    def _load_static_fields(self):
        """Load static fields (bathymetry, land, Cd)"""
        # Load bathymetry
        bathy_file = self.data_dir / "bathymetry.nc"
        if bathy_file.exists():
            ds_bathy = xr.open_dataset(bathy_file)
            lon_b = ds_bathy['lon'].values
            lat_b = ds_bathy['lat'].values
            bathy = ds_bathy['bathymetry'].values
            # Create interpolator
            self.f_bath = RectBivariateSpline(lon_b, lat_b, bathy.T, kx=1, ky=1)
        else:
            print(f"[ERA5Env] Warning: {bathy_file} not found, using default bathymetry")
            self.f_bath = None
        
        # Load land mask
        land_file = self.data_dir / "land.nc"
        if land_file.exists():
            ds_land = xr.open_dataset(land_file)
            lon_l = ds_land['lon'].values
            lat_l = ds_land['lat'].values
            land = ds_land['land'].values
            self.f_land = RectBivariateSpline(lon_l, lat_l, land.T, kx=1, ky=1)
        else:
            print(f"[ERA5Env] Warning: {land_file} not found, using default land mask")
            self.f_land = None
        
        # Load drag coefficient
        cd_file = self.data_dir / "Cd.nc"
        if cd_file.exists():
            ds_cd = xr.open_dataset(cd_file)
            lon_cd = ds_cd['longitude'].values if 'longitude' in ds_cd.coords else ds_cd['lon'].values
            lat_cd = ds_cd['latitude'].values if 'latitude' in ds_cd.coords else ds_cd['lat'].values
            cd = ds_cd['Cd'].values
            # Normalize Cd (as in geo.py)
            cd_gradient = cd / (1 + 250.0 * cd)
            cd_norm = cd_gradient / np.nanmin(cd_gradient)
            cd_normalized = 0.0015 * cd_norm  # Base Cd from constants
            self.f_cd = RectBivariateSpline(lon_cd, lat_cd, cd_normalized.T, kx=1, ky=1)
        else:
            print(f"[ERA5Env] Warning: {cd_file} not found, using default Cd")
            self.f_cd = None

    def _load_thermo_fields(self):
        """Load precomputed thermodynamic fields (vmax, chi, rh_mid)"""
        thermo_files = sorted(self.data_dir.glob("thermo_*.nc"))
        if not thermo_files:
            thermo_files = sorted(self.data_dir.glob("thermo*.nc"))
        if thermo_files:
            thermo_file = thermo_files[0]
            try:
                self.ds_thermo = xr.open_dataset(thermo_file)
                self.thermo_time = self.ds_thermo['time']
                self.thermo_lon = self.ds_thermo['lon'].values
                self.thermo_lat = self.ds_thermo['lat'].values
                print(f"[ERA5Env] Loaded thermo dataset: {thermo_file.name}")
            except Exception as exc:
                print(f"[ERA5Env] Warning: failed to load {thermo_file}: {exc}")
                self.ds_thermo = None
        else:
            print(f"[ERA5Env] Warning: No thermo_*.nc files in {self.data_dir}")
            self.ds_thermo = None
    
    def _interp_monthly_to_time(self, month_data, current_time):
        """Interpolate monthly climatology to specific time"""
        if month_data is None:
            return None
        
        month = current_time.month
        # Use nearest month (simple approach)
        # For more accurate interpolation, could use linear interpolation between months
        month_idx = month - 1  # 0-indexed
        
        return month_idx
    
    def _get_mld(self, current_time, lon, lat):
        """Get mixed layer depth at time and location"""
        if self.mld_data is None:
            return 50.0  # Default MLD
        
        month_idx = self._interp_monthly_to_time(self.mld_data, current_time)
        mld_month = self.mld_data['mixed_layer'][:, :, month_idx].values
        
        # Interpolate spatially
        lon_mld = self.mld_data['lon'].values
        lat_mld = self.mld_data['lat'].values
        
        # Simple nearest neighbor interpolation
        lon_idx = np.argmin(np.abs(lon_mld - lon))
        lat_idx = np.argmin(np.abs(lat_mld - lat))
        
        return float(mld_month[lat_idx, lon_idx])
    
    def _get_strat(self, current_time, lon, lat):
        """Get thermal stratification at time and location"""
        if self.strat_data is None:
            return 0.2  # Default stratification (K/100m)
        
        month_idx = self._interp_monthly_to_time(self.strat_data, current_time)
        strat_month = self.strat_data['strat'][:, :, month_idx].values
        
        # Interpolate spatially
        lon_strat = self.strat_data['lon'].values
        lat_strat = self.strat_data['lat'].values
        
        # Simple nearest neighbor interpolation
        lon_idx = np.argmin(np.abs(lon_strat - lon))
        lat_idx = np.argmin(np.abs(lat_strat - lat))
        
        return float(strat_month[lat_idx, lon_idx])
    
    def _get_bathymetry(self, lon, lat):
        """Get bathymetry at location"""
        if self.f_bath is None:
            return -5000.0  # Default deep ocean
        
        try:
            return float(self.f_bath.ev(lon, lat)[0])
        except:
            return -5000.0
    
    def _get_land(self, lon, lat):
        """Check if location is over land"""
        if self.f_land is None:
            return False
        
        try:
            land_val = self.f_land.ev(lon, lat)[0]
            return land_val > 0.5
        except:
            return False
    
    def _get_cd(self, lon, lat):
        """Get drag coefficient at location"""
        if self.f_cd is None:
            return 0.0015  # Default Cd
        
        try:
            return float(self.f_cd.ev(lon, lat)[0])
        except:
            return 0.0015

    # ------------------------------------------------------------------
    # Vortex removal for environmental wind (vertical shear calculation)
    # ------------------------------------------------------------------
    def _prepare_vortex_lib(self, lon_vals, lat_vals):
        """Initialize or reuse VORTEX_LIB for given grid."""
        key = (len(lon_vals), len(lat_vals), float(lon_vals[0]), float(lat_vals[0]))
        if key not in self._vortex_lib_cache:
            vlib = VORTEX_LIB(lon_vals, lat_vals, xres=1)
            vlib.d_crit = self.vortex_removal_radius_km
            self._vortex_lib_cache[key] = vlib
        return self._vortex_lib_cache[key]

    def _remove_vortex_env_wind(self, current_time, target_lon, target_lat, level_hpa):
        """
        Remove TC vortex circulation and return environmental wind at target point.
        Returns tuple (u_env, v_env) or None on failure.
        """
        try:
            u_layer = self.ds_u[self.u_var].sel(**{self.level_coord: level_hpa,
                                                   self.time_coord: current_time},
                                                method='nearest')
            v_layer = self.ds_v[self.v_var].sel(**{self.level_coord: level_hpa,
                                                   self.time_coord: current_time},
                                                method='nearest')
            lon_vals = u_layer[self.lon_coord].values
            lat_vals = u_layer[self.lat_coord].values

            # Ensure lon in [0, 360)
            lon_vals_wrapped = np.mod(lon_vals, 360.0)
            lon_sorted_idx = np.argsort(lon_vals_wrapped)
            lon_vals_wrapped = lon_vals_wrapped[lon_sorted_idx]

            # If lat is ascending, flip to descending (library expects 90 -> -90)
            lat_vals_desc = lat_vals
            flip_lat = False
            if lat_vals[0] < lat_vals[-1]:
                lat_vals_desc = lat_vals[::-1]
                flip_lat = True

            u_field = u_layer.values
            v_field = v_layer.values
            # Sort lon dimension
            u_field = u_field[:, lon_sorted_idx]
            v_field = v_field[:, lon_sorted_idx]
            # Flip lat if needed
            if flip_lat:
                u_field = u_field[::-1, :]
                v_field = v_field[::-1, :]

            vlib = self._prepare_vortex_lib(lon_vals_wrapped, lat_vals_desc)
            # Wrap target lon to 0-360
            target_lon_wrapped = target_lon
            if np.min(lon_vals_wrapped) >= 0 and target_lon_wrapped < 0:
                target_lon_wrapped += 360.0
            x, y, u_env_grid, v_env_grid, _, _ = vlib.vortex_surgery(
                u_field, v_field, target_lon_wrapped, target_lat
            )

            # Bilinear interpolation using xarray for convenience
            da_u = xr.DataArray(
                u_env_grid,
                coords={self.lat_coord: lat_vals_desc, self.lon_coord: lon_vals_wrapped},
                dims=[self.lat_coord, self.lon_coord]
            )
            da_v = xr.DataArray(
                v_env_grid,
                coords={self.lat_coord: lat_vals_desc, self.lon_coord: lon_vals_wrapped},
                dims=[self.lat_coord, self.lon_coord]
            )
            u_env = float(da_u.interp(**{self.lon_coord: target_lon_wrapped,
                                         self.lat_coord: target_lat},
                                      kwargs={"fill_value": "extrapolate"}).values)
            v_env = float(da_v.interp(**{self.lon_coord: target_lon_wrapped,
                                         self.lat_coord: target_lat},
                                      kwargs={"fill_value": "extrapolate"}).values)
            return u_env, v_env
        except Exception as exc:
            print(f"[ERA5Env] Warning: vortex inversion failed ({exc}); fallback to raw winds.")
            return None
    
    def _get_env_wind(self, current_time, lon, lat):
        """
        Get environment wind profile (250 and 850 hPa) with optional vortex removal
        
        Args:
            current_time: Time to query
            lon: Longitude (degrees)
            lat: Latitude (degrees)
        
        Returns:
            Tuple of (u_250, v_250, u_850, v_850) in m/s
        """
        # Prefer vortex-removed environmental winds if available
        if self.enable_vortex_inversion:
            try:
                res_250 = self._remove_vortex_env_wind(current_time, lon, lat, 250)
                res_850 = self._remove_vortex_env_wind(current_time, lon, lat, 850)
                
                if res_250 is not None and res_850 is not None:
                    u_250, v_250 = res_250
                    u_850, v_850 = res_850
                    self._vortex_removal_count += 1
                    return (float(u_250), float(v_250), float(u_850), float(v_850))
            except Exception:
                # Fall through to fallback
                pass
        
        # Fallback: direct interpolation without vortex removal
        try:
            self._vortex_fallback_count += 1
            
            u_250 = self.ds_u[self.u_var].sel(**{self.level_coord: 250}, method='nearest').sel(
                **{self.time_coord: current_time}, method='nearest'
            ).interp(**{self.lon_coord: lon, self.lat_coord: lat}).values
            v_250 = self.ds_v[self.v_var].sel(**{self.level_coord: 250}, method='nearest').sel(
                **{self.time_coord: current_time}, method='nearest'
            ).interp(**{self.lon_coord: lon, self.lat_coord: lat}).values
            u_850 = self.ds_u[self.u_var].sel(**{self.level_coord: 850}, method='nearest').sel(
                **{self.time_coord: current_time}, method='nearest'
            ).interp(**{self.lon_coord: lon, self.lat_coord: lat}).values
            v_850 = self.ds_v[self.v_var].sel(**{self.level_coord: 850}, method='nearest').sel(
                **{self.time_coord: current_time}, method='nearest'
            ).interp(**{self.lon_coord: lon, self.lat_coord: lat}).values
            
            return (float(u_250), float(v_250), float(u_850), float(v_850))
        except Exception as e:
            print(f"[ERA5Env] Warning: Error getting env wind: {e}")
            return (0.0, 0.0, 0.0, 0.0)
    
    def _get_vpot_chi(self, current_time, lon, lat):
        """Calculate v_pot, chi, and rh_mid"""
        if self.ds_thermo is not None:
            try:
                thermo_point = self.ds_thermo.sel(time=current_time, method='nearest')
                vmax_val = float(thermo_point['vmax'].interp(lon=lon, lat=lat))
                chi_val = float(thermo_point['chi'].interp(lon=lon, lat=lat))
                rh_val = float(thermo_point['rh_mid'].interp(lon=lon, lat=lat))
                scale = 1.0
                if self.tcr_namelist is not None:
                    try:
                        scale = (self.tcr_namelist.PI_reduc *
                                 np.sqrt(self.tcr_namelist.Ck / self.tcr_namelist.Cd))
                    except Exception:
                        scale = 1.0
                v_pot = vmax_val * scale
                # Ensure all values are finite
                if not np.isfinite(v_pot):
                    v_pot = 0.0
                if not np.isfinite(chi_val):
                    chi_val = 0.5
                if not np.isfinite(rh_val):
                    rh_val = None
                return float(v_pot), float(chi_val), rh_val
            except Exception as exc:
                # fall back to dynamic computation
                pass
        
        try:
            # Get data at nearest time with error handling
            try:
                sst = self.ds_sst[self.sst_var].sel(**{self.time_coord: current_time}, method='nearest').interp(**{self.lon_coord: lon, self.lat_coord: lat}, kwargs={"fill_value": "extrapolate"}).values
                sp = self.ds_sp[self.sp_var].sel(**{self.time_coord: current_time}, method='nearest').interp(**{self.lon_coord: lon, self.lat_coord: lat}, kwargs={"fill_value": "extrapolate"}).values
            except Exception:
                # If interpolation fails, return default values
                return 0.0, 0.5, None
            
            # Check if interpolated values are valid
            if np.isnan(sst) or np.isnan(sp) or not np.isfinite(sst) or not np.isfinite(sp):
                return 0.0, 0.5, None
            
            # Get temperature and humidity profiles
            t_profile = self.ds_t[self.t_var].sel(**{self.time_coord: current_time}, method='nearest')
            q_profile = self.ds_q[self.q_var].sel(**{self.time_coord: current_time}, method='nearest')
            
            # Get mid-level (500 hPa) for chi
            p_mid = 50000.0  # 500 hPa in Pa
            try:
                t_mid = t_profile.sel(**{self.level_coord: 500}, method='nearest').interp(**{self.lon_coord: lon, self.lat_coord: lat}, kwargs={"fill_value": "extrapolate"}).values
                q_mid = q_profile.sel(**{self.level_coord: 500}, method='nearest').interp(**{self.lon_coord: lon, self.lat_coord: lat}, kwargs={"fill_value": "extrapolate"}).values
            except Exception:
                return 0.0, 0.5, None
            
            # Check mid-level values
            if np.isnan(t_mid) or np.isnan(q_mid) or not np.isfinite(t_mid) or not np.isfinite(q_mid):
                return 0.0, 0.5, None
            
            # Get full profiles for v_pot
            try:
                levels = t_profile[self.level_coord].values
                t_full = t_profile.interp(**{self.lon_coord: lon, self.lat_coord: lat}, kwargs={"fill_value": "extrapolate"}).values
                q_full = q_profile.interp(**{self.lon_coord: lon, self.lat_coord: lat}, kwargs={"fill_value": "extrapolate"}).values
            except Exception:
                return 0.0, 0.5, None
            
            # Check profile values
            if np.any(np.isnan(t_full)) or np.any(np.isnan(q_full)) or not np.all(np.isfinite(t_full)) or not np.all(np.isfinite(q_full)):
                return 0.0, 0.5, None
            
            p_full = levels * 100.0  # Convert hPa to Pa
            
            # Convert SST to Kelvin if needed
            if sst < 200:
                sst = sst + 273.15
            
            # Calculate v_pot and chi
            try:
                v_pot = self.compute_vpot(sst, sp, t_full, q_full, p_full)
                chi = 0.4 * self.compute_chi(sst, sp, t_mid, p_mid, q_mid)
                rh_mid = float(self._estimate_rh(t_mid, q_mid, p_mid))
            except Exception:
                return 0.0, 0.5, None
            
            # Ensure all values are finite
            v_pot = float(np.clip(v_pot, 0.0, 150.0)) if np.isfinite(v_pot) else 0.0
            chi = float(np.clip(chi, 0.0, 10.0)) if np.isfinite(chi) else 0.5
            rh_mid = float(rh_mid) if np.isfinite(rh_mid) else None
            
            return v_pot, chi, rh_mid
        except Exception as exc:
            # Return default values on any error
            return 0.0, 0.5, None
        except Exception as e:
            print(f"[ERA5Env] Warning: Error calculating v_pot/chi: {e}")
            return 80.0, 0.5, 0.5  # Default values

    def _estimate_rh(self, temperature_k, q, pressure_pa):
        """Approximate relative humidity given temperature, specific humidity, and pressure"""
        try:
            Rd = 287.04
            Rv = 461.5
            e = q * pressure_pa / (0.622 + 0.378 * q)
            e_s = 611.2 * np.exp((17.67 * (temperature_k - 273.15)) /
                                 (temperature_k - 29.65))
            return float(np.clip(e / max(e_s, 1e-3), 0.0, 1.0))
        except Exception:
            return 0.5
    
    def _vpot_approx(self, sst, p_surf, T_env, q_env, p_env):
        """Simplified v_pot approximation"""
        # Very simple approximation
        return np.sqrt(100 * (sst - 273.15) / 30.0) * 20.0
    
    def _chi_approx(self, sst, p_surf, T_mid, p_mid, q_mid):
        """Simplified chi approximation"""
        return 0.5
    
    def get_env(self, t: float, lon: float, lat: float) -> Dict[str, Any]:
        """
        Get environment at time t (seconds) and location (lon, lat)
        
        Args:
            t: Time in seconds since initialization
            lon: Longitude in degrees
            lat: Latitude in degrees
        
        Returns:
            Dictionary with environment parameters
        """
        # Handle NaN or invalid values
        if np.isnan(t) or np.isnan(lon) or np.isnan(lat):
            # Return default values
            return {
                "v_pot": 0.0,
                "h_m": 50.0,
                "t_strat": 0.2,
                "chi": 0.5,
                "C_k": 0.0015,
                "env_wind_profile": (0.0, 0.0, 0.0, 0.0),
                "bathymetry": -5000.0,
                "is_land": False
            }
        
        # Calculate current datetime
        try:
            current_time = self.init_time + timedelta(seconds=float(t))
        except (ValueError, OverflowError):
            # If time conversion fails, use init_time
            current_time = self.init_time
        
        # Convert longitude to 0-360° format if needed
        # Check if ERA5 data uses 0-360° format
        lon_era5_min = float(self.ds_sst[self.lon_coord].min().values)
        if lon_era5_min >= 0 and lon < 0:
            # Convert from -180-180° to 0-360°
            lon = lon + 360.0
        elif lon_era5_min < 0 and lon > 180:
            # Convert from 0-360° to -180-180°
            lon = lon - 360.0
        
        # Get static fields first (needed for default value logic)
        is_land = self._get_land(lon, lat)
        bathymetry = self._get_bathymetry(lon, lat)
        if np.isnan(bathymetry) or not np.isfinite(bathymetry):
            bathymetry = 0.0 if is_land else -5000.0
        
        # Get thermodynamic parameters from ERA5
        v_pot, chi, rh_mid = self._get_vpot_chi(current_time, lon, lat)
        
        # Handle NaN or invalid values from interpolation with improved defaults
        if np.isnan(v_pot) or not np.isfinite(v_pot):
            # Use reasonable default: 0 on land, 40 m/s on ocean (allows system to develop)
            v_pot = 0.0 if is_land else 40.0
        if np.isnan(chi) or not np.isfinite(chi):
            chi = 0.5
        if rh_mid is None or (isinstance(rh_mid, (float, np.floating)) and (np.isnan(rh_mid) or not np.isfinite(rh_mid))):
            rh_mid = None
        
        # Get ocean parameters from climatology
        h_m = self._get_mld(current_time, lon, lat)
        if np.isnan(h_m) or not np.isfinite(h_m):
            h_m = 50.0
        
        t_strat = self._get_strat(current_time, lon, lat)
        if np.isnan(t_strat) or not np.isfinite(t_strat):
            t_strat = 0.2
        
        # If over land, reduce v_pot (ensure this happens after default value assignment)
        if is_land:
            v_pot = 0.0
        
        # Get environment wind profile (with optional vortex removal)
        env_wind_profile = self._get_env_wind(current_time, lon, lat)
        
        # Ensure env_wind_profile is valid
        if env_wind_profile is None or len(env_wind_profile) != 4:
            env_wind_profile = (0.0, 0.0, 0.0, 0.0)
        
        return {
            "v_pot": float(v_pot),
            "h_m": float(h_m),
            "t_strat": float(t_strat),
            "chi": float(chi),
            "C_k": 0.0015,  # Can be made variable
            "env_wind_profile": env_wind_profile,
            "bathymetry": float(bathymetry),
            "is_land": is_land,
            "rh_mid": rh_mid
        }
    
    def get_vortex_removal_stats(self):
        """Get statistics about vortex removal usage"""
        return {
            'vortex_removal_enabled': HAS_VORTEX_LIB,
            'vortex_removal_calls': self._vortex_removal_count,
            'fallback_calls': self._vortex_fallback_count,
            'total_calls': self._vortex_removal_count + self._vortex_fallback_count
        }
