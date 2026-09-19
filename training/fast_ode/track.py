"""
track.py
Author: Shijie Xiao
Email: sxiao73@gatech.edu
Date: 2025-11-15
"""

from typing import Protocol
from pathlib import Path
import numpy as np
import math
import xarray as xr

class Track(Protocol):
    """
    track provider interface
    """
    def get_velocity(self, t: float, lon: float, lat: float) -> np.ndarray:
        """
        get the velocity at time t, lon, lat
        """
        pass

class RandomTrack(Track):
    """
    random track provider implementation
    """
    def __init__(self, u_mean:float = 5.0, v_mean:float = 0.0):
        self.u_mean = float(u_mean)
        self.v_mean = float(v_mean)
    def get_velocity(self, t: float, lon: float, lat: float) -> np.ndarray:
        """
        get the velocity at time t, lon, lat
        """
        period = 24*60*60
        phase = 2*math.pi*t/period

        u_T = self.u_mean * (1.0 + 0.2 * math.sin(phase))
        v_T = self.v_mean *(0.5 * math.sin(phase/2.0))
        return np.array([u_T, v_T],dtype=float)


class NetCDFTrack(Track):
    """
    Track provider that reads synthetic tracks from NetCDF file.
    Supports interpolation for any time point along the track.
    """
    def __init__(self, nc_file: Path, track_id: int = 0):
        """
        Initialize NetCDF track provider
        
        Args:
            nc_file: Path to NetCDF file with synthetic tracks
            track_id: Which track to use (0-indexed)
        """
        self.ds = xr.open_dataset(nc_file)
        self.track_id = track_id
        
        # Extract track data
        self.time_coords = self.ds['time'].values
        self.lon = self.ds['lon'][track_id, :].values
        self.lat = self.ds['lat'][track_id, :].values
        
        # Calculate velocities from positions if not available
        if 'u' in self.ds.data_vars and 'v' in self.ds.data_vars:
            self.u = self.ds['u'][track_id, :].values
            self.v = self.ds['v'][track_id, :].values
        else:
            # Calculate velocities from position differences
            self.u, self.v = self._calculate_velocities()
        
        # Convert time to seconds since first time
        self.t0 = self.time_coords[0]
        
        # Handle different time formats
        if isinstance(self.t0, np.datetime64):
            # numpy datetime64 format
            time_deltas = self.time_coords - self.t0
            # Convert timedelta64 to seconds
            self.time_seconds = time_deltas / np.timedelta64(1, 's')
        elif hasattr(self.t0, 'to_pydatetime'):
            # xarray datetime format
            t0_py = self.t0.to_pydatetime()
            self.time_seconds = np.array([
                (t.to_pydatetime() - t0_py).total_seconds() 
                for t in self.time_coords
            ])
        elif np.issubdtype(type(self.t0), np.number):
            # Numeric time axis; assume units are seconds since start
            self.time_seconds = np.array(self.time_coords, dtype=float)
        else:
            # Python datetime objects
            self.time_seconds = np.array([
                (t - self.t0).total_seconds() 
                for t in self.time_coords
            ])
    
    def _calculate_velocities(self):
        """Calculate velocities from position differences"""
        dt_hours = 6.0  # 6-hourly data
        dt_seconds = dt_hours * 3600.0
        
        # Calculate velocity using spherical geometry
        u = np.zeros_like(self.lon)
        v = np.zeros_like(self.lat)
        
        for i in range(1, len(self.lon)):
            # Longitude velocity (m/s)
            dlon = self.lon[i] - self.lon[i-1]
            # Normalize longitude difference
            if dlon > 180:
                dlon -= 360
            elif dlon < -180:
                dlon += 360
            lat_mean = np.deg2rad((self.lat[i] + self.lat[i-1]) / 2.0)
            u[i] = (dlon * 111.0 * 1000.0 * np.cos(lat_mean)) / dt_seconds
            
            # Latitude velocity (m/s)
            dlat = self.lat[i] - self.lat[i-1]
            v[i] = (dlat * 111.0 * 1000.0) / dt_seconds
        
        # Use first velocity for initial point
        u[0] = u[1] if len(u) > 1 else 0.0
        v[0] = v[1] if len(v) > 1 else 0.0
        
        return u, v
    
    def get_velocity(self, t: float, lon: float, lat: float) -> np.ndarray:
        """
        Get velocity at time t (in seconds since track start)
        
        Args:
            t: Time in seconds since track start
            lon: Current longitude (not used, kept for interface compatibility)
            lat: Current latitude (not used, kept for interface compatibility)
        
        Returns:
            Velocity vector [u, v] in m/s
        """
        # Interpolate velocity
        if t <= self.time_seconds[0]:
            return np.array([self.u[0], self.v[0]], dtype=float)
        if t >= self.time_seconds[-1]:
            return np.array([self.u[-1], self.v[-1]], dtype=float)
        
        u_interp = np.interp(t, self.time_seconds, self.u)
        v_interp = np.interp(t, self.time_seconds, self.v)
        return np.array([u_interp, v_interp], dtype=float)
    
    def get_position(self, t: float) -> tuple:
        """
        Get position at time t (in seconds since track start)
        
        Args:
            t: Time in seconds since track start
        
        Returns:
            Tuple (lon, lat) in degrees
        """
        if t <= self.time_seconds[0]:
            return (self.lon[0], self.lat[0])
        if t >= self.time_seconds[-1]:
            return (self.lon[-1], self.lat[-1])
        
        lon_interp = np.interp(t, self.time_seconds, self.lon)
        lat_interp = np.interp(t, self.time_seconds, self.lat)
        return (lon_interp, lat_interp)




