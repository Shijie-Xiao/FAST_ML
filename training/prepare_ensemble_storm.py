#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ensemble-member track builders shared by the ensemble preparation pipeline.

Provides the three helpers consumed by ensemble/run_prep_div1000.py:
  _load_best_track    -- rebuild hourly (time, lat, lon, vmax_kts) from BT *_dataset.pkl
  _load_synthetic     -- read synthetic_tracks NC, pick N members by seed
  _build_member_track -- splice best-track prepend + synthetic hourly track per member

Track CSV convention (consumed by prepare_complete_training_data.process_track_data):
  columns = time, lat, lon(0..360), vmax(knots)
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

MS_TO_KNOTS = 1.94384
KNOTS_TO_MS = 0.514444


def _load_best_track(best_track_pkl):
    """Rebuild hourly (time, lat, lon, vmax_kts) from *_dataset.pkl."""
    with open(best_track_pkl, 'rb') as f:
        ds = pickle.load(f)
    times = pd.to_datetime(np.asarray(ds['times']).ravel())
    lats = np.asarray(ds['lats']).reshape(-1).astype(float)
    lons = np.asarray(ds['lons']).reshape(-1).astype(float)
    vgt = np.asarray(ds['v_gt']).reshape(-1).astype(float)  # m/s
    vmax = vgt * MS_TO_KNOTS
    n = min(len(times), len(lats), len(lons), len(vmax))
    return pd.DataFrame({
        'time': times[:n], 'lat': lats[:n], 'lon': lons[:n], 'vmax': vmax[:n],
    })


def _load_synthetic(synthetic_nc, n_members, seed):
    """Read synthetic_tracks NC; pick n_members by seed.

    Returns: init_time, dt_hours, lons[N,T](0..360), lats[N,T], t_seconds[T], pick
    """
    ds = xr.open_dataset(synthetic_nc)
    init_time = pd.Timestamp(ds.attrs.get('init_time'))
    dt_hours = float(ds.attrs.get('dt_hours', 6.0))
    n_total = ds.sizes['track']
    if n_members > n_total:
        raise ValueError(f'n_members={n_members} > available {n_total}')
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(n_total, size=n_members, replace=False))
    lons = ds['lon'].values[pick]
    lats = ds['lat'].values[pick]
    lons = np.where(lons < 0, lons + 360.0, lons)
    raw_t = ds['time'].values
    if np.issubdtype(raw_t.dtype, np.datetime64):
        t_sec = (raw_t - np.datetime64(init_time)).astype('timedelta64[s]').astype(np.float64)
    else:
        t_sec = raw_t.astype(np.float64)
    ds.close()
    return init_time, dt_hours, lons, lats, t_sec, pick


def _build_member_track(best_track_df, synth_lons, synth_lats, synth_t_sec,
                        init_time, reference_time, duration_h):
    """Splice best-track prepend [ref,init) + hourly-upsampled synthetic [init, ref+dur].

    vmax column always borrows the best-track value at matching time (ffill if missing).
    """
    synth_times = init_time + pd.to_timedelta(synth_t_sec, unit='s')
    end_time = reference_time + pd.Timedelta(hours=duration_h)
    mask = synth_times <= end_time
    synth_times = synth_times[mask]
    synth_lons = synth_lons[mask]
    synth_lats = synth_lats[mask]

    hourly_synth_times = pd.date_range(init_time, synth_times[-1], freq='h')
    synth_lon_hourly = np.interp(
        hourly_synth_times.view('int64'), synth_times.view('int64'), synth_lons)
    synth_lat_hourly = np.interp(
        hourly_synth_times.view('int64'), synth_times.view('int64'), synth_lats)

    bt = best_track_df[
        (best_track_df['time'] >= reference_time) &
        (best_track_df['time'] < init_time)
    ].copy()
    bt = bt.set_index('time').asfreq('h').interpolate('linear').reset_index()

    syn = pd.DataFrame({
        'time': hourly_synth_times,
        'lat': synth_lat_hourly,
        'lon': synth_lon_hourly,
    })
    track = pd.concat([bt[['time', 'lat', 'lon']], syn], ignore_index=True)
    track = track.drop_duplicates(subset='time').sort_values('time').reset_index(drop=True)

    bt_full = best_track_df.set_index('time').asfreq('h')
    bt_full['vmax'] = bt_full['vmax'].interpolate('linear').ffill().bfill()
    track['vmax'] = track['time'].map(bt_full['vmax']).astype(float)
    track['vmax'] = track['vmax'].ffill().bfill()

    return track
