#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download GEFS ensemble forecast GRIB2 subsets for hurricane cases.

For each storm, downloads:
  - 31 members (c00 + gep01..gep30)
  - 81 forecast hours (f000, f003, ..., f240) at 3-hourly interval -> 10 days
  - Variables: SPFH, TMP, UGRD, VGRD, HGT
  - Levels: 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150 mb
            + surface (PRES ground-or-water-surface from pgrb2ap5,
                       HGT surface from pgrb2bp5)
  - Source: AWS noaa-gefs-pds bucket
      pgrb2ap5 (0p50) for: HGT/TMP/UGRD/VGRD @ 1000,925,850,700,500,300,250,200 mb
                            + PRES @ surface
      pgrb2bp5 (0p50) for: SPFH @ all 11 levels
                            + HGT/TMP/UGRD/VGRD @ 600,400,150 mb
                            + HGT @ surface

Uses S3 Range GET to extract only the requested fields from each GRIB2 file,
greatly reducing both network transfer and disk usage (~5 MB per member-hour
instead of ~100 MB).

Usage:
  python download_gefs_storm.py --storm_csv <path> --out_root <path> [--basin EP]
                                [--year 2024] [--name ERIN] [--members 31]
                                [--forecast_hours 0-240] [--workers 8]
                                [--skip_existing]

Output layout (under out_root/<YEAR>_<NAME>_<BASIN>/):
    grib2/<MEMBER>/<MEMBER>.t<CC>z.f<HHH>.grib2     (subset GRIB2 file)
    manifest.json                                    (download manifest)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import boto3
from botocore import UNSIGNED
from botocore.config import Config


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BUCKET = 'noaa-gefs-pds'
REGION = 'us-east-1'

# Variables and isobaric levels consumed by prepare_complete_training_data.py
# Code uses ALL pressure levels present in the file for the tcpyPI thermodynamic
# profile (T, Q), not just the 7 "PL_LEVELS" used for spatial_3d. To match ERA5
# (29 levels, 50-1000 mb), we download every isobaric level GEFS publishes.
TARGET_VARS = ('HGT', 'TMP', 'SPFH', 'UGRD', 'VGRD')
# All isobaric levels that GEFS publishes across pgrb2a + pgrb2bp5:
#   A: 10, 50, 100, 200, 250, 300, 400, 500, 700, 850, 925, 1000
#   B: 1,2,3,5,7,10,20,30,50,70,100,125,150,200,250,300,350,400,450,475,
#      500,550,600,650,700,750,800,850,900,925,950,975,1000
# Use empty TARGET_LEVELS_MB to mean "take every isobaric level in the idx".
TARGET_LEVELS_MB = ()  # empty = all available isobaric levels

# Surface fields actually consumed by the prep pipeline:
#   - PRES @ surface       -> SP (surface pressure, used by calc_chi via msl_pa path)
#   - PRMSL @ mean sea level -> MSL (mean sea level pressure, primary pressure input)
#   - TMP @ surface        -> SST proxy (skin temperature, when ERA5 SSTK unavailable)
SURFACE_FIELDS_A = (
    ('PRES',  'surface'),          # ground-or-water-surface pressure
    ('PRMSL', 'mean sea level'),   # MSL pressure (gfs_adapter reads 'prmsl')
)
SURFACE_FIELDS_B = (
    ('TMP', 'surface'),            # skin temperature (gfs_adapter reads as SST proxy)
)

# Per-source variable/level matrix.
# When an entry value is (), it means "all isobaric levels present in the idx".
# pgrb2ap5 publishes: HGT/TMP/UGRD/VGRD @ standard levels (no SPFH, no 600mb)
# pgrb2bp5 publishes: HGT/TMP/UGRD/VGRD/SPFH @ all 33 isobaric levels
#
# Variable level requirements (verified from prepare_complete_training_data.py):
#   - T  (tcpyPI CAPE + spatial_3d 7 lev):  needs FULL 31 levels
#   - q  (tcpyPI CAPE + spatial_3d 7 lev):  needs FULL 31 levels
#   - U  (spatial_3d 7 lev + shear 250/850): needs PL_LEVELS = 7 levels
#   - V  (spatial_3d 7 lev + shear 250/850): needs PL_LEVELS = 7 levels
#   - gh (spatial_3d 7 lev):                 needs PL_LEVELS = 7 levels
#
# NOTE: shear only uses 250+850 hPa BUT those are already inside PL_LEVELS, so
# serving U/V at PL_LEVELS covers both spatial_3d and shear.
# T is skipped from pgrb2a because pgrb2b publishes all 31 levels of TMP and
# the merge logic de-dupes anyway (downloading T from both streams is wasteful).
PL_LEVELS_FULL = (1000, 850, 700, 600, 500, 250, 200)
# Note: 600 hPa is NOT in pgrb2a, so the gh/U/V download from pgrb2a picks up
# 6 levels (1000/850/700/500/250/200). The missing 600 hPa is fetched from
# pgrb2b.

LEVELS_FROM_A = {
    'HGT':  PL_LEVELS_FULL,  # 7 PL_LEVELS (a has 6 of them; 600 from b)
    # TMP: skipped in pgrb2a -- T comes from pgrb2b which has all 31 levels
    'UGRD': PL_LEVELS_FULL,  # 7 PL_LEVELS (covers spatial_3d + shear 250/850)
    'VGRD': PL_LEVELS_FULL,  # 7 PL_LEVELS
    # SPFH not in A
}
LEVELS_FROM_B = {
    # HGT/UGRD/VGRD @ 600 hPa: pgrb2a does NOT publish 600 mb. Without this,
    # the adapter would .sel(method='nearest') onto 700 or 500 mb for the
    # spatial_3d 600mb channel, which is a meaningful drift. Download just
    # this one level from pgrb2b (~250 KB each).
    'HGT':  (600,),
    'UGRD': (600,),
    'VGRD': (600,),
    # TMP/SPFH: full 31-level profile from pgrb2b (for tcpyPI CAPE)
    'TMP':  (),
    'SPFH': (),
}

# Forecast hours: 0 to 240 step 3 = 81 hours (10 days)
DEFAULT_FHOURS = list(range(0, 241, 3))
DEFAULT_MEMBERS = ['c00'] + [f'p{int(i):02d}' for i in range(1, 31)]  # 31 total


# ---------------------------------------------------------------------------
# S3 client
# ---------------------------------------------------------------------------

def make_s3():
    return boto3.client('s3', region_name=REGION,
                        config=Config(signature_version=UNSIGNED,
                                      max_pool_connections=32,
                                      connect_timeout=30,
                                      read_timeout=120,
                                      retries={'max_attempts': 5, 'mode': 'adaptive'}))


# ---------------------------------------------------------------------------
# idx parsing
# ---------------------------------------------------------------------------

@dataclass
class IdxRecord:
    seq: int
    pos: int
    var: str
    level: str
    end: int  # byte offset of next record (or file size)


def parse_idx(idx_text: str, file_size: int) -> list[IdxRecord]:
    """Parse a GRIB2 .idx file.

    Format: seq:pos:d=YYYYMMDDHH:VAR:LEVEL:fcst_type:ENS=member[:extra]
    """
    records: list[IdxRecord] = []
    for line in idx_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = line.split(':')
        if len(parts) < 7:
            continue
        try:
            seq = int(parts[0])
            pos = int(parts[1])
        except ValueError:
            continue
        var = parts[3]
        level = parts[4]
        records.append(IdxRecord(seq=seq, pos=pos, var=var, level=level, end=-1))
    # Set end positions
    for i in range(len(records)):
        if i + 1 < len(records):
            records[i].end = records[i + 1].pos
        else:
            records[i].end = file_size
    return records


def fetch_idx(s3, key: str, idx_key: str) -> tuple[list[IdxRecord], int]:
    """Fetch and parse idx file, return records + file size."""
    # Get file size via HEAD
    head = s3.head_object(Bucket=BUCKET, Key=key)
    file_size = head['ContentLength']
    # Get idx via direct HTTPS (faster than SDK for tiny files)
    idx_url = f'https://{BUCKET}.s3.{REGION}.amazonaws.com/{idx_key}'
    req = urllib.request.Request(idx_url, headers={'User-Agent': 'Mozilla/5.0'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                idx_text = r.read().decode('utf-8', errors='ignore')
            break
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(1.0 * (attempt + 1))
    return parse_idx(idx_text, file_size), file_size


def select_ranges(records: list[IdxRecord], fields: Iterable[tuple[str, str]]) -> list[tuple[int, int]]:
    """Return list of (start, end) byte ranges for matching (var, level) pairs.
    If field has level '__ALL_ISOBARIC__', expand to all 'N mb' records of that var."""
    expanded: list[tuple[str, str]] = []
    sentinel_vars: set[str] = set()
    for v, l in fields:
        if l == '__ALL_ISOBARIC__':
            sentinel_vars.add(v)
        else:
            expanded.append((v, l))
    if sentinel_vars:
        for rec in records:
            if rec.var in sentinel_vars and rec.level.endswith(' mb'):
                expanded.append((rec.var, rec.level))
    field_set = set(expanded)
    ranges = []
    for rec in records:
        if (rec.var, rec.level) in field_set:
            ranges.append((rec.pos, rec.end))
    return ranges


# ---------------------------------------------------------------------------
# Range GET download
# ---------------------------------------------------------------------------

def merge_ranges(ranges: list[tuple[int, int]], gap_threshold: int = 64) -> list[tuple[int, int]]:
    """Merge adjacent ranges separated by <= gap_threshold bytes to reduce request count."""
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        ls, le = merged[-1]
        if s <= le + gap_threshold:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def fetch_ranges(s3, key: str, ranges: list[tuple[int, int]], out_path: Path,
                max_retries: int = 4) -> int:
    """Download a list of byte ranges from one S3 object and concatenate to out_path.
    Returns total bytes written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged = merge_ranges(ranges)
    total = 0
    # If a single merged range covers most of the file, fall back to full GET
    with out_path.open('wb') as f:
        for start, end in merged:
            chunk_size = 8 * 1024 * 1024  # 8 MB streaming
            for attempt in range(max_retries):
                try:
                    rng = f'bytes={start}-{end - 1}'
                    resp = s3.get_object(Bucket=BUCKET, Key=key, Range=rng)
                    stream = resp['Body'].iter_chunks(chunk_size)
                    for chunk in stream:
                        f.write(chunk)
                        total += len(chunk)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise RuntimeError(f'failed range {key} {rng}: {e}') from e
                    time.sleep(1.5 * (2 ** attempt))
    return total


# ---------------------------------------------------------------------------
# Storm plan
# ---------------------------------------------------------------------------

@dataclass
class StormInfo:
    basin: str       # 'EP' or 'NA'
    season: int
    name: str        # 'ERIN', 'UNNAMED', etc.
    sid: str         # IBTrACS SID
    first_time: str  # ISO time string

    @property
    def case_name(self) -> str:
        return f'{self.season}_{self.name}_{self.basin}'


def load_storms(csv_path: Path, basin: str | None = None, year: int | None = None,
                name: str | None = None) -> list[StormInfo]:
    import csv as csvmod
    storms: list[StormInfo] = []
    with csv_path.open() as f:
        reader = csvmod.DictReader(f)
        for row in reader:
            if basin and row['basin'] != basin:
                continue
            if year and int(row['season']) != year:
                continue
            if name and row['name'].upper() != name.upper():
                continue
            storms.append(StormInfo(
                basin=row['basin'], season=int(row['season']),
                name=row['name'].upper(), sid=row['sid'],
                first_time=row['first_time'],
            ))
    return storms


def nearest_cycle_before(iso_time: str) -> tuple[str, str, str]:
    """Given an ISO time, find nearest 6-hourly cycle AT OR BEFORE.
    Returns (YYYYMMDD, HH, 'CC')."""
    import datetime as dt
    t = pd_ts(iso_time)
    # Snap to 00/06/12/18Z
    hour = t.hour
    snapped = t.replace(minute=0, second=0, microsecond=0)
    if hour < 6:
        snapped = snapped.replace(hour=0)
    elif hour < 12:
        snapped = snapped.replace(hour=6)
    elif hour < 18:
        snapped = snapped.replace(hour=12)
    else:
        snapped = snapped.replace(hour=18)
    date_str = snapped.strftime('%Y%m%d')
    cycle_str = f'{snapped.hour:02d}'
    return date_str, cycle_str, snapped


def pd_ts(iso: str):
    import datetime as dt
    return dt.datetime.fromisoformat(iso.replace(' ', 'T'))


# ---------------------------------------------------------------------------
# Per-member-hour task
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    member: str
    fhour: int
    a_bytes: int
    b_bytes: int
    a_fields: int
    b_fields: int
    a_key: str
    b_key: str
    a_out: str
    b_out: str
    elapsed: float
    error: str | None = None


def build_fields_for_source(source: str) -> list[tuple[str, str]]:
    """Return list of (var, 'N mb' or 'surface') for the given source.
    If level list is empty, emit a '__ALL_ISOBARIC__' sentinel which
    select_ranges() expands to every 'N mb' record of that var."""
    fields: list[tuple[str, str]] = []
    if source == 'a':
        matrix = LEVELS_FROM_A
        surface = SURFACE_FIELDS_A
    else:
        matrix = LEVELS_FROM_B
        surface = SURFACE_FIELDS_B
    for var, levels in matrix.items():
        if len(levels) == 0:
            fields.append((var, '__ALL_ISOBARIC__'))
        else:
            for lvl in levels:
                fields.append((var, f'{lvl} mb'))
    fields.extend(surface)
    return fields


def download_one_member_hour(s3, storm: StormInfo, cycle_date: str, cycle_hour: str,
                              member_code: str, fhour: int, out_dir: Path,
                              skip_existing: bool = True) -> TaskResult:
    """member_code: 'c00', 'p01', ..., 'p30'
    Downloads both pgrb2ap5 and pgrb2bp5 subsets for one member/forecast-hour."""
    t0 = time.time()
    member_member_grib = f'ge{member_code}'  # gec00, gep01, ...
    fhh = f'{fhour:03d}'
    out_a = out_dir / f'pgrb2a/ge{member_code}.t{cycle_hour}z.pgrb2a.0p50.f{fhh}.grib2'
    out_b = out_dir / f'pgrb2b/ge{member_code}.t{cycle_hour}z.pgrb2b.0p50.f{fhh}.grib2'

    # Check if outputs already exist and have non-trivial size
    if skip_existing and out_a.exists() and out_b.exists() \
            and out_a.stat().st_size > 1000 and out_b.stat().st_size > 1000:
        return TaskResult(member_code, fhour, out_a.stat().st_size, out_b.stat().st_size,
                          0, 0, '', '', str(out_a), str(out_b), time.time() - t0,
                          error='SKIPPED_EXISTING')

    key_a = f'gefs.{cycle_date}/{cycle_hour}/atmos/pgrb2ap5/ge{member_code}.t{cycle_hour}z.pgrb2a.0p50.f{fhh}'
    key_b = f'gefs.{cycle_date}/{cycle_hour}/atmos/pgrb2bp5/ge{member_code}.t{cycle_hour}z.pgrb2b.0p50.f{fhh}'
    idx_a = key_a + '.idx'
    idx_b = key_b + '.idx'

    try:
        # Source A
        recs_a, sz_a = fetch_idx(s3, key_a, idx_a)
        ranges_a = select_ranges(recs_a, build_fields_for_source('a'))
        bytes_a = fetch_ranges(s3, key_a, ranges_a, out_a) if ranges_a else 0

        # Source B
        recs_b, sz_b = fetch_idx(s3, key_b, idx_b)
        ranges_b = select_ranges(recs_b, build_fields_for_source('b'))
        bytes_b = fetch_ranges(s3, key_b, ranges_b, out_b) if ranges_b else 0

        return TaskResult(member_code, fhour, bytes_a, bytes_b,
                          len(ranges_a), len(ranges_b),
                          key_a, key_b, str(out_a), str(out_b),
                          time.time() - t0)
    except Exception as e:
        return TaskResult(member_code, fhour, 0, 0, 0, 0, key_a, key_b,
                          str(out_a), str(out_b), time.time() - t0, error=str(e))


# ---------------------------------------------------------------------------
# Main per-storm driver
# ---------------------------------------------------------------------------

def download_storm(s3, storm: StormInfo, out_root: Path, members: list[str],
                   fhours: list[int], workers: int, skip_existing: bool = True,
                   log_every: int = 25, cycle_override: str | None = None) -> dict:
    case_dir = out_root / storm.case_name
    grib_dir = case_dir / 'grib2'
    case_dir.mkdir(parents=True, exist_ok=True)
    if cycle_override:
        # Format: 'YYYYMMDD/HH' or 'YYYYMMDDHH'
        s = cycle_override.replace('/', '')
        if len(s) != 10 or not s.isdigit():
            raise ValueError(f'cycle_override must be YYYYMMDDHH, got {cycle_override}')
        cycle_date = s[:8]
        cycle_hour = s[8:10]
        import datetime as dt
        snapped = dt.datetime.strptime(s, '%Y%m%d%H')
    else:
        cycle_date, cycle_hour, snapped = nearest_cycle_before(storm.first_time)

    manifest = {
        'storm': asdict(storm),
        'init_date': cycle_date,
        'init_cycle': cycle_hour,
        'init_utc': snapped.isoformat(),
        'first_track_time': storm.first_time,
        'members': members,
        'forecast_hours': fhours,
        'variables': list(TARGET_VARS),
        'levels_mb': list(TARGET_LEVELS_MB),
        'surface_fields': {
            'pgrb2a': [list(f) for f in SURFACE_FIELDS_A],
            'pgrb2b': [list(f) for f in SURFACE_FIELDS_B],
        },
        'bucket': BUCKET,
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'results': [],
    }

    print(f"\n[{storm.case_name}] init={cycle_date}/{cycle_hour}Z "
          f"(track starts {storm.first_time})")
    print(f"[{storm.case_name}] {len(members)} members × {len(fhours)} fhours "
          f"= {len(members) * len(fhours)} tasks")

    # Build all tasks
    tasks = [(m, fh) for m in members for fh in fhours]

    t_start = time.time()
    n_ok = n_skip = n_err = 0
    total_bytes = 0

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(download_one_member_hour, s3, storm, cycle_date, cycle_hour,
                      m, fh, grib_dir, skip_existing): (m, fh)
            for m, fh in tasks
        }
        for i, fut in enumerate(cf.as_completed(futures), 1):
            r: TaskResult = fut.result()
            manifest['results'].append(asdict(r))
            if r.error == 'SKIPPED_EXISTING':
                n_skip += 1
            elif r.error:
                n_err += 1
                print(f'  ERROR {r.member} f{r.fhour:03d}: {r.error}')
            else:
                n_ok += 1
                total_bytes += r.a_bytes + r.b_bytes
            if i % log_every == 0 or i == len(tasks):
                rate = total_bytes / 1e6 / max(time.time() - t_start, 1e-6)
                eta = (len(tasks) - i) / max(i / (time.time() - t_start), 1e-6)
                print(f'  progress {i:5d}/{len(tasks)} ok={n_ok} skip={n_skip} '
                      f'err={n_err} {total_bytes/1e9:.2f}GB '
                      f'{rate:.1f}MB/s ETA {eta/60:.1f}min')

    elapsed = time.time() - t_start
    manifest['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    manifest['elapsed_s'] = elapsed
    manifest['summary'] = {'ok': n_ok, 'skipped': n_skip, 'error': n_err,
                           'total_bytes': total_bytes}

    # Save manifest
    (case_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'[{storm.case_name}] done in {elapsed:.0f}s '
          f'ok={n_ok} skip={n_skip} err={n_err} '
          f'{total_bytes/1e9:.2f}GB saved under {case_dir}')
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_fhours(s: str) -> list[int]:
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a), int(b) + 1, 3))
    return [int(x) for x in s.split(',')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--storm_csv', default=str(Path(__file__).resolve().parent / 'gefs_storms_2024_2026.csv'))
    ap.add_argument('--out_root', required=True,
                    help='Output root, e.g. /global/cfs/cdirs/m5011/Jay/ERA5/GFS')
    ap.add_argument('--basin', choices=['EP', 'NA'])
    ap.add_argument('--year', type=int)
    ap.add_argument('--name')
    ap.add_argument('--members', type=int, default=31,
                    help='Number of ensemble members (1=c00 only, 31=c00+p01..p30)')
    ap.add_argument('--forecast_hours', default='0-240',
                    help='Forecast hours, e.g. 0-240 (step 3) or 0,3,6,...')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--skip_existing', action='store_true', default=True)
    ap.add_argument('--overwrite', action='store_true',
                    help='Re-download even if files exist')
    ap.add_argument('--cycle_override', default=None,
                    help='Force init cycle, e.g. 2025081112 (YYYYMMDDHH). '
                         'Default: nearest 6h cycle at or before storm.first_time.')
    args = ap.parse_args()

    storms = load_storms(Path(args.storm_csv), args.basin, args.year, args.name)
    if not storms:
        print('No matching storms.')
        sys.exit(1)

    members = DEFAULT_MEMBERS[:args.members]
    fhours = parse_fhours(args.forecast_hours)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    skip = not args.overwrite

    print(f'Plan: {len(storms)} storms × {len(members)} members × {len(fhours)} fhours')
    print(f'Output root: {out_root}')
    if args.cycle_override:
        print(f'Init cycle override: {args.cycle_override}')

    s3 = make_s3()
    for storm in storms:
        try:
            download_storm(s3, storm, out_root, members, fhours, args.workers, skip,
                           cycle_override=args.cycle_override)
        except Exception as e:
            print(f'[{storm.case_name}] FAILED: {e}', file=sys.stderr)


if __name__ == '__main__':
    main()
