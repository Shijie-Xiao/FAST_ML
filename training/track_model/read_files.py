"""Read ECMWF 51-member ensemble TC tracks from TIGGE XML (2023-2025 NA storms).

For each auto-discovered storm we scan the TIGGE base times around genesis and
pick the cycle (base time) whose ensemble contains the MOST members tracking the
named storm -- that base time becomes the storm's init_time. The 51 (<=51)
ensemble member tracks for that cycle are saved as a raw pickle.

Data layout: /global/cfs/cdirs/m5011/Jay/ECMF/{year}/{YYYYMMDD}/z_tigge_c_ecmf_*.xml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import datetime, timedelta
import pickle
import re
import xml.etree.ElementTree as ET
import numpy as np

from config import (
    ECMWF_BASE_DIR, PROCESSED_TRACKS_DIR, ALL_STORMS, MIN_MEMBERS,
)
import config as _cfg          # live reference: INIT_SEARCH_* mutable at runtime


def _parse_dt(text):
    """Parse CXML time; tolerant of optional trailing 'Z' (2017 has Z, 2024 doesn't)."""
    if not text:
        return None
    s = text.strip().rstrip("Z")
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def parse_tigge_xml(fp: Path, storm_name: str, base_time_filter: datetime = None):
    """Parse one TIGGE XML; return ensemble-member tracks of the named storm."""
    tracks = []
    try:
        root = ET.parse(fp).getroot()
        header = root.find("header")
        base_time = None
        if header is not None:
            bt = header.find("baseTime")
            if bt is not None and bt.text:
                base_time = _parse_dt(bt.text)
        if base_time_filter and base_time and \
                abs((base_time - base_time_filter).total_seconds()) > 3601:
            return [], base_time

        for data_elem in root.findall("data"):
            if data_elem.get("type", "") != "ensembleForecast":
                continue
            ms = data_elem.get("member")
            if ms is None:
                continue
            member_id = int(ms)
            if member_id < 0 or member_id > 50:
                continue
            for dist in data_elem.findall("disturbance"):
                ne = dist.find("cycloneName")
                cyclone_name = ne.text.strip() if ne is not None and ne.text else None
                if not cyclone_name or storm_name.upper() not in cyclone_name.upper():
                    continue
                num_e = dist.find("cycloneNumber")
                cyclone_num = None
                if num_e is not None and num_e.text:
                    digits = re.sub(r"\D", "", num_e.text)  # '02L' -> '02'
                    cyclone_num = int(digits) if digits else None
                bas_e = dist.find("basin")
                basin = bas_e.text.strip() if bas_e is not None and bas_e.text else None

                lon_list, lat_list, time_list = [], [], []
                for fix in dist.findall("fix"):
                    lat_e, lon_e, time_e = fix.find("latitude"), fix.find("longitude"), fix.find("validTime")
                    if lat_e is None or lon_e is None:
                        continue
                    try:
                        lat_v = float(lat_e.text.strip().rstrip("NS"))
                        if lat_e.text.upper().endswith("S") or (lat_e.get("units") or "").upper() == "S":
                            lat_v = -lat_v
                        lon_v = float(lon_e.text.strip().rstrip("EW"))
                        if lon_e.text.upper().endswith("W") or (lon_e.get("units") or "").upper() == "W":
                            lon_v = -lon_v
                        lon_v = ((lon_v + 180) % 360) - 180
                        fix_time = base_time
                        if time_e is not None and time_e.text:
                            fix_time = _parse_dt(time_e.text) or base_time
                        if fix_time:
                            lon_list.append(lon_v); lat_list.append(lat_v); time_list.append(fix_time)
                    except Exception:
                        continue
                if len(lon_list) >= 2:
                    tracks.append({
                        "ensemble_system": "ecmwf", "member_id": member_id,
                        "storm_name": cyclone_name, "storm_number": cyclone_num,
                        "basin": basin, "init_time": base_time,
                        "lon": np.array(lon_list), "lat": np.array(lat_list),
                        "datetime": time_list,
                    })
    except Exception as e:
        return [], None
    return tracks, base_time


def _candidate_xml_files(year, genesis):
    """All TIGGE XML files whose date dir is within the init search window."""
    ydir = ECMWF_BASE_DIR / str(year)
    if not ydir.is_dir():
        return []
    start = (genesis - timedelta(days=_cfg.INIT_SEARCH_LEAD_DAYS)).date()
    end = (genesis + timedelta(days=_cfg.INIT_SEARCH_TAIL_DAYS)).date()
    files = []
    d = start
    while d <= end:
        ddir = ydir / d.strftime("%Y%m%d")
        if ddir.is_dir():
            files.extend(sorted(ddir.glob("*.xml")))
        d += timedelta(days=1)
    return files


def _candidate_xml_files_center(year, genesis, center="ecmf", base_dir=None):
    """TIGGE XML files for a given producing center within the init window.

    center='ecmf': {base}/ECMF/{year}/{YYYYMMDD}/z_tigge_c_ecmf_*.xml  (51 members)
    center='kwbc': {base}/KWBC/{year}/{YYYYMMDD}/z_tigge_c_kwbc_*_GEFS_*.xml
                   (NCEP GEFS ensemble tracks; also CENS/CMC/GFS products exist,
                   filtered to GEFS only). FHLO's GEFS track source.
    """
    base_dir = Path(base_dir) if base_dir else ECMWF_BASE_DIR
    root = base_dir / str(year)          # base_dir already points at the
    if not root.is_dir():                # center dir (…/TIGGE/ecmf or …/TIGGE/kwbc)
        return []
    start = (genesis - timedelta(days=_cfg.INIT_SEARCH_LEAD_DAYS)).date()
    end = (genesis + timedelta(days=_cfg.INIT_SEARCH_TAIL_DAYS)).date()
    files = []
    d = start
    while d <= end:
        ddir = root / d.strftime("%Y%m%d")
        if ddir.is_dir():
            if center == "kwbc":
                files.extend(sorted(ddir.glob("z_tigge_c_kwbc_*_GEFS_*.xml")))
            else:
                files.extend(sorted(ddir.glob("*.xml")))
        d += timedelta(days=1)
    return files


def read_storm_ensemble_center(storm_cfg: dict, center="ecmf", base_dir=None,
                               min_members=None, save_dir=None):
    """read_storm_ensemble generalized to a producing center (ecmf | kwbc/GEFS).

    kwbc GEFS XMLs use the same CXML schema; member ids 0..30 (31 members,
    control + 30 perturbed). Output raw.pkl identical to the ecmf path, with
    ensemble_system=center and parent_member set for member-paired sampling.
    """
    storm = storm_cfg["storm_name"]
    year = storm_cfg["year"]
    genesis = storm_cfg["genesis"]
    min_members = min_members or MIN_MEMBERS

    per_base = {}
    for fp in _candidate_xml_files_center(year, genesis, center=center,
                                          base_dir=base_dir):
        trks, base_time = parse_tigge_xml(fp, storm)
        if base_time is None:
            continue
        slot = per_base.setdefault(base_time, {})
        for tr in trks:
            mid = tr["member_id"]
            if mid not in slot or len(tr["lon"]) > len(slot[mid]["lon"]):
                slot[mid] = tr

    if not per_base:
        print(f"  [SKIP] {storm} ({year}): no {center} TIGGE tracks near genesis "
              f"{genesis:%Y-%m-%d}")
        return None

    best_base = max(per_base, key=lambda b: len(per_base[b]))
    members = per_base[best_base]
    if len(members) < min_members:
        print(f"  [SKIP] {storm} ({year}): best {center} cycle {best_base:%Y-%m-%d %Hz} "
              f"has only {len(members)} members (<{min_members})")
        return None

    tracks = []
    for mid in sorted(members.keys()):
        tr = dict(members[mid])
        tr["ensemble_system"] = center
        # parent attribution for member-paired sampling (FHLO inheritance)
        tr["parent_member"] = ("c00" if mid == 0 else f"p{mid:02d}") \
            if center == "kwbc" else f"e{mid:02d}"
        tracks.append(tr)

    times = [t["datetime"] for t in tracks if t.get("datetime")]
    print(f"  {storm} ({year}) [{center}]: init={best_base:%Y-%m-%d %Hz}, "
          f"{len(tracks)} ensemble members")

    result = {
        "storm_config": {**storm_cfg, "forced_init_time": best_base},
        "tracks": tracks, "n_tracks": len(tracks),
        "time_range": {
            "init_time": min(min(tt) for tt in times) if times else best_base,
            "end_time": max(max(tt) for tt in times) if times else best_base,
        },
        "forced_init_time": best_base,
        "ensemble_systems": [center],
    }
    out_dir = Path(save_dir) if save_dir else PROCESSED_TRACKS_DIR
    out_dir = out_dir / storm.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{storm.lower()}_{best_base:%Y%m%dT%H%M%S}_raw.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(result, f)
    return result


def read_storm_ensemble(storm_cfg: dict):
    """Find best init cycle for a storm and save its 51-member ensemble."""
    storm = storm_cfg["storm_name"]
    year = storm_cfg["year"]
    genesis = storm_cfg["genesis"]

    # 1) scan candidate base times -> count members per base time
    per_base = {}   # base_time -> {member_id: track}
    for fp in _candidate_xml_files(year, genesis):
        trks, base_time = parse_tigge_xml(fp, storm)
        if base_time is None:
            continue
        slot = per_base.setdefault(base_time, {})
        for tr in trks:
            mid = tr["member_id"]
            if mid not in slot or len(tr["lon"]) > len(slot[mid]["lon"]):
                slot[mid] = tr

    if not per_base:
        print(f"  [SKIP] {storm} ({year}): no TIGGE tracks near genesis {genesis:%Y-%m-%d}")
        return None

    # 2) pick base time with the most members
    best_base = max(per_base, key=lambda b: len(per_base[b]))
    members = per_base[best_base]
    if len(members) < MIN_MEMBERS:
        print(f"  [SKIP] {storm} ({year}): best cycle {best_base:%Y-%m-%d %Hz} "
              f"has only {len(members)} members (<{MIN_MEMBERS})")
        return None

    tracks = [members[k] for k in sorted(members.keys())]
    times = [t["datetime"] for t in tracks if t.get("datetime")]
    print(f"  {storm} ({year}): init={best_base:%Y-%m-%d %Hz}, "
          f"{len(tracks)} ensemble members")

    result = {
        "storm_config": {**storm_cfg, "forced_init_time": best_base},
        "tracks": tracks, "n_tracks": len(tracks),
        "time_range": {
            "init_time": min(min(tt) for tt in times) if times else best_base,
            "end_time": max(max(tt) for tt in times) if times else best_base,
        },
        "forced_init_time": best_base,
        "ensemble_systems": ["ecmwf"],
    }

    out_dir = PROCESSED_TRACKS_DIR / storm.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{storm.lower()}_{best_base:%Y%m%dT%H%M%S}_raw.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(result, f)
    return result


def run_read_tracks(storms=None):
    storm_list = storms or ALL_STORMS
    results = {}
    for cfg in storm_list:
        r = read_storm_ensemble(cfg)
        if r:
            results[cfg["storm_name"]] = r
    print(f"\n=== read_tracks: {len(results)}/{len(storm_list)} storms have ensembles ===")
    return results


if __name__ == "__main__":
    run_read_tracks()
