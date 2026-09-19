"""Configuration for ensemble track model -- 2023-2025 NA hurricanes.

ECMWF TIGGE data layout (external archive, NERSC):
    $TIGGE_BASE_DIR/{year}/{YYYYMMDD}/z_tigge_c_ecmf_*.xml
    (default /global/cfs/cdirs/m5011/Jay/TIGGE/ecmf)
Best-track data (IBTrACS-derived, produced by IBtracs_datasets.py):
    {TRAIN_ROOT}/data/NA/{year}/{IBID}_{NAME}/track_intensity_6h.csv
Synthetic-track output:
    {TRAIN_ROOT}/ensemble_tracks/{name_lower}/synthetic_tracks_1000members.nc

Storms are AUTO-DISCOVERED from the best-track directory (every named NA storm in
2023-2025). The init_time of each storm is auto-determined in read_files.py by
scanning the TIGGE files for the base time that yields the most ensemble members.
"""
from pathlib import Path
from datetime import datetime, timedelta
import os
import re

# ── Base directories ──────────────────────────────────────────────────────────
TRAIN_ROOT = Path(__file__).resolve().parent.parent      # .../FAST_ML/training
PROJECT_ROOT = TRAIN_ROOT
ECMWF_BASE_DIR = Path(os.environ.get(
    "TIGGE_BASE_DIR", "/global/cfs/cdirs/m5011/Jay/TIGGE/ecmf"))
BEST_TRACK_DIR = TRAIN_ROOT / "data" / "NA"
# Output dir for the generated synthetic ensemble tracks:
SYNTH_TRACKS_DIR = TRAIN_ROOT / "ensemble_tracks"
PROCESSED_TRACKS_DIR = Path(__file__).resolve().parent / "processed"

SYNTH_TRACKS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_TRACKS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
Earth_Radius = 6_378_100.0  # meters (WGS84 mean)

YEARS = [2023, 2024, 2025, 2026]
N_TRACKS = 1000          # synthetic members to sample
DURATION_DAYS = 11.0     # synthetic track length (matches IRMA 264h)
MIN_MEMBERS = 20         # minimum ECMWF ensemble members required to keep a storm
INIT_SEARCH_LEAD_DAYS = 2   # search TIGGE base times from (genesis - lead) ...
INIT_SEARCH_TAIL_DAYS = 4   #   ... to (genesis + tail) to pick best init cycle


def _name_from_dir(dirname: str) -> str:
    """'2023232N13300_FRANKLIN' -> 'FRANKLIN' (None for UNNAMED/STORMxxx)."""
    m = re.match(r"^\d{4}\d+[NS]\d+_(.+)$", dirname)
    if not m:
        return None
    name = m.group(1).strip().upper()
    if name.startswith("UNNAMED") or name.startswith("STORM"):
        return None
    return name


def _genesis_from_bt(storm_dir: Path):
    """First timestamp in track_intensity_6h.csv (storm genesis)."""
    csv = storm_dir / "track_intensity_6h.csv"
    if not csv.exists():
        return None
    try:
        with open(csv) as f:
            f.readline()  # header
            first = f.readline().strip().split(",")[0]
        return datetime.strptime(first, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def discover_storms(years=None):
    """Auto-discover every named NA storm with best-track data for given years.

    Returns a list of dicts: {storm_name, storm_dir, year, genesis}.
    init_time is determined later (read_files) by scanning TIGGE.
    """
    years = years or YEARS
    storms = []
    for yr in years:
        ydir = BEST_TRACK_DIR / str(yr)
        if not ydir.is_dir():
            continue
        for sdir in sorted(ydir.iterdir()):
            if not sdir.is_dir():
                continue
            name = _name_from_dir(sdir.name)
            if not name:
                continue
            genesis = _genesis_from_bt(sdir)
            if genesis is None:
                continue
            storms.append({
                "storm_name": name,
                "ibtracs_id": sdir.name,
                "storm_dir": str(sdir),
                "year": yr,
                "basin": "NA",
                "genesis": genesis,
            })
    return storms


# Discovered storm list (computed at import)
ALL_STORMS = discover_storms()
STORMS = ALL_STORMS
