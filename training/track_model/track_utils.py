"""Track processing utilities."""
import sys
from pathlib import Path
# Add parent directory to path to find config
sys.path.append(str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
import xml.etree.ElementTree as ET
from config import TRACKS_DIR

def time_to_step_index(time_val: datetime, reference_time: Optional[datetime],
                       dt_hours: float = 6.0, tolerance_hours: float = 1.5,
                       zero_based: bool = True) -> Optional[int]:
    """Convert absolute time to discrete forecast step index."""
    if reference_time is None or time_val is None:
        return None
    delta_hours = (time_val - reference_time).total_seconds() / 3600.0
    step_float = delta_hours / dt_hours
    step_idx = int(round(step_float))
    step_hours = step_idx * dt_hours
    if abs(delta_hours - step_hours) > tolerance_hours:
        return None
    if zero_based:
        return step_idx if step_idx >= 0 else None
    return step_idx if step_idx >= 1 else None


class TrackPreprocessor:
    """Track preprocessor for TIGGE XML parsing."""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or TRACKS_DIR
        self.output_dir = self.data_dir / "processed"
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def parse_tigge_xml(self, filepath: Path, storm_name: str = None,
                        forced_init_time: Optional[datetime] = None) -> List[Dict]:
        """Parse TIGGE XML file to extract TC tracks."""
        tracks = []
        try:
            root = ET.parse(filepath).getroot()
            filename = filepath.name.lower()
            ensemble_system = 'ecmwf' if ('ecmf' in filename or 'ecmwf' in filename) else \
                            'gefs' if ('kwbc' in filename or 'gefs' in filename) else 'unknown'
            
            header = root.find('header')
            base_time = None
            if header is not None:
                base_time_elem = header.find('baseTime')
                if base_time_elem is not None and base_time_elem.text:
                    try:
                        base_time = datetime.strptime(base_time_elem.text.strip(), "%Y-%m-%dT%H:%M:%SZ")
                    except:
                        base_time = None
            
            if forced_init_time and base_time:
                base_naive = base_time.replace(tzinfo=None) if base_time.tzinfo else base_time
                init_naive = forced_init_time.replace(tzinfo=None) if forced_init_time.tzinfo else forced_init_time
                if abs((base_naive - init_naive).total_seconds()) / 3600.0 > 1.01:
                    return []
            
            for data_elem in root.findall('data'):
                member_id = int(data_elem.get('member')) if 'member' in data_elem.attrib else \
                           -1 if data_elem.get('type') == 'analysis' else len(tracks)
                if ensemble_system == 'ecmwf' and member_id is not None and member_id > 51:
                    continue
                
                for disturbance in data_elem.findall('disturbance'):
                    name_elem = disturbance.find('cycloneName')
                    cyclone_name = name_elem.text.strip() if name_elem is not None and name_elem.text else None
                    if storm_name and cyclone_name and storm_name.upper() not in cyclone_name.upper():
                        continue
                    
                    num_elem = disturbance.find('cycloneNumber')
                    cyclone_num = int(num_elem.text.strip()) if num_elem is not None and num_elem.text else None
                    basin_elem = disturbance.find('basin')
                    basin = basin_elem.text.strip() if basin_elem is not None and basin_elem.text else None
                    
                    lon_list, lat_list, time_list = [], [], []
                    for fix in disturbance.findall('fix'):
                        lat_elem, lon_elem, time_elem = fix.find('latitude'), fix.find('longitude'), fix.find('validTime')
                        if lat_elem is None or lon_elem is None:
                            continue
                        try:
                            lat_str, lon_str = lat_elem.text.strip(), lon_elem.text.strip()
                            lat_units, lon_units = (lat_elem.get('units') or '').upper(), (lon_elem.get('units') or '').upper()
                            
                            lat_val = float(lat_str.rstrip('NS').strip())
                            if lat_str.upper().endswith('S') or 'S' in lat_units:
                                lat_val = -lat_val
                            
                            lon_val = float(lon_str.rstrip('EW').strip())
                            if lon_str.upper().endswith('W') or 'W' in lon_units:
                                lon_val = -lon_val
                            lon_val = ((lon_val + 180) % 360) - 180
                            
                            fix_time = base_time
                            if time_elem is not None and time_elem.text:
                                try:
                                    fix_time = datetime.strptime(time_elem.text.strip(), "%Y-%m-%dT%H:%M:%SZ")
                                except:
                                    hour_attr = fix.get('hour')
                                    if hour_attr and base_time:
                                        fix_time = base_time + timedelta(hours=int(hour_attr))
                            
                            if fix_time:
                                lon_list.append(lon_val)
                                lat_list.append(lat_val)
                                time_list.append(fix_time)
                        except:
                            continue
                    
                    if len(lon_list) >= 2:
                        tracks.append({
                            'ensemble_system': ensemble_system,
                            'member_id': member_id,
                            'storm_name': cyclone_name,
                            'storm_number': cyclone_num,
                            'basin': basin,
                            'init_time': base_time,
                            'lon': np.array(lon_list),
                            'lat': np.array(lat_list),
                            'datetime': time_list
                        })
        except Exception as e:
            raise ValueError(f"Error parsing TIGGE XML file {filepath}: {e}")
        return tracks
