"""
basin_regions.py
Define tropical cyclone basin regions for ERA5 data download.
Based on the basin definitions from generate_land_masks.py.

Each basin is defined by:
- Longitude range (in 0-360°E format for CDS API)
- Latitude range
- Optional: specific box boundaries to exclude land areas
"""

# Basin definitions
# Format: [North, West, South, East] for CDS API area parameter
# Note: CDS API uses [N, W, S, E] format where W/E are in 0-360°E

BASIN_REGIONS = {
    'NA': {  # North Atlantic
        'name': 'North Atlantic',
        'area': [60.0, 255.0, 0.0, 360.0],  # [N, W, S, E] in 0-360°E
        'lon_range_0_360': (255, 360),  # 0-360°E format
        'lon_range_180': (-105, 0),     # -180 to 180°E format
        'lat_range': (0, 60),
        'description': 'Atlantic basin including Caribbean and Gulf of Mexico'
    },
    'EP': {  # Eastern Pacific
        'name': 'Eastern Pacific',
        'area': [60.0, 180.0, 0.0, 290.0],
        'lon_range_0_360': (180, 290),
        'lon_range_180': (-180, -70),
        'lat_range': (0, 60),
        'description': 'Eastern Pacific basin'
    },
    'WP': {  # Western Pacific
        'name': 'Western Pacific',
        'area': [60.0, 100.0, 0.0, 180.0],
        'lon_range_0_360': (100, 180),
        'lon_range_180': (100, 180),
        'lat_range': (0, 60),
        'description': 'Western Pacific basin'
    },
    'NI': {  # Northern Indian
        'name': 'Northern Indian',
        'area': [49.0, 30.0, 0.0, 100.0],
        'lon_range_0_360': (30, 100),
        'lon_range_180': (30, 100),
        'lat_range': (0, 49),
        'description': 'Northern Indian Ocean basin'
    },
    'SI': {  # Southern Indian
        'name': 'Southern Indian',
        'area': [0.0, 10.0, -45.0, 100.0],
        'lon_range_0_360': (10, 100),
        'lon_range_180': (10, 100),
        'lat_range': (-45, 0),
        'description': 'Southern Indian Ocean basin'
    },
    'AU': {  # Australia
        'name': 'Australia',
        'area': [0.0, 100.0, -45.0, 170.0],
        'lon_range_0_360': (100, 170),
        'lon_range_180': (100, 170),
        'lat_range': (-45, 0),
        'description': 'Australia basin'
    },
    'SP': {  # Southern Pacific
        'name': 'Southern Pacific',
        'area': [0.0, 170.0, -45.0, 260.0],
        'lon_range_0_360': (170, 260),
        'lon_range_180': (170, -100),
        'lat_range': (-45, 0),
        'description': 'Southern Pacific basin'
    },
    'GL': {  # Global (tropical, ±50°)
        'name': 'Global Tropical',
        'area': [50.0, 0.0, -50.0, 360.0],
        'lon_range_0_360': (0, 360),
        'lon_range_180': (-180, 180),
        'lat_range': (-50, 50),
        'description': 'Global tropical region (±50° latitude)'
    }
}


def get_basin_area(basin_code: str) -> list:
    """
    Get CDS API area parameter for a basin.
    
    Args:
        basin_code: Basin code ('NA', 'EP', 'WP', etc.)
    
    Returns:
        List [N, W, S, E] for CDS API area parameter
    """
    if basin_code.upper() not in BASIN_REGIONS:
        raise ValueError(f"Unknown basin code: {basin_code}. "
                        f"Available: {list(BASIN_REGIONS.keys())}")
    return BASIN_REGIONS[basin_code.upper()]['area']


def get_basin_info(basin_code: str) -> dict:
    """
    Get full basin information.
    
    Args:
        basin_code: Basin code ('NA', 'EP', 'WP', etc.)
    
    Returns:
        Dictionary with basin information
    """
    if basin_code.upper() not in BASIN_REGIONS:
        raise ValueError(f"Unknown basin code: {basin_code}. "
                        f"Available: {list(BASIN_REGIONS.keys())}")
    return BASIN_REGIONS[basin_code.upper()]


def list_basins():
    """List all available basins."""
    print("Available TC basins:")
    for code, info in BASIN_REGIONS.items():
        print(f"  {code}: {info['name']} - {info['description']}")
        print(f"    Area: {info['area']} [N, W, S, E] in 0-360°E format")


if __name__ == "__main__":
    list_basins()
    print("\nExample: North Atlantic basin area for CDS API:")
    print(f"  {get_basin_area('NA')}")

