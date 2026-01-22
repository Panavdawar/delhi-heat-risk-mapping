# === NEW: Create Dwarka grid and fast spatial join ===
import geopandas as gpd
import numpy as np
from shapely.geometry import box, Point
import logging

def create_grid(bbox, rows=20, cols=20):
    south, west, north, east = bbox  # (south, west, north, east)
    lat_edges = np.linspace(south, north, rows+1)
    lon_edges = np.linspace(west, east, cols+1)
    cells = []
    for i in range(rows):
        for j in range(cols):
            minx, miny = lon_edges[j], lat_edges[i]
            maxx, maxy = lon_edges[j+1], lat_edges[i+1]
            geom = box(minx, miny, maxx, maxy)
            cells.append({'geometry': geom, 'cell_id': f'{i}_{j}'})
    grid = gpd.GeoDataFrame(cells, crs="EPSG:4326")
    logging.info("Created grid with %d cells", len(grid))
    return grid

def spatial_aggregate_points_to_grid(df_points, grid_gdf, agg_col='heat_index'):
    """
    df_points: pandas DataFrame with 'longitude','latitude',agg_col
    grid_gdf: GeoDataFrame of grid cells
    """
    import geopandas as gpd
    pts = gpd.GeoDataFrame(df_points, geometry=gpd.points_from_xy(df_points.longitude, df_points.latitude), crs="EPSG:4326")
    # === use spatial index and sjoin (vectorized) ===
    joined = gpd.sjoin(pts, grid_gdf, how='inner', predicate='within')
    agg = joined.groupby('cell_id')[agg_col].agg(['mean','max','count']).reset_index()
    grid_merged = grid_gdf.merge(agg, on='cell_id', how='left').fillna({'mean':0,'max':0,'count':0})
    return grid_merged

def spatial_join_weather(df, geojson_path):
    """
    Join weather data points with district boundaries to add district_name column.
    df: pandas DataFrame with 'latitude', 'longitude' columns
    geojson_path: path to geojson file with district polygons
    Returns: df with added 'district_name' column
    """
    districts = gpd.read_file(geojson_path)
    # Assume districts have 'name' property
    districts = districts.rename(columns={'name': 'district_name'})  # if needed, but probably already
    # Create GeoDataFrame from df points
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    # Spatial join
    joined = gpd.sjoin(pts, districts, how='left', predicate='within')
    # Add district_name to df
    df = df.copy()
    df['district_name'] = joined['district_name']
    return df

# === END NEW ===
