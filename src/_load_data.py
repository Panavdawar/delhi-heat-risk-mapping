# === NEW: Logging + chunked xarray open ===
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def open_era5_dataset_chunked(netcdf_paths):
    """
    Open many ERA5 NetCDFs with xarray using chunking for Dask.
    netcdf_paths: list of file paths or a glob pattern
    """
    import xarray as xr
    logging.info("Opening ERA5 files with xarray (chunked)")
    ds = xr.open_mfdataset(netcdf_paths, combine="by_coords",
                           chunks={'time': 50, 'latitude': 50, 'longitude': 50},
                           parallel=True)
    logging.info("Opened dataset: dims=%s", ds.dims)
    return ds

def load_netcdf_to_df(netcdf_paths):
    """
    Load NetCDF files and convert to pandas DataFrame.
    netcdf_paths: list of file paths or a glob pattern
    """
    ds = open_era5_dataset_chunked(netcdf_paths)
    df = ds.to_dataframe().reset_index()
    logging.info("Converted to DataFrame with shape %s", df.shape)
    return df

