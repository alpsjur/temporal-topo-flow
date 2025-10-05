import xarray as xr
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.bathymetry import generate_bathymetry
from utils.forcing import generate_forcing

def ensure_dir(directory_path: str):
    """
    Ensure that a directory exists. If it does not, create it.
    """
    path = Path(directory_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {path}")


def read_raw_output(params):
    """
    Load model output from a NetCDF file and attach bathymetry and forcing fields.

    This function reads raw model output from:
        params["filepath"] + "raw/" + params["name"] + ".nc"

    If 'bathymetry_file' is specified in `params`, the bathymetry field is read from
    that file (must contain a variable named 'bath'). Otherwise, the bathymetry is 
    generated from parameters using `generate_bathymetry(params)`.

    If 'forcing_file' is specified in `params`, forcing is loaded from that file. 
    Otherwise, forcing is generated from parameters using default expression.

    Args:
        params (dict): Dictionary of configuration parameters. 

    Returns:
        xr.Dataset: A dataset containing:
            - Raw model output (e.g., u, v, eta, etc.)
            - 'bath' (2D array on ["yC", "xC"])
            - 'forcing_x' and 'forcing_y' (3D arrays on ["time", "yC", "xC"])
    """
    # Load raw model output
    ds = xr.open_dataset("/itf-fi-ml/home/alsjur/temporal-topo-flow/"+params["filepath"] + "raw/" + params["name"] + ".nc").squeeze()

    # Load or generate bathymetry
    if "bathymetry_file" in params:
        bath_ds = xr.open_dataset("/itf-fi-ml/home/alsjur/temporal-topo-flow/"+params["bathymetry_file"])
        if "bath" not in bath_ds:
            raise ValueError(f"Bathymetry file '{params['bathymetry_file']}' does not contain a 'bath' variable.")
        ds["bath"] = bath_ds["bath"]
    else:
        X, Y, bath = generate_bathymetry(params)
        ds["bath"] = (["yC", "xC"], bath)

    # Load or generate forcing
    if "forcing_file" in params:
        f = xr.open_dataset("/itf-fi-ml/home/alsjur/temporal-topo-flow/" + params["forcing_file"])

        # 1) rename x/y -> xC/yC (your generator uses x,y)
        if ("x" in f.dims) or ("x" in f.coords): f = f.rename({"x": "xC"})
        if ("y" in f.dims) or ("y" in f.coords): f = f.rename({"y": "yC"})

        # 2) ensure variable dims are (time, yC, xC) to match the model
        for v in ["forcing_x", "forcing_y"]:
            if v in f and tuple(f[v].dims) != ("time", "yC", "xC"):
                wanted = ("time", "yC", "xC")
                have = f[v].dims
                if set(have) == set(wanted):
                    f[v] = f[v].transpose(*wanted)
                else:
                    raise ValueError(f"{v} has unexpected dims {have}; expected {wanted} (any order ok).")

        # 3) make forcing time timedelta64[ns] 
        import pandas as pd
        t_f = f["time"]
        if np.issubdtype(t_f.dtype, np.timedelta64):
            f = f.assign_coords(time=t_f.astype("timedelta64[ns]"))
        elif np.issubdtype(t_f.dtype, np.datetime64):
            f = f.assign_coords(time=(t_f - t_f.isel(time=0)).astype("timedelta64[ns]"))
        else:
            # numeric -> seconds
            f = f.assign_coords(time=xr.DataArray(pd.to_timedelta(t_f.values, unit="s"), dims="time"))

        # 4) make model time timedelta64[ns] as well
        t_ds = ds["time"]
        if np.issubdtype(t_ds.dtype, np.timedelta64):
            t_model = t_ds.astype("timedelta64[ns]")
        elif np.issubdtype(t_ds.dtype, np.datetime64):
            t_model = (t_ds - t_ds.isel(time=0)).astype("timedelta64[ns]")
        else:
            t_model = xr.DataArray(pd.to_timedelta(t_ds.values, unit=params.get("model_time_unit", "s")), dims="time")

        # 5) align horizontally if grids differ (nearest to avoid smoothing)
        if "xC" in f.coords and "xC" in ds.coords and not np.array_equal(f["xC"], ds["xC"]):
            f = f.interp(xC=ds["xC"], method="nearest")
        if "yC" in f.coords and "yC" in ds.coords and not np.array_equal(f["yC"], ds["yC"]):
            f = f.interp(yC=ds["yC"], method="nearest")

        # 6) interpolate in time
        f = f.interp(time=t_model, method="linear")

        # 7) attach
        for key in ["forcing_x", "forcing_y"]:
            if key not in f:
                raise ValueError(f"'{key}' not found in forcing file: {params['forcing_file']}")
            ds[key] = f[key]
    else:
        forcing = generate_forcing(params) 
        for key in ["forcing_x", "forcing_y"]:
            if key not in forcing:
                raise ValueError(f"Missing expected key '{key}' in output from generate_forcing()")
            ds[key] = forcing[key]
            
    # remove redundant coordinates
    ds = ds.drop_vars([v for v in ["zC", "zF"] if v in ds])

    return ds

def save_processed_ds(ds, params, onH=False):
    """
    Save an xarray dataset of processed results to NetCDF format.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to save.
    params : dict
        Dictionary of parameters containing:
            - "filepath": base path where simulation outputs are stored.
            - "name": configuration name.
    onH : bool, optional
        If True, save the dataset in a "depth-following" subdirectory
        with a `_depth-following.nc` suffix.
        If False (default), save in a "cartesian" subdirectory
        with a `_cartesian.nc` suffix.

    Notes
    -----
    - The function ensures that the appropriate output directory exists
      before attempting to save.
    """

    if onH:
        # Ensure depth-following directory exists
        ensure_dir(params["filepath"] + "processed/depth-following/")
        # Build output filename for depth-following dataset
        output_name = (
            params["filepath"] + "processed/depth-following/" 
            + params["name"] + "_depth-following.nc"
        )
    else:
        # Ensure cartesian directory exists
        ensure_dir(params["filepath"] + "processed/cartesian/")
        # Build output filename for cartesian dataset
        output_name = (
            params["filepath"] + "processed/cartesian/" 
            + params["name"] + "_cartesian.nc"
        )
    
    # Save dataset to NetCDF file
    ds.to_netcdf(output_name, mode="w")