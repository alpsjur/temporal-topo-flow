import xarray as xr
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.bathymetry import generate_bathymetry
from utils.forcing import generate_zonal_forcing

def ensure_dir(directory_path: str):
    """
    Ensure that a directory exists. If it does not, create it.
    """
    path = Path(directory_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"Directory created: {path}")

def resolve_input_path(file_path: str) -> Path:
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        return path

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
    ds = xr.open_dataset(resolve_input_path(params["filepath"] + "raw/" + params["name"] + ".nc"),
                         ).squeeze()

    # Load or generate bathymetry
    if "bathymetry_file" in params:
        bath_ds = xr.open_dataset(resolve_input_path(params["bathymetry_file"]))
        if "bath" not in bath_ds:
            raise ValueError(f"Bathymetry file '{params['bathymetry_file']}' does not contain a 'bath' variable.")
        ds["bath"] = bath_ds["bath"]
    else:
        X, Y, bath = generate_bathymetry(params)
        ds["bath"] = (["yC", "xC"], bath)

    # Load or generate forcing
    if "forcing_file" in params:
        forcing_ds = xr.open_dataset(resolve_input_path(params["forcing_file"]),
                                    ).squeeze()
    
       
        
        if "forcing_x" not in forcing_ds or "forcing_y" not in forcing_ds:
            raise ValueError(f"Forcing file '{params['forcing_file']}' does not contain correct forcing variables.")
        ds["forcing_x"] = forcing_ds["forcing_x"]
        ds["forcing_y"] = forcing_ds["forcing_y"]

    
    else:
        forcing = generate_zonal_forcing(params) 
        for key in ["forcing_x", "forcing_y"]:
            ds[key] = forcing[key]
            
    # remove redundant coordinates
    ds = ds.drop_vars([v for v in ["zC", "zF"] if v in ds])

    return ds


