import xarray as xr
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.bathymetry import generate_bathymetry
from utils.forcing import generate_forcing


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
    ds = xr.open_dataset(params["filepath"] + "raw/" + params["name"] + ".nc").squeeze()

    # Load or generate bathymetry
    if "bathymetry_file" in params:
        bath_ds = xr.open_dataset(params["bathymetry_file"])
        if "bath" not in bath_ds:
            raise ValueError(f"Bathymetry file '{params['bathymetry_file']}' does not contain a 'bath' variable.")
        ds["bath"] = bath_ds["bath"]
    else:
        X, Y, bath = generate_bathymetry(params)
        ds["bath"] = (["yC", "xC"], bath)

    # Load or generate forcing
    if "forcing_file" in params:
        forcing_ds = xr.open_dataset(params["forcing_file"])

        # Interpolate forcing onto model output time
        forcing_interp = forcing_ds.interp(time=ds.time, method="linear")

        for key in ["forcing_x", "forcing_y"]:
            if key not in forcing_interp:
                raise ValueError(f"Variable '{key}' not found in forcing file: {params['forcing_file']}")
            ds[key] = forcing_interp[key]
    else:
        forcing = generate_forcing(params) 
        for key in ["forcing_x", "forcing_y"]:
            if key not in forcing:
                raise ValueError(f"Missing expected key '{key}' in output from generate_forcing()")
            ds[key] = forcing[key]
            
    # remove redundant coordinates
    ds = ds.drop_vars([v for v in ["zC", "zF"] if v in ds])

    return ds
