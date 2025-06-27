import xarray as xr
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.bathymetry import generate_bathymetry, bathymetry_gradient
from utils.config import load_config, default_params
from utils.forcing import generate_sinusoidal_depthfollowing_forcing

def main():
    params = load_config()
    config = params["name"]

    # NOTE: Forcing is intentionally generated using `default_params` with bumps,
    # even if the actual simulation uses a smooth or different bathymetry.
    # This creates a cross-contour component in the forcing. 
    ds = generate_sinusoidal_depthfollowing_forcing(default_params)

    outpath = f"input/forcing/{config}_forcing.nc"
    ds.to_netcdf(outpath)
    print(f"Saved forcing file to: {outpath}")

if __name__ == "__main__":
    main()
