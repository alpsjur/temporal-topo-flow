"""
This script generates a cross-slope wind stress forcing file for a given simulation configuration.
The generated forcing is saved as a NetCDF file in the input/forcing/ directory.

To create the forcing file for a specific simulation, run from the command line:
    python scripts/generate-crosslope-forcing.py configs/{name}.json
"""

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

    # NOTE: This spesific forcing is intentionally generated using `default_params`, which includes bumps
    # even if the actual simulation uses a smooth or different bathymetry.
    # This creates a cross-slope component in the forcing. 
    ds = generate_sinusoidal_depthfollowing_forcing(default_params, params)

    outpath = f"input/forcing/{config}_forcing.nc"
    ds.to_netcdf(outpath)
    print(f"Saved forcing file to: {outpath}")

if __name__ == "__main__":
    main()
