import xarray as xr
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.bathymetry import generate_bathymetry, bathymetry_gradient
from utils.config import load_config, default_params

def main():
    # Load user-defined parameters
    params = load_config()
    config = params["name"]
    T, tau0 = params["T"], params["tau0"]
    time = np.arange(0, params["tmax"], params["outputtime"])

    # Generate bathymetry and compute gradients
    X, Y, h = generate_bathymetry(default_params)
    dh_dx, dh_dy = bathymetry_gradient(h, default_params["dx"], default_params["dy"])

    # Compute magnitude of gradient
    mag = np.sqrt(dh_dx**2 + dh_dy**2)

    # Initialize tangential vectors
    u_tan = np.empty_like(dh_dx)
    v_tan = np.empty_like(dh_dy)

    # Where slope is flat, set direction to x-only
    threshold=1e-10
    flat = mag < threshold
    u_tan[flat] = 1.0
    v_tan[flat] = 0.0

    # Where slope is nonzero, normalize gradient
    u_tan[~flat] = dh_dy[~flat] / mag[~flat]
    v_tan[~flat] = dh_dx[~flat] / mag[~flat]

    # Sinusoidal time variation
    amplitude = tau0 * np.sin(2 * np.pi * time / T)

    tau_x = amplitude[:, None, None] * u_tan[None, :, :]
    tau_y = amplitude[:, None, None] * v_tan[None, :, :]

    # Package into xarray dataset
    ds = xr.Dataset(
        {"forcing_x": (["time", "x", "y"], tau_x),
         "forcing_y": (["time", "x", "y"], tau_y)},
        coords={"time": time, "x": X[:, 0], "y": Y[0, :]}
    )

    # Save to NetCDF
    outpath = f"input/forcing/{config}_forcing.nc"
    ds.to_netcdf(outpath)
    print(f"Saved forcing file to: {outpath}")

if __name__ == "__main__":
    main()
