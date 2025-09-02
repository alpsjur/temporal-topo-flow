import numpy as np
import xarray as xr
from utils.bathymetry import generate_bathymetry, bathymetry_gradient

def generate_sinusoidal_depthfollowing_forcing(params):
    """
    Generate wind stress forcing that follows bathymetry contours,
    with sinusoidal time variation and magnitude tau0.

    Args:
        params (dict): Dictionary of configuration parameters. 

    Returns:
        xr.Dataset: Dataset with 'forcing_x' and 'forcing_y' on (time, x, y)
    """
    T, tau0 = params["T"], params["tau0"]
    time = np.arange(0, params["tmax"], params["outputtime"])

    # Generate bathymetry and gradients
    X, Y, h = generate_bathymetry(params)
    dh_dx, dh_dy = bathymetry_gradient(h, params["dx"], params["dy"])

    # Normalize gradient to get tangential vectors
    mag = np.sqrt(dh_dx**2 + dh_dy**2)
    threshold = 1e-10

    u_tan = np.where(mag < threshold, 1.0, dh_dy / mag)
    v_tan = np.where(mag < threshold, 0.0, dh_dx / mag)

    # Time-dependent amplitude of forcing
    amplitude = tau0 * np.sin(2 * np.pi * time / T)

    tau_x = amplitude[:, None, None] * u_tan[None, :, :]
    tau_y = amplitude[:, None, None] * v_tan[None, :, :]

    return xr.Dataset(
        {
            "forcing_x": (["time", "x", "y"], tau_x),
            "forcing_y": (["time", "x", "y"], tau_y),
        },
        coords={"time": time, "x": X[:, 0], "y": Y[0, :]}
    )


def windforcing(t, params):
    """
    Compute time-dependent x-direction wind forcing averaged over output intervals.

    Args:
        t (1D array): Time points corresponding to model output times.
        params (dict): Dictionary of configuration parameters. 

    Returns:
        ndarray: x-direction wind forcing at each output time (1D array)
    """
    T = params["T"]
    outputtime = params["outputtime"]
    tau0 = params["tau0"]
    dt = params["dt"]

    omega = 2 * np.pi / T
    t_hr = np.arange(0, len(t) * outputtime, dt)
    window = round(outputtime / dt)

    windforce_hr = tau0 * np.sin(omega * t_hr) 
    forcing = windforce_hr.reshape(-1, window).mean(axis=1)

    return forcing


def generate_forcing(params):
    """
    Generate time-dependent wind forcing dataset with zonal wind stress only.

    The x-component is sinusoidal in time and uniform in space.
    The y-component is zero.

    Args:
        params (dict): Dictionary of configuration parameters. 

    Returns:
        xr.Dataset: Dataset with variables:
            - 'forcing_x' (time, yC, xC): Zonal wind stress [m²/s²]
            - 'forcing_y' (time, yC, xC): Meridional wind stress (zero)
    """
    # Time and spatial grids
    time_sec = np.arange(0, params["tmax"]+params["dt"]/2, params["outputtime"])
    time = time_sec.astype("timedelta64[s]").astype("timedelta64[ns]")
    x = np.arange(params["dx"] / 2, params["Lx"], params["dx"])
    y = np.arange(params["dy"] / 2, params["Ly"], params["dy"])

    # Get x-forcing over time
    forcing_x_t = windforcing(time, params)

    # Broadcast to 3D arrays
    forcing_x = forcing_x_t[:, np.newaxis, np.newaxis] * np.ones((1, len(y), len(x)))
    forcing_y = np.zeros_like(forcing_x)

    return xr.Dataset(
        {
            "forcing_x": (["time", "yC", "xC"], forcing_x),
            "forcing_y": (["time", "yC", "xC"], forcing_y)
        },
        coords={
            "time": time,
            "xC": x,
            "yC": y
        }
    )