import numpy as np
import xarray as xr
from utils.bathymetry import generate_bathymetry, bathymetry_gradient

def generate_sinusoidal_depthfollowing_forcing(default_params, params):
    """
    Generate wind stress forcing that follows bathymetry contours,
    with sinusoidal time variation and amplitude tau0.

    Args:
        default_params (dict): Dictionary of default comfiguration parameters. 
                                Used to generate bathymetry with bumps. 
        params (dict): Dictionary of configuration parameters. 

    Returns:
       xr.Dataset: Dataset with variables:
            - 'forcing_x' (time, yC, xC): Zonal wind stress [m²/s²]
            - 'forcing_y' (time, yC, xC): Meridional wind stress [m²/s²]
    """
    T, tau0 = params["T"], params["tau0"]
    
     # Time and spatial grids
    time_sec = np.arange(0, params["tmax"]+params["dt"]/2, params["outputtime"])
    time = time_sec.astype("timedelta64[s]").astype("timedelta64[ns]")
    #time_ = np.arange(0, params["tmax"], params["outputtime"])
    x = np.arange(params["dx"] / 2, params["Lx"], params["dx"])
    y = np.arange(params["dy"] / 2, params["Ly"], params["dy"])

    # Generate bathymetry and gradients
    X, Y, h = generate_bathymetry(default_params)
    dh_dx, dh_dy = bathymetry_gradient(h, default_params["dx"], default_params["dy"])

    # Compute unit tangent vectors to bathymetry contours
    threshold = 1e-10
    u_tan = np.where(dh_dy < threshold, 1.0, dh_dy / dh_dy)
    v_tan = np.where(dh_dy < threshold, 0.0, -dh_dx / dh_dy)

    # Time-dependent amplitude of forcing
    amplitude = tau0 * np.sin(2 * np.pi * time_sec / T)

    forcing_x = amplitude[:, None, None] * u_tan[None, :, :]
    forcing_y = amplitude[:, None, None] * v_tan[None, :, :]

    return xr.Dataset(
        {
            "forcing_x": (["time", "yC", "xC"], forcing_x),
            "forcing_y": (["time", "yC", "xC"], forcing_y)
        },
        coords={
            "time": time,
            "time_sec": time_sec,
            "xC": x,
            "yC": y
        }
    )


def generate_zonal_forcing(params):
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
    
    
def windforcing(t, params):
    """
    Compute time-dependent x-direction wind forcing averaged over output intervals.
    Used to generate timeseries of zonal wind stress forcing for time points `t`.
    Used both for generating forcing files and for plotting and postprocessing.
    
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
