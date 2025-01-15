import xarray as xr
import numpy as np
import json

import xarray as xr
import numpy as np
import json

# ================================
# Default Parameters
# ================================

default_params = {
    # Data storage parameters
    "name": "slope-001",
    "filepath": "slope/output/",

    # Grid configuration
    "dx": 1e3,
    "dy": 1e3,
    "Lx": 120e3,
    "Ly": 90e3,

    # Simulation settings
    "dt": 2.0,
    "tmax": 64 * 86400.0,
    "outputtime": 3 * 3600.0,

    # Forcing parameters
    "rho": 1e3,
    "d": 0.1,
    "T": 4 * 86400.0,
    "dn" : 0,
    "Tn" : 0,
    "R": 5e-4,
    "f": 1e-4,
    "gravitational_acceleration": 9.81,

    # Bathymetry parameters
    "W": 30e3,
    "XC": 45e3,
    "DS": 800.0,
    "DB": 100.0,
    "sigma": 1.0,
    "a": 10e3,
    "lam": 45e3,
    "noise": False,
}

# ================================
# Configuration Loading
# ================================

def load_config(file_path, params):
    """
    Load configuration from a JSON file and overwrite default parameters.

    Parameters:
        file_path (str): Path to the configuration file.
        params (dict): Default parameters dictionary.

    Returns:
        dict: Updated parameters.
    """
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
            params.update(config)  # Overwrite defaults with values from the config
    except FileNotFoundError:
        print(f"Warning: Configuration file '{file_path}' not found. Using default parameters.")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file '{file_path}': {e}")
    return params

# ================================
# Bathymetry Calculations
# ================================

def calculate_bathymetry(x, y, params):
    """Compute bathymetry based on parameters and grid coordinates."""
    a = params["a"]
    lam = params["lam"]
    DB = params["DB"]
    DS = params["DS"]
    XC = params["XC"]
    W = params["W"]

    corr = a * np.sin(2 * np.pi * y / lam)
    slope = DB + 0.5 * DS * (1 + np.tanh(np.pi * (x - XC - corr) / W))
    basin = DB + DS

    return np.where(x < XC + W, slope, basin)

def get_h(params):
    """Generate the bathymetry grid."""
    Lx = params["Lx"]
    Ly = params["Ly"]
    dx = params["dx"]
    dy = params["dy"]

    x = np.arange(0, Lx, dx)
    y = np.arange(0, Ly, dy)

    X, Y = np.meshgrid(x, y)
    h = calculate_bathymetry(X, Y, params)

    return h

# ================================
# Depth-Following Contour Calculations
# ================================

def calculate_xt(y, H, params):
    """Compute x-coordinates along a contour based on depth."""
    W = params["W"]
    DB = params["DB"]
    DS = params["DS"]
    XC = params["XC"]
    a = params["a"]
    lam = params["lam"]

    x0 = -W * np.arctanh((2 * DB - 2 * H + DS) / DS) / np.pi + XC
    xt = a * np.sin(2 * np.pi * y / lam) + x0

    # Constrain xt to valid range
    xt = np.clip(xt, 0, XC + W)
    return xt

def compute_contour_vectors(y, H, params):
    """Compute vectors along depth-following contours."""
    xt = calculate_xt(y, H, params)
    dxt = 2 * np.pi * params["a"] / params["lam"] * np.cos(2 * np.pi * y / params["lam"])

    # Normalize vectors
    dyt = np.ones_like(dxt)
    norm = np.sqrt(dxt**2 + dyt**2)

    return dxt / norm, dyt / norm

def calculate_contour_lengths(x, y, Ny, Nl, upstream=False):
    """Calculate segment lengths along the contour."""
    dx = np.diff(x)
    dy = np.diff(y)
    dl = np.sqrt(dx**2 + dy**2)

    dl_result = np.zeros_like(x)
    if upstream:
        dl_result[1:] = dl
        dl_result[0] = dl[int(Ny / Nl) - 1]
    else:
        dl_result[1:-1] = (dl[:-1] + dl[1:]) / 2
        dl_result[0] = dl[int(Ny / Nl) - 1]
        dl_result[-1] = dl[int(Ny / Nl) - 2]

    return dl_result

def _depth_following_contour_variables(params, H):
    Ly = params["Ly"]
    dy = params["dy"]
    lam = params["lam"]
    
    Ny = Ly//dy 
    Nl = Ly//lam

    yt = np.arange(dy/2, Ly+dy/2, dy)
    
    xt = calculate_xt(yt, H, params)
    dxt, dyt = compute_contour_vectors(yt, H, params)
    dlt = calculate_contour_lengths(xt, yt, Ny, Nl, upstream=True)
    
    return xt, yt, dxt, dyt, dlt 

def _flat_contour_variables(params, xflatidx):
    Ly = params["Ly"]
    dx = params["dx"]
    dy = params["dy"]
    Ny = round(Ly/dy)
    
    xt = np.ones(Ny)*xflatidx*dx 
    yt = np.arange(dy/2, Ly+dy/2, dy)
    
    dxt = np.zeros_like(xt)
    dyt = np.ones_like(xt)
    
    dlt = np.ones_like(xt)*dy
    
    return xt, yt, dxt, dyt, dlt 

def depth_following_contour(params, H, flat=False, xflat=0):
    Ly = params["Ly"]
    dy = params["dy"]
    Ny = round(Ly/dy)
    
    if not flat:
        xt, yt, dxt, dyt, dlt = _depth_following_contour_variables(params, H)
    else:
        xt, yt, dxt, dyt, dlt = _flat_contour_variables(params, xflat)
    
    contour = xr.Dataset(
        data_vars=dict(
            dtdx=(["j"], dxt),
            dtdy=(["j"], dyt),
            dl=(["j"], dlt),
        ),
        coords=dict(
            x=(["j"], xt),
            y=(["j"], yt),
            j = np.arange(Ny, dtype=np.int32)
        )
    )
    
    return contour

def depth_following_grid(params):
    """Construct depth-following grid."""
    Lx = params["Lx"]
    Ly = params["Ly"]
    dx = params["dx"]
    dy = params["dy"]
    lam = params["lam"]

    Nx = Lx // dx
    Ny = Ly // dy
    slope_end_idx, _ = slope_end(params)

    x = np.arange(0, Lx, dx)
    y = np.arange(0, Ly, dy)

    X, Y = np.meshgrid(x, y)
    h = calculate_bathymetry(X, Y, params)

    # Construct contour-following grid
    Xt, Yt, dXt, dYt, dLt = np.zeros_like(X), np.zeros_like(Y), np.zeros_like(X), np.ones_like(Y), np.ones_like(X)

    for i, Ht in enumerate(h.mean(axis=0)):
        xt, yt, dxt, dyt, dlt = _depth_following_contour_variables(params, Ht)
        Xt[:, i], Yt[:, i], dXt[:, i], dYt[:, i], dLt[:, i] = xt, yt, dxt, dyt, dlt

    return xr.Dataset(
        data_vars=dict(
            dtdx=(["j", "i"], dXt),
            dtdy=(["j", "i"], dYt),
            dl=(["j", "i"], dLt),
        ),
        coords=dict(
            x=(["j", "i"], Xt),
            y=(["j", "i"], Yt),
            i=np.arange(Nx, dtype=np.int32),
            j=np.arange(Ny, dtype=np.int32),
        ),
    )

# ================================
# Analytical Models
# ================================

def analytical_circ(params, t, cL, H, nonlin=None):
    """
    Compute the analytical circulation for a given time series.

    Parameters:
        params (dict): Simulation parameters.
        t (array): Time series.
        cL (float): Characteristic length scale.
        H (float): Depth.
        nonlin (float, optional): Non-linear component.

    Returns:
        array: Analytical circulation over time.
    """
    T = params["T"]
    outputtime = params["outputtime"]
    dt = params["dt"]
    Ly = params["Ly"]
    rho = params["rho"]
    R = params["R"]
    d = params["d"]

    omega = 2 * np.pi / T
    t_hr = np.arange(0, len(t) * outputtime, dt)
    window = round(outputtime / dt)

    windforce_hr = -d * Ly * np.sin(omega * t_hr) / (rho * H * cL)
    forcing = windforce_hr.reshape(-1, window).mean(axis=1)

    if nonlin is not None:
        forcing += nonlin / cL

    analytical = np.zeros_like(t)
    for i in range(1, len(t)):
        filtered_forcing = np.exp(-R * (t[i:0:-1]) / H) * forcing[:i]
        analytical[i] = np.sum(filtered_forcing * outputtime)

    return analytical

# ================================
# Helper Functions and Utilities
# ================================

def slope_end(params):
    """Determine slope endpoint."""
    dx = params["dx"]
    XC = params["XC"]
    W = params["W"]
    return XC + W, round((XC + W) / dx)

def compute_alignment(Ut, Vt):
    """Calculate alignment and misalignment metrics."""
    magnitude = np.sqrt(Ut**2 + Vt**2)
    alignment = np.abs(Vt) / (magnitude + 1e-10)  # Add small epsilon to avoid division by zero
    return alignment

def get_contour_following_velocities(contour, ds):
    """Calculate contour-following velocities."""
    ut = ds.u.interp(xF=contour.x, yC=contour.y)
    vt = ds.v.interp(xC=contour.x, yF=contour.y)
    Ut = ut * contour.dtdy - vt * contour.dtdx
    Vt = ut * contour.dtdx + vt * contour.dtdy
    return Ut, Vt

def get_vorticityflux_at_contour(contour, ds):
    """Calculate vorticity flux at the contour."""
    vortu = ds.omegau
    vortv = ds.omegav
    vortut = vortu.interp(xF=contour.x, yF=contour.y)
    vortvt = vortv.interp(xF=contour.x, yF=contour.y)
    vortUt = vortut * contour.dtdy - vortvt * contour.dtdx
    return (vortUt * contour.dl).sum(dim="j")

def load_parameters():
    """Load simulation parameters from configuration file or defaults."""
    import sys
    if len(sys.argv) == 2:
        config_path = sys.argv[1]
        print(f"Loading configuration from {config_path}")
        return load_config(config_path, default_params)
    else:
        print("No configuration file provided. Using default parameters.")
        return default_params

def load_dataset(filepath, name):
    """Load dataset from NetCDF file."""
    return xr.open_dataset(filepath + name + ".nc").squeeze()

def truncate_time_series(ds):
    """Truncate time series if a blow-up in velocity is detected."""
    if np.abs(ds.v).max() > 1.5:
        tstop = np.nonzero(np.abs(ds.v).max(dim=("xC", "yF")).values > 1.5)[0][0]
        print("Blow-up in velocity, truncating time series.")
        return ds.isel(time=slice(None, tstop))
    return ds


