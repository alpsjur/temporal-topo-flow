import xarray as xr
import numpy as np
import json

import xarray as xr
import numpy as np
import json
from scipy.optimize import fsolve

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
    "Lx": 90e3,
    "Ly": 90e3,

    # Simulation settings
    "dt": 4.0,
    "tmax": 128 * 86400.0,
    "outputtime": 3 * 3600.0,

    # Forcing parameters
    "rho": 1e3,
    "d": 0.1,
    "T": 4 * 86400.0,
    "dn" : 0,
    "Tn" : 86400.0,
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

    steepness = (1/np.cosh(np.pi * (x - XC) / W))**2

    corr = a * np.sin(2 * np.pi * y / lam) * steepness
    slope = DB + 0.5 * DS * (1 + np.tanh(np.pi * (x - XC - corr) / W))
    basin = DB + DS

    return np.where(x < XC + W, slope, basin)

def get_h(params):
    """Generate the bathymetry grid."""
    Lx = params["Lx"]
    Ly = params["Ly"]
    dx = params["dx"]
    dy = params["dy"]

    x = np.arange(dx/2, Lx-dx/2, dx)
    y = np.arange(dy/2, Ly-dy/2, dy)

    X, Y = np.meshgrid(x, y)
    h = calculate_bathymetry(X, Y, params)

    return h

# ================================
# Depth-Following Contour Calculations
# ================================

def calculate_analytical_xt(y, H, params):
    """Compute x-coordinates along a contour based on depth
    by inplicitly solving analytical function for bathymetry.
    
    If y is a vector, solve for each element separately.
    """
    W = params["W"]
    DB = params["DB"]
    DS = params["DS"]
    XC = params["XC"]
    a = params["a"]
    lam = params["lam"]

    # Define function to solve for a single y value
    def solve_xt(y_val):
        x0 = XC + W / np.pi * np.arctanh((2 * (H - DB)) / DS - 1)

        def implicit_eq(x):
            x = np.atleast_1d(x)  # Ensure x is array-like
            steepness = 1 / np.cosh(np.pi * (x - XC) / W) ** 2
            corr = a * np.sin(2 * np.pi * y_val / lam) * steepness
            return x - (XC + W / np.pi * np.arctanh((2 * (H - DB)) / DS - 1) + corr)

        xt = fsolve(implicit_eq, np.array([x0]))[0]  # Solve for x
        return np.clip(xt, 0, XC + W)

    # Check if y is a scalar or an array
    if np.isscalar(y):
        return solve_xt(y)  # Solve for a single y
    else:
        return np.array([solve_xt(y_val) for y_val in y])  # Solve for each y separately
    
def calculate_numerical_xt(y, H, params):
    """Compute x-coordinates along a contour based on depth,
    using skimage to find points along the contour.
    """
    from skimage.measure import find_contours
    
    ds = xr.open_dataset(params["bathymetry_file"])
    
    contours = find_contours(ds.h.values, level=H)
    contour = contours[0]

    # reverse first dimension
    contour = contour[::-1,:]
    contour

    jdx = np.arange(0,90,1)

    idx = []
    for row in contour:
        if row[0] in jdx:
            idx.append(row[1])
    idx = np.array(idx)
    
    dx = params["dx"]
    xt = idx*dx 
    
    return xt

def calculate_xt(yt, H, params):
    if "bathymetry_file" in params:
        return calculate_numerical_xt(y, H, params)
    else:
        return calculate_analytical_xt(y, H, params)

def compute_contour_vectors(y, H, params):
    """Compute vectors along depth-following contours."""
    xt = calculate_xt(y, H, params)
 
    xtext = np.insert(xt, 0, xt[-1])
    xtext = np.append(xtext, xt[1])
    dxt = np.gradient(xtext)[1:-1]

    # Normalize vectors
    dyt = np.ones_like(dxt)*params["dy"]
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
        # use dl info from equivalent place at other part along nathymetry wave
        dl_result[0] = dl[int(Ny / (2*Nl)) - 1]
    else:
        dl_result[1:-1] = (dl[:-1] + dl[1:]) / 2
        # use dl info from equivalent place at other part along nathymetry wave
        dl_result[0] = dl[int(Ny / (2*Nl)) - 1]
        dl_result[-1] = dl[int(Ny / (2*Nl)) - 2]

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
    
    
def hnorm_on_xygrid(params, ds):
    dx = params["dx"]
    dy = params["dy"]
    
    h = ds.h.isel(time=1).squeeze().values
    h = np.insert(h, 0, h[-1,:], axis=0)
    h = np.append(h, h[1,:][None,:], axis=0)
    
    
    dh_dy, dh_dx = np.gradient(h, dy, dx, edge_order=2)
    
    dh_dx = dh_dx[1:-1,:]
    dh_dy = dh_dy[1:-1,:]
    
    magnitude = np.sqrt(dh_dy**2 + dh_dx**2) + 1e-12
    dh_dx /= magnitude
    dh_dy /= magnitude
    
    _, slope_end_idx = slope_end(params)
    dh_dx[:,slope_end_idx:] = 1
    dh_dy[:,slope_end_idx:] = 0
    
    hnorm = xr.Dataset(
        data_vars=dict(
            dhdx = (["yC", "xC"], dh_dx),
            dhdy = (["yC", "xC"], dh_dy)
        ),
        coords=dict(
            xC = ds.xC,
            yC = ds.yC
        )
    )
    
    return hnorm
    

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

def get_contour_following_tau(contour, ds):
    """Calculate contour-following stress."""
    tauxt = ds.forcing_x.interp(x=contour.x, y=contour.y)
    tauyt = ds.forcing_y.interp(x=contour.x, y=contour.y)
    Tauxt = tauxt * contour.dtdy - tauyt * contour.dtdx
    Tauyt = tauxt * contour.dtdx + tauyt * contour.dtdy
    return Tauxt, Tauyt

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


def axes_styling(ax):
    #ax.spines['left'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.tick_bottom()
    
    ax.spines['left'].set_color('lightgray')
    ax.spines['bottom'].set_color('lightgray')


# ================================
# Analytical Models
# ================================

def windforcing(t, params):
    
    T = params["T"]
    outputtime = params["outputtime"]
    rho = params["rho"]
    d = params["d"]
    dt = params["dt"]
    
    omega = 2 * np.pi / T
    t_hr = np.arange(0, len(t) * outputtime, dt)
    window = round(outputtime / dt)

    windforce_hr = -d * np.sin(omega * t_hr) / (rho)
    forcing = windforce_hr.reshape(-1, window).mean(axis=1)
    
    return forcing

def analytical_circ(params, t, cL, H, nonlin=None):
    """
    Compute the analytical circulation for a given time series.

    Parameters:
        params (dict): Simulation parameters.
        t (array): Time series.
        cL (float): Contour length.
        H (float): Depth.
        nonlin (float, optional): Non-linear component.

    Returns:
        array: Analytical circulation over time.
    """
    R = params["R"]
    Ly = params["Ly"]
    outputtime = params["outputtime"]
    
    forcing = windforcing(t, params)* Ly/(cL*H)

    if nonlin is not None:
        forcing += nonlin / cL

    analytical = np.zeros_like(t)
    for i in range(1, len(t)):
        filtered_forcing = np.exp(-R * (t[i:0:-1]) / H) * forcing[:i]
        analytical[i] = np.sum(filtered_forcing * outputtime)

    return analytical

def integrated_zonal_momentum_terms(params, ds, xidx):
    f = params["f"]
    R = params["R"]
    g = params["gravitational_acceleration"]

    t = ds.time / np.timedelta64(1, 's')
    uF = ds.u 
    uC = 0.5*(uF.isel(xF = xidx) + uF.isel(xF = xidx+1)).squeeze()
    
    x = ds.xC.isel(xC=xidx).values
    yC = ds.yC.values
    yF = ds.yF.values
    
    hC = calculate_bathymetry(x, yC, params)
    hF = calculate_bathymetry(x, yF, params)
    
    surfstress = -windforcing(t, params)
    nonlin = -ds.duvhdx.isel(xC=xidx).mean("yC")
    massflux = -(f*hC*uC).mean("yC")
    formstress = -(g*hF*ds.detady.isel(xC=xidx)).mean("yF")
    bottomstress = -R*ds.v.isel(xC=xidx).mean("yF")                
    
    terms = xr.Dataset(
        data_vars=dict(
            surfstress = (["time"], surfstress),
            nonlin = nonlin,
            massflux = massflux,
            formstress = formstress,
            bottomstress = bottomstress
        ),
        coords=dict(
            time = ds.time
        ),
    ).squeeze()
    
    # update forcing to forcing prescribed from file if file is defined
    if "forcing_file" in params:

        tau = xr.open_dataset(params["forcing_file"]).squeeze()
        terms["surfstress"][1:] = tau.forcing_y.isel(x=xidx).mean("y").values/params["rho"]


    return terms

def integrated_contour_momentum_terms(params, ds, H):
    Ly = params["Ly"]
    t = ds.time / np.timedelta64(1, 's')
    R = params["R"]
    f = params["f"]
    contour = depth_following_contour(params, H)
    cL = contour.dl.sum(dim=("j")).values
    Ut, Vt =  get_contour_following_velocities(contour, ds)
    
    nonlin = -get_vorticityflux_at_contour(contour, ds)*H/cL
    surfstress = -windforcing(t, params)*Ly/(cL)    
    Rv = R*Vt
    bottomstress = -(Rv*contour.dl).sum(dim="j")/cL   
    massflux = -(f*Ut*contour.dl).sum(dim="j")*H/cL

    terms = xr.Dataset(
        data_vars=dict(
            surfstress = (["time"], surfstress),
            nonlin = nonlin,
            bottomstress = bottomstress,
            #massflux = massflux
        ),
        coords=dict(
            time = ds.time
        ),
    ).squeeze()
    
    # update forcing to forcing prescribed from file if file is defined
    if "forcing_file" in params:
        tauds = xr.open_dataset(params["forcing_file"]).squeeze()
        taui, tauj = get_contour_following_tau(contour, tauds)
        terms["surfstress"][1:] = (tauj*contour.dl).sum(dim="j").values/(cL*params["rho"])

    return terms