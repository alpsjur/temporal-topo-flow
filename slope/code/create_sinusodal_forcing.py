import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from utils import default_params, load_parameters


def h_i(x, y, p):
    """Computes the bathymetry height at (x, y) based on given parameters."""
    steepness = (1 / np.cosh(np.pi * (x - p["XC"]) / p["W"]))**2
    corr = p["a"] * np.sin(2 * np.pi * y / p["lam"]) * steepness
    h = p["DB"] + 0.5 * p["DS"] * (1 + np.tanh(np.pi * (x - p["XC"] - corr) / p["W"]))
    return h


def compute_bathymetry(p):
    """Generate the bathymetry grid."""
    x = np.arange(0, p["Lx"], p["dx"])
    y = np.arange(0, p["Ly"], p["dy"])
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Compute bathymetry at each grid point
    h = np.vectorize(lambda x, y: h_i(x, y, p))(X, Y)

    return X, Y, h


def compute_bathymetry_gradient(X, Y, h, dx, dy):
    """Compute the gradient of the bathymetry."""
    dh_dx, dh_dy = np.gradient(h, dx, dy, edge_order=2)
    return dh_dx, dh_dy


# Compute bathymetry and its gradient
X, Y, h = compute_bathymetry(default_params)
dh_dx, dh_dy = compute_bathymetry_gradient(X, Y, h, default_params["dx"], default_params["dy"])



params = load_parameters()

config = params["name"] 
T = params["T"]
d = params["d"]

# Generate time array
time = np.arange(0, params["tmax"], params["outputtime"])




def compute_time_factor(time, T, d):
    """Compute sinusoidal variation in time."""
    return d * np.sin(2 * np.pi * time / T)

def compute_time_varying_tangential_velocity(dh_dx, dh_dy, time, T, d):
    """Compute velocity field tangential to the bathymetry with sinusoidal time variation."""
    #magnitude = np.sqrt(dh_dx**2 + dh_dy**2)  # Compute magnitude of the normal vector
    #magnitude[magnitude == 0] = 1e-12  # Avoid division by zero

    # Compute unit tangential velocity (constant in space)
    u_tan = dh_dy / dh_dx#magnitude
    v_tan = dh_dx / dh_dx#magnitude

    # Apply sinusoidal variation in time
    time_factor = compute_time_factor(time, T, d)

    return time_factor[:, None, None] * u_tan[None, :, :], time_factor[:, None, None] * v_tan[None, :, :]



# Compute time-varying tangential velocity
tau_x, tau_y = compute_time_varying_tangential_velocity(dh_dx, dh_dy, time, T, d)

# Create an xarray dataset
ds = xr.Dataset(
    {"forcing_x": (["time", "x", "y"], tau_x),
     "forcing_y": (["time", "x", "y"], tau_y)},
    coords={"time": time, "x": X[:, 0], "y": Y[0, :]}
)

# Save to NetCDF
ds.to_netcdf("slope/input/"+config+"_forcing.nc")