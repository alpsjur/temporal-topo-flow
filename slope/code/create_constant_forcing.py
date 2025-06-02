import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from utils import default_params, load_parameters


params = load_parameters()

config = params["name"] 
d = params["d"]

days2sec = 60*60*24

# Define dimensions and time steps
Lx = params["Lx"]
Ly = params["Ly"]

dx = params["dx"]
dy = params["dy"]

tmax = params["tmax"]
dt = params["outputtime"]

x = np.arange(0, Lx, dx)
y = np.arange(0, Ly, dy)
time = np.arange(0, tmax, dt)


tau_y = np.ones((len(time),len(x),len(y)))
tau_y = -d*(np.tanh(time/(10*days2sec))[:,None,None])*tau_y

tau_x = np.zeros_like(tau_y)


# Create an xarray dataset
ds = xr.Dataset(
    {"forcing_x": (["time", "x", "y"], tau_x),
     "forcing_y": (["time", "x", "y"], tau_y)},
    coords={"time": time, "x": x, "y": y}
)

# Save to NetCDF
ds.to_netcdf("slope/input/"+config+"_forcing.nc")