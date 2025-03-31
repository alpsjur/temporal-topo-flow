import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from utils import load_parameters
import cmocean


def h_i(x, y, p):
    """Computes the bathymetry height at (x, y) based on given parameters."""
    steepness = (1 / np.cosh(np.pi * (x - p["XC"]) / p["W"]))**2

    corr = corregation(x,y,p) * steepness
    h = p["DB"] + 0.5 * p["DS"] * (1 + np.tanh(np.pi * (x - p["XC"] - corr) / p["W"]))
    return h

def corregation(x, y, p):
    alist = p["alist"]
    lamlist = p["lamlist"]
    phaselist  = p["phaselist"]
    aglob = p["a"]

    corr = 0
    for a, lam, phase in zip(alist, lamlist,phaselist):
        corr += a*np.sin(2 * np.pi * (y-phase) / lam)
    corr *= aglob/np.max(corr)
    return corr


params = load_parameters()
config = params["name"] 

Lx = params["Lx"]
Ly = params["Ly"]
dx = params["dx"]
dy = params["dy"]

Nx = Lx // dx
Ny = Ly // dy

x = np.arange(0, Lx, dx)
y = np.arange(0, Ly, dy)

X, Y = np.meshgrid(x, y)
h = h_i(X, Y, params)


# Create an xarray dataset
ds = xr.Dataset(
    {"h": (["x", "y"], h)},
    coords={"x": x, "y": y}
)

fig, ax = plt.subplots(figsize=(8, 8))
cm = ax.pcolormesh(x/1e3, y/1e3, h, cmap = cmocean.cm.deep)
fig.colorbar(cm, ax=ax, label='Depth [m]')
ax.set_aspect("equal")
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")

fig.tight_layout()
fig.savefig("slope/figures/bathymetry/" + config + "_bathymetry.png")


# Save to NetCDF
ds.to_netcdf("slope/input/"+config+"_bathymetry.nc")

