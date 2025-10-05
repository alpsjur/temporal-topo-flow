import xarray as xr
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.grid import prepare_dsH, interp_ds
from utils.config import load_config
from utils.io import read_raw_output, save_processed_ds


def diagnose_circulation_onH(ds, variable):
    """Compute circulation of variable along depth-following contours (H-contours)."""
    return (ds[variable] * ds.dl).sum("i") / ds.dl.sum("i")

def diagnose_circulation_onY(ds, variable, xdim="xC"):
    """Compute mean circulation of variable along the y-direction."""
    return ds[variable].mean(xdim)

def calculate_analytical_estimates_xr(
    ds: xr.Dataset,
    forcing_vars: list[str],
    params: dict,
) -> xr.DataArray:
    """
    y(t_j) = ∫_0^{t_j} exp[-(R/H(j)) * (t_j - τ)] * F(τ, j) dτ
    One-pass recurrence (left Riemann per step) with constant Δt = params["outputtime"].
    Recurrence: y_i = decay * y_{i-1} + alpha * F_{i-1},
    where decay = exp(-k*Δt), alpha = (1 - decay)/k, k = R/H(j).
    """
    R = float(params["R"])
    dt = float(params["outputtime"])  # seconds (ensure your units make R/H * dt dimensionless)

    H = ds["depth"]  # dims include "j"
    # Sum forcing fields (must have dims include "time" and "j")
    F = sum(ds[v] for v in forcing_vars)

    t = ds["time"]
    nT = t.sizes["time"]
    if nT == 0:
        raise ValueError("Empty time axis.")
    if nT == 1:
        return xr.zeros_like(F)

    # k(j) = R/H(j), broadcast over all non-time dims of F
    k = (R / H).broadcast_like(F.isel(time=0))

    # Precompute coefficients (constant in time)
    decay = np.exp(-k * dt)
    # Safe alpha for k≈0: limit -> dt
    alpha = xr.where(np.abs(k) > 0, (1.0 - decay) / k, dt)

    # build y 
    y0 = xr.zeros_like(F.isel(time=0, drop=True))
    ys = [y0]
    for i in range(1, nT):
        Fi_1 = F.isel(time=i - 1, drop=True)  # drop time here too
        y_next = decay * ys[-1] + alpha * Fi_1
        ys.append(y_next)
    y = xr.concat(ys, dim="time").assign_coords(time=t) / H

    return y

# read simulation output
params = load_config()
ds = read_raw_output(params)


### H contour postprocessing  ###
# select target depths for contours
H_targets = ds.bath.mean("xC").values

# interpolate dataset onto the selected depth-contours
dsH = prepare_dsH(ds, params, H_targets)

# diagnose circulation
dsH["circulation"] = diagnose_circulation_onH(dsH, variable="ui")

# calculate momentum terms
dsH["BS"] = -dsH.circulation*params["R"]
dsH["TFS"] = dsH.circulation*0
dsH["MFC"] = diagnose_circulation_onH(dsH, variable="zetaflux") * dsH.depth
dsH["SS"] = diagnose_circulation_onH(dsH, variable="forcing_i") 

# estimate linear circulaiton
dsH["linear_estimate"] = calculate_analytical_estimates_xr(dsH, ["SS"], params)
dsH["nonlinear_estimate"] = calculate_analytical_estimates_xr(dsH, ["SS", "MFC"], params)

# save processed dataset
save_processed_ds(dsH, params, onH=True)



### y contour postprocessing 

# interpolate relevant variables to center position
dsY = interp_ds(ds, params, ["u", "v", "forcing_x", "detadx", "duvhdy"])

# calculate momentum terms
dsY["circulation"] = diagnose_circulation_onY(dsY, variable="u")
dsY["uH"] = (dsY.u*dsY.bath).mean("xC")
dsY["BS"] = -(dsY.u).mean("xC")*params["R"]
dsY["TFS"] = (-params["gravitational_acceleration"]*dsY.detadx*dsY.bath).mean("xC")
dsY["MFC"] = (-dsY.duvhdy).mean("xC")
dsY["SS"] = (dsY.forcing_x).mean("xC")

save_processed_ds(dsY, params, onH=False)




