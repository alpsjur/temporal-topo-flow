import numpy as np
import sys
import matplotlib.pyplot as plt
import xarray as xr  
import seaborn as sns
from cmcrameri import cm
from scipy.ndimage import uniform_filter1d

from utils import default_params, load_config, depth_following_grid
sns.set_style("whitegrid")

# Load parameters
if len(sys.argv) == 2:
    config_path = sys.argv[1]
    print(f"Loading configuration from {config_path}")
    params = load_config(config_path, default_params)
else:
    params = default_params
    print("No configuration file provided. Using default parameters.")
    
    
name = params["name"]
filepath = params["filepath"]
outputtime = params["outputtime"]
T = params["T"]
outputtime = params["outputtime"]
dt = params["dt"]
rho = params["rho"]
R = params["R"]
d = params["d"]
Ly = params["Ly"]

omega = 2 * np.pi / T

xvals = (30, 35, 40, 45, 50)
cmap = cm.batlow 


# depth-following grid
grid = depth_following_grid(params)

# Load data 
ds = xr.open_dataset(filepath + name + ".nc")

u = ds.u  # u-component of velocity field
v = ds.v  # v-component of velocity field
t = ds.time / np.timedelta64(1, 's')
t_days =  ds.time / np.timedelta64(1, 'D')
t_hr = np.arange(0, len(t)*outputtime, 2)


Tinc = round(5 * T / outputtime)
if Tinc >= len(t):
    Tinc = len(t)
    
window = round(outputtime / dt)    
    

# contour folowing velocities
ut = u.interp(xF=grid.x, yC=grid.y)
vt = v.interp(xC=grid.x, yF=grid.y)
Ut = ut*grid.dtdy - vt*grid.dtdx
Vt = ut*grid.dtdx + vt*grid.dtdy

# Set up figures and axes
figscatter, axscatter = plt.subplots(figsize=(6, 6))
axscatter.set_xlabel("analytical circ [cm s-1]")
axscatter.set_ylabel("numerical circ [cm s-1]")
axscatter.plot([-10, 10], [-10, 10], color="gray")

figts, axts = plt.subplots(figsize=(12, 4))
axts.set_xlabel("time [days]")
axts.set_ylabel("circ [cm s-1]")

# Generate colors for each deoth
n = len(xvals)
colors = [cmap(1 - i / (n - 1)) for i in range(n)]

Hs = []
for i, xval in enumerate(xvals):
    color = colors[i]
    H = grid.depth.sel(i=xval).values
    Hs.append(H)
    cL = grid.dl.sel(i=xval).sum(dim=("j")).values

    analytical_hr = -d * Ly / (rho * cL ) * (R * np.sin(omega * t_hr) - H * omega * np.cos(omega * t_hr) +
                               np.exp(-R * t_hr / H) * H * omega) / (H**2 * omega**2 + R**2)
    analytical = uniform_filter1d(analytical_hr, size=window)[::window] * 1e2  # Convert to cm/s

    numerical = -Vt.sel(i=xval).mean(dim=("j")) * 1e2  # Convert to cm/s

    axts.plot(t_days[-Tinc:], numerical[-Tinc:], color=color, label=str(int(H)))
    axts.plot(t_days[-Tinc:], analytical[-Tinc:], color=color, linestyle="--")

    # Scatter plot of analytical vs. numerical data
    axscatter.scatter(analytical, numerical, label=str(int(H)), s=16, alpha=0.7, color=color)

# Add legends
axscatter.legend(loc="lower right")
axts.legend(loc="lower right")

figscatter.tight_layout()
figts.tight_layout()

# Save figures
#figscatter.savefig(figurepath  + name + "_scatter_Hcontour.png")
#figts.savefig(figurepath  + name + "_timeseries_Hcontour.png")


# plot also contours
# fig, ax = plt.subplots(figsize=(6,6))
# ax.pcolormesh(h[:,:90], cmap="Grays", alpha=0.7)
# ax.contour(h[:,:90], levels=Hs, colors=colors)
#fig.savefig(figurepath + "Hcontours.png")


# Show plots (optional, for interactive environments)
plt.show()