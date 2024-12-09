# Import necessary packages for visualization, data handling, and numerical operations
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr  
import seaborn as sns
from cmcrameri import cm
from scipy.ndimage import uniform_filter1d

from coordtransform.utils import params, xt_from_y, dl_fromxt_yt, dt, H

sns.set_style("whitegrid")

# Define parameters and file paths
name = "brink_2010-329"

xvals = (30, 35, 40, 45, 50)
Tinc = 12 * 8
full = False

filepath = "output/brink/"
figurepath = "figures/brink/"
cmap = cm.batlow 

# Forcing parameters
rho = 1e3
d = 0.1
T = 4 * 24 * 60 * 60  # forcing period in seconds
R = 0.5e-4
L = 90 * 1e3  # 90 kilometers in meters

outputtime = 3 * 60 * 60  # 3 hours in seconds
delta_t = 2  # 2 seconds

config = name
omega = 2 * np.pi / T
window = int(outputtime / delta_t)

# Load data 
ds = xr.open_dataset(filepath + name + ".nc").squeeze()

rename_map = {
    "xu": "xF",
    "xv": "xC",
    "yu": "yC",
    "yv": "yF"
}

# Check if the coordinates exist and only rename the ones that are present
existing_coords = {key: value for key, value in rename_map.items() if key in ds.coords}

# Rename coordinates
ds = ds.rename(existing_coords)

u = ds.u  
v = ds.v  
vortu = ds.omegau
vortv = ds.omegav
t = ds.time / np.timedelta64(1, 's')
t_days =  ds.time / np.timedelta64(1, 'D')
t_hr = np.arange(0, len(t)*outputtime, 2)
#t_days = t / (24 * 60 * 60)  # Convert time to days

# Construct depth-following grid
Lx = params["grid"]["Lx"]
Ly = params["grid"]["Ly"]
dx = params["grid"]["dx"]
dy = params["grid"]["dy"]

x1 = params["bathymetry"]["x1"]
x2 = params["bathymetry"]["x2"]
h0 = params["bathymetry"]["h0"]
h1 = params["bathymetry"]["h1"]
h2 = params["bathymetry"]["h2"]

g = params["derived"]["g"]
k = params["derived"]["k"]
Nx = params["derived"]["Nx"]
Ny = params["derived"]["Ny"]
Nl = params["derived"]["Nl"]
A = params["derived"]["A"]
B = params["derived"]["B"]

hA = 0    

x = np.arange(0, Lx, dx)+dx/2
y = np.arange(0, Ly, dy)+dy/2

X, Y = np.meshgrid(x, y)
h = H(X, Y, x1, x2, hA, h0, h1, h2, A, B, g, k)

# construct depths from h
depths = h.mean(axis=0)

## Create contour folowing grid
Yt = Y.copy()
Xt = X.copy()

dXt = np.zeros_like(Xt)
dYt = np.ones_like(Yt)
dLt = np.ones_like(Xt)*dy

for i, Ht in enumerate(depths):
    xt = xt_from_y(y, Ht, x1, x2, hA, h0, h1, h2, g, k)
    Xt[:,i] = xt
    
    dxt, dyt = dt(y, Ht, x1, x2, hA,  h0, h1, h2, g, k)
    dXt[:,i] = dxt
    dYt[:,i] = dyt
    
    dlt = dl_fromxt_yt(xt, y, Ny, Nl, upstream=True)
    dLt[:,i] = dlt

contour_grid = xr.Dataset(
    data_vars=dict(
        dtdx=(["j", "i"], dXt),
        dtdy=(["j", "i"], dYt),
        dl=(["j", "i"], dLt),
    ),
    coords=dict(
        x=(["j", "i"], Xt),
        y=(["j", "i"], Yt),
        depth=(["i"], depths),
        i = np.arange(Nx, dtype=np.int32),
        j = np.arange(Ny, dtype=np.int32)
    )
)

# contour folowing velocities
ut = u.interp(xF=contour_grid.x, yC=contour_grid.y)
vt = v.interp(xC=contour_grid.x, yF=contour_grid.y)
Ut = ut*contour_grid.dtdy - vt*contour_grid.dtdx
Vt = ut*contour_grid.dtdx + vt*contour_grid.dtdy

# cross-contour vorticity flux
vortut = vortu.interp(xF=contour_grid.x, yF=contour_grid.y)
vortvt = vortv.interp(xF=contour_grid.x, yF=contour_grid.y)
vortUt = vortut*contour_grid.dtdy - vortvt*contour_grid.dtdx
vortVt = vortut*contour_grid.dtdx + vortvt*contour_grid.dtdy
vortflux = vortUt * contour_grid.dl

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
    H = depths[xval]
    Hs.append(H)
    cL = contour_grid.dl.sel(i=xval).sum(dim=("j")).values
    
    # TODO får drifting-problem... 
    # Enten være nøyere med definering av dl, eller regne ut virvlingsflux på en mer robust måte
    nonlin = vortflux.sel(i=xval).sum(dim="j").values/cL
    
    windforce_hr = -d*L*np.sin(omega * t_hr)/(rho*H*cL)
    windforce = windforce_hr.reshape(-1, window).mean(axis=1)

    
    filtered_forcing = np.exp(-R/H*(t[::-1]))*(windforce+nonlin)
    analytical = np.cumsum(filtered_forcing*outputtime)
    analytical *= 1e2

    numerical = -(Vt*contour_grid.dl).sel(i=xval).sum(dim=("j"))/cL * 1e2 

    if full:
        # Plot analytical and numerical data over time
        axts.plot(t_days, numerical, color=color, label=str(round(H)))
        axts.plot(t_days, analytical, color=color, linestyle="--")
        #axts.plot(t_days, nonlin*1e7, color=color, linestyle="--")
    
    else:
        # Plot analytical and numerical data over time
        axts.plot(t_days[-Tinc:], numerical[-Tinc:], color=color, label=str(round(H)))
        axts.plot(t_days[-Tinc:], analytical[-Tinc:], color=color, linestyle="--")
        #axts.plot(t_days[-Tinc:], nonlin[-Tinc:]*1e7, color=color, linestyle="--")

    # Scatter plot of analytical vs. numerical data
    axscatter.scatter(analytical, numerical, label=str(round(H)), s=16, alpha=0.7, color=color)

# Add legends
axscatter.legend(loc="lower right")
axts.legend(loc="lower right")

figscatter.tight_layout()
figts.tight_layout()

# Save figures
#figscatter.savefig(figurepath + "nonlinear_scatter/" + name + "_scatter_Hcontour.png")
figts.savefig(figurepath + "nonlinear_timeseries/" + name + "_nonlinear_ts_Hcontour.png")


# plot also contours
fig, ax = plt.subplots(figsize=(6,6))
ax.pcolormesh(h[:,:90], cmap="Grays", alpha=0.7)
ax.contour(h[:,:90], levels=Hs, colors=colors)
fig.savefig(figurepath + "nonlinear_timeseries/" + "Hcontours.png")


# Show plots (optional, for interactive environments)
plt.show()
