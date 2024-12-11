import numpy as np
import sys
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
from cmcrameri import cm

from utils import default_params, load_config, get_h, analytical_circ

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
T = params["T"]
d = params["d"]
tmax = params["tmax"]
outputtime = params["outputtime"]
dx = params["dx"]
dy = params["dy"]

# Load data
ds = xr.open_dataset(filepath + name + ".nc").squeeze()

x = ds.xC.values
y = ds.yC.values
u = ds.u.interp(xF=x)
v = ds.v.interp(yF=y)
t = ds.time / np.timedelta64(1, 's')
t_days = ds.time / np.timedelta64(1, 'D')

x /= 1e3
y /= 1e3

nfc = round(np.floor(tmax/T)) # number of forcing cycles
nts_fc = T/outputtime  # number of time steps in a forcing cycle


timesteps = [round(nts_fc*((nfc-1) + i/4)) for i in range(4)]

fig = plt.figure(layout="constrained", figsize=(12,12))
axd = fig.subplot_mosaic(
    [
        ["forcing"]*2,
        ["sl0", "sl1"],
        ["sl0", "sl1"],
        ["sl2", "sl3"],
        ["sl2", "sl3"],
    ],
)

forcing = d*np.sin(2*np.pi*t/T)
analytical = -analytical_circ(params, t, 90e3, 500)
analytical *= d/np.max(analytical[timesteps[0]:timesteps[-1]])

sec2day = 1/86400
tstart = (timesteps[0]-5)*outputtime*sec2day
tstop = (timesteps[-1]+5)*outputtime*sec2day
axd["forcing"].set_xlim(tstart,tstop)

axd["forcing"].plot(t_days[timesteps[0]-5:], forcing[timesteps[0]-5:],
                    color="gray",
                    label="forcing"
                    )

axd["forcing"].plot(t_days[timesteps[0]-5:], analytical[timesteps[0]-5:],
                    color="darkorange",
                    label="scaled linear response\nat H=500m",
                    )
axd["forcing"].legend()

h = get_h(params)

for i, ts in enumerate(timesteps):
    axd["forcing"].axvline(ts*outputtime*sec2day, 
                           color="cornflowerblue",
                           )

    ax = axd[f"sl{i}"]

    # dd time step text in the upper right corner of the subplot
    ax.text(0.95, 0.95, f"Time step {i}", 
            transform=ax.transAxes, 
            ha="right", va="top", 
            fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="lightgray"))

    ax.pcolormesh(x, y, h, cmap="Grays", alpha=0.7)

    U = u.isel(time=ts).values
    V = v.isel(time=ts).values
    ax.streamplot(x, y, U, V, color="cornflowerblue")

    ax.set_aspect("equal")

fig.tight_layout()
fig.savefig("slope/figures/streamlines/" + name + "_evolution.png")

plt.show()
