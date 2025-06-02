import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm as cmc

from utils import integrated_contour_momentum_terms, depth_following_contour, \
    axes_styling, calculate_bathymetry, load_parameters, load_dataset, get_contour_following_velocities
    
    
params = load_parameters()
ds = load_dataset(params["filepath"], params["name"])
config = params["name"]

sec2days = 1/(60*60*24)

### Plot timeseries of momentum terms and circulation for one depth value
# This is mainly a sanity check
DB = params["DB"]
DS = params["DS"]

H = DB + 0.5*DS
terms = integrated_contour_momentum_terms(params, ds, H)

contour = depth_following_contour(params, H)
Ut, Vt =  get_contour_following_velocities(contour, ds)
cL = contour.dl.sum(dim=("j")).values

#fig, [ax, ax2] = plt.subplots(figsize=(10,10), nrows=2, sharex=True)
fig = plt.figure(layout="constrained", figsize=(10,10))
axd = fig.subplot_mosaic(
    [
        ["terms"],
        ["circ"],
    ],
    height_ratios=[2, 1],
    sharex=True
)
ax = axd["terms"]
ax2 = axd["circ"]

ax.set_title(f"Momentum terms and circulation integrated along depth = \
    {H:.0f} m")
axes_styling(ax)
axes_styling(ax2)
ax.axhline(0, color="gray", ls="--")
ax2.axhline(0, color="gray", ls="--")

terms["sum"] = terms["surfstress"] + terms["nonlin"] + terms["bottomstress"]# + terms["massflux"] 
tday = terms.time / np.timedelta64(1,"D")

colors = cmc.batlow(np.linspace(0, 1, 6))
for term, color in zip(terms, colors):
    ax.plot(tday, -terms[term], label=term, color=color)
    
circ_est = -(terms["sum"]*params["outputtime"]).cumsum("time")
circ_diag = -(Vt*contour.dl).sum(dim=("j"))*H/cL 

ax2.plot(tday, circ_est, 
         color = "darkorange",
         label = "estimated circulation\nfrom integrated momentum terms"
         )
ax2.plot(tday, circ_diag,
         color="cornflowerblue",
         label = "actual circulation"
         )

ax.set_ylabel("m2 s-2")
ax2.set_ylabel("m2 s-1")
ax2.set_xlabel("simulation day")

ax.legend(ncols=2)
ax2.legend()

fig.savefig(f"slope/figures/terms/Hcontour/Hcontour_terms_ts_{config}.png")



### Plot Hövmoller diagrams of momentum terms over slope
xstart = 15
xstop = 75
idxs = np.arange(xstart, xstop)

termlist = []
circlist = []
Hs = []
idy = int(params["lam"]/params["dy"]*0.5)
for idx in idxs:
    H = ds.h.isel(time=1,xC=idx, yC=idy).values
    terms = integrated_contour_momentum_terms(params, ds, H).squeeze()
    
    contour = depth_following_contour(params, H)
    Ut, Vt =  get_contour_following_velocities(contour, ds)
    cL = contour.dl.sum(dim=("j")).values
    numerical = -(Vt*contour.dl).sum(dim=("j"))*H/cL 
    
    termlist.append(terms)
    circlist.append(numerical)
    Hs.append(H)
    
results = xr.concat(termlist, dim="H")
results["sum"] = results["surfstress"] + results["nonlin"] + results["bottomstress"]# + results["massflux"] 

circ = xr.concat(circlist, dim="H")

fig = plt.figure(layout="constrained", figsize=(8,8))
axd = fig.subplot_mosaic(
    [
        ["0", "1", "cb"],
        ["2", "C", "cb"],
    ],
    width_ratios=[12, 12, 1],
)

Td = params["T"]*sec2days   
tmaxd = params["tmax"]*sec2days 
vmax = np.max([np.abs(results[term]).quantile(0.98) for term in results])
idh = np.arange(len(idxs))

terms = ["surfstress", "nonlin", "bottomstress"]
axes = [axd[f"{i}"] for i in range(3)]
for term, ax in zip(terms, axes):
    cm = ax.pcolormesh(idh, tday, -results[term].T, vmin=-vmax, vmax=vmax, cmap=cmc.vik)
    ax.set_title(term)
    ax.set_ylim(tmaxd-Td, tmaxd)
    
axd["C"].set_title("integration paths")
h = ds.h.isel(time=1).values
axd["C"].contour(h[:,xstart:xstop], levels=Hs, colors="gray", linewidths=1)
axd["C"].set_xticks([])
axd["C"].set_yticks([])
    
fig.colorbar(cm, cax=axd["cb"], label="m2 s-2")
fig.supxlabel("Contour index")
fig.supylabel("Time [days]")
    
axd["0"].set_xticks([])
axd["1"].set_xticks([])
axd["1"].set_yticks([])
#fig.tight_layout()

fig.savefig(f"slope/figures/terms/Hcontour/Hcontour_terms_cm_{config}.png")



### Integrated momentum terms and Hövmoller of circulation 
fig = plt.figure(layout="constrained", figsize=(8,8))
axd = fig.subplot_mosaic(
    [
        ["tmean", "legend"],
        ["circ", "xmean"],
    ],
    #empty_sentinel="BLANK",
    # set the height ratios between the rowsslice(int(tmax-T), int(tmax))
    height_ratios=[1, 3],
    # set the width ratios between the columns
    width_ratios=[3, 1]
)

cmap = cmc.batlow
n = 5
colors = [cmap(1 - i / (n - 1)) for i in range(n)]

axd["tmean"].sharex(axd["circ"])
axd["xmean"].sharey(axd["circ"])

axd["tmean"].spines['right'].set_color('none')  # Remove the right spine
axd["tmean"].spines['top'].set_color('none')  # Remove the top spine
axd["tmean"].spines['bottom'].set_position('zero')  # Set the bottom spine position
axd["tmean"].spines['left'].set_color('lightgray')
axd["tmean"].spines['bottom'].set_color('lightgray')

axd["xmean"].spines['right'].set_color('none')  # Remove the right spine
axd["xmean"].spines['top'].set_color('none')  # Remove the top spine
axd["xmean"].spines['left'].set_position('zero')  # Set the bottom spine position
axd["xmean"].spines['left'].set_color('lightgray')
axd["xmean"].spines['bottom'].set_color('lightgray')

Tn = int(params["T"]/params["outputtime"])

circT = circ.isel(time=slice(-Tn, None))
resultsT = results.isel(time=slice(-Tn, None))
vmax = np.max(np.abs(circT))
axd["circ"].pcolormesh(idh, tday[-Tn:], circT.T, 
                  vmin=-vmax, vmax=vmax, 
                  cmap=cmc.vik)



terms = ["sum", "surfstress", "nonlin", "bottomstress"]#, "massflux"]
for term, color in zip(terms, colors):
    
    result = -resultsT[term]
    tmean = result.mean("time")
    xmean = result.mean("H")
    
    
    axd["tmean"].plot(idh, tmean, label=term, color=color)
    axd["xmean"].plot(xmean[-Tn:], tday[-Tn:], label=term, color=color)
    axd["legend"].plot([None, None], [None, None], label=term,color=color)
    
    
axd["circ"].set_xlabel("Contour index")
axd["circ"].set_ylabel("Time [days]")
axd["tmean"].set_ylabel("m2 s-2")
axd["xmean"].set_xlabel("m2 s-2")

if config in ["slope-052", "slope-053"]:
    axd["tmean"].set_ylim(-5e-5,5e-5)
  
    
axd["legend"].axis("off")
axd["legend"].legend(loc='upper center', 
                    frameon=False,
                )

fig.savefig(f"slope/figures/terms/Hcontour/Hcontour_terms_integrated_{config}.png")