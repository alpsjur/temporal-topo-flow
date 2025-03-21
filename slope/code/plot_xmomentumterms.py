import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm as cmc

from utils import integrated_zonal_momentum_terms, depth_following_contour, \
    axes_styling, calculate_bathymetry, load_parameters, load_dataset, get_contour_following_velocities
    
    
params = load_parameters()
ds = load_dataset(params["filepath"], params["name"])
config = params["name"]

sec2days = 1/(60*60*24)

### Plot timeseries of momentum terms and circulation for one depth value
# This is mainly a sanity check
DB = params["DB"]
DS = params["DS"]

xidx = 45
terms = integrated_zonal_momentum_terms(params, ds, xidx)


fig, [ax, ax2] = plt.subplots(figsize=(10,10), nrows=2, sharex=True)
ax.set_title(f"Momentum terms and circulation integrated along depth = \
    {xidx*params["dx"]/1e3:.0f} km")
axes_styling(ax)
axes_styling(ax2)
ax.axhline(0, color="gray", ls="--")
ax2.axhline(0, color="gray", ls="--")

terms["sum"] = terms["surfstress"] + terms["nonlin"] + terms["bottomstress"] + terms["formstress"] 
tday = terms.time / np.timedelta64(1,"D")

colors = cmc.batlow(np.linspace(0, 1, 5))
for term, color in zip(terms, colors):
    ax.plot(tday, -terms[term], label=term, color=color)
    
circ_est = -(terms["sum"]*params["outputtime"]).cumsum("time")
circ_diag = -(ds.v.isel(xC=xidx)*calculate_bathymetry(ds.xC.isel(xC=xidx), ds.yF, params)).mean("yF")

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

fig.savefig(f"slope/figures/terms/xcontour/xcontour_terms_ts_{config}.png")



### Plot Hövmoller diagrams of momentum terms over slope
xstart = 15
xstop = 75
idxs = np.arange(xstart, xstop)

termlist = []
circlist = []
for xidx in idxs:
    terms = integrated_zonal_momentum_terms(params, ds, xidx).squeeze()
    numerical = -(ds.v.isel(xC=xidx)*calculate_bathymetry(ds.xC.isel(xC=xidx), ds.yF, params)).mean("yF")
 
    
    termlist.append(terms)
    circlist.append(numerical)
    
results = xr.concat(termlist, dim="x")
results["sum"] = results["surfstress"] + results["nonlin"] + results["bottomstress"] + results["formstress"]

circ = xr.concat(circlist, dim="x")

fig = plt.figure(layout="constrained", figsize=(8,8))
axd = fig.subplot_mosaic(
    [
        ["0", "1", "cb"],
        ["2", "3", "cb"],
    ],
    width_ratios=[12, 12, 1],
)

Td = params["T"]*sec2days   
tmaxd = params["tmax"]*sec2days 
vmax = np.max([np.abs(results[term]).quantile(0.98) for term in results])
idh = np.arange(len(idxs))

terms = ["surfstress", "nonlin", "bottomstress", "formstress"]
axes = [axd[f"{i}"] for i in range(4)]
for term, ax in zip(terms, axes):
    cm = ax.pcolormesh(idxs, tday, -results[term].T, vmin=-vmax, vmax=vmax, cmap=cmc.vik)
    ax.set_title(term)
    ax.set_ylim(tmaxd-Td, tmaxd)
    
fig.colorbar(cm, cax=axd["cb"], label="m2 s-2")
fig.supxlabel("x [km]")
fig.supylabel("Time [days]")
    
axd["0"].set_xticks([])
axd["1"].set_xticks([])
axd["1"].set_yticks([])
axd["3"].set_yticks([])
#fig.tight_layout()

fig.savefig(f"slope/figures/terms/xcontour/xcontour_terms_cm_{config}.png")



### Integrated momentum terms and Hövmoller of circulation 
fig = plt.figure(layout="constrained", figsize=(8,8))
axd = fig.subplot_mosaic(
    [
        ["tmean", "txmean"],
        ["circ", "xmean"],
    ],
    #empty_sentinel="BLANK",
    # set the height ratios between the rowsslice(int(tmax-T), int(tmax))
    height_ratios=[2, 5],
    # set the width ratios between the columns
    width_ratios=[5, 2]
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

axd["txmean"].spines['right'].set_color('none')  # Remove the right spine
axd["txmean"].spines['top'].set_color('none')  # Remove the top spine
axd["txmean"].spines['bottom'].set_position('zero')  # Set the bottom spine position
axd["txmean"].spines['left'].set_color('lightgray')
axd["txmean"].spines['bottom'].set_color('lightgray')

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



terms = ["sum", "surfstress", "nonlin", "bottomstress"]
i = 1
for term, color in zip(terms, colors):
    
    result = -resultsT[term]
    tmean = result.mean("time")
    xmean = result.mean("x")
    txmean = xmean.mean("time")
    
    
    axd["tmean"].plot(idh, tmean, label=term, color=color)
    axd["xmean"].plot(xmean[-Tn:], tday[-Tn:], label=term, color=color)
    axd["txmean"].scatter(i, txmean, label=term, color=color)
    i += 1
    
axd["txmean"].set_xticks([])    

axd["circ"].set_xlabel("x [km]")
axd["circ"].set_ylabel("Time [days]")
axd["tmean"].set_ylabel("m2 s-2")
axd["xmean"].set_xlabel("m2 s-2")
  
    
axd["circ"].legend(loc='upper center', 
                    frameon=False,
                    ncols=3,
                    bbox_to_anchor=(0, 1.5)
                )

fig.savefig(f"slope/figures/terms/xcontour/xcontour_terms_integrated_{config}.png")