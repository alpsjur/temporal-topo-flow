import numpy as np
import sys
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
from cmcrameri import cm
from utils import depth_following_grid, get_h, analytical_circ, \
    load_parameters, load_dataset, truncate_time_series, calculate_bathymetry

# Configure seaborn style
sns.set_style("whitegrid")


def setup_plots():
    """Initialize figures and axes for various plots."""
    figscatter, axscatter = plt.subplots(figsize=(12, 6), ncols=2)
    axscatter[0].set_xlabel("linear analytical circ [cm s-1]")
    axscatter[1].set_xlabel("nonlinear analytical circ [cm s-1]")
    axscatter[0].set_ylabel("numerical circ [cm s-1]")
    axscatter[0].set_aspect("equal")
    axscatter[1].set_aspect("equal")


    figts, axts = plt.subplots(figsize=(12, 4), sharex=True)
    #axts.set_xlabel("time [days]")
    axts.set_ylabel("circ [cm s-1]")
    axts.set_xlabel("time [days]")

    return figscatter, axscatter, figts, axts

def plot_results(params, ds, xvals, t, t_days, cmap):
    """Generate plots for time series, scatter plots, and contours."""
    n = len(xvals)
    colors = [cmap(1 - i / (n - 1)) for i in range(n)]

    X, Y = np.meshgrid(ds.xC, ds.yF)
    H = calculate_bathymetry(X,Y, params)
    h = xr.DataArray(data=H, dims=(["yF", "xC"]))
    
    vh = ds.v*h
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pcolormesh(h, cmap="Grays", alpha=0.7)
    ax.set_aspect("equal")

    figscatter, axscatter, figts, axts = setup_plots()
    
    T = params["T"]
    outputtime = params["outputtime"]
    R = params["R"]
    Tinc = min(round(5 * T / outputtime), len(t))
    
    cmax, cmin = 0, 0
    for i, xval in enumerate(xvals):
        color = colors[i]
       
    return figscatter, figts

def main():
    try:
        
        params = load_parameters()
        ds = load_dataset(params["filepath"], params["name"])
        ds = truncate_time_series(ds)

        t = ds.time / np.timedelta64(1, 's')
        t_days = ds.time / np.timedelta64(1, 'D')

        xvals = (10, 30, 40, 50, 80)
        cmap = cm.batlow

        figscatter, figts = plot_results(params, ds, xvals, t, t_days, cmap)


        # Save figures
        figscatter.savefig(f"slope/figures/scatters/{params['name']}_scatter_xcontour.png")
        figts.savefig(f"slope/figures/timeseries/{params['name']}_timeseries_xcontour.png")


        # Optional: Show plots interactively
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
