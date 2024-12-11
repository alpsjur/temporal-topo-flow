import numpy as np
import sys
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
from cmcrameri import cm
from utils import default_params, load_config, depth_following_grid, get_h, analytical_circ

# Configure seaborn style
sns.set_style("whitegrid")

def load_parameters():
    """Load simulation parameters either from a configuration file or defaults."""
    if len(sys.argv) == 2:
        config_path = sys.argv[1]
        print(f"Loading configuration from {config_path}")
        return load_config(config_path, default_params)
    else:
        print("No configuration file provided. Using default parameters.")
        return default_params

def load_dataset(filepath, name):
    """Load the dataset from a NetCDF file."""
    return xr.open_dataset(filepath + name + ".nc").squeeze()

def truncate_time_series(ds):
    """Truncate time series if a blow-up in velocity is detected."""
    if np.abs(ds.v).max() > 1.5:
        tstop = np.nonzero(np.abs(ds.v).max(dim=("xC", "yF")).values > 1.5)[0][0]
        print("Blow-up in velocity, truncating time series.")
        return ds.isel(time=slice(None, tstop))
    return ds

def compute_contour_following_velocities(u, v, grid):
    """Calculate contour-following velocities."""
    ut = u.interp(xF=grid.x, yC=grid.y)
    vt = v.interp(xC=grid.x, yF=grid.y)
    Ut = ut * grid.dtdy - vt * grid.dtdx
    Vt = ut * grid.dtdx + vt * grid.dtdy
    return Ut, Vt

def compute_vorticity_flux(ds, grid):
    """Calculate vorticity flux contributions."""
    vortu = ds.omegau
    vortv = ds.omegav
    vortut = vortu.interp(xF=grid.x, yF=grid.y)
    vortvt = vortv.interp(xF=grid.x, yF=grid.y)
    vortUt = vortut * grid.dtdy - vortvt * grid.dtdx
    vortVt = vortut * grid.dtdx + vortvt * grid.dtdy
    vortflux = vortUt * grid.dl
    return vortflux

def compute_alignment(Ut, Vt):
    """Calculate alignment and misalignment metrics for along and cross-contour velocities."""
    magnitude = np.sqrt(Ut**2 + Vt**2)
    alignment = np.abs(Vt) / (magnitude + 1e-10) # Add small epsilon to avoid division by zero
    return alignment

def setup_plots():
    """Initialize figures and axes for various plots."""
    figscatter, axscatter = plt.subplots(figsize=(12, 6), ncols=2)
    axscatter[0].set_xlabel("linear analytical circ [cm s-1]")
    axscatter[1].set_xlabel("nonlinear analytical circ [cm s-1]")
    axscatter[0].set_ylabel("numerical circ [cm s-1]")
    axscatter[0].set_aspect("equal")
    axscatter[1].set_aspect("equal")

    figvort, axvort = plt.subplots(figsize=(12, 6), ncols=2)
    axvort[0].set_ylabel("vorticity contribution [cm s-1]")
    axvort[0].set_xlabel("numerical circ [cm s-1]")
    #axvort[0].set_aspect("equal")
    axvort[1].set_ylabel("vorticity flux x H/R [cm s-1]")
    axvort[1].set_xlabel("numerical circ [cm s-1]")
    #axvort[1].set_aspect("equal")

    figts, [axts, axalign] = plt.subplots(figsize=(12, 4), nrows=2, sharex=True)
    #axts.set_xlabel("time [days]")
    axts.set_ylabel("circ [cm s-1]")
    axalign.set_xlabel("time [days]")
    axalign.set_ylabel("mean alignment []")

    return figscatter, axscatter, figvort, axvort, figts, axts, axalign

def plot_results(params, ds, grid, vortflux, Ut, Vt, xvals, t, t_days, cmap):
    """Generate plots for time series, scatter plots, and contours."""
    n = len(xvals)
    colors = [cmap(1 - i / (n - 1)) for i in range(n)]

    h = xr.DataArray(data=get_h(params), dims=(["yF", "xC"]))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pcolormesh(h, cmap="Grays", alpha=0.7)
    ax.set_aspect("equal")

    figscatter, axscatter, figvort, axvort, figts, axts, axalign = setup_plots()
    
    T = params["T"]
    outputtime = params["outputtime"]
    R = params["R"]
    Tinc = min(round(5 * T / outputtime), len(t))
    
    
    
    cmax, cmin = 0, 0

    for i, xval in enumerate(xvals):
        color = colors[i]
        H = grid.depth.sel(i=xval).values
        cL = grid.dl.sel(i=xval).sum(dim=("j")).values

        nonlin = vortflux.sel(i=xval).sum(dim="j").values / cL
        alignments = compute_alignment(Ut.sel(i=xval), Vt.sel(i=xval))
        alignment = (alignments*grid.dl.sel(i=xval)).sum(dim=("j"))/cL
        alignment = alignment.where(alignment != 0, other=np.nan)

        analytical_linear = analytical_circ(params, t, cL, H) * 1e2
        analytical_nonlinear = analytical_circ(params, t, cL, H, nonlin) * 1e2
        numerical = -(Vt.sel(i=xval)*grid.dl.sel(i=xval)).sum(dim=("j"))/cL * 1e2  # Convert to cm/s

        axts.plot(t_days[-Tinc:], numerical[-Tinc:], color=color, label=f"{int(H)}m depth")
        axts.plot(t_days[-Tinc:], analytical_linear[-Tinc:], color=color, linestyle="--")
        axts.plot(t_days[-Tinc:], analytical_nonlinear[-Tinc:], color=color, linestyle="-." )

        axscatter[0].scatter(analytical_linear, numerical, s=16, alpha=0.7, color=color, label=f"{int(H)}m depth")
        axscatter[1].scatter(analytical_nonlinear, numerical, s=16, alpha=0.7, color=color)

        vorticity_contribution = numerical - analytical_linear
        axvort[0].scatter(numerical, vorticity_contribution, s=16, alpha=0.7, color=color, label=f"{int(H)}m depth")
        axvort[1].scatter(numerical, nonlin * H / R * 1e2, s=16, alpha=0.7, color=color)

        axalign.plot(t_days[-Tinc:], alignment[-Tinc:], color=color, label=f"{int(H)}m depth")

        ax.plot(grid.sel(i=xval).x * 1e-3, grid.sel(i=xval).y * 1e-3, color=color)

        cmax, cmin = max(cmax, np.max(numerical)), min(cmin, np.min(numerical))

    axscatter[0].plot([cmin, cmax], [cmin, cmax], color="gray")
    axscatter[1].plot([cmin, cmax], [cmin, cmax], color="gray")

    axscatter[0].legend(loc="lower right")
    axvort[0].legend(loc="best")
    axts.legend(loc="best")
    #axalign.legend(loc="best")

    return figscatter, figvort, figts, fig

def main():
    try:
        params = load_parameters()
        ds = load_dataset(params["filepath"], params["name"])
        ds = truncate_time_series(ds)

        grid = depth_following_grid(params)
        Ut, Vt = compute_contour_following_velocities(ds.u, ds.v, grid)
        vortflux = compute_vorticity_flux(ds, grid)

        t = ds.time / np.timedelta64(1, 's')
        t_days = ds.time / np.timedelta64(1, 'D')

        xvals = (10, 30, 40, 50, 80)
        cmap = cm.batlow

        figscatter, figvort, figts, fig = plot_results(
            params, ds, grid, vortflux, Ut, Vt, xvals, t, t_days, cmap
        )


        # Save figures
        figscatter.savefig(f"slope/figures/scatters/{params['name']}_scatter_Hcontour.png")
        figvort.savefig(f"slope/figures/vorticity(u)/{params['name']}_nonlin_scatter_Hcontour.png")
        figts.savefig(f"slope/figures/timeseries/{params['name']}_timeseries_Hcontour.png")
        fig.savefig("slope/figures/timeseries/Hcontours.png")

        # Optional: Show plots interactively
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
