import numpy as np
import sys
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
from cmcrameri import cm as cmc
from utils import get_h, analytical_circ, load_parameters, load_dataset, \
    truncate_time_series, get_vorticityflux_at_contour, compute_alignment, \
        depth_following_contour, slope_end, get_contour_following_velocities

# Configure seaborn style
sns.set_style("whitegrid")


def setup_plots():
    """Initialize figures and axes for various plots."""
    figscatter, axscatter = plt.subplots(figsize=(12, 6), 
                                         ncols=2)
    axscatter[0].set_xlabel("linear analytical circ [cm m s-1]")
    axscatter[1].set_xlabel("nonlinear analytical circ [cm m s-1]")
    axscatter[0].set_ylabel("numerical circ [cm m s-1]")
    axscatter[0].set_aspect("equal")
    axscatter[1].set_aspect("equal")

    figvort, axvort = plt.subplots(figsize=(12, 6), ncols=2)
    axvort[0].set_ylabel("vorticity contribution [cm m s-1]")
    axvort[0].set_xlabel("linear circ [cm m s-1]")
    #axvort[0].set_aspect("equal")
    axvort[1].set_ylabel("vorticity flux x H/R [cm m s-1]")
    axvort[1].set_xlabel("linear circ [cm m s-1]")
    #axvort[1].set_aspect("equal")

    #figts, [axts, axalign] = plt.subplots(figsize=(16, 10), nrows=2, sharex=True)
    figts, axts = plt.subplots(figsize=(16, 10))
    #axts.set_xlabel("time [days]")
    axts.set_ylabel("circ * H [cm m s-1]")
    # axalign.set_xlabel("time [days]")
    # axalign.set_ylabel("mean alignment []")
    
    figalsc, axalsc = plt.subplots(figsize=(6, 6))
    axalsc.set_xlabel("circ * H [cm m s-1]")
    axalsc.set_ylabel("alignment []")
    
    # Dummy label
    axts.plot([None, None],[None,None], color="gray", label="numerical")
    axts.plot([None, None],[None,None], color="gray", label="non-linear", ls="-.")
    axts.plot([None, None],[None,None], color="gray", label="linear", ls="--")

    return figscatter, axscatter, figvort, axvort, figts, axts, \
        figalsc, axalsc
    # return figscatter, axscatter, figvort, axvort, figts, axts, \
    #     axalign, figalsc, axalsc

def plot_results(params, ds, xvals, t, t_days, cmap):
    """Generate plots for time series, scatter plots, and contours."""
    n = len(xvals)
    colors = [cmap(1 - i / (n - 1)) for i in range(n)]

    h = get_h(params)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.pcolormesh(h, cmap="Grays", alpha=0.7)
    ax.set_aspect("equal")

    figscatter, axscatter, figvort, axvort, figts, axts, \
        figalsc, axalsc = setup_plots()
    
    T = params["T"]
    outputtime = params["outputtime"]
    R = params["R"]
    Tinc = min(round(5 * T / outputtime), len(t))
    
    _, slopeend = slope_end(params)
    depths = np.mean(h, axis=0)
    
    cmax, cmin = 0, 0

    for i, xval in enumerate(xvals):
        color = colors[i]
        H = depths[xval]
        
        if xval > slopeend:
            flat = True
            
        else:
            flat = False
        
        contour = depth_following_contour(params, H, flat=flat, xflat=xval)
        
        cL = contour.dl.sum(dim=("j")).values

        nonlin = get_vorticityflux_at_contour(contour, ds).values   
        Ut, Vt =  get_contour_following_velocities(contour, ds)
        alignments = compute_alignment(Ut, Vt)
        alignment = (alignments*contour.dl).sum(dim=("j"))/cL
        alignment = alignment.where(alignment != 0, other=np.nan)

        analytical_linear = analytical_circ(params, t, cL, H) * 1e2
        analytical_nonlinear = analytical_circ(params, t, cL, H, nonlin) * 1e2
        numerical = -(Vt*contour.dl).sum(dim=("j"))/cL * 1e2  # Convert to cm/s

        axts.plot(t_days[:Tinc], numerical[:Tinc]*H, 
                  color=color, 
                  label=f"{int(H)}m depth",
                  zorder = len(xvals)-i,
                  )
        axts.plot(t_days[:Tinc], analytical_linear[:Tinc]*H, 
                  color=color, 
                  linestyle="--",
                  zorder = len(xvals)-i,
                  )
        axts.plot(t_days[:Tinc], analytical_nonlinear[:Tinc]*H, 
                  color=color, 
                  linestyle="-." ,
                  zorder = len(xvals)-i,
                  )

        axscatter[0].scatter(analytical_linear*H, numerical*H, 
                             s=16, 
                             alpha=0.7, 
                             marker = "x",
                             color=color, 
                             zorder = len(xvals)-i,
                             label=f"{int(H)}m depth"
                             )
        axscatter[1].scatter(analytical_nonlinear*H, numerical*H, 
                             s=16, 
                             alpha=0.7, 
                             marker = "x",
                             color=color,
                             zorder = len(xvals)-i,
                             )

        vorticity_contribution = numerical - analytical_linear
        axvort[0].scatter(#numerical*H, vorticity_contribution*H, 
                          analytical_linear*H, vorticity_contribution*H,
                          s=16, 
                          alpha=0.7, 
                          marker = "x",
                          color=color, 
                          zorder = len(xvals)-i,
                          label=f"{int(H)}m depth"
                          )
        axvort[1].scatter(#numerical*H, nonlin * H * H / R * 1e2, 
                          analytical_linear*H, nonlin * H * H / R * 1e2,
                          s=16, 
                          alpha=0.7, 
                          marker = "x",
                          color=color,
                          zorder = len(xvals)-i,
                          )

        # axalign.plot(t_days[-Tinc:], alignment[-Tinc:], 
        #              color=color, 
        #              zorder = len(xvals)-i,
        #              label=f"{int(H)}m depth"
        #              )
        axalsc.scatter(numerical, alignment, 
                       color=color, 
                       label=f"{int(H)}m depth", 
                       s=16, 
                       alpha=0.7,
                       marker = "x", 
                       zorder = len(xvals)-i,
                       )

        ax.plot(contour.x * 1e-3, contour.y * 1e-3, color=color)

        cmax, cmin = max(cmax, np.max(numerical)), min(cmin, np.min(numerical))

    axscatter[0].plot([cmin, cmax], [cmin, cmax], color="gray")
    axscatter[1].plot([cmin, cmax], [cmin, cmax], color="gray")

    axscatter[0].legend(loc="lower right")
    axvort[0].legend(loc="best")
    axts.legend(loc="best")
    axalsc.legend()

    return figscatter, figvort, figts, figalsc, fig

def main():
    try:
        params = load_parameters()
        ds = load_dataset(params["filepath"], params["name"])
        ds = truncate_time_series(ds)

        t = ds.time / np.timedelta64(1, 's')
        t_days = ds.time / np.timedelta64(1, 'D')

        xvals = (10, 40, 45, 50, 55, 80)
        cmap = cmc.batlow

        figscatter, figvort, figts, figalsc, fig = plot_results(
            params, ds, xvals, t, t_days, cmap
        )

        # Save figures
        figscatter.savefig(f"slope/figures/scatters/circ/{params['name']}_scatter_Hcontour.png")
        figvort.savefig(f"slope/figures/vorticity(u)/{params['name']}_nonlin_scatter_Hcontour.png")
        figts.savefig(f"slope/figures/timeseries/{params['name']}_timeseries_Hcontour.png")
        figalsc.savefig(f"slope/figures/scatters/alignment/{params['name']}_scatter_alignment.png")
        fig.savefig("slope/figures/timeseries/Hcontours.png")

        # Optional: Show plots interactively
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
