"""
Creates an animation of simulation output. 
To create a animation of a specific simulation, run from the command line:
    python scripts/animate_fields.py configs/{name}.json

Animate velocity (left panel) and vorticity fields (right panel) from raw model output. 
Velocity vectors are downsampled for clarity. Speed is represented both by vector length and color.

- Loads configuration, reads raw output, and trims the time window to match publication figures.
- Interpolates u/v to cell centers, downsamples vectors, and computes arrow scaling for consistent vector length.
- Produces an MP4 (or GIF fallback) saved under animations/animation_{name}.ext.
"""

import xarray as xr
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from matplotlib.animation import FFMpegWriter
import os

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.grid import prepare_dsH, interp_ds
from utils.config import load_config
from utils.io import read_raw_output
from utils.plotting import create_figure, palette, colorwheel, customize_axis, get_figure_dimensions


# Parameters for arrow density and scaling, for readability
ARROW_DENSITY = 5  # Number of cells per bin (controls vector density)


def load_and_prepare():
    # Load user config, raw fields, and restrict time window to match figure settings
    # config is provided as command-line argument or defaults to configs/default.json
    params = load_config()
    ds = read_raw_output(params)
    ds = interp_ds(ds, params, ["u", "v"])

    # Focus on the same time period as in paper figures
    if params["name"] == "long":
        start = -(128 + 64) * 8
        stop = -64 * 8
        ds = ds.isel(time=slice(start, stop))
    elif params["name"] == "short":
        start = -(128 + 8) * 8
        stop = -8 * 8
        ds = ds.isel(time=slice(start, stop))
    else: # defaults to last 128 output days
        dt = params.get("outputtime")
        if dt is None:
            raise RuntimeError("outputtime missing from config for generic case")
        ds = ds.isel(time=slice(-128 * int(86400 / dt), None))

    return params, ds


def bin_average(data, bins_y, bins_x, arrow_density):
    # Average vectors inside density blocks to thin quiver arrows without aliasing flow
    trim_y = bins_y * arrow_density
    trim_x = bins_x * arrow_density
    data = data[:, :trim_y, :trim_x]
    # reshape -> (time, bins_y, arrow_density, bins_x, arrow_density)
    binned = data.reshape(data.shape[0], bins_y, arrow_density, bins_x, arrow_density)
    binned = binned.mean(axis=2).mean(axis=3)
    return binned


def compute_binning(u, v, x, y, arrow_density):
    # Downsample velocities and coordinates so quiver is readable
    bins_y = len(y) // arrow_density
    bins_x = len(x) // arrow_density
    x_binned = x[:bins_x * arrow_density].reshape(bins_x, arrow_density).mean(axis=1)
    y_binned = y[:bins_y * arrow_density].reshape(bins_y, arrow_density).mean(axis=1)
    u_binned = bin_average(u, bins_y, bins_x, arrow_density)
    v_binned = bin_average(v, bins_y, bins_x, arrow_density)
    Xb, Yb = np.meshgrid(x_binned, y_binned)
    return u_binned, v_binned, Xb, Yb


def compute_arrow_gain(x, u_binned, v_binned, target_frac=0.08):
    # Scale arrows so a 90th percentile speed spans a fixed fraction of domain width
    domain_w = float(x.max() - x.min())
    L_target = target_frac * domain_w
    speed_binned = np.hypot(u_binned, v_binned)
    s_ref = np.nanpercentile(speed_binned, 90)
    return 0.0 if s_ref == 0 else (L_target / s_ref)


def make_axes(extent, x, y, xb, yb, tau, bathymetry, H_targets, i0, speed, zeta, u_binned, v_binned, gain):
    # Build the static artists (two panels) reused across animation frames
    fig, axes = plt.subplot_mosaic(
        [
         ['ax0', 'ax1'],
         ['tau', 'tau'],
         ],
        figsize=(6.4, 5),
        height_ratios=[3, 1],
        constrained_layout=True
    )
    ax_tau = axes['tau']
    ax0 = axes['ax0']
    ax1 = axes['ax1']

    # overwrite default axis customization (imported from utils.plotting)
    for ax in (ax0, ax1):
        ax.spines["right"].set_visible(True)
        ax.spines["top"].set_visible(True)

    ax0.set_title("Velocity field: speed (color) + velocity (vectors)", loc="left")
    ax1.set_title("Relative vorticity field", loc="left")
    ax0.set_xlabel("x [km]")
    ax1.set_xlabel("x [km]")
    ax0.set_ylabel("y [km]")
    
    ### forcing panel
    #ax_tau.set_axis_off()
    tau_plot = tau / 1e3  # convert to N m⁻²
    t_plot = np.linspace(0, 128, len(tau_plot))
    ax_tau.plot(t_plot, tau_plot, color="gray")
    ax_tau.set_ylabel(r"Surface stress [N m⁻²]")
    ax_tau.set_xlabel("Time [days]")
    ax_tau.set_xlim(t_plot.min(), t_plot.max())
    ax_tau.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax_tau.set_xticks(np.arange(0, 129, 16))
    
    # plot forcign for initial time as a scatter point
    sc = ax_tau.scatter(tau_plot[i0], tau_plot[i0], color=colorwheel[2], zorder=10)

    
    ### Velocity panel

    # Remove poutlier values for better color scaling 
    vmax = np.nanpercentile(speed, 99)
    vmin = 0.0

    im0 = ax0.imshow(
        speed[i0], origin="lower", extent=extent, aspect="equal",
        interpolation="lanczos", vmin=vmin, vmax=vmax, cmap=palette["cmseq"]
    )
    cb0 = fig.colorbar(im0, ax=ax0, pad=0.02, label="|u| [cm s⁻¹]", orientation="horizontal", shrink=0.8)

    ax0.contour(
        x, y, bathymetry,
        levels=H_targets[2::4],
        colors="gray",
        linewidths=0.5,
        zorder=10
    )

    quiv = ax0.quiver(
        xb, yb,
        gain * u_binned[i0],
        gain * v_binned[i0],
        angles="xy", scale_units="xy", scale=1.0,
        width=0.006, pivot="mid",
        color=colorwheel[2],
        headwidth=4.0, headlength=6.0, headaxislength=4.0, minlength=0.0,
        zorder=12
    )
    
    ### Vorticity panel

    # remove outlier values for better color scaling
    zeta_lim = np.nanpercentile(np.abs(zeta), 98)
    im1 = ax1.imshow(
        zeta[i0], origin="lower", extent=extent, aspect="equal",
        interpolation="lanczos", cmap=palette["cmdiv"],
        vmin=-zeta_lim, vmax=zeta_lim
    )
    cb1 = fig.colorbar(im1, ax=ax1, pad=0.02, label=r"$\zeta$ [s⁻¹]", orientation="horizontal", extend="both", shrink=0.8)

    ax1.contour(
        x, y, bathymetry,
        levels=H_targets[2::4],
        colors="gray",
        linewidths=0.5,
        zorder=10
    )

    return fig, ax0, ax1, sc, im0, im1, quiv, (cb0, cb1)


def make_time_string(time):
    # Factory: return a formatter matching time coordinate dtype
    def _time_str(t):
        if np.issubdtype(time.dtype, np.datetime64):
            return str(np.asarray(t).astype("datetime64[s]"))
        if np.issubdtype(time.dtype, np.timedelta64):
            return f"{((t - time[0]) / np.timedelta64(1,'D')):.0f} days"
        return str(t - time[0])
    return _time_str


def animate(params, ds):
    # Prepare arrays (km for coords, cm/s for velocities) and derived fields
    x = ds["xC"].values / 1e3
    y = ds["yC"].values / 1e3
    time = ds["time"].values
    u = ds["u"].values * 1e2
    v = ds["v"].values * 1e2
    bathymetry = ds["h"].isel(time=1).values
    H_targets = bathymetry.mean(axis=1)  # same depth contour as in paper figures

    u_binned, v_binned, xb, yb = compute_binning(u, v, x, y, ARROW_DENSITY)
    
    speed = np.hypot(u, v)
    zeta = ds["zeta"].values
    tau = ds["forcing_x"].mean(dim=["xC", "yC"]).values

    gain = compute_arrow_gain(x, u_binned, v_binned)
 
    extent = [0, 90, 0, 90]  # domain extent in km
    i0 = 0

    fig, ax0, ax1, sc, im0, im1, quiv, _ = make_axes(
        extent, x, y, xb, yb, tau, bathymetry, H_targets, i0, speed, zeta, u_binned, v_binned, gain
    )

    # Time annotation
    _time_str = make_time_string(time)
    ttl0 = ax0.text(0.02, 0.98, f"t = {_time_str(time[i0])}", transform=ax0.transAxes, va="top", ha="left",
                    zorder=20, bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=2))
    ttl1 = ax1.text(0.02, 0.98, f"t = {_time_str(time[i0])}", transform=ax1.transAxes, va="top", ha="left",
                    zorder=20, bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=2))

    outdir = Path("animations")
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"static_frame_{params['name']}.png", dpi=150)

    def update(frame):
        sc.set_offsets([frame*params["outputtime"]/86400, tau[frame] / 1e3])
        im0.set_data(speed[frame])
        quiv.set_UVC(gain * u_binned[frame], gain * v_binned[frame])
        im1.set_data(zeta[frame])
        ttl0.set_text(f"t = {_time_str(time[frame])}")
        ttl1.set_text(f"t = {_time_str(time[frame])}")
        return im0, quiv, ttl0, im1, ttl1

    ani = animation.FuncAnimation(fig, update, 
                                  #frames=speed.shape[0], 
                                  frames = 50,  # limit frames for faster rendering
                                  interval=100, 
                                  blit=False
                                  )

    out_path = outdir / f"animation_{params['name']}.mp4"

    # writer = FFMpegWriter(
    #     fps=20,
    #     codec="libx264",
    #     extra_args=[
    #         "-crf", "18",
    #         "-preset", "slow",
    #         "-pix_fmt", "yuv444p",      # <- key: no chroma subsampling
    #         "-profile:v", "high444"
    #     ],
    # )
    #ani.save(str(out_path), writer=writer, dpi=200)
    ani.save(str(out_path), dpi=200, fps=20, codec="h264", 
             extra_args=[
                 "-crf", "18",
                 "-preset", "slow",
                 "-pix_fmt", "yuv444p",      # <- key: no chroma subsampling
                 "-profile:v", "high444"
             ],
             )



def main():
    params, ds = load_and_prepare()
    animate(params, ds)


if __name__ == "__main__":
    main()
