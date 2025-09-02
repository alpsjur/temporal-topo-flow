import matplotlib.pyplot as plt
import plotly.graph_objects as go
from cmcrameri import cm as cmc
from cmocean import cm as cmo
import numpy as np

### Global style of figures ###

# Default print resolution
figure_dpi = 300

# Standard journal widths in inches
figure_sizes_in = {
    "single": 3.5,
    "1.5col": 5.0,
    "double": 7.0
}

# Article-appropriate color palette
palette = {
    "background": "#ffffff",
    "text": "#000000",
    "accent1": "#E64A19",  
    "accent2": "#6495ED",  
    "accent3": "#09B1A3",
    "cmdiv": cmc.vik,      # muted, diverging
    "cmcat": cmc.batlow,   # perceptually uniform, colorblind-friendly
    "cmseq": cmc.buda,      # good for 1D fields (e.g., SSH, T, etc.)
}

# Global style settings for article figures
plt.rcParams.update({
    "lines.linewidth": 1.5,
    "font.family": "sans-serif",
    "font.size": 10,                  # base size: smaller, journal-appropriate
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 10,

    # Light background for printing
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",

    # Text and axes colors
    "axes.edgecolor": palette["text"],
    "axes.labelcolor": palette["text"],
    "xtick.color": palette["text"],
    "ytick.color": palette["text"],
    "text.color": palette["text"],
    "axes.titlecolor": palette["text"],

    # Gridlines (optional)
    "grid.color": "#cccccc",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,

    # Legend style
    "legend.edgecolor": palette["text"],
    "legend.facecolor": "white",
    "legend.framealpha": 1.0
})

# Color list for line/marker plots
n = 4
colorwheel = [palette["cmcat"](i / (n - 1)) for i in range(n)]

def customize_axis(ax):
    # Customize the appearance of the axes
    ax.spines['right'].set_color('none')  # Remove the right spine
    ax.yaxis.tick_left()  # Keep the left ticks

    # Set the position of the bottom and left spines to zero
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    # Remove the top spine and corresponding ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()  # Keep the bottom ticks

def get_figure_dimensions(width="single", aspect_ratio=0.6):
    """
    Return figure size in both inches (for Matplotlib) and pixels (for Plotly).

    Parameters:
        width (str or float): 'single', '1.5col', 'double', or numeric width in inches
        aspect_ratio (float): height / width

    Returns:
        (fig_width_in, fig_height_in), (fig_width_px, fig_height_px)
    """
    if isinstance(width, str):
        fig_width_in = figure_sizes_in.get(width, figure_sizes_in["single"])
    else:
        fig_width_in = float(width)

    fig_height_in = fig_width_in * aspect_ratio
    fig_width_px = int(fig_width_in * figure_dpi)
    fig_height_px = int(fig_height_in * figure_dpi)

    return (fig_width_in, fig_height_in), (fig_width_px, fig_height_px)


def create_figure(width="single", aspect_ratio=0.6, **kwargs):
    """
    Create a Matplotlib figure with predefined size based on journal column width.

    Returns:
        fig, ax
    """

    (fig_width_in, fig_height_in), _ = get_figure_dimensions(width, aspect_ratio)
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), **kwargs)
    return fig, ax

def plot_3D_bathymetry(
    X, Y, h, 
    width="single",
    aspect_ratio=0.6
):

    _, (fig_width_px, fig_height_px) = get_figure_dimensions(width, aspect_ratio)

    # Ensure 2D grids
    if np.ndim(X) == 1 and np.ndim(Y) == 1:
        Xg, Yg = np.meshgrid(X, Y, indexing="xy")
    else:
        Xg, Yg = X, Y

    H = np.asarray(h, dtype=float)
    cmin, cmax = np.nanmin(H), np.nanmax(H)

    fig = go.Figure([
        go.Surface(
            z=H, x=Xg, y=Yg,
            colorscale="deep",
            cmin=cmin, cmax=cmax,
            showscale=False,
            lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.5),
            contours_z=dict(show=False)
        )
    ])

    fig.update_layout(
        font=dict(color=palette["text"], size=16),
        width=fig_width_px,
        height=fig_height_px,
        scene=dict(
            xaxis=dict(title="x (km)", color=palette["text"], gridcolor=palette["text"]),
            yaxis=dict(title="y (km)", color=palette["text"], gridcolor=palette["text"]),
            zaxis=dict(
                title="Depth (m)", autorange="reversed",
                color=palette["text"], gridcolor=palette["text"],
                range=[0, cmax * 1.1], nticks=5
            ),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=aspect_ratio)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, b=0, t=40),
        scene_camera=dict(
            eye=dict(x=1.2, y=1.2, z=0.7),
            center=dict(x=0, y=0, z=-0.2)
        )
    )

    return fig

def add_surface_contours3d(
    fig,
    X, Y, h,
    levels,
    *,
    color=palette["accent1"],
    width=6,
    eps="auto",           # small z-offset to avoid Z-fighting ("auto" ~ 1e-3 data range)
    resample_factor=1,    # >1 to upsample before contour extraction (optional)
):
    """
    Overlay 3D contour polylines (single style) on an existing go.Figure with a Surface.

    Parameters
    ----------
    fig : go.Figure
    X, Y : 1D or 2D arrays
    h : 2D array
    levels : sequence of float
        Exact z-levels to draw as contours.
    color : str
        Color for contour lines.
    width : float
        Line width (pixels).
    eps : float or "auto"
        Small positive lift above the surface to prevent Z-fighting.
    resample_factor : int
        If >1, upsample X, Y, h for smoother contour paths (bilinear).
    """
    # Ensure 2D grids
    if np.ndim(X) == 1 and np.ndim(Y) == 1:
        Xg, Yg = np.meshgrid(X, Y, indexing="xy")
    else:
        Xg, Yg = X, Y
    H = np.asarray(h, dtype=float)

    cmin, cmax = np.nanmin(H), np.nanmax(H)
    eps_val = max(1e-6, (cmax - cmin) * 1e-3) if eps == "auto" else float(eps)

    # Optional upsampling
    if resample_factor and resample_factor > 1:
        from scipy.ndimage import zoom
        zy = zx = int(resample_factor)
        H  = zoom(H,  (zy, zx), order=1)
        Xg = zoom(Xg, (zy, zx), order=1)
        Yg = zoom(Yg, (zy, zx), order=1)

    levels = np.asarray(levels, dtype=float)
    cs = plt.contour(Xg, Yg, H, levels=levels)

    for level, coll in zip(cs.levels, cs.collections):
        for path in coll.get_paths():
            v = path.vertices
            if v.shape[0] < 2:
                continue
            fig.add_trace(go.Scatter3d(
                x=v[:, 0],
                y=v[:, 1],
                z=np.full(v.shape[0], level - eps_val),
                mode="lines",
                line=dict(width=float(width), color=color),
                showlegend=False
            ))
    plt.close()
    return fig

def plot_bathymetry(ax, X, Y, h):
    """
    Plot 2D bathymetry on a given axis using imshow (no contours).

    Parameters
    ----------
    ax : matplotlib axis object
    X, Y : 2D meshgrid arrays (in km)
    h : 2D array of bathymetry (in m, positive downward)
    """
    ax.set_aspect("equal")

    Z = h.T[::-1]  # transpose and flip y for correct orientation
    extent = [X.min(), X.max(), Y.min(), Y.max()]

    img = ax.imshow(
        Z,
        extent=extent,
        #origin="lower",  
        cmap=cmo.deep,
        vmin=np.nanmin(h),
        vmax=np.nanmax(h),
        aspect="auto"
    )

    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")

    ax.set_xticks(np.linspace(X.min(), X.max(), 4))
    ax.set_yticks(np.linspace(Y.min(), Y.max(), 4))

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())

    return img


def add_bathymetry_contours(ax, X, Y, h, contours, color=palette["accent1"], linewidth=1):
    """
    Add bathymetry contours to an existing axis.
    """
    ax.contour(
        X, Y, h,
        levels=contours,
        colors=color,
        linewidths=linewidth
    )

def plot_scatter(ax, x, y):

    ax.scatter(x, y,
               color=palette["accent1"],
               zorder=-1
               )

    # # Plot a reference line (y=x) to help assess the agreement between estimates and simulations
    lmax = np.max(x)
    lmin = np.min(y)
    
    llim = np.max(np.abs(y))

    
    ax.plot(#(lmin, lmax), (lmin, lmax), 
            (-llim, llim), (-llim, llim),
            palette["text"],
            #lw=0.5,
            zorder=21,
            )

    customize_axis(ax)

    # Set the aspect ratio of the plot to be equal
    ax.set_aspect('equal')


def plot_circulation_timeseries(ax, ts):
    
    t = ts.time / np.timedelta64(1, "D")
    
    ax.plot(t, ts.circulation*1e2,
            color=palette["accent1"],
            zorder=20,
            label = "simulations"
            )
    
    ax.plot(t, ts.linear_estimate*1e2,
            color=palette["accent2"],
            zorder=21,
            label="linear estimates"
            )
    
    ax.plot(t, ts.nonlinear_estimate*1e2,
            color=palette["accent3"],
            zorder=22,
            linestyle="--",
            label = "estimates including\nvorticity flux"
            )
    
    # Customize the axis appearance
    customize_axis(ax)
    
    
    
def initialize_momentum_diagrams():
    (fig_width_in, fig_height_in), _ = get_figure_dimensions("single", aspect_ratio=1)
    fig = plt.figure(layout="constrained", figsize=(fig_width_in, fig_height_in))
    axd = fig.subplot_mosaic(
        [
            ["ymean", "tymean"],
            ["circ", "tmean"],
        ],
        #empty_sentinel="BLANK",
        # set the height ratios between the rowsslice(int(tmax-T), int(tmax))
        height_ratios=[2, 4],
        # set the width ratios between the columns
        width_ratios=[4, 2]
    )

    axd["tmean"].sharey(axd["circ"])
    axd["ymean"].sharex(axd["circ"])

    axd["tmean"].spines['right'].set_color('none')  # Remove the right spine
    axd["tmean"].spines['top'].set_color('none')  # Remove the top spine
    #axd["tmean"].spines['bottom'].set_position('zero')  # Set the bottom spine position
    axd["tmean"].spines['left'].set_color('lightgray')
    axd["tmean"].spines['bottom'].set_color('lightgray')

    axd["tymean"].spines['right'].set_color('none')  # Remove the right spine
    axd["tymean"].spines['top'].set_color('none')  # Remove the top spine
    #axd["tymean"].spines['bottom'].set_position('zero')  # Set the bottom spine position
    axd["tymean"].spines['left'].set_color('lightgray')
    axd["tymean"].spines['bottom'].set_color('lightgray')

    axd["ymean"].spines['right'].set_color('none')  # Remove the right spine
    axd["ymean"].spines['top'].set_color('none')  # Remove the top spine
    #axd["ymean"].spines['left'].set_position('zero')  # Set the bottom spine position
    axd["ymean"].spines['left'].set_color('lightgray')
    axd["ymean"].spines['bottom'].set_color('lightgray')
    
    return fig, axd