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
    "cmdiv": cmc.vik,      # muted, diverging
    "cmcat": cmc.batlow,   # perceptually uniform, colorblind-friendly
    "cmseq": cmc.buda      # good for 1D fields (e.g., SSH, T, etc.)
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

def plot_3D_bathymetry(X, Y, h, show_contour=False, width="single", aspect_ratio=0.6):
    _, (fig_width_px, fig_height_px) = get_figure_dimensions(width, aspect_ratio)

    fig = go.Figure(data=[
        go.Surface(
            z=h,
            x=X,
            y=Y,
            colorscale="deep",
            cmin=np.nanmin(h),
            cmax=np.nanmax(h),
            showscale=False,
            lighting=dict(ambient=0.5, diffuse=0.8, roughness=0.5),
            contours_z=dict(
                show=show_contour,
                color=palette["accent1"],
                width=6,
                start=500,
                end=501,
                size=20
            )
        )
    ])

    fig.update_layout(
        font=dict(color=palette["text"], size=16),
        width=fig_width_px,
        height=fig_height_px,
        scene=dict(
            xaxis=dict(title="x (km)", color=palette["text"], gridcolor=palette["text"]),
            yaxis=dict(title="y (km)", color=palette["text"], gridcolor=palette["text"]),
            zaxis=dict(title="Depth (m)", autorange="reversed",
                       color=palette["text"], gridcolor=palette["text"],
                       range=[0, np.nanmax(h) * 1.1], nticks=5),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=aspect_ratio)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, b=0, t=40),
        scene_camera=dict(
        eye=dict(x=1, y=-1.5, z=0.7),
        center=dict(x=0, y=0, z=-0.2)
    )
    )

    return fig

def plot_bathymetry(ax, X, Y, h, contours=None):
    """
    Plot 2D bathymetry on a given axis using imshow and optional contours.

    Parameters:
        ax: matplotlib axis object
        X, Y: 2D meshgrid arrays (in km)
        h: 2D array of bathymetry (in m, positive downward)
        contours: list of contour levels (in m)
    """
    ax.set_aspect("equal")

    Z = h.T[::-1]  # transpose and flip y for correct orientation
    extent = [X.min(), X.max(), Y.min(), Y.max()]

    img = ax.imshow(
        Z,
        extent=extent,
        cmap=cmo.deep,
        vmin=np.nanmin(h),
        vmax=np.nanmax(h),
        aspect="auto"
    )

    if contours is not None:
        ax.contour(
            X, Y, h,
            levels=contours,
            colors=palette["accent1"],
            linewidths=1.5
        )

    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")

    ax.set_xticks(np.linspace(X.min(), X.max(), 4))
    ax.set_yticks(np.linspace(Y.min(), Y.max(), 4))

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())

    return img
