import matplotlib.pyplot as plt
import plotly.graph_objects as go
from cmcrameri import cm as cmc
from cmocean import cm as cmo
import numpy as np
from matplotlib.cm import get_cmap 
import matplotlib

### Global style of figures ###

# Default print resolution
figure_dpi = 300

# Standard journal widths in cm
figure_sizes_cm = {
    "single": 8.3,
    "double": 12
}

# Make a sliced colormap (20–100% of the Blues colormap)
orig_cmap = plt.get_cmap("Blues")
new_cmap = orig_cmap(np.linspace(0.2, 1.0, 256))
new_cmap = matplotlib.colors.ListedColormap(new_cmap)

# Color palette
palette = {
    "background": "#ffffff",
    "text": "#000000",
    #"accent2": "#6495ED",  
    #"accent3": "#09B1A3",
    "cmdiv": cmo.balance,      # muted, diverging
    #"cmcat": cmc.batlow,   # perceptually uniform, colorblind-friendly
    "cmseq": new_cmap#cmo.deep,      # good for 1D fields (e.g., SSH, T, etc.)
}


colorwheel = [
    (000, 000, 000),   # black
    (000,158,115),  # green
    (213,94,0),    # orange
    (000,114,178), # blue
    # gray scale    
    (85,85,85),    # dark gray
    (170,170,170), # light gray
    
]

#transform to hex
colorwheel = ['#%02x%02x%02x' % color for color in colorwheel]

# Global style settings for article figures
plt.rcParams.update({
    # --- Geometry & export ---
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    #"constrained_layout.use": True,  # more reliable than tight_layout

    # --- Lines & markers ---
    "lines.linewidth": 1.0,          # thinner for small figures
    "lines.markersize": 3.0,
    "errorbar.capsize": 2.5,

    # --- Font sizes (tuned for 8.3 cm width) ---
    "font.family": "sans-serif",
    "font.size": 6,                  # base font
    "axes.titlesize": 7,
    "axes.labelsize": 6,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "legend.fontsize": 5.5,
    "figure.titlesize": 7,

    # --- Colors & faces ---
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",

    "axes.edgecolor": '#555555',#palette["text"],
    "axes.labelcolor": '#555555',#palette["text"],
    "xtick.color": '#555555',#palette["text"],
    "ytick.color": '#555555',#palette["text"],
    "text.color": '#555555',#palette["text"],
    "axes.titlecolor": '#555555',#palette["text"],

    # --- Axes & ticks ---
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,

    # --- Gridlines (subtle) ---
    "axes.grid": False,              # enable per-axis if needed
    "grid.color": "#cccccc",
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,

    # --- Legend ---
    "legend.frameon": False,
    "legend.edgecolor": palette["text"],
    "legend.facecolor": "white",
    "legend.framealpha": 1.0,

    # --- Fonts in vector outputs ---
    "pdf.fonttype": 42,              # embed fonts as TrueType
    "ps.fonttype": 42,
    "mathtext.default": "regular",
    "axes.formatter.use_mathtext": True,  # for consistent exponents, 1e−3 etc.
})


def customize_axis(ax, yzero=True, xzero=True):
    # Customize the appearance of the axes
    
    # Remove the top spine and corresponding ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()  # Keep the bottom ticks
    ax.spines['right'].set_color('none')  # Remove the right spine
    ax.yaxis.tick_left()  # Keep the left ticks

    # Set the position of the bottom and left spines to zero
    if yzero:
        ax.spines['bottom'].set_position('zero')
    if xzero:
        ax.spines['left'].set_position('zero')


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
        fig_width_cm = figure_sizes_cm.get(width, figure_sizes_cm["single"])
    else:
        fig_width_cm = float(width)

    fig_height_cm = fig_width_cm * aspect_ratio
    
    fig_width_in = (fig_width_cm / 2.54)
    fig_height_in = (fig_height_cm / 2.54)
    
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



    
    
    
