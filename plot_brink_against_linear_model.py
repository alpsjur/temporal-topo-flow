# Import necessary packages for visualization, data handling, and numerical operations
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr  # For loading data in NetCDF format (replace if needed)
from scipy.ndimage import uniform_filter1d

# Define parameters and file paths
name = "329"
depths = []
Tinc = 12 * 8

filepath = "output/brink/"
figurepath = "figures/brink/"
cmap = plt.cm.get_cmap('viridis')  # Replace with appropriate colormap

# Forcing parameters
rho = 1e3
d = 0.1
T = 4 * 24 * 60 * 60  # 4 days in seconds
R = 5e-4
L = 90 * 1e3  # 90 kilometers in meters

outputtime = 3 * 60 * 60  # 3 hours in seconds
delta_t = 2  # 2 seconds

config = name
omega = 2 * np.pi / T
window = int(outputtime / delta_t)

# Load data (assuming NetCDF format for Python compatibility)
ds = xr.open_dataset(filepath + name + ".nc")
u = ds.u  # u-component of velocity field
v = ds.v  # v-component of velocity field
h = ds.h
t = ds.time
t_hr = np.arange(0, t[-1] + 1, 2)
t_days = t[:-1] / (24 * 60 * 60)  # Convert time to days

# Set up figures and axes
figscatter, axscatter = plt.subplots(figsize=(6, 6))
axscatter.set_xlabel("analytical circ [cm s-1]")
axscatter.set_ylabel("numerical circ [cm s-1]")
axscatter.plot([-10, 10], [-10, 10], color="gray")

figts, axts = plt.subplots(figsize=(12, 4))
axts.set_xlabel("time [days]")
axts.set_ylabel("circ [cm s-1]")

# Generate colors for each xval
n = len(depths)
colors = [cmap(1 - i / (n - 1)) for i in range(n)]

for i, H in enumerate(depths):
    color = colors[i]
    H = np.mean(h[xval, :])

    analytical_hr = -d / rho * (R * np.sin(omega * t_hr) - H * omega * np.cos(omega * t_hr) +
                                np.exp(-R * t_hr / H) * H * omega) / (H**2 * omega**2 + R**2)
    analytical = uniform_filter1d(analytical_hr, size=window)[::window] * 1e2  # Convert to cm/s

    numerical = -np.mean(v[xval, :, 0, :], axis=(0, 1))[:-1] * 1e2  # Convert to cm/s

    # Plot analytical and numerical data over time
    axts.plot(t_days, numerical, color=color, label=str(xval))
    axts.plot(t_days, analytical, color=color, linestyle="--")

    # Scatter plot of analytical vs. numerical data
    axscatter.scatter(analytical, numerical, label=str(xval), s=16, alpha=0.7, color=color)

# Add legends
figscatter.legend(loc="lower right")
figts.legend(loc="lower right")

# Save figures
#figscatter.savefig(figurepath + name + "_scatter.png")
#figts.savefig(figurepath + name + "_analytical_ts.png")

# Show plots (optional, for interactive environments)
plt.show()
