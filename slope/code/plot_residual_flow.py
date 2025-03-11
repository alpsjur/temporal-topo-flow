import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cmocean

from utils import load_parameters, load_dataset, calculate_bathymetry

# Load parameters and dataset
params = load_parameters()
data = load_dataset(params["filepath"], params["name"])

# Extract variables
x = data['xC'].values 
y = data['yC'].values 
u = data['u'].values  # Shape: (time, y, x)
v = data['v'].values  # Shape: (time, y, x)
h = data['h'].values
X, Y = np.meshgrid(x, y)
bath = calculate_bathymetry(X, Y, params)

x /= 1e3
y /= 1e3

# Define time period T (e.g., average over all time steps or a subset)
Tn = int(params["T"]/params["outputtime"])

# Compute time-averaged velocity field
u_mean = np.mean(u[-Tn:], axis=0)
v_mean = np.mean(v[-Tn:], axis=0)

# Interpolate velocities to cell centers
u_centered = 0.5 * (u_mean[:, :-1] + u_mean[:, 1:])
v_centered = 0.5 * (v_mean[:-1, :] + v_mean[1:, :])
v_centered = np.concatenate((v_centered, 0.5 * (v_mean[-1:, :] + v_mean[:1, :])), axis=0)

# Bin the velocity field for quiver plot
arrow_density = 3  # Number of bins for averaging
bins_y = len(y) // arrow_density
bins_x = len(x) // arrow_density

# Function to bin data
def bin_average(data, bins_y, bins_x):
    ny, nx = data.shape  # Spatial dimensions
    data = data[:bins_y * arrow_density, :bins_x * arrow_density]  # Trim to fit bins
    binned = data.reshape(bins_y, arrow_density, bins_x, arrow_density)
    binned = binned.mean(axis=1).mean(axis=2)  # Average over bins
    return binned

# Compute binned velocity field
u_binned = bin_average(u_centered, bins_y, bins_x)
v_binned = bin_average(v_centered, bins_y, bins_x)
x_binned = x[:bins_x * arrow_density].reshape(bins_x, arrow_density).mean(axis=1) 
y_binned = y[:bins_y * arrow_density].reshape(bins_y, arrow_density).mean(axis=1) 

# Compute speed for scaling arrows
speed_binned = np.sqrt(u_binned**2 + v_binned**2)
print(np.max(speed_binned))
speed_scale = np.max([round(np.max(speed_binned)*1e2),1])
print(speed_scale)
scale_factor = 1 / (np.max(speed_binned) + 1e-10)  # Scale by max speed

# Adjust arrow length by speed
u_scaled = u_binned * scale_factor
v_scaled = v_binned * scale_factor

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bathymetry
cmap = cmocean.cm.deep
bath_plot = ax.pcolormesh(x, y, bath, cmap=cmap, shading='auto')
fig.colorbar(bath_plot, ax=ax, label='Depth [m]')

# Plot streamlines
#ax.streamplot(x, y, u_centered, v_centered, color='k', density=1.5, linewidth=0.8, arrowsize=0)

# Overlay quiver plot with scaled arrows at binned locations
quiver = ax.quiver(x_binned, y_binned, 
                   u_scaled, v_scaled, 
                   scale = 20,
                   color='r', 
                   alpha=0.7,
                   zorder = 20
                   )
#ax.quiverkey(quiver, 0.9, 1.05, 1, 'Velocity vectors', labelpos='E')

ax.quiverkey(quiver, 0.9, 1.05, speed_scale*scale_factor/1e2, f'{speed_scale} cm s-1', labelpos='E')

# Labels and formatting
ax.set_title("Recidual flow")
ax.set_xlabel("x [km]")
ax.set_ylabel("y [km]")
ax.set_aspect('equal')

fig.tight_layout()
fig.savefig("slope/figures/streamlines/" + params["name"] + "_residual.png")


plt.show()
