import xarray as xr
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.grid import prepare_dsH, interp_ds
from utils.config import load_config
from utils.io import read_raw_output


# Parameters for arrow density and scaling
arrow_density = 5  # Number of bins for averaging

# read simulation output
params = load_config()
ds = read_raw_output(params)
ds = interp_ds(ds, params, ["u", "v"])

# Extract variables
x = ds['xC'].values 
y = ds['yC'].values 
time = ds['time'].values
u = ds['u'].values  # Shape: (time, y, x)
v = ds['v'].values  # Shape: (time, y, x)


# ds = xr.open_dataset("/itf-fi-ml/home/alsjur/temporal-topo-flow/input/forcing/long_nobumps_crosswind_forcing.nc")
# x = ds.x.values
# y = ds.y.values
# u = ds.forcing_x.values
# v = ds.forcing_y.values
# time = ds.time.values

# Calculate binned averages
def bin_average(data, bins_y, bins_x):
    # Ensure the data can be evenly reshaped
    ny, nx = data.shape[1], data.shape[2]  # Original spatial dimensions
    data = data[:, :bins_y * arrow_density, :bins_x * arrow_density]  # Trim to fit bins
    binned = data.reshape(data.shape[0], bins_y, arrow_density, bins_x, arrow_density)
    binned = binned.mean(axis=2).mean(axis=3)  # Average over bins
    return binned

bins_y = len(y) // arrow_density
bins_x = len(x) // arrow_density
x_binned = x[:bins_x * arrow_density].reshape(bins_x, arrow_density).mean(axis=1)
y_binned = y[:bins_y * arrow_density].reshape(bins_y, arrow_density).mean(axis=1)

u_binned = bin_average(u, bins_y, bins_x)
v_binned = bin_average(v, bins_y, bins_x)


# --- full-field speed (for colors) ---
speed = np.hypot(u, v)  # (t,y,x)

# Fixed color scaling (stable across time)
vmax = np.nanpercentile(speed, 99)
vmin = 0.0

# --- quiver grid from your binning ---
Xb, Yb = np.meshgrid(x_binned, y_binned)

# ===== Arrow scaling that looks good =====
# 1) pick a target arrow length as a fraction of domain width
domain_w = float(x.max() - x.min())
target_frac = 0.08  # ~8% of domain width looks readable; tweak 0.05–0.12
L_target = target_frac * domain_w  # arrow length in axis x-units

# 2) measure a robust "typical" binned speed
speed_binned = np.hypot(u_binned, v_binned)  # (t,By,Bx)
s_ref = np.nanpercentile(speed_binned, 90)  # 90th keeps a few long arrows

# avoid divide-by-zero
gain = 0.0 if s_ref == 0 else (L_target / s_ref)

# We'll **multiply** velocities by 'gain' so that a 90th-percentile arrow spans ~L_target
# Use 'scale_units="xy", scale=1.0' so U,V are interpreted directly in axis units
# =======================================

# --- figure / artists ---
fig, ax = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
ax.set_title("Velocity field: speed (color) + binned vectors")
ax.set_xlabel("x")
ax.set_ylabel("y")
extent = [x.min(), x.max(), y.min(), y.max()]

i0 = 0
im = ax.imshow(
    speed[i0], origin='lower', extent=extent, aspect='auto',
    interpolation='nearest', vmin=vmin, vmax=vmax
)
cb = fig.colorbar(im, ax=ax, pad=0.02, label='|u| (speed)')

quiv = ax.quiver(
    Xb, Yb,
    gain * u_binned[i0],
    gain * v_binned[i0],
    angles='xy', scale_units='xy', scale=1.0,
    width=0.0025, pivot='mid',
    headwidth=3.5, headlength=5.5, headaxislength=4.0, minlength=0.0
)

def _time_str(t):
    if np.issubdtype(time.dtype, np.datetime64):
        return str(np.asarray(t).astype('datetime64[s]'))
    if np.issubdtype(time.dtype, np.timedelta64):
        return f"{(t / np.timedelta64(1,'D')):.2f} days"
    return str(t)

ttl = ax.text(
    0.02, 0.98, f"t = {_time_str(time[i0])}",
    transform=ax.transAxes, va='top', ha='left', fontsize=10,
    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2)
)

def update(frame):
    im.set_data(speed[frame])
    quiv.set_UVC(gain * u_binned[frame], gain * v_binned[frame])
    ttl.set_text(f"t = {_time_str(time[frame])}")
    return im, quiv, ttl

ani = animation.FuncAnimation(fig, update, frames=speed.shape[0], interval=100, blit=False)


# --- optional: save ---
ani.save("uv_speed_quiver.mp4", dpi=150, fps=10, codec="h264")
# or:
#ani.save("uv_speed_quiver.gif", dpi=120, fps=10, writer="pillow")
