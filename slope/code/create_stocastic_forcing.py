import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from utils import load_parameters

kappa_inv = 3 * 24 * 3600  # Forcing timescale (3 days)
r =  30                    # Energy control parameter
Cd = 1.5e-3                # Drag coefficient
rho_air =  1.225           # Air density (kg/m³)
num_modes = 12             # Number of wind patterns
L_c = 70e3                 # Cutoff length scale 

params = load_parameters()
config = params["name"]

# Define dimensions and time steps
Lx = params["Lx"]
Ly = params["Ly"]

dx = params["dx"]
dy = params["dy"]

tmax = params["tmax"]
dt = params["outputtime"]

x = np.arange(0, Lx, dx)
y = np.arange(0, Ly, dy)
time = np.arange(0, tmax, dt)

nx, ny, nt = len(x), len(y), len(time)

# Define wavenumbers TODO skal det være pi her?
kx = np.fft.fftfreq(nx, dx) #* 2 * np.pi
ky = np.fft.fftfreq(ny, dy) #* 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing="ij")
K2 = KX**2 + KY**2  # Squared total wavenumber

# Define modal spectral density function Φ(k)
C = 1/L_c
K2_safe = np.copy(K2)
K2_safe[K2 == 0] = 1e-12  # Replace zero values with a small number

Phi_k = C**3 / (K2_safe * (np.sqrt(K2_safe) + C)**3)  # Spectral shape, eq. 69


Phi_k[0, 0] = 0  # Ensure zero mean component

# Tapering function
F = np.ones((nx, ny))
F[:15,:] = (np.sin(x[:15]*np.pi/30e3)**2)[:,None]
F[-15:,:] = (np.sin(x[:15]*np.pi/30e3)**2)[::-1,None]

# Generate 12 independent spatial wind patterns W_i(x, y)
W_x_modes = []
W_y_modes = []

fig, axes = plt.subplots(figsize=(12,10), nrows=3, ncols=4)
axes = axes.flatten()
i = 0

for _ in range(num_modes):
    # Generate a random phase field
    phase = np.exp(2j * np.pi * np.random.rand(nx, ny))

    # Create streamfunction in Fourier space
    Psi_k = np.sqrt(Phi_k) * phase

    # Transform back to physical space
    Psi = np.real(np.fft.ifft2(Psi_k)) 
    
    axes[i].pcolormesh(Psi)
    i += 1

    # Compute divergence-free wind components
    dPsidx, dPsidy = np.gradient(Psi, dx, dy)
    W_x = -dPsidy 
    W_y = dPsidx
    # W_x = np.gradient(Psi, axis=1) / dy  # ∂Ψ/∂y
    # W_y = -np.gradient(Psi, axis=0) / dx   # ∂Ψ/∂x

    W_x_modes.append(W_x)
    W_y_modes.append(W_y)

W_x_modes = np.array(W_x_modes)  # Shape: (num_modes, nx, ny)
W_y_modes = np.array(W_y_modes)

fig.savefig("Psi.png")


# Define the Markov process for time-dependent weights
kappa = 1 / kappa_inv
m = np.zeros((num_modes, nt))
m[:, 0] = 0  # Initial condition: all weights start at 0

# Generate white noise process R_i(t)
R = np.random.normal(0, 1, (num_modes, nt))


# Syclic markov chain, to ensure the field integrates to zero
# Solve the Markov equation for m_i(t)
nnt = int(nt/16)
for t in range(1, nt):
    m[:, t] = m[:, t-1] - kappa * m[:, t-1] * dt + kappa * r * R[:, t] * dt

# m[:, nnt:nnt*2] = m[:, nnt-1::-1]
# m[:, nnt*2:nnt*4] = -m[:, 0:nnt*2]
# m[:, nnt*4:nnt*8] = -m[:, 0:nnt*4]
# m[:, nnt*8:] = m[:, 0:nnt*8]

fig, ax = plt.subplots()

for i in range(num_modes):
    ax.plot(m[i], alpha=0.2)
fig.savefig("markov.png")


# Reshape Markov process weights to allow broadcasting
m_reshaped = m[:, :, None, None]  # Shape: (num_modes, nt, 1, 1)

# Ensure `W_y_modes` integrates to zero in x and y
# W_y_modes -= np.mean(W_y_modes, axis=1, keepdims=True)  
# W_y_modes -= np.mean(W_y_modes, axis=2, keepdims=True) 
# W_x_modes -= np.mean(W_x_modes, axis=1, keepdims=True) 
# W_x_modes -= np.mean(W_x_modes, axis=2, keepdims=True)  

# Compute wind field by summing over modes
u = np.mean(m_reshaped * W_x_modes[:, None, :, :], axis=0) * 2e3  # Shape: (nt, nx, ny)
v = np.mean(m_reshaped * W_y_modes[:, None, :, :], axis=0) * 2e3

u = u*F 
v = v*F

# Compute wind stress using quadratic drag law
wind_speed = np.sqrt(u**2 + v**2)

tau_x = Cd * rho_air * wind_speed * u
tau_y = Cd * rho_air * wind_speed * v

# Remove the mean in time and y-direction to ensure integral is zero
# tau_x -= np.mean(tau_x, axis=(0, 2), keepdims=True)
# tau_y -= np.mean(tau_y, axis=(0, 2), keepdims=True)


# Should be of order ~0.1
print(np.max(tau_x))
print(np.min(tau_x))
print(np.mean(tau_x))

# Create an xarray dataset
ds = xr.Dataset(
    {"forcing_x": (["time", "x", "y"], tau_x),
     "forcing_y": (["time", "x", "y"], tau_y)
     },
    coords={"time": time, "x": x, "y": y}
)

# Save to NetCDF
ds.to_netcdf("slope/input/"+config+"_forcing.nc")