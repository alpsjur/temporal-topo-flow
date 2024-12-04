"""
Recreate Figure 3 from Brink (2010).
"""
# Import necessary packages for visualization, ocean simulation, formatted output, and data handling
using CairoMakie        # For creating visualizations and animations
using Oceananigans      # For simulating ocean dynamics
using Printf            # For formatted string output
using JLD2              # For saving and loading simulation data in Julia format
using Statistics        # For basic statistical operations

# Define file paths and filenames
filepath = "output/"                # Path to simulation output
figurepath = "figures/"             # Path for saving figures
filename = "brink_2010-300"         # Base name for simulation files

# Define physical constants
R = 5e-4                            # Bottom friction coefficient
ρ = 1e3                             # Reference density of seawater (kg/m³)
g = 9.81                            # Gravitational acceleration (m/s²)
dy = 1e3                            # Grid spacing in y-direction (m)
dx = 1e3                            # Grid spacing in x-direction (m)

# Load simulation data
full_output = FieldDataset(filepath * filename * ".jld2")
u = full_output.u                   # u-component of velocity field
v = full_output.v                   # v-component of velocity field
η = full_output.η                   # Sea surface height field
h = -FieldDataset(filepath * filename * "_bathymetry.jld2").bath[:, :, 1, 1]  # Bathymetry
u∂v∂x = full_output.u∂v∂x           # Term for momentum transport
∂v∂x = full_output.∂v∂x             # Derivative of v with respect to x
∂η∂y = full_output.∂η∂y             # Derivative of sea surface height with respect to y

# Extract time points from simulation data
times = u.times

# Get grid coordinates from the first timestep of the height field
xc, yc, zc = nodes(η[1]) 
xc /= 1e3  # Convert x-coordinates to kilometers
yc /= 1e3  # Convert y-coordinates to kilometers

# Define time-averaging parameters
T = 4                           # Average over 4 days (1 timestep per day)
Tend = length(times)            # Final timestep index

# Initialize centered fields for interpolation
uc = deepcopy(η)                # Interpolated u-component
vc = deepcopy(η)                # Interpolated v-component
u∂v∂xc = deepcopy(η)            # Interpolated u∂v/∂x
∂v∂xc = deepcopy(η)             # Interpolated ∂v/∂x
∂η∂yc = deepcopy(η)             # Interpolated ∂η/∂y

# Interpolate fields to grid centers for all timesteps
for i in 1:length(times)
    uc[i] .= @at (Center, Center, Center) u[i]              # u at grid centers
    vc[i] .= @at (Center, Center, Center) v[i]              # v at grid centers
    u∂v∂xc[i] .= @at (Center, Center, Center) u∂v∂x[i]      # u∂v/∂x at grid centers
    ∂v∂xc[i] .= @at (Center, Center, Center) u∂v∂x[i]       # ∂v/∂x at grid centers
    ∂η∂yc[i] .= @at (Center, Center, Center) ∂η∂y[i]        # ∂η/∂y at grid centers
end  

# Compute momentum transport (MT) terms
MT1 = mean(u∂v∂xc.data[:, :, 1, Tend-T:Tend], dims=(2, 3))[:, 1, 1]
MT2 = mean(∂v∂xc.data[:, :, 1, Tend-T:Tend], dims=(2, 3))[:, 1, 1] .* 
      mean(uc.data[:, :, 1, Tend-T:Tend], dims=(2, 3))[:, 1, 1]
MT = (MT1 - MT2) .* mean(h, dims=2)[:, 1]

# Compute form stress (TFS) terms
TFS1 = (∂η∂yc .* h)[:, :, 1, :]
TFS2 = mean(∂η∂yc.data[:, :, 1, Tend-T:Tend], dims=(2, 3))[:, 1, 1] .* mean(h, dims=2)[:, 1]
TFS = (mean(TFS1, dims=(2, 3))[:, 1, 1] - TFS2) .* g

# Compute bottom stress (BS) terms
BS = mean(vc.data[:, :, 1, Tend-T:Tend], dims=(2, 3))[:, 1, 1] .* R

# Plot results
fig = Figure(size=(800, 600))
ax = Axis(fig[1, 1], xlabel="x [km]", ylabel="Terms [m²/s²]") 

# Add individual terms to the plot
lines!(ax, xc, TFS, label="Form stress")                   # Form stress
lines!(ax, xc, MT, label="Momentum transport")             # Momentum transport
lines!(ax, xc, BS, label="Bottom stress")                  # Bottom stress
lines!(ax, xc, TFS + MT + BS, label="Sum", color=:gray, linestyle=:dash)  # Total sum

# Set axis limits and legend
limits!(ax, 0, 90, nothing, nothing)                       # Set x-axis limits
axislegend(position=:rb)                                   # Place legend at bottom-right

# Save the figure
save(figurepath * filename * "_fig3.png", fig)
