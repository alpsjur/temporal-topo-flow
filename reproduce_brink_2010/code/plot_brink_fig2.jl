"""
Recreate Figure 2 from Brink (2010).
"""
# Import necessary packages for visualization, ocean simulation, formatted output, and data handling
using CairoMakie        # For creating visualizations and animations
using Oceananigans      # For simulating ocean dynamics
using Printf            # For formatted string output
using JLD2              # For saving and loading simulation data in Julia format
using Statistics        # For basic statistical operations

# Define file paths and names for simulation output and figures
filepath = "reproduce_brink_2010/output/"          # Path to simulation output
figurepath = "reproduce_brink_2010/figures/"       # Path for saving figures
filename = "brink_2010-300"         # Base name for simulation files

# Define visualization parameters
step = 3                            # Interval for vector field visualization
stepstart = 1                       # Starting point for vector field visualization

# Load simulation data
full_output = FieldDataset(filepath * filename * ".jld2")  # Load full simulation output
u_timeseries = full_output.u                              # u-component of velocity field
v_timeseries = full_output.v                              # v-component of velocity field
η_timeseries = full_output.η                              # Surface height/depth field
bath = FieldDataset(filepath * filename * "_bathymetry.jld2").bath[:, :, 1, 1] # Bathymetry data

# Extract time points from simulation
times = u_timeseries.times

# Get grid coordinates from the first timestep of the height field
xc, yc, zc = nodes(η_timeseries[1])
xc /= 1e3   # Convert x-coordinates to kilometers
yc /= 1e3   # Convert y-coordinates to kilometers

# Initialize fields for interpolated velocity components and derived fields
uc_timeseries = deepcopy(η_timeseries)  # Interpolated u-component on the η grid
vc_timeseries = deepcopy(η_timeseries)  # Interpolated v-component on the η grid

# Interpolate velocity components to grid centers for all timesteps
for i in 1:length(times)
    uᵢ = u_timeseries[i]   # u-component at timestep `i`
    vᵢ = v_timeseries[i]   # v-component at timestep `i`

    # Interpolate to grid centers
    uc_timeseries[i] .= @at (Center, Center, Center) uᵢ
    vc_timeseries[i] .= @at (Center, Center, Center) vᵢ
end

# Define averaging parameters for time-averaged fields
T = 4                       # Average over 4 days (1 timestep per day)
Tend = length(times)        # Final timestep index

# Compute time-averaged fields
U = collect(mean(uc_timeseries.data[:, :, 1, Tend-T:Tend], dims=3)[:, :, 1])  # Averaged u-component
V = collect(mean(vc_timeseries.data[:, :, 1, Tend-T:Tend], dims=3)[:, :, 1])  # Averaged v-component
Η = collect(mean(η_timeseries.data[:, :, 1, Tend-T:Tend], dims=3)[:, :, 1])   # Averaged surface height

# Calculate depth extremes for color scaling
Ηmax = maximum(Η)           # Maximum depth
Ηmin = minimum(Η)           # Minimum depth

# Create figure for visualization
fig = Figure(size=(600, 600))
ax = Axis(fig[1, 1], xlabel="x [km]", ylabel="y [km]") 
limits!(ax, 0, 90, 0, yc[end])  # Set axis limits

# Plot surface height contours
contour!(ax, xc, yc, Η; color=:black, linewidth=5)

# Overlay arrows representing velocity vectors
arrows!(
    ax,
    xc[stepstart:step:end], yc[stepstart:step:end],
    U[stepstart:step:end, stepstart:step:end],
    V[stepstart:step:end, stepstart:step:end],
    lengthscale=8e1  # Scale arrow length
)

# Add bathymetry contours
contour!(ax, xc, yc, bath, levels=0:-100:-1000, color=:gray)

# Save the figure
save(figurepath * filename * "_fig2.png", fig)
