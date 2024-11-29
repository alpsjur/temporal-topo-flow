# Import necessary packages for visualization, ocean simulation, formatted output, and data handling
using CairoMakie        # For creating visualizations and animations
using Oceananigans      # For simulating ocean dynamics
using Printf            # For formatted string output
using JLD2              # For saving and loading simulation data in Julia format
using Statistics        # For statistical calculations
using RollingFunctions  # For applying rolling window functions

# Define file path and name of the saved simulation output
filepath = "output/brink/"
#fullfilename = ARGS[1]
#filename = split(fullfilename, "/")[end][1:end-3]
filename = "brink_2010-361"

# Visualization step interval for vector fields
step = 4

# Load time series data from the saved JLD2 file
u_timeseries = FieldTimeSeries(filepath * filename * ".jld2", "u")  # u-component of velocity field
v_timeseries = FieldTimeSeries(filepath * filename * ".jld2", "v")  # v-component of velocity field
η_timeseries = FieldTimeSeries(filepath * filename * ".jld2", "η")  # height (or depth) field
ω_timeseries = FieldTimeSeries(filepath * filename * ".jld2", "ω")  

# Extract time points from the simulation data
times = u_timeseries.times
days = times / (3600 * 24)  # Convert time from seconds to days for easier interpretation


# Get coordinate arrays from the first timestep of height field data
# `xv`, `yv`, `zv` are not used, so only `xc`, `yc`, `zc` are extracted
xc, yc, zc = nodes(η_timeseries[1])
xc = xc/1e3
yc = yc/1e3


# Initialize fields for interpolated velocity components and derived fields
uc_timeseries = deepcopy(η_timeseries)  # u-component interpolated on the h grid
vc_timeseries = deepcopy(η_timeseries)  # v-component interpolated on the h grid
s_timeseries = deepcopy(η_timeseries)  # vorticity interpolated on the h grid
ωc_timeseries = deepcopy(η_timeseries)  # vorticity interpolated on the h grid

# Loop over all time steps to interpolate `u` and `v` and calculate derived fields
for i in 1:length(times)
    uᵢ = u_timeseries[i]  # u-component at time `i`
    vᵢ = v_timeseries[i]  # v-component at time `i`
    ωᵢ = ω_timeseries[i]

    # Interpolate `u` and `v` to grid centers and calculate speed and vorticity
    uc_timeseries[i] .= @at (Center, Center, Center) uᵢ
    vc_timeseries[i] .= @at (Center, Center, Center) vᵢ
    s_timeseries[i] .= @at (Center, Center, Center) sqrt(uᵢ^2 + vᵢ^2)
    ωc_timeseries[i] .= @at (Center, Center, Center) ωᵢ
end

# Initialize logging for the animation creation process
@info "Animating "*filename

# Create an observable integer `n` to index the time series data during animation
n = Observable(1)

# Define a dynamic title for the animation that updates based on the current time step
title = @lift @sprintf("%15i days", times[$n] / (3600 * 24))

# Extract the interior data for the height, speed, vorticity, and velocity components at the current time step
# These values will be updated dynamically during the animation
ηₙ = @lift interior(η_timeseries[$n], :, :)  # Height field
sₙ = @lift interior(s_timeseries[$n], :, :)                  # Speed field
ucₙ = @lift interior(uc_timeseries[$n], Integer(step/2):step:length(xc), Integer(step/2):step:length(yc))  # u-component of velocity
vcₙ = @lift interior(vc_timeseries[$n], Integer(step/2):step:length(xc), Integer(step/2):step:length(yc))  # v-component of velocity
ωₙ = @lift interior(ωc_timeseries[$n], :, :)  # Vorticity field

# Set limits for the color scales used in the plots
ηmax = maximum(interior(η_timeseries))           # Maximum depth for color scaling
ηmin = minimum(interior(η_timeseries))
slim = maximum(interior(s_timeseries))           # Maximum speed for color scaling
ωlim = maximum(abs, ωc_timeseries)           # Maximum speed for color scaling

# Create a figure object to hold the plots for the animation
fig = Figure(size = (1500, 700))

# Create axes for each subplot with appropriate titles
ax2 = Axis(fig[2:4, 1]; title = "velocity [m s-1]", xlabel="x [km]", ylabel="y [km]", aspect=DataAspect())            # Axis for velocity plot
ax3 = Axis(fig[2:4, 2]; title = "sea surface height [m]", xlabel="x [km]", ylabel="y [km]", aspect=DataAspect())      # Axis for ssh plot
ax4 = Axis(fig[2:4, 3]; title = "vorticity [s-1]", xlabel="x [km]", ylabel="y [km]", aspect=DataAspect())             # Axis for vorticity plot
limits!(ax2, 0, 90, 0, yc[end])
limits!(ax3, 0, 90, 0, yc[end])
limits!(ax4, 0, 90, 0, yc[end])

# Add a label for the figure title that will dynamically update with the time variable
fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)

hm_s = heatmap!(ax2, xc, yc, sₙ; colorrange = (0, slim), colormap = :speed)
Colorbar(fig[5, 1], hm_s, vertical=false)

# Overlay arrows representing velocity vectors on the speed plot
ar = arrows!(ax2, xc[Integer(step/2):step:end], yc[Integer(step/2):step:end], ucₙ, vcₙ, 
    lengthscale = 2,
    normalize = true,
)

hm_η = heatmap!(ax3, xc, yc, ηₙ; colorrange = (ηmin, ηmax), colormap = :thermal)
Colorbar(fig[5, 2], hm_η, vertical=false)

hm_ω = heatmap!(ax4, xc, yc, ωₙ; colorrange = (-ωlim, ωlim), colormap = :curl)
Colorbar(fig[5, 3], hm_ω, vertical=false)

# Define the frame range for the animation
frames = 1:length(times)

# Record the animation, updating the figure for each time step
CairoMakie.record(fig, "animations/brink/" * filename * ".mp4", frames, framerate = 8) do i
    #msg = string("Plotting frame ", i, " of ", frames[end])
    #print(msg * " \r")  # Log progress without creating a new line for each frame
    n[] = i             # Update the observable `n` to the current frame index

end