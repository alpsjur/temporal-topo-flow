# Import necessary packages for visualization, ocean simulation, formatted output, and data handling
using CairoMakie        # For creating visualizations and animations
using Oceananigans      # For simulating ocean dynamics
using Printf            # For formatted string output
using JLD2              # For saving and loading simulation data in Julia format
using Statistics        # For statistical calculations
using RollingFunctions  # For applying rolling window functions

# Define file path and name of the saved simulation output
filepath = "output/"
filename = "slope-no_bumps-noise-quadratic_drag"  # Name of the file containing simulation data

# Visualization step interval for vector fields
step = 14  # Interval for vector field visualization (used for arrow plotting)

# Load time series data from the saved JLD2 file
u_timeseries = FieldTimeSeries(filepath * filename * ".jld2", "u")  # u-component of velocity field
v_timeseries = FieldTimeSeries(filepath * filename * ".jld2", "v")  # v-component of velocity field
h_timeseries = FieldTimeSeries(filepath * filename * ".jld2", "h")  # height (or depth) field

# Extract time points from the simulation data
times = u_timeseries.times
days = times / (3600 * 24)  # Convert time from seconds to days for easier interpretation

# Compute the mean along-slope velocity with rolling mean smoothing
T = 10 * 2  # Rolling window size
U = mean(u_timeseries.data, dims=(1,2,3))[1,1,1,:]  # Mean u-velocity across spatial dimensions
Ures = rollmean(collect(U), T)  # Smoothed velocity using rolling mean
fill_start = fill(-100, T - 1)  # Fill the start with placeholder values for alignment
Ures = vcat(fill_start, Ures)   # Concatenate the placeholder values with the rolling mean

# Get coordinate arrays from the first timestep of height field data
# `xv`, `yv`, `zv` are not used, only `xc`, `yc`, `zc` are extracted
xc, yc, zc = nodes(h_timeseries[1])

# Initialize timeseries for derived quantities: speed and vorticity
s_timeseries = sqrt(u_timeseries^2 + v_timeseries^2)  # Speed (magnitude of velocity)
ω_timeseries = ∂x(v_timeseries) - ∂y(u_timeseries)    # Vorticity (curl of velocity)

# Initialize fields for interpolated velocity components and derived fields
uc_timeseries = deepcopy(h_timeseries)  # Interpolated u-component on the h grid
vc_timeseries = deepcopy(h_timeseries)  # Interpolated v-component on the h grid
s_timeseries = deepcopy(h_timeseries)   # Interpolated speed on the h grid
ω_timeseries = deepcopy(h_timeseries)   # Interpolated vorticity on the h grid

# Loop over all time steps to interpolate `u` and `v` and calculate derived fields
for i in 1:length(times)
    uᵢ = u_timeseries[i]  # u-component at time `i`
    vᵢ = v_timeseries[i]  # v-component at time `i`

    # Interpolate `u` and `v` to grid centers and calculate speed and vorticity
    uc_timeseries[i] .= @at (Center, Center, Center) uᵢ  # Interpolated u at grid centers
    vc_timeseries[i] .= @at (Center, Center, Center) vᵢ  # Interpolated v at grid centers
    s_timeseries[i] .= @at (Center, Center, Center) sqrt(uᵢ^2 + vᵢ^2)  # Speed at grid centers
    ω_timeseries[i] .= @at (Center, Center, Center) ∂x(vᵢ) - ∂y(uᵢ)    # Vorticity at grid centers
end

# Initialize logging for the animation creation process
@info "Making an animation from saved data..."

# Create an observable integer `n` to index the time series data during animation
n = Observable(1)  # Observable to track current time step for animation
pointsU = Observable(Point2f[(days[1], U[1])])  # Observable for plotting raw velocity data
pointsUres = Observable(Point2f[(days[1], Ures[1])])  # Observable for plotting smoothed velocity

# Define a dynamic title for the animation that updates based on the current time step
title = @lift @sprintf("%20i days", times[$n] / (3600 * 24))  # Title displaying current time in days

# Extract the interior data for speed, vorticity, and velocity components for the current time step
# These values are updated dynamically during the animation
sₙ = @lift interior(s_timeseries[$n], :, :)  # Speed field at current time step
ωₙ = @lift interior(ω_timeseries[$n], :, :)  # Vorticity field at current time step
ucₙ = @lift interior(uc_timeseries[$n], Integer(step/2):step:length(xc), Integer(step/2):step:length(yc))  # Interpolated u-component
vcₙ = @lift interior(vc_timeseries[$n], Integer(step/2):step:length(xc), Integer(step/2):step:length(yc))  # Interpolated v-component

# Set limits for the color scales used in the plots
slim = maximum(interior(s_timeseries))  # Max speed for color scaling
ωlim = maximum(abs, interior(ω_timeseries))  # Max absolute vorticity for color scaling
Ulim = maximum(abs, U) * 1.01  # Max along-slope velocity for scaling

# Create a figure object to hold the plots for the animation
fig = Figure(size = (800, 800))

# Create axes for each subplot with appropriate titles
ax1 = Axis(fig[2, 1]; title = "area mean u [m s-1]")  # Axis for mean u-velocity plot
ax2 = Axis(fig[3, 1]; title = "velocity [m s-1]")     # Axis for velocity plot
ax3 = Axis(fig[4, 1]; title = "vorticity [s-1]")      # Axis for vorticity plot

# Add a label for the figure title that dynamically updates with the current time
fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)

# Scatter plots for raw and smoothed velocity data
scatter!(ax1, pointsU)
scatter!(ax1, pointsUres)
limits!(ax1, 0, days[end], -Ulim, Ulim)  # Set axis limits based on time and velocity ranges

# Create heatmap for the speed field with color scaling
hm_s = heatmap!(ax2, xc, yc, sₙ; colorrange = (0, slim), colormap = :speed)
Colorbar(fig[3, 2], hm_s)  # Add colorbar for the speed plot

# Overlay arrows representing velocity vectors on the speed plot
ar = arrows!(ax2, xc[Integer(step/2):step:end], yc[Integer(step/2):step:end], ucₙ, vcₙ, 
    lengthscale = 1e6
)

# Create heatmap for the vorticity field with color scaling
hm_ω = heatmap!(ax3, xc, yc, ωₙ; colorrange = (-ωlim, ωlim), colormap = :curl)
Colorbar(fig[4, 2], hm_ω)  # Add colorbar for the vorticity plot

# Define the frame range for the animation
frames = 1:length(times)

# Record the animation, updating the figure for each time step
CairoMakie.record(fig, "animations/" * filename * ".mp4", frames, framerate = 4) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")  # Log progress without creating a new line for each frame
    n[] = i  # Update the observable `n` to the current frame index

    # Update points for raw and smoothed velocity plots
    new_pointU = Point2f((days[i], U[i]))
    new_pointUres = Point2f((days[i], Ures[i]))
    pointsU[] = push!(pointsU[], new_pointU)
    pointsUres[] = push!(pointsUres[], new_pointUres)
end
