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
filename = "brink_2010-300-period_02"

# Visualization step interval for vector fields
step = 4

# Load time series data from the saved JLD2 file
u_timeseries = FieldTimeSeries(filepath * filename * ".jld2", "u")  # u-component of velocity field
v_timeseries = FieldTimeSeries(filepath * filename * ".jld2", "v")  # v-component of velocity field
η_timeseries = FieldTimeSeries(filepath * filename * ".jld2", "η")  # height (or depth) field

# Extract time points from the simulation data
times = u_timeseries.times
days = times / (3600 * 24)  # Convert time from seconds to days for easier interpretation

# Get coordinate arrays from the first timestep of height field data
# `xv`, `yv`, `zv` are not used, so only `xc`, `yc`, `zc` are extracted
xc, yc, zc = nodes(η_timeseries[1])
xc = xc/1e3
yc = yc/1e3

dx = xc[2]-xc[1]

# Compute the mean along-slope velocity with rolling mean smoothing
T = parse(Int16,split(filename, "_")[end]) * 24  # Rolling window size
v_slope = v_timeseries.data[1:Int(dx*60),:,1,:]
V = mean(v_slope, dims=(1,2))[1,1,:]  # Mean v-velocity across spatial dimensions
Vres = rollmean(collect(V), T)  # Smoothed velocity using rolling mean
fill_start = fill(-100, T - 1)  # Fill the start with placeholder values for alignment
Vres = vcat(fill_start, Vres)   # Concatenate the placeholder values with the rolling mean


# Initialize fields for interpolated velocity components and derived fields
uc_timeseries = deepcopy(η_timeseries)  # u-component interpolated on the h grid
vc_timeseries = deepcopy(η_timeseries)  # v-component interpolated on the h grid
s_timeseries = deepcopy(η_timeseries)   # Speed interpolated on the h grid

# Loop over all time steps to interpolate `u` and `v` and calculate derived fields
for i in 1:length(times)
    uᵢ = u_timeseries[i]  # u-component at time `i`
    vᵢ = v_timeseries[i]  # v-component at time `i`

    # Interpolate `u` and `v` to grid centers and calculate speed and vorticity
    uc_timeseries[i] .= @at (Center, Center, Center) uᵢ
    vc_timeseries[i] .= @at (Center, Center, Center) vᵢ
    s_timeseries[i] .= @at (Center, Center, Center) sqrt(uᵢ^2 + vᵢ^2)
end

# Initialize logging for the animation creation process
@info "Animating "*filename

# Create an observable integer `n` to index the time series data during animation
n = Observable(1)
pointsV = Observable(Point2f[(days[1], V[1])])  # Observable for plotting raw velocity data
pointsVres = Observable(Point2f[(days[1], Vres[1])])  # Observable for plotting smoothed velocity

# Define a dynamic title for the animation that updates based on the current time step
title = @lift @sprintf("%15i days", times[$n] / (3600 * 24))

# Extract the interior data for the height, speed, vorticity, and velocity components at the current time step
# These values will be updated dynamically during the animation
ηₙ = @lift interior(η_timeseries[$n], :, :)  # Height field
sₙ = @lift interior(s_timeseries[$n], :, :)  # Speed field
ucₙ = @lift interior(uc_timeseries[$n], Integer(step/2):step:length(xc), Integer(step/2):step:length(yc))  # u-component of velocity
vcₙ = @lift interior(vc_timeseries[$n], Integer(step/2):step:length(xc), Integer(step/2):step:length(yc))  # v-component of velocity

# Set limits for the color scales used in the plots
ηmax = maximum(interior(η_timeseries))           # Maximum depth for color scaling
ηmin = minimum(interior(η_timeseries))
slim = maximum(interior(s_timeseries))           # Maximum speed for color scaling
Vlim = maximum(abs, V) * 1.01  # Max along-slope velocity for scaling

# Create a figure object to hold the plots for the animation
fig = Figure(size = (1200, 800))

# Create axes for each subplot with appropriate titles
ax2 = Axis(fig[2:4, 1]; title = "velocity [m s-1]", xlabel="x [km]", ylabel="y [km]", aspect=DataAspect())     # Axis for velocity plot
ax3 = Axis(fig[2:4, 2]; title = "sea surface height [m]", xlabel="x [km]", ylabel="y [km]", aspect=DataAspect())      # Axis for vorticity plot
ax4 = Axis(fig[6, 1:2]; title = "area mean v", xlabel="time [days]", ylabel="[m s-1]")  # Axis for mean u-velocity plot

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

# Scatter plots for raw and smoothed velocity data
scatter!(ax4, pointsV)
scatter!(ax4, pointsVres)
limits!(ax4, 0, days[end], -Vlim, Vlim)  # Set axis limits based on time and velocity ranges

# Define the frame range for the animation
frames = 1:length(times)

# Record the animation, updating the figure for each time step
CairoMakie.record(fig, "animations/brink/" * filename * ".mp4", frames, framerate = 24) do i
    #msg = string("Plotting frame ", i, " of ", frames[end])
    #print(msg * " \r")  # Log progress without creating a new line for each frame
    n[] = i             # Update the observable `n` to the current frame index

    # Update points for raw and smoothed velocity plots
    new_pointV = Point2f((days[i], V[i]))
    new_pointVres = Point2f((days[i], Vres[i]))
    pointsV[] = push!(pointsV[], new_pointV)
    pointsVres[] = push!(pointsVres[], new_pointVres)
end