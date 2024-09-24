# Import required packages for visualization, ocean simulation, formatted output, and data handling
using CairoMakie
using Oceananigans
using Printf
using JLD2

# Define the path to the saved output file containing simulation data
filepath = "output/"
filename = "test_shallow_water"

# Open the JLD2 file and extract time series data 
u_timeseries = FieldTimeSeries(filepath*filename*".jld2", "u")
v_timeseries = FieldTimeSeries(filepath*filename*".jld2", "v")
h_timeseries = FieldTimeSeries(filepath*filename*".jld2", "h")


# Extract time points and bottom height 
times = u_timeseries.times
#h = b_timeseries.grid.immersed_boundary.bottom_height
#h = interior(h,1,:,1)  # Adjust the bottom height array for visualization

# Get coordinate arrays 
#xv, yv, zv = nodes(v_timeseries[1])  
xc, yc, zc = nodes(h_timeseries[1]) 

# Shift u and v to same grid point, calculate speed
vc_timeseries = deepcopy(b_timeseries)
wc_timeseries = deepcopy(b_timeseries)
s_timeseries = deepcopy(b_timeseries)


for i in 1:length(times)
    uᵢ = u_timeseries[i]
    vᵢ = v_timeseries[i]

    uc_timeseries[i] .= @at (Center, Center, Center) uᵢ
    vc_timeseries[i] .= @at (Center, Center, Center) vᵢ
    s_timeseries[i] .= @at (Center, Center, Center) sqrt(uᵢ^2+wᵢ^2)
end


# Initialize logging for the animation creation process
@info "Making an animation from saved data..."

# Create an observable integer for indexing through time series data
n = Observable(1)

# Define a title for the animation with the time variable dynamically updated
title = @lift @sprintf("%20s", prettytime(times[$n]))

# Extract the interior data for the u and b fields at the current time step, dynamically updated
sₙ = @lift interior(s_timeseries[$n], :, :)

# Set limits for the velocity color scale
slim = maximum(interior(s_timeseries))


# Create a figure object for the animation
fig = Figure(size = (1200, 1100))

# Create axes 
ax_vs = Axis(fig[1, 1]; 
    title = "velocity [m/s]", 
    )

# Add a title 
fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)

# Create a heatmap for the surface velocity field
hm_s = heatmap!(ax_v, yc, zc, sₙ; colorrange = (0, slim), colormap = :speed)

#lines!(ax_v, yc, h, color=:gray)
Colorbar(fig[1, 2], hm_s)

# Define the frame range for the animation
frames = 1:length(times)

# Record the animation, updating the figure for each time step
CairoMakie.record(fig, "animations/"*filename*".mp4", frames, framerate=1) do i
    msg = string("Plotting frame ", i, " of ", frames[end])
    print(msg * " \r")  # Log progress without creating a new line for each frame
    n[] = i             # Update the observable to the current frame index
end