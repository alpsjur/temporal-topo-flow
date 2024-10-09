# Import necessary packages for visualization, ocean simulation, formatted output, and data handling
using CairoMakie        # For creating visualizations and animations
using Oceananigans      # For simulating ocean dynamics
using Printf            # For formatted string output
using JLD2              # For saving and loading simulation data in Julia format
using Statistics

# Define file path and name of the saved simulation output
filepath = "output/"
figurepath = "figures/"
#fullfilename = ARGS[1]
#filename = split(fullfilename, "/")[end][1:end-3]
filename = "brink_2010-300"

# Visualization step interval for vector fields
step = 4


# Load time series data from the saved JLD2 file
full_output = FieldDataset(filepath * filename * ".jld2")
u_timeseries = full_output.u # u-component of velocity field
v_timeseries = full_output.v  # v-component of velocity field
η_timeseries = full_output.η  # height (or depth) field
bath = FieldDataset(filepath * filename * "_bathymetry.jld2").bath[:,:,1,1]

# Extract time points from the simulation data
times = u_timeseries.times

# Get coordinate arrays from the first timestep of height field data
# `xv`, `yv`, `zv` are not used, so only `xc`, `yc`, `zc` are extracted
xc, yc, zc = nodes(η_timeseries[1]) 
xc = xc/1e3
yc = yc/1e3

# Initialize timeseries for derived quantities: speed and vorticity
s_timeseries = sqrt(u_timeseries^2 + v_timeseries^2)  # Speed (magnitude of velocity vector)

# Initialize fields for interpolated velocity components and derived fields
uc_timeseries = deepcopy(η_timeseries)  # u-component interpolated on the h grid
vc_timeseries = deepcopy(η_timeseries)  # v-component interpolated on the h grid
s_timeseries = deepcopy(η_timeseries)   # Speed interpolated on the h grid

# Loop over all time steps to interpolate `u` and `v` and calculate derived fields
for i in 1:length(times)
    uᵢ = u_timeseries[i]  # u-component at time `i`
    vᵢ = v_timeseries[i]  # v-component at time `i`

    # Interpolate `u` and `v` to grid centers
    uc_timeseries[i] .= @at (Center, Center, Center) uᵢ
    vc_timeseries[i] .= @at (Center, Center, Center) vᵢ
end


T = 4*24
Tend = length(times)

U = collect(mean(uc_timeseries.data[:,:,1,Tend-T:Tend], dims=3)[:,:,1])
V = collect(mean(vc_timeseries.data[:,:,1,Tend-T:Tend], dims=3)[:,:,1])
Η = collect(mean(η_timeseries.data[:,:,1,Tend-T:Tend], dims=3)[:,:,1])

Ηmax = maximum(Η)           # Maximum depth for color scaling
Ηmin = minimum(Η)


fig = Figure(size = (600, 600));
ax = Axis(fig[1, 1]) 
limits!(ax, 0, xc[end], 0, yc[end])

#hm_η = heatmap!(ax, xc, yc, Η; colorrange = (Ηmin, Ηmax), colormap = :balance)
#Colorbar(fig[2, 1], hm_η, vertical=false)
contour!(ax, xc, yc, Η; color=:black, linewidth = 5)

# Overlay arrows representing velocity vectors on the speed plot
ar = arrows!(ax, xc[2:step:end], yc[2:step:end], U[2:step:end,2:step:end], V[2:step:end,2:step:end], 
    lengthscale = 5e2,
    #normalize = true,
)

contour!(ax, xc, yc, bath, levels = 0:-100:-1000, color=:gray)

save(figurepath*filename*"_fig2.png", fig)