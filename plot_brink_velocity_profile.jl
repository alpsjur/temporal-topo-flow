# Import necessary packages for visualization, ocean simulation, formatted output, and data handling
using CairoMakie        # For creating visualizations and animations
using Oceananigans      # For simulating ocean dynamics
using Printf            # For formatted string output
using JLD2              # For saving and loading simulation data in Julia format
using Statistics

# Define file path and name of the saved simulation output
filepath = "output/brink/"
figurepath = "figures/brink/"

filename = "brink_2010-300-period_002"

# Load time series data from the saved JLD2 file
full_output = FieldDataset(filepath * filename * ".jld2")
u = full_output.u # u-component of velocity field
v = full_output.v  # v-component of velocity field
η = full_output.η  # sea surface height field
h = - FieldDataset(filepath * filename * "_bathymetry.jld2").bath[:,:,1,1]

# Extract time points from the simulation data
times = u.times

# Get coordinate arrays from the first timestep of height field data
# `xv`, `yv`, `zv` are not used, so only `xc`, `yc`, `zc` are extracted
xc, yc, zc = nodes(η[1]) 
xc = xc/1e3
yc = yc/1e3


T = 64*8
Tend = length(times)

# Center all fields 
uc = deepcopy(η)  
vc = deepcopy(η) 

# Loop over all time steps to interpolate to center
for i in 1:length(times)
    uc[i] .= @at (Center, Center, Center) u[i]  # Interpolated u at grid centers
    vc[i] .= @at (Center, Center, Center) v[i]  # Interpolated v at grid centers
end  


U = mean(uc.data[:,:,1,Tend-T:Tend], dims=(2,3))[:,1,1]
Ustd = std(mean(uc.data[:,:,1,Tend-T:Tend], dims=(3))[:,:,1], dims=2)[:,1]

V = mean(vc.data[:,:,1,Tend-T:Tend], dims=(2,3))[:,1,1]
Vstd = std(mean(vc.data[:,:,1,Tend-T:Tend], dims=(3))[:,:,1], dims=2)[:,1]

fig = Figure(size = (800, 600));
ax = Axis(fig[1, 1], xlabel="x [km]", ylabel="Velocity [m s-1]") 

lines!(ax, xc, U, label="U")
band!(ax, xc, U+Ustd, U-Ustd)

lines!(ax, xc, V, label="V")
band!(ax, xc, V+Vstd, V-Vstd)

limits!(ax, 0, 90 , nothing, nothing)

axislegend(position = :rb)

save(figurepath*filename*"_velprofile.png", fig)