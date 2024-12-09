# Import necessary packages for visualization, ocean simulation, formatted output, and data handling
using CairoMakie        # For creating visualizations and animations
using Oceananigans      # For simulating ocean dynamics
using Printf            # For formatted string output
using JLD2              # For saving and loading simulation data in Julia format
using Statistics
using ColorSchemes
using Random 

# Define file path and name of the saved simulation output
filepath = "output/brink/"
figurepath = "figures/brink/"

fig = Figure(size = (800, 600));
ax = Axis(fig[1, 1], xlabel="x [km]", ylabel="Velocity [m s-1]") 
cmap = ColorSchemes.batlow

T = 4*2*2
basefilename = "brink_2010-300-closure" 
closures = ("1", "2", "3", "4", "5", "6", "7", "8", "9")
n = length(closures)
colors = [ColorSchemes.get(cmap, i / (n - 1)) for i in 0:n-1]

# Shuffle the list of colors
#colors = shuffle(colors)

for i in eachindex(closures)
    closure = closures[i]
    filename = basefilename * closure
    color = colors[i]

    # Load time series data from the saved JLD2 file
    full_output = FieldDataset(filepath * filename * ".jld2")
    u = full_output.u # u-component of velocity field
    v = full_output.v  # v-component of velocity field
    η = full_output.η  # sea surface height field

    # Extract time points from the simulation data
    times = u.times

    # Get coordinate arrays from the first timestep of height field data
    # `xv`, `yv`, `zv` are not used, so only `xc`, `yc`, `zc` are extracted
    xc, yc, zc = nodes(η[1]) 
    xc = xc/1e3
    yc = yc/1e3

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



    lines!(ax, xc, U, linestyle=:dash, color=color)
    #band!(ax, xc, U+Ustd, U-Ustd)

    lines!(ax, xc, V, label=closure, color=color)
    #band!(ax, xc, V+Vstd, V-Vstd)

    limits!(ax, 0, 90 , nothing, nothing)
end

axislegend(position = :rb)

save(figurepath*"brink_test_closures.png", fig)