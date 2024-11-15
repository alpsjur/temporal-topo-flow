# Import necessary packages for visualization, ocean simulation, formatted output, and data handling
using CairoMakie        # For creating visualizations and animations
using Oceananigans      # For simulating ocean dynamics
using Oceananigans.Units
using Printf            # For formatted string output
using JLD2              # For saving and loading simulation data in Julia format
using Statistics
using ColorSchemes
using Random 
using RollingFunctions  

name = "329.jl"
xvals = (30, 35, 40, 45, 50)
#xvals = (45)

Tinc = 12*8

filepath = "output/brink/"
figurepath = "figures/brink/"
cmap = ColorSchemes.batlow

# Forcing parameters
ρ   = 1e3
d   = 0.1
T   = 4days
R   = 5e-4  
L   = 90kilometers

outputtime = 3hours
Δt = 2seconds

#config = "period-doubling/"*name
config = name
include("configs/brink/"*config)

ω   = 2π/T
window = Integer(outputtime/Δt)

# Load time series data from the saved JLD2 file
full_output = FieldDataset(filepath *  name * ".jld2")
u = full_output.u # u-component of velocity field
v = full_output.v  # v-component of velocity field
η = full_output.η  # sea surface height field
h = - FieldDataset(filepath *  name * "_bathymetry.jld2").bath[:,:,1,1]
t = collect(u.times)[1:end]
t_hr = collect(0:2:t[end])
t = t[1:end-1]/1day

figscatter = Figure(size = (600, 600))
axscatter = Axis(figscatter[1, 1], ylabel="numerical circ [cm s-1]", xlabel="analytical circ [cm s-1]", 
            aspect=DataAspect()
            ) 

figts = Figure(size = (1200, 400))
axts = Axis(figts[1, 1], xlabel="time [days]", ylabel="circ [cm s-1]", 
            ) 

lines!(axscatter,[-10, 10], [-10, 10], color=:gray)

n = length(xvals)
colors = [ColorSchemes.get(cmap, 1-i / (n - 1)) for i in 0:n-1]
#colors=[:blue]


for i in eachindex(xvals)
#for xval in xvals
    xval = xvals[i]
    color = colors[i]

    H = mean(h[xval,:])

    #analytical = -d/ρ * (R*sin.(ω*t)-H*ω*cos.(ω*t)+exp.(-R*t/H)*H*ω)/(H^2*ω^2+R^2)
    analytical_hr = -d/ρ * (R*sin.(ω*t_hr)-H*ω*cos.(ω*t_hr)+exp.(-R*t_hr/H)*H*ω)/(H^2*ω^2+R^2)
    analytical = rollmean(analytical_hr, window)[1:window:end]*1e2

    numerical = -mean(v[xval,:,1,:], dims=(1))[1,1:end-1]*1e2

    lines!(axts, t, numerical,  color=color, label = string(xval))
    lines!(axts, t, analytical,  color=color, linestyle=:dash)

    # lines!(axts, t[1: Tinc], numerical[1: Tinc], color=color, label = string(xval))
    # lines!(axts, t[1: Tinc], analytical[1: Tinc],  color=color, linestyle=:dash)

    # lines!(axts, t[end-Tinc: end], numerical[end-Tinc: end], color=color, label = string(xval))
    # lines!(axts, t[end-Tinc: end], analytical[end-Tinc: end],  color=color, linestyle=:dash)
    

    scatter!(axscatter, analytical, numerical, 
        label = string(xval),
        markersize = 4,
        alpha = 0.7, 
        color=color,
        )

end

axislegend(axscatter, position = :rb)
axislegend(axts, position = :rb)

save(figurepath*name*"_scatter.png", figscatter)
save(figurepath*name*"_analytical_ts.png", figts)