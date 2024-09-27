"""
Validation criteria
- Wave speed and shape : The waves should propagate outward symmetrically 
    from the initial Gaussian bump. Compare the numerical wave speed to 
    the theoretical wave speed given by: √(gh) ≈ 3.13 m/s 
- Wave amplitude decays : The initial bump should spread and decrease in 
    amplitude as the wave propagates, conserving the total water mass.
- Energy conservation : Track the total energy (potential + kinetic). 
    While some numerical dissipation is expected, the energy should be 
    largely conserved.
"""

using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.ShallowWaterModels
using Statistics
using Printf                   # formatting text
using CUDA                     # for running on GPU
using CairoMakie               # plotting

# output filename
filename = "propagating_wave"


# Run on GPU (wow, fast!) if available. Else run on CPU
if CUDA.functional()
    architecture = GPU()
    @info "Running on GPU"
else
    architecture = CPU()
    @info "Running on CPU"
end

# Grid parameters
const Lx = 100meters
const Ly = 100meters          
const dx = 0.1meters                  # Grid spacing in x-direction
const dy = 0.1meters                  # Grid spacing in y-direction
const Nx = Int(Lx/dx)                   # Number of grid cells in x-direction
const Ny = Int(Ly/dy)                   # Number of grid cells in y-direction



# bump parameters
const h₀ =    1meters           # depth without bump
const A  =  0.1meters           # bump amplitude
const x₀ =   50meters            # initial x position
const y₀ =   50meters            # initial y position
const σ  =    5meters            # bump width parameter


gravitational_acceleration = 9.81
const ρ = 1e3
const g = 9.81

tmax =   60seconds           
Δt   = 0.01second                 

# create grid
grid = RectilinearGrid(architecture,
                       size = (Nx, Ny),
                       x = (0, Lx), y = (0, Ly),
                       topology = (Bounded, Bounded, Flat))


# define gausian bump
η₀(x, y) = A*exp(-((x-x₀)^2 + (y-y₀)^2)/(2*σ^2))
hᵢ(x, y) = h₀ + η₀(x, y)

model = ShallowWaterModel(; grid, 
                          gravitational_acceleration,
                          momentum_advection = VectorInvariant(),
                          closure = ShallowWaterScalarDiffusivity(ν=1e-4),
                          formulation = VectorInvariantFormulation(),                  
                          )


# set initial conditions
set!(model, h=hᵢ)

# initialize simulations
simulation = Simulation(model, Δt=Δt, stop_time=tmax)


# logging simulation progress
start_time = time_ns()
progress(sim) = @printf(
    "i: %d, sim time: % 8s, wall time: %s\n",
    sim.model.clock.iteration,
    prettytime(sim.model.clock.time),
    prettytime(1e-9 * (time_ns() - start_time))
)

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10seconds/Δt))

# output
u, v, h = model.solution

vel = Field(u^2+v^2)
KE = Field(Integral(0.5*vel, dims=(1,2)))
PE = Field(Integral(0.5*g*(h-h₀)^2, dims=(1,2)))

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, v, h, KE, PE),
                                                    schedule = TimeInterval(Δt*10),
                                                    filename = "output/" * filename * ".jld2",
                                                    overwrite_existing = true)
nothing

run!(simulation)


# Plot result
h_timeseries = FieldTimeSeries("output/" * filename * ".jld2", "h")
KE_timeseries = FieldTimeSeries("output/" * filename * ".jld2", "KE")
PE_timeseries = FieldTimeSeries("output/" * filename * ".jld2", "PE")

times = h_timeseries.times

x, y, z = nodes(h_timeseries)

set_theme!(Theme(fontsize = 24))

fig = Figure(size = (600, 600))

ax1 = Axis(fig[2, 1]; xlabel = "x", ylabel = "y", aspect = DataAspect())
ax2 = Axis(fig[3, 1]; xlabel = "t", ylabel = "E")

lines!(ax2, times, KE_timeseries[1,1,1,:]; color = :blue)
lines!(ax2, times, PE_timeseries[1,1,1,:]; color = :tomato)
lines!(ax2, times, KE_timeseries[1,1,1,:]+PE_timeseries[1,1,1,:]; color = :gray)

hmax = maximum(h_timeseries)
hmin = minimum(h_timeseries)

n = Observable(1)
heatmap!(ax1, h; colormap = :deep, colorrange = (hmin, hmax))

title = @lift "t = " * string(round(times[$n], digits=2))
Label(fig[1, 1], title, fontsize=24, tellwidth=false)

frames = 1:length(times)
CairoMakie.record(fig, "animations/" * filename * ".mp4", frames, framerate=10) do i
    n[] = i
end