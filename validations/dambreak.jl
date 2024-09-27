using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.ShallowWaterModels
using Statistics
using Printf                   # formatting text
using CUDA                     # for running on GPU
using CairoMakie               # plotting

# output filename
filename = "dambreak"


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
const Ly =  10meters          
const dx = 0.1meters                  # Grid spacing in x-direction
const dy = 0.1meters                  # Grid spacing in y-direction
const Nx = Int(Lx/dx)                   # Number of grid cells in x-direction
const Ny = Int(Ly/dy)                   # Number of grid cells in y-direction


gravitational_acceleration = 9.81

tmax =  60seconds           
Δt   = 0.01second                 

# create grid
grid = RectilinearGrid(architecture,
                       size = (Nx, Ny),
                       x = (0, Lx), y = (0, Ly),
                       topology = (Bounded, Bounded, Flat))


# define water column depth right after dam breakage
function hᵢ(x, y)
    if x > Lx/2
        return 1
    else 
        return 0.01
    end
end

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
simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, v, h),
                                                    schedule = TimeInterval(Δt*10),
                                                    filename = "output/" * filename * ".jld2",
                                                    overwrite_existing = true)
nothing

run!(simulation)


# Plot result
h_timeseries = FieldTimeSeries("output/" * filename * ".jld2", "h")

times = h_timeseries.times

x, y, z = nodes(h_timeseries)

set_theme!(Theme(fontsize = 24))

fig = Figure(size = (1200, 400))

ax = Axis(fig[2, 1]; xlabel = "x [m]", ylabel = "y [m]", aspect = DataAspect())

n = Observable(1)
hₙ = @lift interior(h_timeseries[$n], :, :)

hm = heatmap!(ax, x, y, hₙ; colormap = :deep, colorrange = (0, 1))
Colorbar(fig[2, 2], hm, label="water height [m]")

title = @lift "t = " * string(round(times[$n], digits=2))
Label(fig[1, 1], title, fontsize=24, tellwidth=false)

frames = 1:length(times)
CairoMakie.record(fig, "animations/" * filename * ".mp4", frames, framerate=10) do i
    n[] = i
end