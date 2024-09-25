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
const Lx = 48e3
const Ly = 48e3          
const dx = 2kilometers                  # Grid spacing in x-direction
const dy = 2kilometers                  # Grid spacing in y-direction
const Nx = Int(Lx/dx)                   # Number of grid cells in x-direction
const Ny = Int(Ly/dy)                   # Number of grid cells in y-direction


gravitational_acceleration = 9.81
coriolis = FPlane(f=10e-4)

tmax = 10days                
Δt   = 10second                 

# create grid
grid = RectilinearGrid(architecture,
                       size = (Nx, Ny),
                       x = (0, Lx), y = (0, Ly),
                       topology = (Bounded, Bounded, Flat))


# define water column depth right after dam breakage
function hᵢ(x, y)
    if x > Lx/2
        return 500
    else 
        return 550
    end
end

model = ShallowWaterModel(; grid, 
                          #coriolis, 
                          gravitational_acceleration,
                          momentum_advection = VectorInvariant(),
                          closure = ShallowWaterScalarDiffusivity(ν=1e-4, ξ=1e-4),
                          formulation = VectorInvariantFormulation(),                  
                          )


# set initial conditions
set!(model, h=hᵢ)


# initialize simulations
simulation = Simulation(model, Δt=Δt, stop_time=tmax)


# logging simulation progress
start_time = time_ns()
progress(sim) = @printf(
    "i: %d, sim time: % 8s, min(u): %.3f ms⁻¹, max(u): %.3f ms⁻¹, min(h): %.1f m, max(h): %.1f m, wall time: %s\n",
    sim.model.clock.iteration,
    prettytime(sim.model.clock.time),
    minimum(s),
    maximum(s),
    minimum(h),
    maximum(h),
    prettytime(1e-9 * (time_ns() - start_time))
)

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1day/Δt))

# output
u, v, h = model.solution
s = sqrt(u^2 + v^2)
simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, v, h),
                                                    schedule = AveragedTimeInterval(1hours),
                                                    filename = "output/" * filename * ".jld2",
                                                    overwrite_existing = true)
nothing

run!(simulation)