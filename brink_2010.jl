"""
Validating Shallow Water by recreating Brink (2010)
"""

using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.ShallowWaterModels
using Statistics
using Printf                   # formatting text
using CUDA                     # for running on GPU
using CairoMakie               # plotting

name = "brink_2010-300"

# Run on GPU (wow, fast!) if available. Else run on CPU
if CUDA.functional()
    architecture = GPU()
    @info "Running on GPU"
else
    architecture = CPU()
    @info "Running on CPU"
end

# Grid parameters
dx =   1kilometer 
dy =   1kilometers 
Lx =  90kilometers
Ly =  90kilometers
Nx = Int(Lx/dx)
Ny = Int(Ly/dy)

# Simulation parameters           
Δt   =    2seconds     
tmax =  200days       
#tmax =  3*Δt       

# create grid
grid = RectilinearGrid(architecture,
                       size = (Nx, Ny),
                       x = (0, Lx), y = (0, Ly),
                       topology = (Bounded, Periodic, Flat),
                       )

# Forcing parameters
const ρ   = 1e3
const d   = 0.1
const T   = 4days
const ω   = 2π/T
const R   = 5e-4  

τS(t) = d*sin(ω*t)/ρ
drag_u(x, y, t, u, v, h) = -R*u/h
drag_v(x, y, t, u, v, h) = -R*v/h

# define total forcing
τx(x, y, t, u, v, h) = drag_u(x, y, t, u, v, h) 
τy(x, y, t, u, v, h) = drag_v(x, y, t, u, v, h) + τS(t)/h
u_forcing = Forcing(τx, field_dependencies=(:u, :v, :h))
v_forcing = Forcing(τy, field_dependencies=(:u, :v, :h))

# Bathymetry parameters
const hA = 0
const h0 = 25meters
const h1 = 100meters
const h2 = 1000meters
const x1 = 40kilometers 
const x2 = 60kilometers
const A  = (h1-h0)/x1
const B  = (h2-h1)/(x2-x1)

const λ  = 45kilometers
const k  = 2π/λ
const hc = 59meters
const γ  = hc/h1

G(y) = γ*sin(k*y)

function hᵢ(x, y)
    if x < x1
        h = hA + h0 + A*x + h1*G(y)*x/x1
    elseif x < x2
        h = hA + h1 + B*(x-x1) + h1*G(y)*(x2-x)/(x2-x1)
    else
        h = hA + h2
    end
    return h
end

b(x, y) = -hᵢ(x, y)

# Model parameters
gravitational_acceleration = 9.81
coriolis = FPlane(f=10e-4)

# Create model
model = ShallowWaterModel(; grid, coriolis, gravitational_acceleration,
                          momentum_advection = VectorInvariant(),
                          bathymetry = b,
                          #closure = ShallowWaterScalarDiffusivity(ν=1e-4, ξ=1e-4),
                          formulation = VectorInvariantFormulation(),                  
                          forcing = (u=u_forcing,v=v_forcing),
                          )


# set initial conditions
set!(model, h=hᵢ)

# plot bathymetry
figurepath = "figures/"
fig = Figure(size = (800, 800))
axis = Axis(fig[1,1], 
        aspect = DataAspect(),
        title = "Model bathymetry",
        xlabel = "x [m]",
        ylabel = "y [m]",
        )

depth = model.solution.h



hm = heatmap!(axis, depth, colormap=:deep)
Colorbar(fig[1, 2], hm, label = "Depth [m]")
#contour!(axis, depth, levels=0:100:1000)
save(figurepath*name*"_bathymetry.png", fig)
                         

# initialize simulations
simulation = Simulation(model, Δt=Δt, stop_time=tmax)


# logging simulation progress
start_time = time_ns()
progress(sim) = @printf(
    "i: %10d, sim time: % 12s, min(u): %4.3f ms⁻¹, max(u): %4.3f ms⁻¹, wall time: %12s\n",
    sim.model.clock.iteration,
    prettytime(sim.model.clock.time),
    minimum(u),
    maximum(u),
    prettytime(1e-9 * (time_ns() - start_time))
)

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10day/Δt))

#output

u, v, h = model.solution
bath = model.bathymetry
η = h + bath

# momentum terms
∂η∂y = ∂y(η)
∂v∂x = ∂x(v)
u∂v∂x = u*∂v∂x


simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, v, η, ∂η∂y, ∂v∂x, u∂v∂x),
                                                    schedule = AveragedTimeInterval(1hours),
                                                    filename = "output/" * name * ".jld2",
                                                    overwrite_existing = true)
nothing

simulation.output_writers[:bathymetry] = JLD2OutputWriter(model, (; bath),
                                                    schedule = TimeInterval(tmax-Δt),
                                                    filename = "output/" * name * "_bathymetry.jld2",
                                                    overwrite_existing = true)
nothing

run!(simulation)