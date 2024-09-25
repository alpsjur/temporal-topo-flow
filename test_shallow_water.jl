"""
units [L] = m, [T] = s,


There are two different formulations of the Shallow water model:
ConservativeFormulation : Solving for uh and vh, this does currently not work with bathymetry
VectorInvariantFormulation : Solving for u and v

Total depth h is defined as 
h = η - b, 
where η is the free surface, b is the bathymetry, both relative to the ocean at rest 

The forcing is added to the right side of the Shallow Water equation
Linear bottom drag is included in the forcing.

PROBLEM: get very high velocities, negative h 
"""

using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.ShallowWaterModels
using Statistics
using Printf                   # formatting text
using CUDA                     # for running on GPU
using CairoMakie               # plotting

# output filename
filename = "test_shallow_water"


# Run on GPU (wow, fast!) if available. Else run on CPU
if CUDA.functional()
    architecture = GPU()
    @info "Running on GPU"
else
    architecture = CPU()
    @info "Running on CPU"
end

# bathymetric parameters based on Haidvogel & Brink (1986), with adjustments
const h₀ = 500                      # minimum depth, need to change this?
const α  = 1e-5                     # e-folding scale
const δ  = 0.2                      # corresponding to a midshelf depth pertubation of ~40 m
const k  = 2π/(150e3)               # wave length of 150km
const θ  = 0                        # Some kind of phase shift? Relevant for non-monocromatic bathymetry?
const Lx, Ly = 416e3, 90e3          # domain length

# Grid parameters
const dx = 2kilometers                  # Grid spacing in x-direction
const dy = 2kilometers                  # Grid spacing in y-direction
const Nx = Int(Lx/dx)                   # Number of grid cells in x-direction
const Ny = Int(Ly/dy)                   # Number of grid cells in y-direction

const ρ₀ = 1026.5                 # mean density


# Forcing parameters
const τ = 0.05/ρ₀                      # Wind stress (kinematic forcing)

# Bottom friction
const Cd = 3e-3#0.002                      # Quadratic drag coefficient []

gravitational_acceleration = 9.81
coriolis = FPlane(f=10e-4)

tmax = 120days                
Δt   = 10second                 

# create grid
grid = RectilinearGrid(architecture,
                       size = (Nx, Ny),
                       x = (0, Lx), y = (0, Ly),
                       topology = (Periodic, Bounded, Flat))


# define bathymetry
# hᵢ(x, y) = h₀ + (h₁-h₀)*y/Ly

# define bathymetry, equation 2.4 in Haidvogel & Brink
hᵢ(x, y) = h₀*exp(α*y + sin(π*y/Ly)*δ*sin(k*x+θ)) 

b(x, y) = -hᵢ(x, y)
            

# define bottom drag for vector invariant formulation
drag_u(x, y, t, u, v, h) = -Cd*√(u^2+v^2)*u/h
drag_v(x, y, t, u, v, h) = -Cd*√(u^2+v^2)*v/h

# define surface stress functions
increasing_surface_stress(t) = τ*tanh(t/(10days))
varying_surface_stress(t) = τ*sin(2π*t/(20days))

# define total forcing
τx(x, y, t, u, v, h) = drag_u(x, y, t, u, v, h) + varying_surface_stress(t)/h
τy(x, y, t, u, v, h) = drag_v(x, y, t, u, v, h)     
u_forcing = Forcing(τx, field_dependencies=(:u, :v, :h))
v_forcing = Forcing(τy, field_dependencies=(:u, :v, :h))

# boundary conditions 
no_slip_bc = ValueBoundaryCondition(0.0)
no_slip_field_bcs = FieldBoundaryConditions(no_slip_bc)

model = ShallowWaterModel(; grid, coriolis, gravitational_acceleration,
                          momentum_advection = VectorInvariant(),
                          bathymetry = b,
                          closure = ShallowWaterScalarDiffusivity(ν=1e-4, ξ=1e-4),
                          formulation = VectorInvariantFormulation(),                  
                          forcing = (u=u_forcing,v=v_forcing),
                          #boundary_conditions=(u=no_slip_field_bcs, v=no_slip_field_bcs),
                          )


# set initial conditions
set!(model, h=hᵢ)


# plot bathymetry
figurepath = "figures/"
fig = Figure()
axis = Axis(fig[1,1], 
        aspect = DataAspect(),
        title = "Model bathymetry",
        xlabel = "x [m]",
        ylabel = "y [m]",
        )

depth = model.bathymetry

hm = heatmap!(depth, colormap=:deep)
Colorbar(fig[1, 2], hm, label = "Depth [m]")

save(figurepath*"bathymetry.png", fig)
                         

# initialize simulations
simulation = Simulation(model, Δt=Δt, stop_time=tmax)


# logging simulation progress
start_time = time_ns()
progress(sim) = @printf(
    "i: %d, sim time: % 8s, min(u): %.3f ms⁻¹, max(u): %.3f ms⁻¹, min(h): %.1f m, max(h): %.1f m, wall time: %s\n",
    sim.model.clock.iteration,
    prettytime(sim.model.clock.time),
    minimum(u),
    maximum(u),
    minimum(h),
    maximum(h),
    prettytime(1e-9 * (time_ns() - start_time))
)

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1day/Δt))

# output

u, v, h = model.solution
#bath = model.bathymetry
simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, v, h),
                                                    schedule = AveragedTimeInterval(12hours),
                                                    filename = "output/" * filename * ".jld2",
                                                    overwrite_existing = true)
nothing

run!(simulation)