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

# bathymetric parameters
const h₀ = 1000                      # minimum depth
const h₁ = 1200                      # maximum depth

# Grid parameters
const Lx = 416kilometers                # Domain length in x-direction
const Ly = 208kilometers                # Domain length in y-direction

const dx = 2kilometers                  # Grid spacing in x-direction
const dy = 2kilometers                  # Grid spacing in y-direction
const Nx = Int(Lx/dx)                   # Number of grid cells in x-direction
const Ny = Int(Ly/dy)                   # Number of grid cells in y-direction

const ρ₀ = 1026.5                 # mean density

# forcing parameters
const τ = 5e-2/ρ₀                 # 1 dyn cm⁻2  = 10^-5 N 10^4 m^-2 

const r = 3e-3                    # Bottom drag coefficient m/s

gravitational_acceleration = 9.81
coriolis = FPlane(f=10e-4)

tmax = 100days                
Δt   = 10second                 

# create grid
grid = RectilinearGrid(GPU(),
                       size = (Nx, Ny),
                       x = (0, Lx), y = (0, Ly),
                       topology = (Periodic, Bounded, Flat))


# define bathymetry
hᵢ(x, y) = h₀ + (h₁-h₀)*y/Ly
b(x, y) = -hᵢ(x, y)
            

# define bottom drag for vector invariant formulation
drag_u(x, y, t, u, v) = -r*u/(h₀ + (h₁-h₀)*y/Ly)
drag_v(x, y, t, u, v) = -r*v/(h₀ + (h₁-h₀)*y/Ly)


# define forcing, equation 2.5 in Haidvogel & Brink, + a linear drag
τx(x, y, t, u, v) = drag_u(x, y, t, u, v) + τ*tanh(t/(10days))/(h₀ + (h₁-h₀)*y/Ly)
τy(x, y, t, u, v) = drag_v(x, y, t, u, v)     
u_forcing = Forcing(τx, field_dependencies=(:u, :v))
v_forcing = Forcing(τy, field_dependencies=(:u, :v))

# boundary conditions 
no_slip_bc = ValueBoundaryCondition(0.0)
no_slip_field_bcs = FieldBoundaryConditions(no_slip_bc)

# hmin_bc = ValueBoundaryCondition(h₀)
# hmax_bc = ValueBoundaryCondition(h₁)
# h_bc = FieldBoundaryConditions(south=hmin_bc, north=hmax_bc)


model = ShallowWaterModel(; grid, coriolis, gravitational_acceleration,
                          momentum_advection = VectorInvariant(),
                          bathymetry = b,
                          closure = ShallowWaterScalarDiffusivity(ν=1e-4, ξ=1e-4),
                          formulation = VectorInvariantFormulation(),                  # Only this working
                          forcing = (u=u_forcing,v=v_forcing),
                          #boundary_conditions=(u=no_slip_field_bcs, v=no_slip_field_bcs),
                          )


# set initial conditions
set!(model, h=hᵢ)
                         

simulation = Simulation(model, Δt=Δt, stop_time=tmax)


# logging simulation progress
start_time = time_ns()
progress(sim) = @printf(
    "i: %d, sim time: % 8s, min(u): %.3f ms⁻¹, max(u): %.3f ms⁻¹, min(h): %.1f m, max(h): %.1f m\n",#, wall time: %s\n",
    sim.model.clock.iteration,
    prettytime(sim.model.clock.time),
    minimum(u),
    maximum(u),
    minimum(h),
    maximum(h),
    #prettytime(1e-9 * (time_ns() - start_time))
)

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1day/Δt))

# output
filename = "alternating_wind_basic_case"
u, v, h = model.solution
#bath = model.bathymetry
simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, v, h),
                                                    schedule = AveragedTimeInterval(1hour),
                                                    filename = "output/" * filename * ".jld2",
                                                    overwrite_existing = true)
nothing

run!(simulation)