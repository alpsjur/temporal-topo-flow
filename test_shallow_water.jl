"""
Model experiment based on the basic case in Haidvogel & Brink (1986) https://doi.org/10.1175/1520-0485(1986)016%3C2159:MCDBTD%3E2.0.CO;2

units [L] = m, [T] = s,


There are two different formulations of the Shallow water model:
ConservativeFormulation : Solving for uh and vh
VectorInvariantFormulation : Solving for u and v

Total depth h is defined as 
h = η - b, 
where η is the free surface, b is the bathymetry, both relative to the ocean at rest 

The forcing is added to the right side of the Shallow Water equation
Linear bottom drag is included in the forcing.

PROBLEM: get very high velocities, negative h 
"""

using Oceananigans
using Oceananigans.Models.ShallowWaterModels
using Statistics

# bathymetric parameters
h₀ = 20                       # minimum depth
α  = 5.2e-5                   # e-folding scale
δ  = 0.636                    # corresponding to a midshelf depth pertubation of ~40 m
k  = 2π/(150e3)               # wave length of 150km
θ  = 0                        # Some kind of phase shift? Relevant for non-monocromatic bathymetry?
Lx, Ly = 450e3, 90e3          # domain length


Nx, Ny = 128, 128           # number of grid points
ρ₀ = 1026.5                 # mean density

# forcing parameters
τ = 1e-1/ρ₀                      # 1 dyn cm⁻2  = 10^-5 N 10^4 m^-2 
ω = 2π/(60*60*24*10)          # period of 10 days              
μ = 0                         # Some kind of phase shift? Relevant for non-monocromatic forcing?

r = 3e-2                      # Bottom drag coefficient m/s

gravitational_acceleration = 9.81
coriolis = FPlane(f=10e-4)

tmax = 60*60*24*100              # integrated for 100 days
Δt   = 60*60*24*10/1e1           # 1/100 of wind forcing period 

# create grid
grid = RectilinearGrid(CPU(),
                       size = (Nx, Ny),
                       x = (0, Lx), y = (0, Ly),
                       topology = (Periodic, Bounded, Flat))


# define bathymetry, equation 2.4 in Haidvogel & Brink
hᵢ(x, y) = h₀*exp(α*y + sin(π*y/Ly)*δ*sin(k*x+θ)) 
b(x, y) = -hᵢ(x, y)
            

# define bottom drag for vector invariant formulation
drag_u(x, y, t, u, v, r) = -r*u/hᵢ(x, y)
drag_v(x, y, t, u, v, r) = -r*v/hᵢ(x, y)

# define bottom drag for conservative formulation
drag_uh(x, y, t, uh, vh, r) = -r*uh/hᵢ(x, y)
drag_vh(x, y, t, uh, vh, r) = -r*vh/hᵢ(x, y)

# define forcing, equation 2.5 in Haidvogel & Brink, + a linear drag
τx(x, y, t, u, v, r) = τ*sin(ω*t+μ)/hᵢ(x, y) + drag_u(x, y, t, u, v, r) 
τy(x, y, t, u, v, r) = drag_v(x, y, t, u, v, r)     
u_forcing = Forcing(τx, parameters=r, field_dependencies=(:u, :v))
v_forcing = Forcing(τy, parameters=r, field_dependencies=(:u, :v))

τxh(x, y, t, uh, vh, r) = τ*sin(ω*t+μ) + drag_uh(x, y, t, uh, vh, r)
τyh(x, y, t, uh, vh, r) = drag_vh(x, y, t, uh, vh, r)
uh_forcing = Forcing(τxh, parameters=r, field_dependencies=(:uh, :vh))
vh_forcing = Forcing(τyh, parameters=r, field_dependencies=(:uh, :vh))


# boundary conditions 
no_slip_bc = ValueBoundaryCondition(0.0)
no_slip_field_bcs = FieldBoundaryConditions(no_slip_bc)

hmin_bc = ValueBoundaryCondition(h₀)
hmax_bc = ValueBoundaryCondition(2150)
h_bc = FieldBoundaryConditions(south=hmin_bc, north=hmax_bc)


model = ShallowWaterModel(; grid, coriolis, gravitational_acceleration,
                          timestepper = :RungeKutta3,
                          bathymetry = b,
                          closure = ShallowWaterScalarDiffusivity(ν=1e-6, ξ=1e-6),
                          
                        #   formulation = ConservativeFormulation(),
                        #   momentum_advection = WENO(),
                        #   forcing = (uh=uh_forcing,vh=vh_forcing),
                        #   boundary_conditions=(uh=no_slip_field_bcs, vh=no_slip_field_bcs),

                          formulation = VectorInvariantFormulation(),
                          momentum_advection = VectorInvariant(),
                          forcing = (u=u_forcing,v=v_forcing),
                          boundary_conditions=(u=no_slip_field_bcs, v=no_slip_field_bcs),
                          )


# set initial conditions
set!(model, h=hᵢ)
                         

simulation = Simulation(model, Δt=Δt, stop_time=tmax)

# Logging simulation progress
progress(sim) = @info string("Iteration: ", iteration(sim), ", time: ", time(sim))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(50))

# output
filename = "alternating_wind_basic_case"
u, v, h = model.solution
#bath = model.bathymetry
simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, v, h),
                                                    schedule = TimeInterval(Δt),
                                                    filename = "output/" * filename * ".jld2",
                                                    overwrite_existing = true)
nothing

run!(simulation)