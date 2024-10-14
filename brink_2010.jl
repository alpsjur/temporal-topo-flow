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

# Run on GPU (wow, fast!) if available. Else run on CPU
architecture = CUDA.functional() ? GPU() : CPU()
@info "Running on $(CUDA.functional() ? "GPU" : "CPU")"

# Run name
name = "brink_2010-300"

# Grid parameters
dx =   1kilometer 
dy =   1kilometers 
Lx =  90kilometers
Ly =  90kilometers

# Simulation parameters           
Δt   =    2seconds     
tmax =  200days      

# Forcing parameters
ρ   = 1e3
d   = 0.1
T   = 4days
R   = 5e-4  
   
# Bathymetry parameters
hA = 0
h0 = 25meters
h1 = 100meters
h2 = 1000meters
x1 = 40kilometers 
x2 = 60kilometers

λ  = 45kilometers
hc = 59meters

# overwrite parameters with configuration file, if provided
if length(ARGS)==1
    include(ARGS[1])
end

# parameters based on provided parameters
ω   = 2π/T
γ  = hc/h1
k  = 2π/λ
A  = (h1-h0)/x1
B  = (h2-h1)/(x2-x1)
Nx = Int(Lx/dx)
Ny = Int(Ly/dy)

# create grid
grid = RectilinearGrid(architecture,
                       size = (Nx, Ny),
                       x = (0, Lx), y = (0, Ly),
                       topology = (Bounded, Periodic, Flat),
                       )

# define forcing functions                       
@inline τS(t, p) = p.d*sin(p.ω*t)/p.ρ
@inline drag_u(x, y, t, u, v, h, p) = -p.R*u/h
@inline drag_v(x, y, t, u, v, h, p) = -p.R*v/h

# define total forcing
@inline τx(x, y, t, u, v, h, p) = drag_u(x, y, t, u, v, h, p) 
@inline τy(x, y, t, u, v, h, p) = drag_v(x, y, t, u, v, h, p) + τS(t, p)/h
u_forcing = Forcing(τx, field_dependencies=(:u, :v, :h), parameters=(; R))
v_forcing = Forcing(τy, field_dependencies=(:u, :v, :h), parameters =(; ρ, d, T, ω, R))


# define bathymetry functions
G(y, p) = p.γ*sin(p.k*y)

function hᵢ(x, y, p)
    if x < p.x1
        h = p.hA + p.h0 + p.A*x + p.h1*G(y, p)*x/p.x1
    elseif x < p.x2
        h = p.hA + p.h1 + B*(x-p.x1) + p.h1*G(y, p)*(p.x2-x)/(p.x2-p.x1)
    else
        h = p.hA + p.h2
    end
    return h
end

@inline hᵢ(x, y) = hᵢ(x, y, (; hA, h0, h1, h2, x1, x2, A, B, λ, k, γ))
@inline b(x, y) = -hᵢ(x, y)


# Model parameters
gravitational_acceleration = 9.81
coriolis = FPlane(f=10e-4)

# Define speed of gravity wave
c = sqrt(gravitational_acceleration*(hA + h2))


# Trying to implement gradient boundary condition
# auxiliary_u = XFaceField(grid)
# auxiliary_v = YFaceField(grid)
# auxiliary_h = CenterField(grid)

# function copy_fields!(sim)
#     model = sim.model
#     solution = model.solution
#     parent(auxiliary_u) .= parent(solution.u)
#     parent(auxiliary_v) .= parent(solution.v)
#     parent(auxiliary_h) .= parent(solution.h)
#     return nothing
# end

# @inline dhdx(j, k, grid, clock, model_fields, c) = -5e3
# @inbounds (auxiliary_u[90, j, k]*auxiliary_h[90, j, k]
#     -auxiliary_u[89, j, k]*auxiliary_h[89, j, k]
#     +auxiliary_v[90, j, k]*auxiliary_h[90, j, k]
#     -auxiliary_v[90, j-1, k]*auxiliary_h[90, j-1, k])/(1e3*c)

# radiative_bc = GradientBoundaryCondition(dhdx, discrete_form=true, parameters=c)


@inline hflux(y, t, h, p) = (h - p.hA - p.h2)*p.c
    

flux_bc = FluxBoundaryCondition(hflux, field_dependencies=:h, parameters=(; c, hA, h2))
h_bcs = FieldBoundaryConditions(FluxBoundaryCondition(nothing), east=flux_bc)

free_slip_bc = FluxBoundaryCondition(nothing)
free_slip_field_bcs = FieldBoundaryConditions(free_slip_bc)

# Create model
model = ShallowWaterModel(; grid, coriolis, gravitational_acceleration,
                          momentum_advection = VectorInvariant(
                            #vorticity_scheme=WENO()
                            ),
                          bathymetry = b,
                          boundary_conditions = (u = free_slip_field_bcs, 
                                                 v = free_slip_field_bcs, 
                                                 h = h_bcs
                                                 ),
                          #closure = ShallowWaterScalarDiffusivity(ν=1e-4, ξ=1e-4),
                          formulation = VectorInvariantFormulation(),                  
                          forcing = (u=u_forcing,v=v_forcing),
                          )


# set initial conditions
set!(model, h=hᵢ)

# plot bathymetry
figurepath = "figures/brink/"
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

# Trying to implement gradient boundary condition
# using Oceananigans: UpdateStateCallsite
#simulation.callbacks[:copy_fields] = Callback(copy_fields!, callsite=UpdateStateCallsite())


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

@info "Starting configuration " * name

run!(simulation)