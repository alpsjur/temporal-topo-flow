"""
Validating Shallow Water model by recreating Brink (2010).
The original article can be found here: 
https://www.researchgate.net/publication/50405295_Topographic_rectification_in_a_forced_dissipative_barotropic_ocean
"""

using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.ShallowWaterModels
using Statistics
using Printf                   
using CUDA                     
using CairoMakie               

# Run on GPU (wow, fast!) if available. Else run on CPU
if CUDA.functional()
    architecture = GPU()
    @info "Running on GPU"
else
    architecture = CPU()
    @info "Running on CPU"
end

# Run name
name = "brink_2010-300"

# Grid parameters
dx =   1kilometer 
dy =   1kilometer 
Lx = 120kilometers
Ly =  90kilometers

# Simulation parameters           
Δt         =    2seconds     
tmax       =  200days      
outputtime =    3hours

# Forcing parameters
ρ   = 1e3
d   = 0.1
T   = 4days
R   = 5e-4  
switch = nothing

# Coriolis parameter
f = 1e-4

# Closure parameter
ν = 0
   
# Bathymetry parameters
hA = 0
h0 = 25meters
h1 = 100meters
h2 = 1000meters
x1 = 40kilometers 
x2 = 60kilometers

λ  = 45kilometers
hc = 59meters

# define total forcing
τx(x, y, t, u, v, h, p) = -p.R*u/h
τy(x, y, t, u, v, h, p) = -p.R*v/h + p.d*sin(p.ω*t)/(p.ρ*h)

# overwrite parameters and functions with configuration file, if provided
if length(ARGS)==1
    include("../"*ARGS[1])
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

# Set up parameters given to forcing function                   
τx_parameters = (; R)
τy_parameters = (; ρ, d, ω, R, switch)

# define forcing field from forcing functions
u_forcing = Forcing(τx, field_dependencies=(:u, :v, :h), parameters=τx_parameters)
v_forcing = Forcing(τy, field_dependencies=(:u, :v, :h), parameters=τy_parameters)


# define bathymetry functions
G(y, γ, k) = γ*sin(k*y)

function hᵢ(x, y, p)
    if x < p.x1
        h = p.hA + p.h0 + p.A*x + p.h1*G(y, p.γ, p.k)*x/p.x1
    elseif x < p.x2
        h = p.hA + p.h1 + B*(x-p.x1) + p.h1*G(y, p.γ, p.k)*(p.x2-x)/(p.x2-p.x1)
    else
        h = p.hA + p.h2
    end
    return h
end

hᵢ(x, y) = hᵢ(x, y, (; hA, h0, h1, h2, x1, x2, A, B, λ, k, γ))
b(x, y) = -hᵢ(x, y)


# Model parameters
gravitational_acceleration = 9.81
coriolis = FPlane(f=f)

# turbulence closure, by default set to zero 
closure = ShallowWaterScalarDiffusivity(ν = ν)

# Create model
model = ShallowWaterModel(; grid, coriolis, gravitational_acceleration,
                          momentum_advection = VectorInvariant(
                            #vorticity_scheme=WENO()
                            ),
                          bathymetry = b,
                          closure = closure,
                          formulation = VectorInvariantFormulation(),                  
                          forcing = (u=u_forcing,v=v_forcing),
                          )


# set initial conditions
set!(model, h=hᵢ)

# plot bathymetry 
figurepath = "../figures/brink/bathymetry/"
fig = Figure(size = (800, 800))
axis = Axis(fig[1,1], 
        aspect = DataAspect(),
        title = "Model bathymetry",
        xlabel = "x [m]",
        ylabel = "y [m]",
        )

depth = model.solution.h                    

# initialize simulations
simulation = Simulation(model, Δt=Δt, stop_time=tmax)

# logging simulation progress
start_time = time_ns()
progress(sim) = @printf(
    "i: %10d, sim time: % 12s, min(v): %4.3f ms⁻¹, max(v): %4.3f ms⁻¹, wall time: %12s\n",
    sim.model.clock.iteration,
    prettytime(sim.model.clock.time),
    minimum(v),
    maximum(v),
    prettytime(1e-9 * (time_ns() - start_time))
)

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10days/Δt))

#output
u, v, h = model.solution
bath = model.bathymetry
η = h + bath

# ∂b∂x = ∂x(bath)
# ∂b∂y = ∂y(bath)

# # momentum terms
# ∂η∂y = ∂y(η)
# ∂v∂x = ∂x(v)

# u∂v∂x = u*∂v∂x
# v∂u∂y = v*∂u∂y    # TODO do I need this therm for contour following stuff?

ω = Field(∂x(v) - ∂y(u))
ωu = Field(ω*u) 
ωv = Field(ω*v)

divωflux = Field(∂x(ωu) + ∂y(ωv))

fields = Dict("u" => u, "v" => v, 
              "h" => h, "omega" => ω,
              "omegau" => ωu, "omegav" => ωv,
              "divomegaflux" => divωflux
              )

simulation.output_writers[:field_writer] = NetCDFOutputWriter(model, fields, 
                        filename = "output/brink/netCDF/"*name*".nc",
                        schedule = AveragedTimeInterval(outputtime),
                        overwrite_existing = true
                        )


# simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, v, η, ω, ∂η∂y, ∂v∂x, u∂v∂x),
#                                                     schedule = AveragedTimeInterval(outputtime),
#                                                     filename = "output/brink/" * name * ".jld2",
#                                                     overwrite_existing = true)
# nothing

# simulation.output_writers[:bathymetry] = JLD2OutputWriter(model, (; bath, ∂b∂x, ∂b∂y),
#                                                     schedule = TimeInterval(tmax-Δt),
#                                                     filename = "output/brink/" * name * "_bathymetry.jld2",
#                                                     overwrite_existing = true)
# nothing

@info "Starting configuration " * name

run!(simulation)