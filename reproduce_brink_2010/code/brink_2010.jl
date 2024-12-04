"""
Validating Shallow Water model by recreating Brink (2010).
The original article can be found here:
https://www.researchgate.net/publication/50405295_Topographic_rectification_in_a_forced_dissipative_barotropic_ocean

Parameters are for run 300. 
"""

using Oceananigans                    # For ocean dynamics simulation
using Oceananigans.Units              # For unit handling
using Oceananigans.Models.ShallowWaterModels  # Shallow water modeling
using Statistics                      # For statistical calculations
using Printf                          # For formatted output
using CUDA                            # For GPU support
using CairoMakie                      # For visualizations

# Determine computational architecture (GPU or CPU)
if CUDA.functional()
    architecture = GPU()              # Use GPU if available
    @info "Running on GPU"
else
    architecture = CPU()              # Default to CPU
    @info "Running on CPU"
end

# Simulation name
name = "brink_2010-300"

# Grid parameters
dx = 1kilometer                       # Grid spacing in x-direction
dy = 1kilometer                       # Grid spacing in y-direction
Lx = 120kilometers                    # Domain size in x-direction
Ly = 90kilometers                     # Domain size in y-direction

# Simulation parameters
Δt = 2seconds                         # Timestep size
tmax = 32days                         # Simulation duration
outputtime = 1day                     # Interval for output writing

# Forcing parameters
ρ = 1e3                               # Density (kg/m³)
d = 0.1                               # Forcing amplitude
T = 4days                             # Forcing period
R = 5e-4                              # Bottom friction coefficient

# Coriolis parameter
f = 1e-4                              # Coriolis frequency

# Closure (turbulence/diffusion) parameter
ν = 0                                 # Default viscosity (no diffusion)

# Bathymetry parameters
hA = 0                                # Base height
h0 = 25meters                         # Minimum depth
h1 = 100meters                        # Intermediate depth
h2 = 1000meters                       # Maximum depth
x1 = 40kilometers                     # Transition point 1
x2 = 60kilometers                     # Transition point 2
λ = 45kilometers                      # Wavelength of bathymetry feature
hc = 59meters                         # Critical depth for G function

# Define total forcing functions
τx(x, y, t, u, v, h, p) = -p.R * u / h
τy(x, y, t, u, v, h, p) = -p.R * v / h + p.d * sin(p.ω * t) / (p.ρ * h)

# Overwrite parameters and functions with configuration file, if provided
if length(ARGS) == 1
    include(ARGS[1])
end

# Derived parameters
ω = 2π / T                            # Angular frequency of forcing
γ = hc / h1                           # Bathymetry scaling
k = 2π / λ                            # Wavenumber
A = (h1 - h0) / x1                    # Bathymetry slope 1
B = (h2 - h1) / (x2 - x1)             # Bathymetry slope 2
Nx = Int(Lx / dx)                     # Number of grid points in x
Ny = Int(Ly / dy)                     # Number of grid points in y

# Create grid
grid = RectilinearGrid(architecture,
                       size=(Nx, Ny),
                       x=(0, Lx), y=(0, Ly),
                       topology=(Bounded, Periodic, Flat))

# Set up parameters for forcing functions
τx_parameters = (; R)
τy_parameters = (; ρ, d, ω, R)

# Define forcing fields
u_forcing = Forcing(τx, field_dependencies=(:u, :v, :h), parameters=τx_parameters)
v_forcing = Forcing(τy, field_dependencies=(:u, :v, :h), parameters=τy_parameters)

# Bathymetry functions
G(y, γ, k) = γ * sin(k * y)

function hᵢ(x, y, p)
    if x < p.x1
        h = p.hA + p.h0 + p.A * x + p.h1 * G(y, p.γ, p.k) * x / p.x1
    elseif x < p.x2
        h = p.hA + p.h1 + p.B * (x - p.x1) + p.h1 * G(y, p.γ, p.k) * (p.x2 - x) / (p.x2 - p.x1)
    else
        h = p.hA + p.h2
    end
    return h
end

hᵢ(x, y) = hᵢ(x, y, (; hA, h0, h1, h2, x1, x2, A, B, λ, k, γ))
b(x, y) = -hᵢ(x, y)

# Model parameters
gravitational_acceleration = 9.81    # Gravitational acceleration (m/s²)
coriolis = FPlane(f=f)               # F-plane model

# Turbulence closure
closure = ShallowWaterScalarDiffusivity(ν=ν)

# Create Shallow Water model
model = ShallowWaterModel(; grid, coriolis, gravitational_acceleration,
                          momentum_advection=VectorInvariant(),
                          bathymetry=b,
                          closure=closure,
                          formulation=VectorInvariantFormulation(),
                          forcing=(u=u_forcing, v=v_forcing))

# Set initial conditions
set!(model, h=hᵢ)

# Plot bathymetry
figurepath = "figures/brink/bathymetry/"
fig = Figure(size=(800, 800))
axis = Axis(fig[1, 1],
            aspect=DataAspect(),
            title="Model Bathymetry",
            xlabel="x [m]",
            ylabel="y [m]")
depth = model.solution.h

# Initialize simulation
simulation = Simulation(model, Δt=Δt, stop_time=tmax)

# Progress logging
start_time = time_ns()
progress(sim) = @printf(
    "i: %10d, sim time: %12s, min(v): %4.3f ms⁻¹, max(v): %4.3f ms⁻¹, wall time: %12s\n",
    sim.model.clock.iteration,
    prettytime(sim.model.clock.time),
    minimum(model.solution.v),
    maximum(model.solution.v),
    prettytime(1e-9 * (time_ns() - start_time))
)

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10days / Δt))

# Define output writers
u, v, h = model.solution
bath = model.bathymetry
η = h + bath

# Momentum terms
∂η∂y = ∂y(η)
∂v∂x = ∂x(v)
u∂v∂x = u*∂v∂x

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, v, η, ∂η∂y, ∂v∂x, u∂v∂x),
    schedule=AveragedTimeInterval(outputtime),
    filename="reproduce_brink_2010/output/" * name * ".jld2",
    overwrite_existing=true)

simulation.output_writers[:bathymetry] = JLD2OutputWriter(model, (; bath),
    schedule=TimeInterval(tmax - Δt),
    filename="reproduce_brink_2010/output/" * name * "_bathymetry.jld2",
    overwrite_existing=true)

@info "Starting configuration " * name

# Run the simulation
run!(simulation)
