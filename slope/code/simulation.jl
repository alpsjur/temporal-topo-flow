using Oceananigans
using Random
using Statistics
using Printf                   
using CUDA 
using CairoMakie 
using JSON

# Run on GPU (wow, fast!) if available. Else run on CPU
if CUDA.functional()
    architecture = GPU()
    @info "Running on GPU"
else
    architecture = CPU()
    @info "Running on CPU"
end


# Default parameters 
default_params = Dict(
    # Run name
    "name" => "slope-001",
    "filepath" => "slope/output/",

    # Grid parameters
    "dx" => 1e3,             
    "dy" => 1e3,
    "Lx" => 120e3,           
    "Ly" => 90e3,

    # Simulation parameters
    "dt" => 2.0,              
    "tmax" => 64 * 86400.0,      # 64 days in seconds
    "outputtime" => 3 * 3600.0,  # 3 hours in seconds

    # Forcing parameters
    "rho" => 1e3,
    "d" => 0.1,
    "T" => 4 * 86400.0,          # 4 days in seconds
    "R" => 5e-4,

    # Coriolis and gravity
    "f" => 1e-4,
    "gravitational_acceleration" => 9.81,

    # Bathymetry parameters
    "W" => 30e3,                # Width of slope
    "XC" => 45e3,               # Center coodinate of slope
    "DS" => 800.0,              # Depth change between shelf and basin
    "DB" => 100.0,              # Depth of shelf
    "sigma" => 1.0,             # Standard deviation of noise
    "a" => 1e3,                 # Horizontal length scale of corrigations
    "lam" => 45e3,              # Wave length of corriations
    "noise" => false
)

# Function to load and selectively overwrite parameters from JSON
function load_config(file_path, defaults)
    if isfile(file_path)
        config = JSON.parsefile(file_path)
        merge!(defaults, config)  # Overwrite only specified parameters
    else
        @warn "Configuration file not found. Using default parameters."
    end
    return defaults
end

# Overwrite default parameters if a configuration file is provided
if length(ARGS) == 1
    config_path = ARGS[1]
    @info "Loading configuration from $config_path"
    params = load_config(config_path, default_params)
else
    params = default_params
end

# Access parameters
name = params["name"]
filepath = params["filepath"]
dx = params["dx"]
dy = params["dy"]
Lx = params["Lx"]
Ly = params["Ly"]
dt = params["dt"]
tmax = params["tmax"]
outputtime = params["outputtime"]
rho = params["rho"]
d = params["d"]
T = params["T"]
R = params["R"]
f = params["f"]
gravitational_acceleration = params["gravitational_acceleration"]
W = params["W"]
XC = params["XC"]
DS = params["DS"]
DB = params["DB"]
sigma = params["sigma"]
a = params["a"]
lambda = params["lam"]
noise = params["noise"]

# Define bathymetry
function h_i(x, y, p)
    if x < (p.XC + p.W)                # slope
        corr = p.a * sin.(2 * pi * y / p.lam)    
        h = p.DB + 0.5 * p.DS * (1 + tanh.(pi * (x - p.XC - corr) / p.W))
    else                               # central basin
        h = p.DB + p.DS
    end
    if p.noise
        h += randn() * p.sigma
    end
    return h
end

# Parameters based on provided configuration
omega = 2 * pi / T
Nx = Int(Lx / dx)
Ny = Int(Ly / dy)

# Create grid
grid = RectilinearGrid(architecture,
                       size=(Nx, Ny),
                       x=(0, Lx), y=(0, Ly),
                       topology=(Bounded, Periodic, Flat))

# Set up parameters given to forcing function                   
tx_parameters = (; R)
ty_parameters = (; rho, d, omega, R)

# Define forcing functions
function tx(x, y, t, u, v, h, p)
    return -p.R * u / h
end

function ty(x, y, t, u, v, h, p)
    return -p.R * v / h + p.d * sin(p.omega * t) / (p.rho * h)
end

# Define bathymetry functions
h_i_func(x, y) = h_i(x, y, (; XC=params["XC"], W=params["W"], DS=params["DS"], DB=params["DB"], a=params["a"], lambda=params["lambda"], sigma=params["sigma"], noise=params["noise"]))
b_func(x, y) = -h_i_func(x, y)

# Coriolis
coriolis = FPlane(f=f)

# Create model
model = ShallowWaterModel(; grid, coriolis, gravitational_acceleration,
                          momentum_advection=VectorInvariant(),
                          bathymetry=b_func,
                          formulation=VectorInvariantFormulation(),
                          forcing=(u=Forcing(tx, field_dependencies=(:u, :v, :h), parameters=tx_parameters),
                                   v=Forcing(ty, field_dependencies=(:u, :v, :h), parameters=ty_parameters)))

# Set initial conditions
set!(model, h=h_i_func)

# Plot bathymetry
figurepath = "slope/figures/bathymetry/"
fig = Figure(size=(1200, 800))
axis = Axis(fig[1, 1], 
            aspect=DataAspect(),
            title="Model bathymetry",
            xlabel="x [m]",
            ylabel="y [m]")

depth = model.solution.h   
hm = heatmap!(axis, depth, colormap=:deep)
Colorbar(fig[1, 2], hm, label="Depth [m]")
save(figurepath * name * "-bathymetry.png", fig)

# Initialize simulation
simulation = Simulation(model, Δt=dt, stop_time=tmax)

# Logging simulation progress
start_time = time_ns()
progress(sim) = @printf(
    "i: %10d, sim time: % 12s, min(v): %4.3f ms⁻¹, max(v): %4.3f ms⁻¹, wall time: %12s\n",
    sim.model.clock.iteration,
    prettytime(sim.model.clock.time),
    minimum(sim.model.solution.v),
    maximum(sim.model.solution.v),
    prettytime(1e-9 * (time_ns() - start_time)))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10days / dt))

# Output
u, v, h = model.solution
bath = model.bathymetry
eta = h + bath

omega_field = Field(∂x(v) - ∂y(u))
omega_u = Field(omega_field * u)
omega_v = Field(omega_field * v)

divomega_flux = Field(∂x(omega_u) + ∂y(omega_v))

fields = Dict("u" => u, "v" => v, 
              "h" => h, "omega" => omega_field,
              "omegau" => omega_u, "omegav" => omega_v,
              "divomegaflux" => divomega_flux)

simulation.output_writers[:field_writer] = NetCDFOutputWriter(model, fields, 
                        filename=filepath * name * ".nc",
                        schedule=AveragedTimeInterval(outputtime),
                        overwrite_existing=true)

@info "Starting configuration " * name

run!(simulation)
