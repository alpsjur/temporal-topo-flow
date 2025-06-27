using Oceananigans
using Oceananigans.Units
using Printf                   
using CUDA 
using JSON
using NCDatasets
using Oceananigans.Architectures: on_architecture


# Default parameters 
default_params = Dict(
    # Run name
    "name" => "default",
    "filepath" => "output/",

    # Grid parameters
    "dx" => 1e3,             
    "dy" => 1e3,
    "Lx" => 90e3,           
    "Ly" => 90e3,

    # Simulation parameters
    "dt" => 4.0,              
    "tmax" => 4 * 86400.0,        # 4 days in seconds
    "outputtime" => 3 * 3600.0,   # 3 hours in seconds

    # Forcing parameters
    "tau0" => 0.0001,              # maximum kinematic forcing [m2 s-2]
    "T" => 4 * 86400.0,            # 4 days in seconds
    "R" => 5e-4,

    # Coriolis and gravity
    "f" => 1e-4,
    "gravitational_acceleration" => 9.81,

    # Bathymetry parameters
    "W" => 30e3,                # Width of slope
    "yc" => 45e3,               # Center coodinate of slope
    "Hsh" => 900.0,             # Depth of shelf
    "Hbs" => 100.0,             # Depth of deep basin
    "Acorr" => 10e3,            # Horizontal length scale of corrigations
    "lam" => 45e3,              # Wave length of corriations
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


# Check GPU flag in config
use_gpu = get(params, "GPU", true)  # Default is true if not in config

if use_gpu && CUDA.functional()
    architecture = GPU()
    @info "Running on GPU"
else
    architecture = CPU()
    @info "Running on CPU"
end

# Access parameters
runname = params["name"]
filepath = params["filepath"]
dx = params["dx"]
dy = params["dy"]
Lx = params["Lx"]
Ly = params["Ly"]
dt = params["dt"]
tmax = params["tmax"]
outputtime = params["outputtime"]

tau0 = params["tau0"]
T = params["T"]
R = params["R"]
f = params["f"]
gravitational_acceleration = params["gravitational_acceleration"]

W = params["W"]
yc = params["yc"]
Hsh = params["Hsh"]
Hbs = params["Hbs"]
Acorr = params["Acorr"]
lam = params["lam"]

# Parameters based on provided configuration
omega = 2 * pi / T
Nx = Int(Lx / dx)
Ny = Int(Ly / dy)

# Define bathymetry
function h_i(x, y, p)
    steepness = (sech.(pi * (y - p.yc) / p.W).^2)
    delta = p.Acorr * sin.(2 * pi * x / p.lam) * steepness
    h = p.Hsh + 0.5 * (p.Hbs - p.Hsh) * (1 + tanh.(pi * (y - p.yc - delta) / p.W))
    return h
end



# Create grid
grid = RectilinearGrid(architecture,
                       size=(Nx, Ny),
                       x=(0, Lx), y=(0, Ly),
                       topology=(Periodic, Bounded, Flat))



# Define bathymetry functions
h_i_func(x, y) = h_i(x, y, (; yc, W, Hsh, Hbs, Acorr, lam))
b_func(x, y) = -h_i_func(x, y)


# === Load Forcing Data if Available === #
forcing_from_file = haskey(params, "forcing_file")
if forcing_from_file
    forcing_file = params["forcing_file"]
    @info "Loading forcing file: $forcing_file"

    ds = Dataset(forcing_file)

    # Ensure forcing data is a Float64 array
    forcing_x_data = convert(Array{Float64, 3}, coalesce.(ds["forcing_x"][:, :, :], NaN))
    forcing_y_data = convert(Array{Float64, 3}, coalesce.(ds["forcing_y"][:, :, :], NaN))
    time = convert(Vector{Float64}, coalesce.(ds["time"][:], NaN))

    close(ds)

    # Move forcing data to GPU if available
    forcing_x_data = on_architecture(architecture, forcing_x_data)
    forcing_y_data = on_architecture(architecture, forcing_y_data)
    time_gpu = on_architecture(architecture, time)

end

# === GPU-Safe Interpolation Function === #
if forcing_from_file
    @inline function interpolate_forcing(i, j, t, forcing_data, p)
        # Compute time indices safely using integer division and modulo
        idx = min(unsafe_trunc(Int32, t / p.fdt) + 1, length(p.time)-1)

        # Performinterpolation in time
        @inbounds begin
            t1 = p.time[idx]
            t2 = p.time[idx+1]
            f1 = forcing_data[j, i, idx]
            f2 = forcing_data[j, i, idx+1]
        end
        return f1 + (f2 - f1) * (t - t1) / (t2 - t1)
    end

    tx_parameters = (; R=params["R"],
                     time=time_gpu, fdt=params["outputtime"], forcing_data=forcing_x_data)

    ty_parameters = (; R=params["R"],
                     time=time_gpu, fdt=params["outputtime"], forcing_data=forcing_y_data)

    @inline function tx(i, j, k, grid, clock, model_fields, p)
        u = @inbounds model_fields.u[i, j, k]
        h = @inbounds model_fields.h[i, j, k]
        return h <= 0 ? 0.0 : -p.R * u / h + interpolate_forcing(i, j, clock.time, p.forcing_data, p) /  h
    end

    @inline function ty(i, j, k, grid, clock, model_fields, p)
        v = @inbounds model_fields.v[i, j, k]
        h = @inbounds model_fields.h[i, j, k]
        return h <= 0 ? 0.0 : -p.R * v / h + interpolate_forcing(i, j, clock.time, p.forcing_data, p) /  h
    end

    forcing_u = Forcing(tx, discrete_form=true, parameters=tx_parameters)
    forcing_v = Forcing(ty, discrete_form=true, parameters=ty_parameters)
else
    @info "No forcing file specified. Running with defaul forcing."

    # Set up parameters
    tx_parameters = (; tau0, omega, R)
    ty_parameters = (; R)

    function tx(x, y, t, u, v, h, p)
        return -p.R * u / h + p.tau0 * sin(p.omega * t) /  h
    end

    function ty(x, y, t, u, v, h, p)
        return -p.R * v / h
    end

    forcing_u = Forcing(tx, field_dependencies=(:u, :v, :h), parameters=tx_parameters)
    forcing_v = Forcing(ty, field_dependencies=(:u, :v, :h), parameters=ty_parameters)
end


# === Define Model === #
# Set initial conditions
h_from_file = haskey(params, "bathymetry_file")
if h_from_file
    h_file = params["bathymetry_file"]
    @info "Loading bathymetry file: $h_file"

    ds = Dataset(h_file)

    # Ensure forcing data is a Float64 array
    h_data = convert(Array{Float64, 2}, coalesce.(ds["h"][:, :], NaN))

    bathymetry = - h_data
    h_initial = h_data
else    
    @info "No bathymetry file specified. Running with default bathymetry."
    bathymetry = b_func
    h_initial = h_i_func
end


coriolis = FPlane(f=params["f"])
model = ShallowWaterModel(; grid, coriolis, gravitational_acceleration=params["gravitational_acceleration"],
                          momentum_advection=VectorInvariant(),
                          bathymetry=bathymetry,
                          formulation=VectorInvariantFormulation(),
                          forcing=(u=forcing_u, v=forcing_v))
set!(model, h=h_initial)

# TODO remove plotting 
# Plot bathymetry
using CairoMakie 
figurepath = "figures/bathymetry/"
fig = Figure(size=(800, 800))
axis = Axis(fig[1, 1], 
            aspect=DataAspect(),
            title="Model bathymetry",
            xlabel="x [m]",
            ylabel="y [m]")

depth = model.solution.h   
hm = heatmap!(axis, depth, colormap=:deep)
Colorbar(fig[1, 2], hm, label="Depth [m]")
save(figurepath * runname * "-bathymetry.png", fig)


# Initialize simulation
simulation = Simulation(model, Δt=dt, stop_time=tmax)

# Logging simulation progress
start_time = time_ns()
progress(sim) = @printf(
    "i: %6d, sim time: % 12s, wall time: %12s\n",
    sim.model.clock.iteration,
    prettytime(sim.model.clock.time),
    #minimum(sim.model.solution.v),
    #maximum(sim.model.solution.v),
    prettytime(1e-9 * (time_ns() - start_time)))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(20days / dt))

# Output
u, v, h = model.solution
bath = model.bathymetry
eta = h + bath

uvh = Field(u*v*h)
#uuh = Field(u*u*h)        # Do I need these?
#vvh = Field(v*v*h)        #
# duvhdx = Field(∂x(uvh))   #
duvhdy = Field(∂y(uvh))   
# duuhdx = Field(∂x(uuh))   #
# duuhdy = Field(∂y(uuh))   #
# dvvhdx = Field(∂x(vvh))   #
# dvvhdy = Field(∂y(vvh))   #

detadx = Field(∂x(eta))   
#detady = Field(∂y(eta))  #

zeta_field = Field(∂x(v) - ∂y(u))
zetau = Field(zeta_field * u)
zetav = Field(zeta_field * v)


fields = Dict("u" => u, "v" => v, 
              "h" => h, "zeta" => zeta_field,
              "zetau" => zetau, "zetav" => zetav,
              "duvhdy" => duvhdy, "detadx" => detadx,
              #"duvhdx" => duvhdx, "duvhdy" => duvhdy, 
              #"duuhdx" => duuhdx, "duuhdy" => duuhdy, 
              #"dvvhdx" => dvvhdx, "dvvhdy" => dvvhdy, 
              #"detadx" => detadx, "detady" => detady,
              )

simulation.output_writers[:field_writer] = NetCDFOutputWriter(model, fields, 
                        filename=filepath * "raw/" * runname * ".nc",
                        schedule=AveragedTimeInterval(outputtime),
                        overwrite_existing=true)

@info "Starting configuration " * runname

run!(simulation)
