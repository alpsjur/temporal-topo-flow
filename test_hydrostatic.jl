# Import necessary packages
using Oceananigans              # Main package for ocean simulation
using Oceananigans.Units        # For physical units (e.g., meters, seconds)
using CUDA
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, PartialCellBottom  # For handling complex geometries
using Printf                    # For formatted output



# Run on GPU (wow, fast!) if available. Else run on CPU
if CUDA.functional()
    architecture = GPU()
    @info "Running on GPU"
else
    architecture = CPU()
    @info "Running on CPU"
end


# Simulation parameters
Δt = 30seconds                     # Time step size
stop_time = 30days                 # Simulation stop time

# Forcing parameters
const τ = -0.05/1000               # Wind stress (kinematic forcing)

# Bottom friction
const Cd = 3e-3#0.002                  # Quadratic drag coefficient []

# Grid parameters
Lx = 416kilometers              # Domain length in x-direction
Ly = 208kilometers             # Domain length in y-direction
Lz = 2300meters                 # Domain depth
dx = 2kilometers                # Grid spacing in x-direction
dy = 2kilometers                # Grid spacing in y-direction
Nx = Int(Lx/dx)                 # Number of grid cells in x-direction
Ny = Int(Ly/dy)                 # Number of grid cells in y-direction
Nz = 2

# define bathymetry
hᵢ(x, y) = -Lz #+ 500*y/Ly

# spesify bottom drag 
drag_u(x, y, t, u, v, Cd) = -Cd*√(u^2+v^2)*u
drag_v(x, y, t, u, v, Cd) = -Cd*√(u^2+v^2)*v


# spesify drag on immersed boundary, note different arguments from bottom drag above
immersed_drag_u(x, y, z, t, u, v, Cd) = -Cd*√(u^2+v^2)*u
immersed_drag_v(x, y, z, t, u, v, Cd) = -Cd*√(u^2+v^2)*v


# create bottom boundary conditions
drag_u_bc = FluxBoundaryCondition(drag_u, field_dependencies=(:u, :v), parameters=Cd)
drag_v_bc = FluxBoundaryCondition(drag_v, field_dependencies=(:u, :v), parameters=Cd)

immersed_drag_u_bc = FluxBoundaryCondition(immersed_drag_u, field_dependencies=(:u, :v), parameters=Cd)
immersed_drag_v_bc = FluxBoundaryCondition(immersed_drag_v, field_dependencies=(:u, :v), parameters=Cd)

immersed_u_bc = ImmersedBoundaryCondition(bottom = immersed_drag_u_bc)
immersed_v_bc = ImmersedBoundaryCondition(bottom = immersed_drag_v_bc)


# spesify surface forcing. Gradualy increase forcing over 10 days, then constant
τx(x, y, t, τ) = τ*tanh(t/(10days))
τy(x, y, t, τ) = 0

# create surface boundary conditions
τx_bc = FluxBoundaryCondition(τx, parameters=τ)
τy_bc = FluxBoundaryCondition(τy, parameters=τ)

# Define horizontal boundary condition
horizontal_bc = ValueBoundaryCondition(0.0)  # No-slip boundary condition
#horizontal_bc = FluxBoundaryCondition(0.0)    # Free-slip boundary condition


# collect boundary conditions
u_bc = FieldBoundaryConditions(
                                bottom=drag_u_bc, 
                                immersed=immersed_u_bc, 
                                top=τx_bc,
                                north = horizontal_bc,
                                south = horizontal_bc,
                                )
v_bc = FieldBoundaryConditions(
                                bottom=drag_v_bc, 
                                immersed=immersed_v_bc, 
                                top=τy_bc,
                                )

# Turbulence closures parameters for vertical and horizontal mixing 
κh =   0      # [m²/s] horizontal diffusivity (tracers)
νh = 100      # [m²/s] horizontal viscocity   (momentum)
κz = 1e-2     # [m²/s] vertical diffusivity
νz = 1e-2     # [m²/s] vertical viscocity

# Scalar diffusivity
vertical_closure = VerticalScalarDiffusivity(ν = νz, κ = κz)                
horizontal_closure = HorizontalScalarDiffusivity(ν = νh, κ = κh)
closure = (horizontal_closure, vertical_closure)

# Rotation
f = 1e-4
coriolis = FPlane(f)
                                
# Create grid
underlying_grid = RectilinearGrid(
        architecture;
        size=(Nx, Ny, Nz), 
        x = (0, Lx),
        y = (0, Ly),
        z = (-Lz, 0),
        #halo = (4, 4, 4),
        topology=(Periodic, Bounded, Bounded)
)



# create grid with immersed bathymetry 
grid = ImmersedBoundaryGrid(underlying_grid, 
                            #GridFittedBottom(hᵢ)
                            PartialCellBottom(hᵢ, minimum_fractional_cell_height=0.1)
                            )


# create model
model = HydrostaticFreeSurfaceModel(; 
        grid,
        #grid=underlying_grid,
        boundary_conditions=(u=u_bc, v=v_bc),
        free_surface = ImplicitFreeSurface(),
        #momentum_advection = = WENO(),
        #tracer_advection = WENO(),
        closure = closure,
        coriolis = coriolis,
)

println(model)

# create simulations
simulation = Simulation(model, Δt=Δt, stop_time=stop_time)

# logging simulation progress
start_time = time_ns()
progress(sim) = @printf(
    "i: %d, sim time: % 8s, min(u): %.1f m, max(u): %.1f m\n",#, wall time: %s\n",
    sim.model.clock.iteration,
    prettytime(sim.model.clock.time),
    minimum(u),
    maximum(u),
    #prettytime(1e-9 * (time_ns() - start_time))
)

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1))

# output
filename = "alternating_wind_basic_case"
u, v, z = model.velocities
simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, v),
                                                    schedule = TimeInterval(Δt),
                                                    filename = "output/" * filename * ".jld2",
                                                    overwrite_existing = true)
nothing

run!(simulation)