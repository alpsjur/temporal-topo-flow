using JLD2
using NCDatasets
using Oceananigans

# Define file path and name of the saved simulation output
filepath = "output/brink/"
filename = "brink_2010-300-period_016"

# Open the JLD2 file and load data
jld2_file = filepath * filename * ".jld2"
nc_file = filepath * filename * ".nc"  # Note: Use ".nc" extension for NetCDF file

# Load time series data from the saved JLD2 file
u_timeseries = FieldTimeSeries(jld2_file, "u")  # u-component of velocity field
v_timeseries = FieldTimeSeries(jld2_file, "v")  # v-component of velocity field
η_timeseries = FieldTimeSeries(jld2_file, "η")  # height (or depth) field
ω_timeseries = FieldTimeSeries(jld2_file, "ω") 
h = - FieldDataset(filepath * filename * "_bathymetry.jld2").bath[:,:,1,1]

# Get the size of the grid and time series for defining NetCDF dimensions
time_steps = length(u_timeseries.times)   # Number of time steps

# Get the staggered coordinates for `u` and `v` fields (C-grid)
xu, yu, _ = nodes(u_timeseries[1])    # x and y coordinates for u field
xv, yv, _ = nodes(v_timeseries[1])    # x and y coordinates for v field
#xc, yc, _ = nodes(η_timeseries[1])    # x and y coordinates for u field
#xs, ys, _ = nodes(ω_timeseries[1])    # x and y coordinates for v field

# Create the NetCDF file and define dimensions and variables
NCDataset(nc_file, "c") do ds  # "c" mode creates a new file
    # Define dimensions
    defDim(ds, "time", time_steps)
    defDim(ds, "xu", length(xu))        # x-dimension for u field
    defDim(ds, "yu", length(yu))        # y-dimension for u field
    defDim(ds, "xv", length(xv))        # x-dimension for v field
    defDim(ds, "yv", length(yv))        # y-dimension for v field

    # Define coordinate variables for staggered grids
    xu_var = defVar(ds, "xu", Float64, ("xu",))
    yu_var = defVar(ds, "yu", Float64, ("yu",))
    xv_var = defVar(ds, "xv", Float64, ("xv",))
    yv_var = defVar(ds, "yv", Float64, ("yv",))

    # Write coordinate data
    xu_var[:] = xu
    yu_var[:] = yu
    xv_var[:] = xv
    yv_var[:] = yv

    # Define time and velocity variables for the u and v fields
    time_var = defVar(ds, "time", Float64, ("time",))
    u_var = defVar(ds, "u", Float64, ("time", "xu", "yu"))
    v_var = defVar(ds, "v", Float64, ("time", "xv", "yv"))
    eta_var = defVar(ds, "eta", Float64, ("time", "xv", "yu"))
    omega_var = defVar(ds, "omega", Float64, ("time", "xu", "yv"))
    h_var = defVar(ds, "h", Float64, ("xv", "yu"))

    # Write time data
    time_var[:] = u_timeseries.times

    # write depth data
    h_var[:] = h

    # Write u and v data over time for each time step
    for t in 1:time_steps
        u_var[t, :, :] = u_timeseries.data[:,:,1,t]  # Assign each time slice of u data
        v_var[t, :, :] = v_timeseries.data[:,:,1,t]  # Assign each time slice of v data
        eta_var[t, :, :] = η_timeseries.data[:,:,1,t]
        omega_var[t, :, :] = ω_timeseries.data[:,:,1,t]
    end
end

println("Conversion complete: Saved to $nc_file")
