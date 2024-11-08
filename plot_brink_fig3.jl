# Import necessary packages for visualization, ocean simulation, formatted output, and data handling
using CairoMakie        # For creating visualizations and animations
using Oceananigans      # For simulating ocean dynamics
using Printf            # For formatted string output
using JLD2              # For saving and loading simulation data in Julia format
using Statistics

# Define file path and name of the saved simulation output
filepath = "output/brink/"
figurepath = "figures/brink/"

filename = "brink_2010-300-period_128"

R = 5e-4
ρ = 1e3
g = 9.81
dy = 1e3
dx = 1e3

# Load time series data from the saved JLD2 file
full_output = FieldDataset(filepath * filename * ".jld2")
u = full_output.u # u-component of velocity field
v = full_output.v  # v-component of velocity field
η = full_output.η  # sea surface height field
h = - FieldDataset(filepath * filename * "_bathymetry.jld2").bath[:,:,1,1]
u∂v∂x = full_output.u∂v∂x
∂v∂x = full_output.∂v∂x
∂η∂y = full_output.∂η∂y

# Extract time points from the simulation data
times = u.times

# Get coordinate arrays from the first timestep of height field data
# `xv`, `yv`, `zv` are not used, so only `xc`, `yc`, `zc` are extracted
xc, yc, zc = nodes(η[1]) 
xc = xc/1e3
yc = yc/1e3


T = 128*2*8
Tend = length(times)


# Center all fields 
uc = deepcopy(η)  
vc = deepcopy(η) 
u∂v∂xc = deepcopy(η) 
∂v∂xc = deepcopy(η) 
∂η∂yc = deepcopy(η) 

# Loop over all time steps to interpolate to center
for i in 1:length(times)
    uc[i] .= @at (Center, Center, Center) u[i]  # Interpolated u at grid centers
    vc[i] .= @at (Center, Center, Center) v[i]  # Interpolated v at grid centers
    u∂v∂xc[i] .= @at (Center, Center, Center) u∂v∂x[i]  
    ∂v∂xc[i] .= @at (Center, Center, Center) u∂v∂x[i]  
    ∂η∂yc[i] .= @at (Center, Center, Center) ∂η∂y[i]  
end  

MT1 = mean(u∂v∂xc.data[:,:,1,Tend-T:Tend], dims=(2,3))[:,1,1]
MT2 = mean(∂v∂xc.data[:,:,1,Tend-T:Tend], dims=(2,3))[:,1,1] .*mean(uc.data[:,:,1,Tend-T:Tend], dims=(2,3))[:,1,1]
MT = (MT1-MT2).*mean(h, dims=2)[:,1]

TFS1 = (∂η∂yc.*h)[:,:,1,:]
TFS2 = mean(∂η∂yc.data[:,:,1,Tend-T:Tend], dims=(2,3))[:,1,1].*mean(h,dims=2)
TFS = (mean(TFS1, dims=(2,3))[:,1,1] - TFS2)[:,1].*g

BS = mean(vc.data[:,:,1,Tend-T:Tend], dims=(2,3))[:,1,1].*R

# ubar = mean(u.data[:,:,1,Tend-T:Tend], dims=(2,3))[:,1,1]
# vbar = mean(v.data[:,:,1,Tend-T:Tend], dims=(2,3))[:,1,1]
# ηbar = mean(η.data[:,:,1,Tend-T:Tend], dims=(2,3))[:,1,1]
# hbar = mean(H[:,:,1], dims=2)[:,1]

# unod = u.data[:,:,1,Tend-T:Tend] .- ubar
# vnod = v.data[:,:,1,Tend-T:Tend] .- vbar
# ηnod = η.data[:,:,1,Tend-T:Tend] .- ηbar
# hnod = H[:,:,1] .- hbar

# ηnoddy =  (ηnod[:,2:end,:] - ηnod[:,1:end-1,:])/dy
# ηnoddy_bp = (ηnod[:,1,:] - ηnod[:,end,:])/dy
# ηnoddy_bp = reshape(ηnoddy_bp, 90, 1, T+1)
# ηnoddy = cat(ηnoddy_bp, ηnoddy, ηnoddy_bp, dims=2)
# ηnoddy = (ηnoddy[:,2:end,:] + ηnoddy[:,1:end-1,:])/2

# vnoddx = (vnod[2:end,:,:] - vnod[1:end-1,:,:])/dx
# vnoddx_bp = reshape(vnoddx[:,1,:], 89, 1, T+1)
# vnoddx = cat(vnoddx, vnoddx_bp, dims=2)
# vnoddx = (vnoddx[2:end,:,:] + vnoddx[1:end-1,:,:])/2
# vnoddx = (vnoddx[:,2:end,:] + vnoddx[:,1:end-1,:])/2
# vnoddx = cat(fill(NaN, 1, 90, T+1), vnoddx,fill(NaN, 1, 90, T+1), dims=1)

# unod = (unod[2:end,:,:] + unod[1:end-1,:,:])/2

#TFS = mean(hnod.*ηnoddy.*g, dims=(2,3))[:,1,1]
#MT = mean(unod.*vnoddx, dims=(2,3))[:,1,1].*hbar
#BS = R*vbar


fig = Figure(size = (800, 600));
ax = Axis(fig[1, 1], xlabel="x [km]", ylabel="Terms [m2 s-2]") 
lines!(ax, xc, TFS, label="Form stress")
lines!(ax, xc, MT, label="Momentum transport")
lines!(ax, xc, BS, label="Bottom stress")
limits!(ax, 0, 90, nothing, nothing)

axislegend(position = :rb)

save(figurepath*filename*"_fig3.png", fig)