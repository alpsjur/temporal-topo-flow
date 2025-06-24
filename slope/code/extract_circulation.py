import numpy as np
import xarray as xr
from utils import analytical_circ, load_parameters, load_dataset, \
    get_vorticityflux_at_contour, \
        depth_following_contour, get_contour_following_velocities


params = load_parameters()
ds = load_dataset(params["filepath"], params["name"])
t = ds.time / np.timedelta64(1, 's')
t_days = ds.time / np.timedelta64(1, 'D')

xvals = 45
depths = np.arange(200,900,100)
#depths = np.arange(101,901,100)

als = []
ans = []
ns = []
for depth in depths:
    contour = depth_following_contour(params, depth)
            
    cL = contour.dl.sum(dim=("j")).values

    nonlin = get_vorticityflux_at_contour(contour, ds).values   
    Ut, Vt =  get_contour_following_velocities(contour, ds)

    analytical_linear = analytical_circ(params, t, cL, depth) * 1e2
    analytical_nonlinear = analytical_circ(params, t, cL, depth, nonlin) * 1e2
    numerical = -(Vt*contour.dl).sum(dim=("j")).data/cL * 1e2  # Convert to cm/s
    
    als.append(analytical_linear)
    ans.append(analytical_nonlinear)
    ns.append(numerical)

als = np.array(als)
ans = np.array(ans)
ns = np.array(ns)

ds_out = xr.Dataset(
    data_vars=dict(
         circulation=(["depth", "time"], ns),
         linear_estimate=(["depth", "time"], als),
         nonlinear_estimate=(["depth", "time"], ans)
     ),
     coords=dict(
         time=t_days.data,
         depth=depths
     ),
)

ds_out.to_netcdf(f"slope/output/extracted_circ/{params["name"]}_circ.nc")