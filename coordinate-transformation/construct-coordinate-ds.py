import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd

# Grid parameters
dx =   1e3
dy =   1e3
Lx = 120e3
Ly =  90e3

# Bathymetry parameters
hA = 0
h0 = 25
h1 = 100
h2 = 1000
x1 = 40e3
x2 = 60e3

l  = 45e3
hc = 59

g  = hc/h1
k  = 2*np.pi/l
A  = (h1-h0)/x1
B  = (h2-h1)/(x2-x1)
Nx = Lx//dx
Ny = Ly//dy
Nl = Ly//l

def G(y, g, k):
    return g*np.sin(k*y)


# Define the vectorized H function
def H(x, y, x1, x2, hA, h0, h1, h2, A, B, g, k, G):
    # Calculate the three possible cases
    case1 = hA + h0 + A * x + h1 * G(y, g, k) * x / x1
    case2 = hA + h1 + B * (x - x1) + h1 * G(y, g, k) * (x2 - x) / (x2 - x1)
    case3 = hA + h2
    
    # Use np.where to choose values based on conditions
    h = np.where(
        x < x1,
        case1,
        np.where(
            x < x2,
            case2,
            case3
        )
    )
    return h


x = np.arange(0, Lx+1, dx)
y = np.arange(0, Ly+1, dy)

X, Y = np.meshgrid(x, y)
bath = H(X, Y, x1, x2, hA, h0, h1, h2, A, B, g, k, G)

xs = np.arange(0,60,1)*1e3
Hs = H(xs, l/4, x1, x2, hA, h0, h1, h2, A, B, g, k, G)
#levels = np.append(np.linspace(hA+h0, hA+h1-0.1, 10), np.linspace(hA+h1, hA+h2, 10))


### Find x and y as function of new coordinates
def xt_from_y(y, H, x1, x2, h0, h1, h2, g, k):
    # First, calculate xt using the initial formula everywhere
    xt = x1*(h0 - H) / (h0 - h1 - h1*g*np.sin(k * y))
    
    # Now, calculate the second formula only for elements where xt > x1
    xt_update = (H*x1 - H*x2 + h1*g*x2*np.sin(k*y) + h1*x2 - h2*x1) / (h1*g*np.sin(k*y) + h1 - h2)
    
    # Update xt where the condition xt > x1 is met
    xt = np.where(xt > x1, xt_update, xt)
    
    return xt

def dt(y, H, x1, x2, h0, h1, h2, g, k):
    # First, calculate xt and dxt using the initial formula everywhere
    xt = x1*(h0 - H) / (h0 - h1 - h1*g*np.sin(k * y))
    dxt = h1*k*g*x1*(h0-H)*np.cos(k*y)/((h0-h1*g*np.sin(k*y)-h1)**2)
    
    # Now, calculate the second formula only for elements where xt > x1
    dxt_update = -h1*k*g*(H-h2)*(x1-x2)*np.cos(k*y)/((h1*g*np.sin(k*y)+h1-h2)**2)
    
    # Update xt where the condition xt > x1 is met
    dxt = np.where(xt > x1, dxt_update, dxt)

    # initialize dyt and normalize dxt and dyt
    dyt = np.ones_like(dxt)
    
    norm = np.sqrt(dxt**2+dyt**2)
    
    dxt /= norm
    dyt /= norm
    
    return dxt, dyt

def dl_fromxt_yt(x, y, Ny, Nl, upstream=False):
    dx = x[1:]-x[:-1]
    dy = y[1:]-y[:-1]
    
    dl1 = np.sqrt(dx**2 + dy**2)
    dl2 = (dl1[1:]+dl1[:-1])/2
    
    dl = np.zeros_like(x)
    
    if upstream:
        dl[1:] = dl1 
        dl[0] = dl1[int(Ny/Nl)-1]
    
    else:
        dl[1:-1] = dl2 

        dl[0]    = dl2[int(Ny/Nl)-1]
        dl[-1]   = dl2[int(Ny/Nl)-2]
        
    return dl

## Create contour folowing grid
Yt = Y.copy()
Xt = X.copy()

dXt = np.zeros_like(Xt)
dYt = np.ones_like(Yt)
dLt = np.ones_like(Xt)*dy

for i, Ht in enumerate(Hs):
    xt = xt_from_y(y, Ht, x1, x2, h0, h1, h2, g, k)
    Xt[:,i] = xt
    
    dxt, dyt = dt(y, Ht, x1, x2, h0, h1, h2, g, k)
    dXt[:,i] = dxt
    dYt[:,i] = dyt
    
    dlt = dl_fromxt_yt(xt, y, Ny, Nl, upstream=True)
    dLt[:,i] = dlt
    #ax.plot(xt, y, color="red")

contour_grid = xr.Dataset(
    data_vars=dict(
        dtdx=(["j", "i"], dXt),
        dtdy=(["j", "i"], dYt),
        dl=(["j", "i"], dLt),
    ),
    coords=dict(
        x=(["j", "i"], Xt),
        y=(["j", "i"], Yt),
        i = np.arange(Nx+1, dtype=np.int32),
        j = np.arange(Ny+1, dtype=np.int32)
    )
)

fig, ax = plt.subplots()
ax.pcolormesh(X, Y, bath, cmap="Blues")

ax.contour(X, Y, bath, levels = Hs, colors="gray")
ax.axhline(l/4, color="gray")

pstep = 1
U = dXt[::pstep,::pstep]*dLt[::pstep,::pstep]
V = dYt[::pstep,::pstep]*dLt[::pstep,::pstep]
ax.quiver(Xt[::pstep,::pstep], Yt[::pstep,::pstep], U, V, 
          zorder=10,
          #scale = 50,
          angles='xy', scale_units='xy', scale=1,
          )


ax.set_xlim(0, 65e3)
ax.set_aspect("equal")


### Create rotated grid
dXr = np.zeros_like(X)
dYr = np.ones_like(Y)

for i, xi in enumerate(xs):
    for j, yi in enumerate(y):
        Hi = bath[j,i]
        dxr, dyr = dt(yi, Hi, x1, x2, h0, h1, h2, g, k)
        dXr[j,i] = dxr
        dYr[j,i] = dyr
    #ax.plot(xt, y, color="red")

rotate_grid = xr.Dataset(
    data_vars=dict(
        dtdx=(["j", "i"], dXr),
        dtdy=(["j", "i"], dYr),
    ),
    coords=dict(
        x=(["j", "i"], Xt),
        y=(["j", "i"], Yt),
        i = np.arange(Nx+1, dtype=np.int32),
        j = np.arange(Ny+1, dtype=np.int32)
    )
)


### 

fig, ax = plt.subplots()
ax.pcolormesh(X, Y, bath, cmap="Blues")

ax.contour(X, Y, bath, levels = Hs, colors="gray")
ax.axhline(l/4, color="gray")

pstep = 1
U = dXr[::pstep,::pstep]*1e3
V = dYr[::pstep,::pstep]*1e3
ax.quiver(X[::pstep,::pstep], Y[::pstep,::pstep], U, V, 
          zorder=10,
          #scale = 50,
          angles='xy', scale_units='xy', scale=1,
          )


ax.set_xlim(0, 65e3)
ax.set_aspect("equal")

### Open dataset

ds = xr.open_dataset("output/brink/brink_2010-300-period_004.nc", decode_times=True)

time_units = "s"  # Adjust if the units are different (e.g., 'D' for days, 'm' for minutes)
time_origin = "2024-01-01"

# Convert the time variable to datetime objects
ds['time'] = pd.to_datetime(ds['time'].values, unit=time_units, origin=pd.Timestamp(time_origin))


# Get the last timestamp in the dataset
end_time = ds.time.max().values
start_time = pd.to_datetime(end_time) - pd.Timedelta(days=64)

dss = ds.sel(time=slice(start_time, end_time)).mean(dim="time")

#fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True)
#ax11, ax12, ax21, ax22 = axes.flatten()


fig = plt.figure(layout="constrained")
axd = fig.subplot_mosaic(
    """
    ABC
    ABC
    abc
    """,
    sharex = True
)

pstep = 3 

# original grid
X, Y = np.meshgrid(x, y)
Uc = dss.u.interp(xu=x, yu=y)
Vc = dss.v.interp(xv=x, yv=y)

axd["A"].contour(X*1e-3, Y*1e-3, bath, levels = Hs, colors="gray")
axd["A"].quiver(X[::pstep,::pstep]*1e-3, Y[::pstep,::pstep]*1e-3, Uc[::pstep,::pstep], Vc[::pstep,::pstep], zorder=20)

axd["A"].set_xlim(0, 65)
axd["A"].set_ylim(0, 90)
axd["A"].set_aspect("equal")

axd["a"].plot(x*1e-3, Vc.mean(dim="yv"))


### rotatet velocities
uc = dss.u.interp(xu=rotate_grid.x, yu=rotate_grid.y)
vc = dss.v.interp(xv=rotate_grid.x, yv=rotate_grid.y)

Ur = uc*rotate_grid.dtdy - vc*rotate_grid.dtdx
Vr = uc*rotate_grid.dtdx + vc*rotate_grid.dtdy

axd["B"].quiver(X[::pstep,::pstep]*1e-3, Y[::pstep,::pstep]*1e-3, Ur[::pstep,::pstep], Vr[::pstep,::pstep], zorder=20)

axd["B"].set_xlim(0, 65)
axd["B"].set_ylim(0, 90)
axd["B"].set_aspect("equal")

axd["b"].plot(x*1e-3, Vr.mean(dim="j"))


### contour folowing velocities
ut = dss.u.interp(xu=contour_grid.x, yu=contour_grid.y)
vt = dss.v.interp(xv=contour_grid.x, yv=contour_grid.y)

I, J = np.meshgrid(contour_grid.i, contour_grid.j)

Ut = ut*contour_grid.dtdy - vt*contour_grid.dtdx
Vt = ut*contour_grid.dtdx + vt*contour_grid.dtdy
axd["C"].quiver(I[::pstep,::pstep], J[::pstep,::pstep], Ut[::pstep,::pstep], Vt[::pstep,::pstep], zorder=20)

axd["C"].set_xlim(0, 65)
axd["C"].set_ylim(0, 90)
axd["C"].set_aspect("equal")

Vmean = (Vt*contour_grid.dl).sum(dim="j")/contour_grid.dl.sum(dim="j")
axd["c"].plot(contour_grid.i, Vmean)

fig.tight_layout()

plt.show()