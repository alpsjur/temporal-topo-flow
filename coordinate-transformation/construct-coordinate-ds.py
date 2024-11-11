import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import h5py

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

#Hs = np.mean(bath, axis=0)[:60]


fig, ax = plt.subplots()
ax.pcolormesh(X, Y, bath, cmap="Blues")

ax.contour(X, Y, bath, levels = Hs, colors="gray")
ax.axhline(l/4, color="gray")



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


new_grid = xr.Dataset(
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



ds = xr.open_dataset("output/brink/brink_2010-300-period_004.nc", decode_times=True)

dss = ds.isel(time=-1)

fig, [ax1, ax2] = plt.subplots(ncols=2, sharex=True, sharey=True)
pstep = 3 

X, Y = np.meshgrid(x, y)
Uc = dss.u.interp(xu=x, yu=y)
Vc = dss.v.interp(xv=x, yv=y)

ax1.contour(X, Y, bath, levels = Hs, colors="gray")
ax1.quiver(X[::pstep,::pstep], Y[::pstep,::pstep], Uc[::pstep,::pstep], Vc[::pstep,::pstep])

ax1.set_xlim(0, 65e3)
ax1.set_aspect("equal")





ut = dss.u.interp(xu=new_grid.x, yu=new_grid.y)
vt = dss.v.interp(xv=new_grid.x, yv=new_grid.y)

I, J = np.meshgrid(new_grid.i, new_grid.j)

Ut = ut*new_grid.dtdy - vt*new_grid.dtdx
Vt = ut*new_grid.dtdx + vt*new_grid.dtdy
ax2.quiver(I[::pstep,::pstep]*1e3, J[::pstep,::pstep]*1e3, Ut[::pstep,::pstep], Vt[::pstep,::pstep])

ax2.set_xlim(0, 65e3)
ax2.set_aspect("equal")

fig.tight_layout()

plt.show()