import xarray as xr
import numpy as np

params = {
    # Grid parameters
    "grid": {
        "dx": 1e3,
        "dy": 1e3,
        "Lx": 120e3,
        "Ly": 90e3,
    },

    # Bathymetry parameters
    "bathymetry": {
        "hA": 0,
        "h0": 25,
        "h1": 100,
        "h2": 1000,
        "x1": 40e3,
        "x2": 60e3,
        "l": 45e3,
        "hc": 59,
    },
}

# Add derived parameters
params["derived"] = {
    "g": params["bathymetry"]["hc"] / params["bathymetry"]["h1"],
    "k": 2 * np.pi / params["bathymetry"]["l"],
    "A": (params["bathymetry"]["h1"] - params["bathymetry"]["h0"]) / params["bathymetry"]["x1"],
    "B": (params["bathymetry"]["h2"] - params["bathymetry"]["h1"]) / 
         (params["bathymetry"]["x2"] - params["bathymetry"]["x1"]),
    "Nx": params["grid"]["Lx"] // params["grid"]["dx"],
    "Ny": params["grid"]["Ly"] // params["grid"]["dy"],
    "Nl": params["grid"]["Ly"] // params["bathymetry"]["l"],
}

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

### Find x and y as function of new coordinates
def xt_from_y(y, H, x1, x2, hA, h0, h1, h2, g, k):
    H = H - hA
    
    # First, calculate xt using the initial formula everywhere
    xt = x1*(h0 - H) / (h0 - h1 - h1*g*np.sin(k * y))
    
    # Now, calculate the second formula only for elements where xt > x1
    xt_update = (H*x1 - H*x2 + h1*g*x2*np.sin(k*y) + h1*x2 - h2*x1) / (h1*g*np.sin(k*y) + h1 - h2)
    
    # Update xt where the condition xt > x1 is met
    xt = np.where(xt > x1, xt_update, xt)
    
    return xt

def dt(y, H, x1, x2, hA, h0, h1, h2, g, k):
    H = H - hA
    
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

