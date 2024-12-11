import xarray as xr
import numpy as np
import json


# Default parameters
default_params = {
    # data storage parameters
    "name": "slope-001",
    "filepath": "slope/output/",
    
    # grid
    "dx": 1e3,              
    "dy": 1e3,
    "Lx": 120e3,              
    "Ly": 90e3,
    
    #simulation
    "dt": 2.0,                
    "tmax": 64 * 86400.0,     
    "outputtime": 3 * 3600.0, 
    
    #forcing
    "rho": 1e3,
    "d": 0.1,
    "T": 4 * 86400.0,         
    "R": 5e-4,
    "f": 1e-4,
    "gravitational_acceleration": 9.81,
    
    # bathymetry
    "W": 30e3,
    "XC": 45e3,
    "DS": 800.0,
    "DB": 100.0,
    "sigma": 1.0,            
    "a": 10e3,               
    "lam": 45e3,           
    "noise": False,
}

def load_config(file_path, params):
    """
    Loads configuration from a JSON file and overwrites the default parameters.

    Parameters:
        file_path (str): Path to the configuration file.
        params (dict): Default parameters dictionary.

    Returns:
        dict: Updated parameters.
    """
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
            params.update(config)  # Overwrite defaults with values from the config
    except FileNotFoundError:
        print(f"Warning: Configuration file '{file_path}' not found. Using default parameters.")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file '{file_path}': {e}")
    return params



def H(x, y, params):
    a = params["a"]
    lam = params["lam"]
    DB = params["DB"]
    DS = params["DS"]
    XC = params["XC"]
    W = params["W"]
    
    corr = a*np.sin(2*np.pi*y/lam)
    slope = DB + 0.5*DS*(1+np.tanh(np.pi*(x-XC-corr)/W))
    basin = DB + DS 
    
    h = np.where(x < XC+W, slope, basin)

    return h


def xt_from_y(y, H, params):
    W = params["W"]
    DB = params["DB"]
    DS = params["DS"]
    XC = params["XC"]
    a = params["a"]
    lam = params["lam"]
    
    # find x0 from depth H
    x0 = -W*np.arctanh((2*DB-2*H+DS)/DS)/np.pi + XC
    xt = a*np.sin(2*np.pi*y/lam) + x0 
    
    # update xt for flat basin
    xt = np.where(xt >= XC+W, XC+W, xt)
    xt = np.where(xt <= 0, 0, xt)
    
    return xt

def dt(y, H, params):
    XC = params["XC"]
    W = params["W"]
    a = params["a"]
    lam = params["lam"]
    
    # First, calculate xt and dxt for the slope
    xt = xt_from_y(y, H, params)
    dxt = 2*np.pi*a/lam*np.cos(2*np.pi*y/lam)
    
    # Update xt for the flat basin
    dxt = np.where(xt > XC+W, 0, dxt)
    dxt = np.where(xt < 0, 0, dxt)

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

def depth_following_grid(params):
    # Construct depth-following grid
    Lx = params["Lx"]
    Ly = params["Ly"]
    dx = params["dx"]
    dy = params["dy"]
    lam = params["lam"]
    XC = params["XC"]
    W = params["W"]
    DB = params["DB"]
    DS = params["DS"]
    
    Nx = Lx//dx 
    Ny = Ly//dy 
    Nl = Ly//lam
    
    slopeend = round((XC+W)/dx)

    x = np.arange(0, Lx, dx)
    y = np.arange(0, Ly, dy)

    X, Y = np.meshgrid(x, y)

    h = H(X, Y, params)

    # construct depths over slope  from h
    hslope = h[:,:slopeend]
    sdepths = hslope.mean(axis=0)

    ## Create contour folowing grid
    Yt = Y.copy()
    Xt = X.copy()

    dXt = np.zeros_like(Xt)
    dYt = np.ones_like(Yt)
    dLt = np.ones_like(Xt)*dy
    
    depths = np.ones_like(x)*(DB + DS)

    for i, Ht in enumerate(sdepths):
        depths[i] = Ht
        
        xt = xt_from_y(y, Ht, params)
        Xt[:,i] = xt
        
        dxt, dyt = dt(y, Ht, params)
        dXt[:,i] = dxt
        dYt[:,i] = dyt
        
        dlt = dl_fromxt_yt(xt, y, Ny, Nl, upstream=True)
        dLt[:,i] = dlt

    contour_grid = xr.Dataset(
        data_vars=dict(
            dtdx=(["j", "i"], dXt),
            dtdy=(["j", "i"], dYt),
            dl=(["j", "i"], dLt),
        ),
        coords=dict(
            x=(["j", "i"], Xt),
            y=(["j", "i"], Yt),
            depth=(["i"], depths),
            i = np.arange(Nx, dtype=np.int32),
            j = np.arange(Ny, dtype=np.int32)
        )
    )
    
    return contour_grid

def get_h(params):
    Lx = params["Lx"]
    Ly = params["Ly"]
    dx = params["dx"]
    dy = params["dy"]
    
    x = np.arange(0, Lx, dx)
    y = np.arange(0, Ly, dy)

    X, Y = np.meshgrid(x, y)

    h = H(X, Y, params)
    
    return h
    
def analytical_circ(params, t, cL, H, nonlin=None):
    #from scipy.ndimage import uniform_filter1d
    T = params["T"]
    outputtime = params["outputtime"]
    dt = params["dt"]
    Ly = params["Ly"]
    rho = params["rho"]
    R = params["R"]
    d = params["d"]

    omega = 2 * np.pi / T
    t_hr = np.arange(0, len(t)*outputtime, dt)
    window = round(outputtime / dt)  
    
    windforce_hr = -d*Ly*np.sin(omega * t_hr)/(rho*H*cL)
    forcing = windforce_hr.reshape(-1, window).mean(axis=1)
    #forcing = uniform_filter1d(windforce_hr, size=window)[::window]
    
    if nonlin is not None:
        forcing += nonlin
        
    analytical = np.zeros_like(t)
    for i in np.arange(1,len(t)):
        filtered_forcing = np.exp(-R*(t[i:0:-1])/H)*forcing[:i]
        analytical[i] = np.sum(filtered_forcing*outputtime)
    return analytical