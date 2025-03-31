import numpy as np
import os
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
from cmcrameri import cm as cmc
import matplotlib.cm as cm
from utils import load_config, calculate_bathymetry, default_params, load_dataset, truncate_time_series 

# Configure seaborn style
sns.set_style("whitegrid")

def loop_over_configs(configs, ax):
    colors = cmc.batlow(np.linspace(0, 1, len(configs)))#cm.tab10(np.linspace(0, 1, len(configs)))
    for config, color in zip(configs, colors):
        params = load_config(config, default_params.copy())
        ds = load_and_prepare_data(params)
        
        
        x, Vh = preprocess_data(params, ds)
        
        name = params["T"]/(60*60*24)
        lam = params["lam"]/1e3
        
        #label = f"{name:03n} days"
        #label = f"{lam:.1f} km"
        label = params["name"]
        
        ax.plot(x, Vh, label=label, color=color)
        ax.axhline(np.mean(Vh), color=color, ls="--")
    return

def load_and_prepare_data(params):
    """Load parameters, dataset, and truncate the time series."""
    ds = load_dataset(params["filepath"], params["name"])
    ds = truncate_time_series(ds)
    return ds


def setup_plots():
    """Initialize figures and axes for various plots."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel("x-position [km]")
    ax.set_ylabel("Residual volume transport")

    return fig, ax

def preprocess_data(params, ds):
    x = ds.xC.values / 1e3  # Convert x-coordinates to kilometers
    
    T = params["T"]
    outputtime = params["outputtime"]
    N = 1              # number of forcing periods to average over
    
    n = int(T * N / outputtime) 
    #X, Y = np.meshgrid(ds.xC, ds.yF)
    #h = calculate_bathymetry(X,Y, params)
    
    ds_slize = ds.isel(time=slice(-n, -1))
    
    V = ds_slize.v#.mean("time")
    h = ds_slize.h.values
    h = np.insert(h, 0, h[:,-1,:], axis=1)
    h = 0.5*(h[:,1:,:]+h[:,:-1,:])
    
    Vh = (V*h).mean(("yF", "time"))
    
    return x, Vh 


def main():
    try:
        fig, ax = setup_plots()
        
        path = "slope/configs/"
        #configs = os.listdir(path)
        configs = [f"slope-{i:03}.json" for i in [201, 203]]
        configs = [path+config for config in configs]
        loop_over_configs(configs, ax)
        ax.legend(
            #title="Forcing period"
            #title="Topography wavelength"
            title="config"
            )
        
        #fig.savefig("slope/figures/profiles/Vtransport_profiles_16day_varying_lambda.png")
        #fig.savefig("slope/figures/profiles/Vtransport_profiles_varying_T.png")
        fig.savefig("slope/figures/profiles/Vtransport_profiles_sinusodal.png")
        
        # Optional: Show plots interactively
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()