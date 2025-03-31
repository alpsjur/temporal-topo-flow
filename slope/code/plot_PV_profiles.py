import numpy as np
import os
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
from cmcrameri import cm as cmc
import matplotlib.cm as cm
from utils import load_config, default_params, load_dataset, truncate_time_series 

# Configure seaborn style
sns.set_style("whitegrid")

def loop_over_configs(configs, ax):
    colors = cmc.batlow(np.linspace(0, 1, len(configs)))#cm.tab10(np.linspace(0, 1, len(configs)))
    for config, color in zip(configs, colors):
        params = load_config(config, default_params.copy())
        ds = load_and_prepare_data(params)
        
        
        x, PVmean, PVstd= preprocess_data(params, ds)
        
        name = params["T"]/(60*60*24)
        
        label = f"{name:03n} days"
        label = params["name"]
        
        ax.plot(x, PVmean, label=label, color=color)
        ax.fill_between(x, PVmean-PVstd, PVmean+PVstd, color=color, alpha=0.3)
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
    ax.set_ylabel("potential vorticity [s-1 m-1]")

    return fig, ax

def preprocess_data(params, ds):
    x = ds.xC.values / 1e3  # Convert x-coordinates to kilometers
    
    T = params["T"]
    outputtime = params["outputtime"]
    N = 1              # number of forcing periods to average over
    
    n = int(T * N / outputtime) 
    
    
    X, Y = np.meshgrid(ds.xC, ds.yF)
    
    ds_slice = ds.isel(time=slice(-n, -1))
    
    f = params["f"]
    omegaF = ds_slice.omega.values
    omegaF = np.insert(omegaF, 0, omegaF[:,-1,:], axis=1)
    omega = 0.5*(omegaF[:,1:,1:]+omegaF[:,:-1,-1:])

    h = ds_slice.h
    PV = ((f+omega)/h)
    PVmean = PV.mean(["time", "yC"])
    PVstd = PV.std(["time", "yC"])
    
    return x, PVmean, PVstd


def main():
    try:
        fig, ax = setup_plots()
        
        path = "slope/configs/"
        #configs = os.listdir(path)
        configs = [f"slope-{i:03}.json" for i in [103,104]]
        configs = [path+config for config in configs]
        loop_over_configs(configs, ax)
        ax.legend(
            #title="Forcing period"
            title="Configuration"
            )
        
        #fig.savefig("slope/figures/profiles/PV_profiles_varying_T.png")
        fig.savefig("slope/figures/profiles/PV_profiles_stochastic_tau_bumps.png")
        
        # Optional: Show plots interactively
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()