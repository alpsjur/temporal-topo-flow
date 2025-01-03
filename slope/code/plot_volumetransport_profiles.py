import numpy as np
import os
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
from cmcrameri import cm
from utils import load_config, get_h, default_params, load_dataset, truncate_time_series 

# Configure seaborn style
sns.set_style("whitegrid")

def loop_over_configs(configs, ax):
    for config in configs:
        params = load_config(config, default_params.copy())
        ds = load_and_prepare_data(params)
        
        
        x, Vh = preprocess_data(params, ds)
        
        name = params["T"]/(60*60*24)
        
        ax.plot(x, Vh, label=name)
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
    
    h = get_h(params)
    
    ds_slize = ds.isel(time=slice(-n, -1))
    
    V = ds_slize.v.mean("time")
    Vh = (V*h).mean("yF")
    
    return x, Vh 


def main():
    try:
        fig, ax = setup_plots()
        
        path = "slope/configs/"
        #configs = os.listdir(path)
        configs = [f"slope-00{i}.json" for i in [1,2,5,6]]
        configs = [path+config for config in configs]
        loop_over_configs(configs, ax)
        ax.legend()
        
        # Optional: Show plots interactively
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()