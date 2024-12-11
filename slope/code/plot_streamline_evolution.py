import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
from cmcrameri import cm
from utils import get_h, analytical_circ, load_dataset, load_parameters, truncate_time_series

# Set seaborn style
sns.set_style("whitegrid")

def load_and_prepare_data():
    """Load parameters, dataset, and truncate the time series."""
    params = load_parameters()
    ds = load_dataset(params["filepath"], params["name"])
    ds = truncate_time_series(ds)
    return params, ds

def preprocess_data(params, ds):
    """Extract and preprocess data from the dataset."""
    x = ds.xC.values / 1e3  # Convert x-coordinates to kilometers
    y = ds.yC.values / 1e3  # Convert y-coordinates to kilometers
    u = ds.u.interp(xF=ds.xC.values)
    v = ds.v.interp(yF=ds.yC.values)
    t = ds.time / np.timedelta64(1, 's')  # Time in seconds
    t_days = ds.time / np.timedelta64(1, 'D')  # Time in days
    tmax = len(t) * params["outputtime"]
    return x, y, u, v, t, t_days, tmax

def calculate_timesteps(tmax, T, outputtime):
    """Calculate key timesteps for visualization."""
    nfc = round(np.floor(tmax / T))  # Number of forcing cycles
    nts_fc = T / outputtime  # Number of time steps in a forcing cycle
    timesteps = [round(nts_fc * ((nfc - 1) + i / 4)) for i in range(4)]
    return timesteps

def plot_results(params, x, y, u, v, t_days, timesteps, tmax):
    """Create and save plots for the analysis."""
    T = params["T"]
    d = params["d"]
    outputtime = params["outputtime"]

    # Setup figure and layout
    fig = plt.figure(layout="constrained", figsize=(12, 12))
    axd = fig.subplot_mosaic(
        [
            ["forcing"] * 2,
            ["sl0", "sl1"],
            ["sl0", "sl1"],
            ["sl2", "sl3"],
            ["sl2", "sl3"],
        ],
    )

    # Calculate forcing and analytical response
    forcing = d * np.sin(2 * np.pi * t_days / T)
    analytical = -analytical_circ(params, t_days, 90e3, 500)
    analytical *= d / np.max(analytical[timesteps[0]:timesteps[-1]])

    # Plot forcing
    sec2day = 1 / 86400
    tstart = (timesteps[0] - 5) * outputtime * sec2day
    tstop = (timesteps[-1] + 5) * outputtime * sec2day
    axd["forcing"].set_xlim(tstart, tstop)

    axd["forcing"].plot(t_days[timesteps[0] - 5:], forcing[timesteps[0] - 5:],
                         color="gray", label="forcing")
    axd["forcing"].plot(t_days[timesteps[0] - 5:], analytical[timesteps[0] - 5:],
                         color="darkorange", label="scaled linear response\nat H=500m")
    axd["forcing"].legend()

    # Retrieve depth field
    h = get_h(params)

    for i, ts in enumerate(timesteps):
        # Add vertical line for each timestep
        axd["forcing"].axvline(ts * outputtime * sec2day, color="cornflowerblue")

        ax = axd[f"sl{i}"]

        # Add time step annotation
        ax.text(0.95, 0.95, f"Time step {i}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="lightgray"))

        # Plot depth field and streamlines
        ax.pcolormesh(x, y, h, cmap="Grays", alpha=0.7)
        U = u.isel(time=ts).values
        V = v.isel(time=ts).values
        ax.streamplot(x, y, U, V, color="cornflowerblue")

        # Set aspect ratio to equal
        ax.set_aspect("equal")

    # Finalize and save the figure
    fig.tight_layout()
    fig.savefig("slope/figures/streamlines/" + params["name"] + "_evolution.png")
    plt.show()

def main():
    """Main function to execute the pipeline."""
    try:
        # Load and preprocess data
        params, ds = load_and_prepare_data()
        x, y, u, v, t, t_days, tmax = preprocess_data(params, ds)

        # Calculate timesteps
        timesteps = calculate_timesteps(tmax, params["T"], params["outputtime"])

        # Plot results
        plot_results(params, x, y, u, v, t_days, timesteps, tmax)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except KeyError as e:
        print(f"Error: Missing parameter - {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
