import json

default_params = {
    # Run name and output path
    "name": "default",
    "filepath": "output/",

    # Grid parameters
    "dx": 1e3,
    "dy": 1e3,
    "Lx": 90e3,
    "Ly": 90e3,

    # Simulation parameters
    "dt": 4.0,
    "tmax": 4 * 86400.0,        # 16 days in seconds
    "outputtime": 3 * 3600.0,   # 3 hours in seconds

    # Forcing parameters
    "tau0": 0.0001,             # maximum kinematic forcing [m2 s-2]
    "T": 4 * 86400.0,           # 16 days in seconds
    "R": 5e-4,

    # Coriolis and gravity
    "f": 1e-4,
    "gravitational_acceleration": 9.81,

    # Bathymetry parameters
    "W": 30e3,                  # Width of slope
    "yc": 45e3,                 # Center coordinate of slope
    "Hsh": 900.0,               # Depth of shelf
    "Hbs": 100.0,               # Depth of deep basin
    "Acorr": 10e3,              # Horizontal length scale of corrugations
    "lam": 45e3                 # Wavelength of corrugations
}


def overwrite_config(file_path, default_params):
    """
    Load configuration from a JSON file and overwrite default parameters.
    Only prameters included in the configuration file will be overwritten.

    Parameters:
        file_path (str): Path to the configuration file.
        default_params (dict): Default parameters dictionary.

    Returns:
        dict: Updated parameters.
    """
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
            default_params.update(config)  # Overwrite defaults with values from the config
    except FileNotFoundError:
        print(f"Warning: Configuration file '{file_path}' not found. Using default parameters.")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file '{file_path}': {e}")
    return default_params

def load_config():
    """Load simulation parameters from configuration file 
    (if runtime argument provided), else defaults."""
    import sys
    if len(sys.argv) == 2:
        config_path = sys.argv[1]
        print(f"Loading configuration from {config_path}")
        return overwrite_config(config_path, default_params)
    else:
        print("No configuration file provided. Using default parameters.")
        return default_params