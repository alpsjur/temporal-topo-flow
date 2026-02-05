import json
import sys
import copy

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
    "tmax": 4 * 86400.0,        # 4 days in seconds
    "outputtime": 3 * 3600.0,   # 3 hours in seconds

    # Forcing parameters
    "tau0": 0.0001,             # maximum kinematic forcing [m2 s-2]
    "T": 4 * 86400.0,           # 4 days in seconds
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


def update_config(file_path, default_params):
    """
    Load configuration from a JSON file and apply overrides 
    to a copy of the default parameters.

    Only keys present in the config file will be used to overwrite 
    values in the copy. The original `default_params` remains unchanged.

    Args:
        file_path (str): Path to the JSON configuration file.
        default_params (dict): Dictionary of default parameter values.

    Returns:
        dict: New parameters dictionary with file overrides applied.
    """
    params = copy.deepcopy(default_params)  # Make a safe copy
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
            params.update(config)  # Update the copy
    except FileNotFoundError:
        print(f"Warning: Configuration file '{file_path}' not found. Using default parameters.")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file '{file_path}': {e}")
    return params


def load_config(config_path=None):
    """
    Load simulation parameters from either a given JSON file, 
    from the command-line argument, or fall back to defaults.

    Priority:
    1. Explicit config_path argument
    2. sys.argv[1] if provided
    3. default_params

    Args:
        config_path (str, optional): Path to a JSON config file.

    Returns:
        dict: Dictionary of simulation parameters.
    """
    if config_path is not None:
        print(f"Loading configuration from {config_path}")
        return update_config(config_path, default_params)

    elif len(sys.argv) > 1:
        config_path = sys.argv[1]
        print(f"Loading configuration from {config_path}")
        return update_config(config_path, default_params)

    else:
        print("No configuration file provided. Using default parameters.")