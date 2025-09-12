import numpy as np

def bathymetry_xy(x, y, p):
    """
    Evaluate the parametric bathymetry h(x, y) at a single point.

    Args:
        x (float): x-coordinate (horizontal position).
        y (float): y-coordinate (meridional position).
        p (dict): Dictionary of configuration parameters.

    Returns:
        float: Depth h(x, y) at the specified point.
    """
    steepness = 1 / np.cosh(np.pi * (y - p["yc"]) / p["W"])**2
    delta = p["Acorr"] * np.sin(2 * np.pi * x / p["lam"]) * steepness
    h = p["Hsh"] + 0.5 * (p["Hbs"] - p["Hsh"]) * (1 - np.tanh(np.pi * (y - p["yc"] - delta) / p["W"]))

    return h


def generate_bathymetry(p, full=False):
    """
    Generate a 2D bathymetry field h(x, y) over the model grid.

    Args:
        p (dict): Dictionary of configuration parameters.
        full (bool): If True, the grid will extend from 0 to 90km in both direction.
            Else, use center grid points.  

    Returns:
        Tuple[ndarray, ndarray, ndarray]: A tuple (X, Y, h) where:
            - X (ndarray): x-coordinate meshgrid (shape Ny x Nx).
            - Y (ndarray): y-coordinate meshgrid (shape Ny x Nx).
            - h (ndarray): Bathymetry depth field on the grid.
    """
    if full:
        x = np.arange(0, p["Lx"]+p["dx"]/2, p["dx"])
        y = np.arange(0, p["Ly"]+p["dy"]/2, p["dy"])
    else:
        x = np.arange(p["dx"]/2, p["Lx"], p["dx"])
        y = np.arange(p["dy"]/2, p["Ly"], p["dy"])
    X, Y = np.meshgrid(x, y
                       #, indexing="xy"
                       )

    h_func = np.vectorize(lambda x, y: bathymetry_xy(x, y, p))
    h = h_func(X, Y)
    return X, Y, h


def bathymetry_gradient(h, dx, dy):
    """
    Compute the gradients of the bathymetry field in x and y directions.

    Uses periodic boundary conditions in the x-direction (axis=1).

    Args:
        h (ndarray): 2D bathymetry array with shape (Ny, Nx).
        dx (float): Grid spacing in the x-direction (assumed periodic).
        dy (float): Grid spacing in the y-direction (non-periodic).

    Returns:
        Tuple[ndarray, ndarray]: A tuple (dh_dx, dh_dy) where:
            - dh_dx is the partial derivative ∂h/∂x using periodic boundaries.
            - dh_dy is the partial derivative ∂h/∂y using second-order central differences.
    """
    # Periodic derivative in x (axis=1)
    dh_dx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2 * dx)

    # Standard centered differences in y (axis=0, non-periodic)
    dh_dy = np.empty_like(h)
    dh_dy[1:-1,:] = (h[2:,:] - h[:-2,:]) / (2 * dy)
    dh_dy[0, :] = (h[1,:] - h[0,:]) / dy           # Forward difference at lower boundary
    dh_dy[-1, :] = (h[-1,:] - h[-2,:]) / dy        # Backward difference at upper boundary

    return dh_dx, dh_dy
