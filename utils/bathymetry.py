import numpy as np

def bathymetry_xy(x, y, p):
    """
    Evaluate the parametric bathymetry h(x, y) at a single point.
    """
    if y > (p["yc"] - p["W"]):  # Slope region
        steepness = 1 / np.cosh(np.pi * (y - p["yc"]) / p["W"])**2
        delta = p["Acorr"] * np.sin(2 * np.pi * x / p["lam"]) * steepness
        h = p["Hsh"] + 0.5 * (p["Hbs"] - p["Hsh"]) * (1 + np.tanh(np.pi * (p["yc"] - y - delta) / p["W"]))
    else:  # Central basin
        h = p["Hbs"]
    return h


def generate_bathymetry(p):
    """
    Generate a 2D bathymetry field h(x, y) over the model grid.
    Returns meshgrids X, Y and bathymetry h.
    """
    x = np.arange(0, p["Lx"]+p["dx"]/2, p["dx"])
    y = np.arange(0, p["Ly"]+p["dy"]/2, p["dy"])
    X, Y = np.meshgrid(x, y, indexing="ij")

    h_func = np.vectorize(lambda x, y: bathymetry_xy(x, y, p))
    h = h_func(X, Y)
    return X, Y, h


def bathymetry_gradient(h, dx, dy):
    """
    Compute gradients of the bathymetry field.
    """
    dh_dx, dh_dy = np.gradient(h, dx, dy, edge_order=2)
    return dh_dx, dh_dy
