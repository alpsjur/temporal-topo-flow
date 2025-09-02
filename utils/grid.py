from skimage import measure
from scipy.interpolate import interp1d
import numpy as np
import xarray as xr
import xgcm
from copy import deepcopy
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.config import default_params

def extract_uniform_contour(bath, x, y, H_target, N_points):
    """
    Extract a depth contour at a target depth and resample it to uniform arclength.
    If no contour is found (e.g., flat bathymetry), return a horizontal line.

    Args:
        bath (2D ndarray): Bathymetry array of shape (Ny, Nx).
        x (1D ndarray): x-coordinates corresponding to axis 1 of `bath`.
        y (1D ndarray): y-coordinates corresponding to axis 0 of `bath`.
        H_target (float): Target depth at which to extract the contour.
        N_points (int): Number of points to sample uniformly along the contour.

    Returns:
        Tuple[ndarray, ndarray, ndarray]: A tuple (x_uniform, y_uniform, ds) where:
            - x_uniform (1D ndarray): x-coordinates of the uniform contour.
            - y_uniform (1D ndarray): y-coordinates of the uniform contour.
    """
    # Find contours at the specified depth level
    contours = measure.find_contours(bath, level=H_target)
    if not contours:
        # No contour found – return a straight line
        x_uniform = np.linspace(x[0], x[-1], N_points)
        y_level = y[np.argmin(np.abs(np.mean(bath, axis=1) - H_target))]
        y_uniform = np.full_like(x_uniform, y_level)
        
        return x_uniform, y_uniform

    # Choose the longest contour
    contour = max(contours, key=len)
    
    # Contour start to the far right, so we need to revert it
    contour = contour[::-1]

    # Convert from index space to physical coordinates
    y_contour = np.interp(contour[:, 1], np.arange(len(y)), y)
    x_contour = np.interp(contour[:, 0], np.arange(len(x)), x)

    if len(x_contour) < len(x):
        # Degenerate case – fallback to straight line
        x_uniform = np.linspace(x[0], x[-1], N_points)
        y_level = y[np.argmin(np.abs(np.mean(bath, axis=1) - H_target))]
        y_uniform = np.full_like(x_uniform, y_level)

        return x_uniform, y_uniform

    # Arclength along the contour
    s_raw = np.insert(np.cumsum(np.sqrt(np.diff(x_contour)**2 + np.diff(y_contour)**2)), 0, 0)
    s_uniform = np.linspace(s_raw[0], s_raw[-1], N_points)

    # Interpolate along arclength
    x_interp = interp1d(s_raw, x_contour, kind='cubic')
    y_interp = interp1d(s_raw, y_contour, kind='cubic')

    x_uniform = x_interp(s_uniform)
    y_uniform = y_interp(s_uniform)

    return x_uniform, y_uniform

def contour_segments(x_contour, y_contour, Lx=default_params["Lx"]):
    """
    Compute centered unit tangent vectors and segment lengths for a closed contour
    in a periodic x-domain of length Lx.

    Args:
        x_contour (ndarray): x-coordinates of the contour points (1D array).
        y_contour (ndarray): y-coordinates of the contour points (1D array).
        Lx (float): Periodic domain length in the x-direction.

    Returns:
        Tuple[ndarray, ndarray, ndarray]: A tuple (dtx, dty, dl) where:
            - dtx (ndarray): x-component of the unit tangent vector at each point.
            - dty (ndarray): y-component of the unit tangent vector at each point.
            - dl (ndarray): Length of the centered segment at each point.
    """
    def periodic_diff(a, axis_shift, period):
        """Compute periodic-aware difference with wraparound correction."""
        diff = np.roll(a, axis_shift) - a
        diff = (diff + 0.5 * period) % period - 0.5 * period
        return diff

    # Forward and backward differences, corrected for periodic jumps in x
    dx_forward = periodic_diff(x_contour, -1, Lx)
    dx_backward = -periodic_diff(x_contour, 1, Lx)
    dx = 0.5 * (dx_forward + dx_backward)

    dy_forward = np.roll(y_contour, -1) - y_contour
    dy_backward = y_contour - np.roll(y_contour, 1)
    dy = 0.5 * (dy_forward + dy_backward)

    # Segment lengths and unit tangent vectors
    dl = np.sqrt(dx**2 + dy**2)
    dtx = dx / dl
    dty = dy / dl

    return dtx, dty, dl

    
    

def depth_following_grid(ds, H_targets, N_points=120):
    """
    Construct a depth-following grid aligned with contours of bathymetry.

    For each target depth in `H_targets`, this function extracts a constant depth contour 
    from the bathymetry field and samples it uniformly in arc length. It then computes 
    unit tangent vectors and segment lengths along each contour.

    Args:
        ds (xr.Dataset): Dataset containing coordinates `xC`, `yC`, and a 2D bathymetry variable `bath`.
        H_targets (array-like): List or array of target depths (in the same units as `bath`) 
                                at which to extract depth contours.
        N_points (int, optional): Number of equally spaced points to sample along each contour. Default is 120.

    Returns:
        xr.Dataset: A new dataset representing the depth-following grid. Contains:
            - x, y: coordinates of contour points (shape: len(ds.yC) × N_points)
            - dtx, dty: unit tangent vectors along the contours
            - dl: local segment lengths
            - depth: 1D array of target depths
            - i, j: index dimensions for contour points and depth levels
    """
    x = ds.xC.values
    y = ds.yC.values
    bath = ds.bath.values


    Ny = len(H_targets)

    # Preallocate arrays for grid structure
    Xc = np.zeros((Ny, N_points))       # x-coordinates of contour points
    Yc = np.zeros((Ny, N_points))       # y-coordinates of contour points
    dtX = np.ones((Ny, N_points))       # x-component of tangent vector
    dtY = np.zeros((Ny, N_points))      # y-component of tangent vector
    dL = np.ones((Ny, N_points))        # segment lengths

    # Construct contour-following grid
    for j, H_target in enumerate(H_targets):
        x_contour, y_contour = extract_uniform_contour(bath, x, y, H_target, N_points)
        dtx, dty, dl = contour_segments(x_contour, y_contour, Lx=x[-1] - x[0])  # assuming periodicity in x
        Xc[j] = x_contour
        Yc[j] = y_contour
        dtX[j] = dtx
        dtY[j] = dty
        dL[j] = dl

    return xr.Dataset(
        data_vars=dict(
            dtx=(["j", "i"], dtX),         # unit tangent x-component
            dty=(["j", "i"], dtY),         # unit tangent y-component
            dl=(["j", "i"], dL),           # segment length
            depth=(["j"], H_targets)       # contour target depths
        ),
        coords=dict(
            x=(["j", "i"], Xc),            # x-coordinate of contour points
            y=(["j", "i"], Yc),            # y-coordinate of contour points
            i=np.arange(N_points, dtype=np.int32),
            j=np.arange(Ny, dtype=np.int32),
        ),
    )

def construct_xgcm_grid(ds, params):
    """
    Construct an xgcm.Grid object from a dataset with specified grid metrics.

    Adds `dx`, `dy`, and `area` as DataArrays to the dataset and defines
    grid axes and metric relationships for interpolation and integration.

    Args:
        ds (xr.Dataset): Simulation output.
        params (dict): Dictionary of configuration parameters.

    Returns:
        xgcm.Grid: An xgcm Grid object configured with metrics and periodicity in x.
    """    
    dx = params["dx"]
    dy = params["dy"]
    
    # === Broadcast grid metrics
    dx_2d = xr.DataArray(dx * np.ones((len(ds['yC']), len(ds['xF']))),
                            dims=('yC', 'xF'), coords={'yC': ds['yC'], 'xF': ds['xF']})
    dy_2d = xr.DataArray(dy * np.ones((len(ds['yF']), len(ds['xC']))),
                            dims=('yF', 'xC'), coords={'yF': ds['yF'], 'xC': ds['xC']})
    area = xr.DataArray(dx * dy * np.ones((len(ds['yC']), len(ds['xC']))),
                        dims=('yC', 'xC'), coords={'yC': ds['yC'], 'xC': ds['xC']})

    ds['dx'] = dx_2d
    ds['dy'] = dy_2d
    ds['area'] = area

    # === Define grid and attach metrics
    coords = {
        'X': {'center': 'xC', 'left': 'xF'},
        'Y': {'center': 'yC', 'left': 'yF'}
    }

    metrics = {
        ('X',): ['dx'],
        ('Y',): ['dy'],
        ('X', 'Y'): ['area']
    }

    grid = xgcm.Grid(ds, coords=coords, metrics=metrics, periodic=['X'])
    
    return grid

def interp_var(grid, da, shiftdict, params=default_params):
    """
    Interpolate a DataArray along specified axes using xgcm.

    This function handles edge trimming and extrapolation to maintain consistent array size,
    particularly for bounded or staggered grids in the y-direction.

    Args:
        grid (xgcm.Grid): Grid object used to perform the interpolation.
        da (xr.DataArray): Data array to interpolate.
        shiftdict (dict): Dictionary mapping old dimensions to new ones, e.g.:
            - {"xF": "xC"} for face-to-center interpolation,
            - {"yC": "yF"} for center-to-face interpolation.
        params (dict): Dictionary of configuration parameters.

    Returns:
        xr.DataArray: Interpolated DataArray with shifted dimensions.
    """
    for old_dim, new_dim in shiftdict.items():
        if old_dim not in da.dims:
            continue

        axis = "X" if old_dim.startswith("x") else "Y"
        to = "left" if new_dim.endswith("F") else "center"

        # === Handle bounded y-axis: yF → yC
        if axis == "Y" and old_dim == "yF" and new_dim == "yC":
            da = da.isel(yF=slice(None, -1))  # remove last point

        # === Interpolate
        da_interp = grid.interp(da, axis=axis, to=to)
        #da_interp = da_interp.rename({old_dim: new_dim})

        # === Handle bounded y-axis: yC → yF
        if axis == "Y" and old_dim == "yC" and new_dim == "yF":
            # Interpolated data is N, we want N+1: extrapolate last point
            last_val = da_interp.isel(yF=-1)
            last_val = last_val.expand_dims(yF=[da_interp.yF[-1] + params["dy"]])
            da_interp = xr.concat([da_interp, last_val], dim="yF")

        da = da_interp
    return da

def interp_ds(ds, params, varnames, shiftdict={"xF": "xC", "yF": "yC"}):
    """
    Interpolate selected variables in a dataset from one grid position to another.

    This wraps `interp_var` for multiple variables, applying the same shift logic.

    Args:
        ds (xr.Dataset): Dataset containing variables to be interpolated.
        params (dict): Dictionary of configuration parameters, passed to interpolation functions.
        varnames (list of str): Names of variables to interpolate.
        shiftdict (dict): Mapping of dimensions to shift, e.g.:
            - {"xF": "xC", "yF": "yC"} for face-to-center interpolation.

    Returns:
        xr.Dataset: A copy of `ds` with selected variables interpolated to new grid positions.
    """
    ds_out = deepcopy(ds)  # preserve input
    grid = construct_xgcm_grid(ds, params)

    for var in varnames:
        da = ds[var]
        da_interp = interp_var(grid, da, shiftdict)
        ds_out[var] = da_interp

    return ds_out



def interp2H(dsC, Hgrid):
    """
    Interpolate a dataset onto a depth-following horizontal contour grid.

    This performs bilinear interpolation of `dsC` from a structured grid
    (defined by `xC`, `yC`) onto the 2D contour positions defined in `Hgrid`.

    Args:
        dsC (xr.Dataset): Dataset defined on a regular grid using coordinates `xC` and `yC`.
        Hgrid (xr.Dataset): Depth-following grid returned by `depth_following_grid`, containing
                            2D coordinate fields `x` and `y`.

    Returns:
        xr.Dataset: Interpolated dataset with variables aligned along the same contours
                    as `Hgrid`, using dimensions `j` and `i`.
    """
    # Bilinear interpolation of dataset to contour grid points
    dsH = dsC.interp(
        xC=Hgrid['x'],
        yC=Hgrid['y'],
        method="linear"
    )
    return dsH


def rotate_dsH_tangential(dsH, Hgrid, map={"ui": ("u", "v"), "forcing_i": ("forcing_x", "forcing_y")}):
    """
    Compute tangential components of vector fields on a depth-following contour grid.

    For each variable pair in `map`, this function computes the projection onto the
    local tangent direction defined by `Hgrid.dtx` and `Hgrid.dty`. Skips any pair
    if one or both input variables are missing from `dsH`.

    Args:
        dsH (xr.Dataset): Dataset on the depth-following grid (dims: ["j", "i"]).
        Hgrid (xr.Dataset): Grid dataset with unit tangents `dtx`, `dty`.
        map (dict): Mapping of output variable name to (u, v) input pair.

    Returns:
        xr.Dataset: Updated dataset with new tangential components added.
    """
    for outvar, (u_name, v_name) in map.items():
        if u_name not in dsH or v_name not in dsH:
            print(f"Skipping '{outvar}': missing '{u_name}' or '{v_name}'")
            continue

        u = dsH[u_name]
        v = dsH[v_name]
        dsH[outvar] = u * Hgrid.dtx + v * Hgrid.dty

    return dsH

def rotate_dsH_normal(dsH, Hgrid, map={"zetaflux": ("zetau", "zetav")}):
    """
    Compute normal components of vector fields on a depth-following contour grid.

    For each variable pair in `map`, this function computes the projection onto the
    direction normal to the local tangent vector. Skips any pair if variables are missing.

    Normal vector is defined as:
        n̂ = (-dty, dtx)

    Args:
        dsH (xr.Dataset): Dataset on the depth-following grid (dims: ["j", "i"]).
        Hgrid (xr.Dataset): Grid dataset with unit tangents `dtx`, `dty`.
        map (dict): Mapping of output variable name to (u, v) input pair.

    Returns:
        xr.Dataset: Updated dataset with new normal components added.
    """
    for outvar, (u_name, v_name) in map.items():
        if u_name not in dsH or v_name not in dsH:
            print(f"Skipping '{outvar}': missing '{u_name}' or '{v_name}'")
            continue

        u = dsH[u_name]
        v = dsH[v_name]
        # Normal projection: u * (-dty) + v * dtx
        dsH[outvar] = -u * Hgrid.dty + v * Hgrid.dtx

    return dsH


def prepare_dsH(ds, params, H_targets):
    """
    Interpolate a dataset onto a depth-following (contour-aligned) grid and 
    attach geometric metadata from the grid.

    This function performs the following steps:
        1. Interpolates all variables in `ds` to the C-grid using `interp_ds`.
        2. Constructs a depth-following grid based on the bathymetry and specified target depths.
        3. Interpolates all fields from the C-grid onto the contour-following grid.
        4. Computes tangential and normal components of selected vector fields.
        5. Attaches all geometric variables from the depth-following grid (e.g. x, y, dtx, dty, dl, depth) to the result.

    Args:
        ds (xr.Dataset): Original dataset defined on the model grid 
                         (may include staggered variables).
        params (dict): Model configuration dictionary with grid and geometry settings.
        H_targets (array-like): List or array of target depth levels (in meters) 
                                at which to extract horizontal bathymetric contours.

    Returns:
        xr.Dataset: Dataset `dsH` interpolated onto the depth-following grid,
                    with dimensions ["time", "j", "i"]. Includes:
                        - Interpolated physical variables
                        - Tangential and normal projections (e.g., 'ui', 'un')
                        - Geometric metadata from the contour grid:
                          'x', 'y', 'depth', 'dtx', 'dty', 'dl'
    """
    # interpolate all variables to C-grid
    dsC = interp_ds(ds, params, varnames=list(ds.data_vars))

    # Create depth-following grid for H_targets, and interpolate ds to that grid
    Hgrid = depth_following_grid(dsC, H_targets)
    dsH = interp2H(dsC, Hgrid)

    # rotate vectors tangential and normal to grid
    # which vectors to rotate can be spesified using the map arg
    rotate_dsH_tangential(dsH, Hgrid)
    rotate_dsH_normal(dsH, Hgrid)
    
    for var in list(Hgrid.data_vars):
        dsH[var] = Hgrid[var]
    
    # remove redundant coordinates
    dsH = dsH.drop_vars([v for v in ["xC", "yC", "yF", "xF"] if v in dsH])
    
    return dsH