import xarray as xr
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.grid import prepare_dsH
from utils.config import load_config
from utils.io import read_raw_output


# read simulation output
params = load_config()
ds = read_raw_output(params)


### H contour postprocessing  ###
H_targets = ds.bath.mean("xC").values
dsH = prepare_dsH(ds, params, H_targets)



print(dsH)




