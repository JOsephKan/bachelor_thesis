# This program is to compute power spectrum of the CNTL and Temperature
# %% Section 1
# Import package
import numpy as np;
import netCDF4 as nc;
from matplotlib import pyplot as plt;

# %% Section 2
# Load data
path: str = "/work/b11209013/2024_Research/MPAS/merged_data/"

dims: dict[str, np.ndarray] = dict();
data: dict[str, np.ndarray] = dict();

with nc.Dataset(f"{path}CNTL/theta.nc", "r") as f:
    for key in f.dimensions.keys():
        dims[key] = f.variables[key][...];

    lat_lim: tuple[int] = np.where((dims["lat"] >= -5) & (dims["lat"] <= 5))[0];
    dims["lat"] = dims["lat"][lat_lim];

    data["cntl"] = f.variables["theta"][:, :, lat_lim, :] * (1000/dims["lev"])[None, :, None, None]**(-0.286);

with nc.Dataset(f"{path}NCRF/theta.nc", "r") as f:
    data["ncrf"] = f.variables["theta"][:, :, lat_lim, :] * (1000/dims["lev"])[None, :, None, None]**(-0.286);
