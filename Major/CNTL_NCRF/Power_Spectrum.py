# This program is to compute the powr spefctrum of temperature anomaly of CNTL and NCRF
# %% Section 1
# Import packages
import sys
import numpy as np;
import netCDF4 as nc;
import matplotlib.pyplot as plt;
from scipy.ndimage import convolve1d;

sys.path.append("/home/b11209013/Package");
import Theory as th; #type: ignore

# %% Section 2
# Load the data
path: str = "/work/b11209013/2024_Research/MPAS/merged_data/";

dims: dict[str, np.ndarray] = dict();
data: dict[str, np.ndarray] = dict();


# # CNTL
with nc.Dataset(f"{path}CNTL/theta.nc", "r") as cntl:
    for key in cntl.dimensions.keys():
        dims[key] = cntl.variables[key][:];
    
    lat_lim: tuple[int] = np.where((dims["lat"] >= -5) & (dims["lat"] <= 5))[0];

    convert: np.ndarray = (1000. / dims["lev"][None, :, None, None]) ** (-0.286);
    
    data["cntl"] = cntl.variables["theta"][:, :, lat_lim, :] * convert;
    
# # NCRF
with nc.Dataset(f"{path}NCRF/theta.nc", "r") as ncrf:
    data["ncrf"] = ncrf.variables["theta"][:, :, lat_lim, :] * convert;

ltime, llev, llat, llon = data["cntl"].shape;

# %% Section 3
# Processing data
# # Remove climatology and zonal mean
data_rm_cli: dict[str, np.ndarray] = dict(
    (exp, data[exp] - np.mean(data[exp], axis=(0, 3), keepdims=True))
    for exp in data.keys()
);

# # Construct symmetric data
data_sym: dict[str, np.ndarray] = dict(
    (exp, ((data_rm_cli[exp] + np.flip(data_rm_cli[exp], axis=2)) / 2).sum(axis=2))
    for exp in data_rm_cli.keys()
);

data_asy: dict[str, np.ndarray] = dict(
    (exp, ((data_rm_cli[exp] - np.flip(data_rm_cli[exp], axis=2)) / 2).sum(axis=2))
    for exp in data_rm_cli.keys()
);

# # Windowing
lsec: int = 120;
hanning: np.ndarray = np.hanning(lsec)[:, None, None];

sym_window: dict[str, np.ndarray] = dict(
    (exp, np.array([
        data_sym[exp][i*60:i*60+lsec, :, :] * hanning
        for i in range(5)
    ]))
    for exp in data_sym.keys()
);

asy_window: dict[str, np.ndarray] = dict(
    (exp, np.array([
        data_asy[exp][i*60:i*60+lsec, :, :] * hanning
        for i in range(5)
    ]))
    for exp in data_asy.keys()
);

# %% Section 4
# Compute power spectrum
def power_spec(
        data: np.ndarray,
) -> np.ndarray:
    fft: np.ndarray = np.fft.fft(data, axis=1);
    fft: np.ndarray = np.fft.ifft(fft, axis=3) * data.shape[3];

    ps : np.ndarray = (fft * fft.conj()) / (data.shape[1] * data.shape[3]) * 2;

    return ps.mean(axis=0).real;

sym_ps: dict[str, np.ndarray] = dict(
    (exp, power_spec(sym_window[exp]))
    for exp in data.keys()
);

asy_ps: dict[str, np.ndarray] = dict(
    (exp, power_spec(asy_window[exp]))
    for exp in data.keys()
);

# # Vertical average with mass wighted
def vertical_avg(
        data: np.ndarray,
        lev : np.ndarray,
) -> np.ndarray:
    data_ave : np.ndarray = (data[:, 1:] + data[:, :-1]) /2.;
    data_vint: np.ndarray = -np.sum(data_ave * np.diff(lev*100.)[None, :, None], axis=1) / -np.sum(np.diff(lev*100.));

    return data_vint;

sym_ps_weight: dict[str, np.ndarray] = dict(
    (exp, vertical_avg(sym_ps[exp], dims["lev"]))
    for exp in data.keys()
);
asy_ps_weight: dict[str, np.ndarray] = dict(
    (exp, vertical_avg(asy_ps[exp], dims["lev"]))
    for exp in data.keys()
);

# # Compute background
def background(data, nsmooth=20):
    kernel = np.array([1, 2, 1])
    kernel = kernel / kernel.sum()

    for _ in range(10):
        data = convolve1d(data, kernel, mode='nearest')

    data_low  = data[:data.shape[0]//2]
    data_high = data[data.shape[0]//2:]

    for _ in range(10):
        data_low = convolve1d(data_low, kernel, mode='nearest')

    for _ in range(40):
        data_high = convolve1d(data_high, kernel, mode='nearest')

    data = np.concatenate([data_low, data_high], axis=0)

    return data

bg: np.ndarray = background(
    (sym_ps_weight["cntl"] + asy_ps_weight["cntl"])/2
);

sym_peak: dict[str, np.ndarray] = dict(
    (exp, sym_ps_weight[exp] / bg)
    for exp in data.keys()
);

wn: np.ndarray = np.fft.fftshift(np.fft.fftfreq(llon, d=1/llon).astype(int));
fr: np.ndarray = np.fft.fftshift(np.fft.fftfreq(lsec, d=1/4));

fr_ana, wn_ana = th.genDispersionCurves(Ahe=[8, 25, 90]);
e_cond = np.where(wn_ana[3, 0] <= 0)[0];

plt.rcParams["font.family"] = "serif";

fig, ax = plt.subplots(1, 2, figsize=(12, 7), sharey=True);
plt.subplots_adjust(left=0.08, right=0.96, bottom=0.03, top=0.9);
cntl_ps = ax[0].contourf(
    wn, fr[fr>0],
    np.fft.fftshift(sym_peak["cntl"])[fr>0],
    cmap="Blues",
    levels=np.linspace(1, 10, 19),
    extend="max",
);
for i in range(3):
    ax[0].plot(wn_ana[3, i, e_cond], fr_ana[3, i, e_cond], color="black", linewidth=1);
    ax[0].plot(wn_ana[4, i], fr_ana[4, i], color="black", linewidth=1);
    ax[0].plot(wn_ana[3, i], fr_ana[5, i], color="black", linewidth=1);
ax[0].set_xticks(np.linspace(-14, 14, 8, dtype=int));
ax[0].set_yticks(np.linspace(0, 0.5, 6));
ax[0].axvline(0, linestyle="--", color="black")
ax[0].axhline(1/3 , linestyle="--", color="black");
ax[0].axhline(1/8 , linestyle="--", color="black");
ax[0].axhline(1/20, linestyle="--", color="black");
ax[0].text(15, 1/3 , "3 Days", ha="right", va="bottom");
ax[0].text(15, 1/8 , "8 Days", ha="right", va="bottom");
ax[0].text(15, 1/20, "20 Days", ha="right", va="bottom");
ax[0].text(0, -0.06, "Zonal Wavenumber", ha="center", fontsize=14);
ax[0].text(-20, 0.25, "Freqquency [CPD]", va="center", rotation=90, fontsize=14);
ax[0].set_xlim(-15, 15);
ax[0].set_ylim(0, 1/2);
ax[0].text(0, 0.52, "CNTL", ha="center", fontsize=16)



ncrf_ps = ax[1].contourf(
    wn, fr[fr>0],
    np.fft.fftshift(sym_peak["ncrf"])[fr>0],
    cmap="Blues",
    levels=np.linspace(1, 10, 19),
    extend="max",
);
for i in range(3):
    ax[1].plot(wn_ana[3, i, e_cond], fr_ana[3, i, e_cond], color="black", linewidth=1);
    ax[1].plot(wn_ana[4, i], fr_ana[4, i], color="black", linewidth=1);
    ax[1].plot(wn_ana[3, i], fr_ana[5, i], color="black", linewidth=1);
ax[1].set_xticks(np.linspace(-14, 14, 8, dtype=int));
ax[1].set_yticks(np.linspace(0, 0.5, 6));
ax[1].axvline(0, linestyle="--", color="black")
ax[1].axhline(1/3 , linestyle="--", color="black");
ax[1].axhline(1/8 , linestyle="--", color="black");
ax[1].axhline(1/20, linestyle="--", color="black");
ax[1].text(15, 1/3 , "3 Days", ha="right", va="bottom");
ax[1].text(15, 1/8 , "8 Days", ha="right", va="bottom");
ax[1].text(15, 1/20, "20 Days", ha="right", va="bottom");
ax[1].text(0, -0.06, "Zonal Wavenumber", ha="center", fontsize=14);
ax[1].set_xlim(-15, 15);
ax[1].set_ylim(0, 1/2);
ax[1].text(0, 0.52, "NCRF", ha="center", fontsize=16)

cbar = plt.colorbar(ncrf_ps, ax=ax, orientation="horizontal", aspect=40, shrink=0.7)
cbar.set_label("Normalized Power", fontsize=14);

plt.savefig("/home/b11209013/Bachelor_Thesis/Major/Figure/Figure04.png", dpi=300);