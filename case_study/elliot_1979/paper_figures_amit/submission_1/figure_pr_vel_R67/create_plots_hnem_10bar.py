import os
import glob
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import nozzlex as nzx

# ======================================================
# CONFIGURATION
# ======================================================
SAVE_FIGS    = True
PLOT_HEM     = False

_HERE    = os.path.dirname(os.path.abspath(__file__))
HNEM_DIR = os.path.join(_HERE, "..", "..", "..", "hnem_solutions", "final_solutions")
HEM_DIR  = os.path.join(_HERE, "..", "..", "..", "hem_solutions")

PRESSURE_TAG    = "10.3bar"        # string that must appear in the HNEM filename
MIXTURE_RATIO   = 39.1             # single mixture ratio for this pressure condition

# nb=nd combinations present in the 10.3 bar solutions
NB_ND_LINESTYLES = {
    1e8:  "--",
    1e10: "-.",
    1e12: (0, (3, 1, 1, 1, 1, 1)),  # dense dash-dot-dot
}

HEM_LINESTYLE = "-"
HEM_LINEWIDTH = 1.6

COLOR = cm.magma(0.45)             # single color for this mixture ratio


# ======================================================
# HELPERS
# ======================================================

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_hnem_filename(path):
    """Return (mr, nb, nd) from filename, or None if no match."""
    name = os.path.basename(path)
    m = re.search(r"mr([0-9]+\.[0-9]+)_nb([0-9.e+E]+)_nd([0-9.e+E]+)\.pkl", name)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def parse_hem_filename(path):
    """Return mixture ratio float from HEM filename, or None."""
    name = os.path.basename(path)
    m = re.search(r"solution_([0-9]+\.[0-9]+)\.pkl", name)
    return float(m.group(1)) if m else None


# ======================================================
# LOAD HNEM
# ======================================================
pkl_files = sorted(glob.glob(os.path.join(HNEM_DIR, "*.pkl")))
if not pkl_files:
    raise FileNotFoundError(f"No .pkl files found in {os.path.abspath(HNEM_DIR)}")

solutions = []
for path in pkl_files:
    if PRESSURE_TAG not in os.path.basename(path):
        continue
    parsed = parse_hnem_filename(path)
    if parsed is None:
        print(f"Skipping (no match): {os.path.basename(path)}")
        continue
    mr, nb, nd = parsed
    if mr != MIXTURE_RATIO:
        continue
    if nb != nd:
        print(f"Skipping (nb != nd): {os.path.basename(path)}")
        continue
    if nb not in NB_ND_LINESTYLES:
        print(f"Skipping (unknown nb={nb:.2e}): {os.path.basename(path)}")
        continue
    try:
        sol = load_pickle(path)
    except Exception as e:
        print(f"Could not load {os.path.basename(path)}: {e}")
        continue
    solutions.append(dict(nb=nb, sol=sol))

print(f"Loaded {len(solutions)} HNEM solutions.")

# ======================================================
# LOAD HEM
# ======================================================
hem_sol = None
for path in sorted(glob.glob(os.path.join(HEM_DIR, "*.pkl"))):
    mr = parse_hem_filename(path)
    if mr is None:
        continue
    if abs(mr - MIXTURE_RATIO) > 0.05:
        continue
    try:
        hem_sol = load_pickle(path)
        print(f"Loaded HEM: {os.path.basename(path)}")
    except Exception as e:
        print(f"Could not load {os.path.basename(path)}: {e}")
    break

# ======================================================
# PLOT
# ======================================================
nzx.set_plot_options(minor_ticks=False, grid=False)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.5))

ax1.set_xlabel("Axial distance [m]")
ax1.set_ylabel("Pressure [kPa]")
ax2.set_xlabel("Axial distance [m]")
ax2.set_ylabel("Velocity [m/s]")

# --- HEM curve ---
if PLOT_HEM and hem_sol is not None:
    x = np.asarray(hem_sol["x"])
    p = np.asarray(hem_sol["p"]) / 1e3
    v = np.asarray(hem_sol["v"])
    ax1.plot(x, p, color=COLOR, linestyle=HEM_LINESTYLE, linewidth=HEM_LINEWIDTH, label=f"HEM R={MIXTURE_RATIO:g}")
    ax2.plot(x, v, color=COLOR, linestyle=HEM_LINESTYLE, linewidth=HEM_LINEWIDTH)

# --- HNEM curves ---
for entry in sorted(solutions, key=lambda e: e["nb"]):
    nb  = entry["nb"]
    sol = entry["sol"]

    x = np.asarray(sol["x"])
    p = np.asarray(sol["p"]) / 1e3
    v = np.asarray(sol["v"])

    ls      = NB_ND_LINESTYLES[nb]
    lbl_nb  = f"$n_b=n_d={nb:.0e}$"

    ax1.plot(x, p, color=COLOR, linestyle=ls, linewidth=1.0)
    ax2.plot(x, v, color=COLOR, linestyle=ls, linewidth=1.0, label=lbl_nb)

# --- Legend ---
import matplotlib.lines as mlines
hnem_handles = [
    mlines.Line2D([], [], color="gray", linestyle=ls, linewidth=1.0,
                  label=f"HNEM $n_b=n_d={nb:.0e}$")
    for nb, ls in NB_ND_LINESTYLES.items()
]
if PLOT_HEM:
    hem_handle = mlines.Line2D([], [], color="gray", linestyle=HEM_LINESTYLE,
                                linewidth=HEM_LINEWIDTH, label="HEM")
    legend_handles = [hem_handle] + hnem_handles
else:
    legend_handles = hnem_handles
ax2.legend(handles=legend_handles, title="Model", loc="upper left",
           fontsize=8, title_fontsize=8)

ax1.set_title(f"p = 10.3 bar,  R = {MIXTURE_RATIO:g}")
fig.tight_layout(pad=1)

if SAVE_FIGS:
    base_filename = os.path.join(os.getcwd(), "hnem_10bar_solutions")
    nzx.savefig_in_formats(fig, base_filename, formats=[".png", ".svg"])

plt.show()
