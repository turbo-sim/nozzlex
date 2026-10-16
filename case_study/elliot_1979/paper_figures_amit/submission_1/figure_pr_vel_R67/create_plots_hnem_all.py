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
SAVE_FIGS = True

_HERE    = os.path.dirname(os.path.abspath(__file__))
HNEM_DIR = os.path.join(_HERE, "..", "..", "..", "hnem_solutions", "final_solutions")
HEM_DIR  = os.path.join(_HERE, "..", "..", "..", "hem_solutions")

# Case number -> mixture ratio  (1=22.45, 2=29.26, 3=37.96, 4=48.85,
#                                5=60.76, 6=67.88, 7=76.46, 8=94.48)
CASES = [1,2,3,4,5,6,7,8]

CASE_MIXTURE_RATIO = {
    1: 22.45, 2: 29.26, 3: 37.96, 4: 48.85,
    5: 60.76, 6: 67.88, 7: 76.46, 8: 94.48,
}
MIXTURE_RATIOS = [CASE_MIXTURE_RATIO[c] for c in CASES]

# nb=nd combinations in increasing order -> linestyles (HNEM)
NB_ND_LINESTYLES = {
    1e8:  "--",
    1e10: "-.",
    1e11: ":",
    1e12: (0, (3, 1, 1, 1, 1, 1)),  # dense dash-dot-dot
}

# HEM uses a solid line (thicker) — same color as the matching mixture ratio
HEM_LINESTYLE   = "-"
HEM_LINEWIDTH   = 1.6

COLORMAP = cm.magma
# Avoid the very dark/light extremes of magma
_n = max(len(MIXTURE_RATIOS) - 1, 1)
_MR_COLORS = {
    mr: COLORMAP(0.15 + 0.70 * i / _n)
    for i, mr in enumerate(MIXTURE_RATIOS)
}


# ======================================================
# HELPERS
# ======================================================

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_hem_filename(path):
    """Return mixture ratio float from HEM filename, or None."""
    name = os.path.basename(path)
    m = re.search(r"solution_([0-9]+\.[0-9]+)\.pkl", name)
    return float(m.group(1)) if m else None


def parse_filename(path):
    """Return (mr, nb, nd) floats from the filename, or None if not matched."""
    name = os.path.basename(path)
    m = re.search(
        r"mr([0-9]+\.[0-9]+)_nb([0-9.e+E]+)_nd([0-9.e+E]+)\.pkl", name
    )
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


# ======================================================
# LOAD
# ======================================================
pkl_files = sorted(glob.glob(os.path.join(HNEM_DIR, "*.pkl")))
if not pkl_files:
    raise FileNotFoundError(f"No .pkl files found in {os.path.abspath(HNEM_DIR)}")

solutions = []
for path in pkl_files:
    parsed = parse_filename(path)
    if parsed is None:
        print(f"Skipping (no match): {os.path.basename(path)}")
        continue
    mr, nb, nd = parsed
    if mr not in MIXTURE_RATIOS:
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
    solutions.append(dict(mr=mr, nb=nb, sol=sol))

print(f"Loaded {len(solutions)} HNEM solutions.")

# --- HEM ---
hem_solutions = []
for path in sorted(glob.glob(os.path.join(HEM_DIR, "*.pkl"))):
    mr = parse_hem_filename(path)
    if mr is None:
        print(f"Skipping HEM (no match): {os.path.basename(path)}")
        continue
    if mr not in MIXTURE_RATIOS:
        continue
    try:
        sol = load_pickle(path)
    except Exception as e:
        print(f"Could not load {os.path.basename(path)}: {e}")
        continue
    hem_solutions.append(dict(mr=mr, sol=sol))

print(f"Loaded {len(hem_solutions)} HEM solutions.")

# ======================================================
# PLOT
# ======================================================
nzx.set_plot_options(minor_ticks=False, grid=False)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.5))

ax1.set_xlabel("Axial distance [m]")
ax1.set_ylabel("Pressure [kPa]")
ax2.set_xlabel("Axial distance [m]")
ax2.set_ylabel("Velocity [m/s]")

# Track which (mr, nb) combinations we have already labelled to avoid duplicates
labelled_mr = set()
labelled_nb = set()

# --- HEM curves (solid, thicker) ---
for entry in sorted(hem_solutions, key=lambda e: e["mr"]):
    mr  = entry["mr"]
    sol = entry["sol"]

    x = np.asarray(sol["x"])
    p = np.asarray(sol["p"]) / 1e3
    v = np.asarray(sol["v"])

    color  = _MR_COLORS.get(mr, "gray")
    lbl_mr = f"R={mr:g}" if mr not in labelled_mr else None

    ax1.plot(x, p, color=color, linestyle=HEM_LINESTYLE, linewidth=HEM_LINEWIDTH, label=lbl_mr)
    ax2.plot(x, v, color=color, linestyle=HEM_LINESTYLE, linewidth=HEM_LINEWIDTH)

    labelled_mr.add(mr)

# --- HNEM curves ---
for entry in sorted(solutions, key=lambda e: (e["nb"], e["mr"])):
    mr   = entry["mr"]
    nb   = entry["nb"]
    sol  = entry["sol"]

    x = np.asarray(sol["x"])
    p = np.asarray(sol["p"]) / 1e3
    v = np.asarray(sol["v"])

    color  = _MR_COLORS.get(mr, "gray")
    ls     = NB_ND_LINESTYLES[nb]
    lbl_mr = f"R={mr:g}" if mr not in labelled_mr else None
    lbl_nb = f"$n_b=n_d={nb:.0e}$" if nb not in labelled_nb else None

    ax1.plot(x, p, color=color, linestyle=ls, linewidth=1.0, label=lbl_mr)
    ax2.plot(x, v, color=color, linestyle=ls, linewidth=1.0, label=lbl_nb)

    labelled_mr.add(mr)
    labelled_nb.add(nb)

# --- Linestyle legend (model type) on ax2 ---
import matplotlib.lines as mlines
hem_handle  = mlines.Line2D([], [], color="gray", linestyle=HEM_LINESTYLE,
                             linewidth=HEM_LINEWIDTH, label="HEM")
hnem_handles = [
    mlines.Line2D([], [], color="gray", linestyle=ls, linewidth=1.0,
                  label=f"HNEM $n_b=n_d={nb:.0e}$")
    for nb, ls in NB_ND_LINESTYLES.items()
]
ax2.legend(handles=[hem_handle] + hnem_handles, title="Model", loc="upper left",
           fontsize=8, title_fontsize=8)

ax1.legend(title="Mixture ratio", loc="upper right", fontsize=8, title_fontsize=8)

fig.tight_layout(pad=1)

if SAVE_FIGS:
    base_filename = os.path.join(os.getcwd(), "hnem_all_solutions")
    nzx.savefig_in_formats(fig, base_filename, formats=[".png", ".svg"])

plt.show()
