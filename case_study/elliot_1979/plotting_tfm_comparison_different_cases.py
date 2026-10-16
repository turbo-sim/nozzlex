import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# --------------------------------------------------
# Files to compare — edit this list freely
# --------------------------------------------------
TFM_FILES = [
    # ("tfm_solutions/solution_case1_20.0bar_mr22.45_nb1.00e+10_nd1.00e+10.pkl", "Nb=1e10, Nd=1e10"),
    ("tfm_solutions/solution_elliot_subsonic_20.0bar_mr22.45_nb1.00e+12_nd1.00e+12.pkl", "Nb=1e12, Nd=1e12"),

    # ("tfm_solutions/solution_case5_20.0bar_mr60.76_nb1.00e+10_nd1.00e+10.pkl", "Nb=1e10, Nd=1e10"),
]

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def concat_passes(raw):
    """Merge pass1/pass2/pass3 into a single dict of 1-D numpy arrays."""
    passes = [raw.get("pass1"), raw.get("pass2"), raw.get("pass3")]
    passes = [p for p in passes if p is not None]
    merged = {}
    for k in passes[0].keys():
        arrays = [np.asarray(p[k]).ravel() for p in passes if k in p]
        if arrays:
            merged[k] = np.concatenate(arrays)
    merged["v_mix"] = merged["Ma_mix"] * merged["a_mix"]
    return merged

# --------------------------------------------------
# Load all solutions
# --------------------------------------------------
solutions = []
for path, label in TFM_FILES:
    try:
        sol = concat_passes(load_pickle(path))
        solutions.append((label, sol))
    except FileNotFoundError:
        print(f"Not found (skipped): {path}")

if not solutions:
    raise RuntimeError("No TFM files loaded — check TFM_FILES paths.")

# --------------------------------------------------
# Plot
# --------------------------------------------------
colors = plt.cm.tab10.colors

fig = plt.figure(figsize=(7, 10))
gs  = gridspec.GridSpec(4, 1)
ax0 = fig.add_subplot(gs[0])                   # mixture velocity
ax1 = fig.add_subplot(gs[1], sharex=ax0)       # v1 and v2
ax2 = fig.add_subplot(gs[2], sharex=ax0)       # volume fraction alpha1
ax3 = fig.add_subplot(gs[3], sharex=ax0)       # pressure

for idx, (label, sol) in enumerate(solutions):
    c = colors[idx % len(colors)]
    x = sol["x"]
    ax0.plot(x, sol["v_mix"],  color=c, label=label)
    ax1.plot(x, sol["v1"],     color=c, linestyle="-",  label=f"{label} (f1)")
    ax1.plot(x, sol["v2"],     color=c, linestyle="--", label=f"{label} (f2)")
    ax2.plot(x, sol["alpha1"], color=c, label=label)
    ax3.plot(x, sol["p"] * 1e-5, color=c, label=label)

ax0.set_ylabel("$v_{mix}$ (m/s)")
ax0.legend(fontsize=7)

ax1.set_ylabel("Velocity (m/s)")
# Shared linestyle legend entry
ax1.plot([], [], "k-",  label="fluid 1")
ax1.plot([], [], "k--", label="fluid 2")
ax1.legend(fontsize=7, ncol=2)

ax2.set_ylabel("Volume fraction $\\alpha_1$")
ax2.legend(fontsize=7)

ax3.set_ylabel("Pressure (bar)")
ax3.set_xlabel("Axial coordinate $x$ (m)")
ax3.legend(fontsize=7)

fig.tight_layout()
plt.show()
