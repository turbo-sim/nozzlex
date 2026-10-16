import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import nozzlex as nzx

_HERE = os.path.dirname(os.path.abspath(__file__))

# ======================================================
# PATHS
# ======================================================
HNEM_DIR = os.path.join(_HERE, "hnem_solutions", "final_solutions")
TFM_DIR  = os.path.join(_HERE, "tfm_solutions")
EXP_FILE = os.path.join(_HERE, "pressure_elliot_big_in-psia_p150psia_T69F_R39.1.csv")

velocity_experiment = 92.90 # m/s
mass_flow_exp = 72.76 # kg/s

HNEM_FILES = [
    ("solution_elliot_10.3bar_mr39.1_nb1.00e+08_nd1.00e+08.pkl", r"HNEM $n=10^{8}$"),
    ("solution_elliot_10.3bar_mr39.1_nb1.00e+10_nd1.00e+10.pkl", r"HNEM $n=10^{10}$"),
    ("solution_elliot_10.3bar_mr39.1_nb1.00e+12_nd1.00e+12.pkl", r"HNEM $n=10^{12}$"),
]

# TFM_FILES = [
#     ("solution_elliot_10.3bar_mr39.1_nb1.00e+08_nd1.00e+08.pkl", r"TFM $n=10^{8}$"),
#     ("solution_elliot_10.3bar_mr39.1_nb1.00e+10_nd1.00e+10.pkl", r"TFM $n=10^{10}$"),
#     ("solution_elliot_10.3bar_mr39.1_nb1.00e+11_nd1.00e+11.pkl", r"TFM $n=10^{11}$"),
# ]
TFM_FILES = [
    ("solution_elliot_big_transport_10.3bar_mr39.1_nb1.00e+08_nd1.00e+08.pkl", r"TFM"),
]

SAVE_FIGS = False

# ======================================================
# LOAD
# ======================================================

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def concat_passes(sol_dict):
    """Concatenate pass1/pass2/pass3 arrays, skipping None passes."""
    keys = None
    arrays = {}
    for pk in ["pass1", "pass2", "pass3"]:
        p = sol_dict.get(pk)
        if p is None:
            continue
        if keys is None:
            keys = [k for k in p if np.asarray(p[k]).ndim == 1]
        for k in keys:
            arrays.setdefault(k, []).append(np.asarray(p[k]))
    return {k: np.concatenate(v) for k, v in arrays.items()}


# --- HNEM ---
hnem_data = []
for fname, label in HNEM_FILES:
    path = os.path.join(HNEM_DIR, fname)
    sol  = load_pickle(path)
    hnem_data.append({
        "x":     np.asarray(sol["x"]),
        "p":     np.asarray(sol["p"]),
        "v":     np.asarray(sol["v"]),   # mixture velocity in HNEM
        "m_dot": np.asarray(sol["m_dot"]),
        "label": label,
    })

# --- TFM ---
tfm_data = []
for fname, label in TFM_FILES:
    raw = load_pickle(os.path.join(TFM_DIR, fname))
    tfm = concat_passes(raw)
    entry = {"x": tfm["x"], "p": tfm["p"], "v": tfm["v_mix"], "m_dot": tfm["m_dot"], "label": label}
    for key in ("d_droplet", "d_bubble", "rho1", "rho2", "v1", "v2"):
        if key in tfm:
            entry[key] = tfm[key]
    tfm_data.append(entry)

# --- Experiment ---
import csv
exp_x, exp_p = [], []
with open(EXP_FILE) as f:
    reader = csv.DictReader(f)
    reader.fieldnames = [h.strip() for h in reader.fieldnames]
    for row in reader:
        row = {k.strip(): v.strip() for k, v in row.items()}
        exp_x.append(float(row["position"]) * 0.0254)    # inches -> m
        exp_p.append(float(row["pressure"]) * 6894.757)  # psia   -> Pa
exp_x = np.array(exp_x)
exp_p = np.array(exp_p)

# ======================================================
# PRINT: mass flow comparison
# ======================================================
print(f"\n--- Mass flow rate (outlet) ---")
print(f"  Experimental: {mass_flow_exp:.4f} kg/s")
print(f"  {'Model':<36} {'m_dot [kg/s]':>13} {'err [kg/s]':>12} {'err [%]':>9}")
print(f"  {'-'*72}")
for entry in hnem_data:
    m = float(entry["m_dot"][-1])
    err = m - mass_flow_exp
    print(f"  {entry['label']:<36} {m:>13.4f} {err:>+12.4f} {err/mass_flow_exp*100:>+9.2f}%")
for entry in tfm_data:
    m = float(entry["m_dot"][-1])
    err = m - mass_flow_exp
    print(f"  {entry['label']:<36} {m:>13.4f} {err:>+12.4f} {err/mass_flow_exp*100:>+9.2f}%")

# ======================================================
# PRINT: exit velocity comparison
# ======================================================
print(f"\n--- Exit mixture velocity (outlet) ---")
print(f"  Experimental: {velocity_experiment:.4f} m/s")
print(f"  {'Model':<36} {'v_mix [m/s]':>13} {'err [m/s]':>12} {'err [%]':>9}")
print(f"  {'-'*72}")
for entry in hnem_data:
    v = float(entry["v"][-1])
    err = v - velocity_experiment
    print(f"  {entry['label']:<36} {v:>13.4f} {err:>+12.4f} {err/velocity_experiment*100:>+9.2f}%")
for entry in tfm_data:
    v = float(entry["v"][-1])
    err = v - velocity_experiment
    print(f"  {entry['label']:<36} {v:>13.4f} {err:>+12.4f} {err/velocity_experiment*100:>+9.2f}%")
print()

# ======================================================
# PLOT
# ======================================================
nzx.set_plot_options(minor_ticks=False, grid=False)

fig, (ax_p, ax_v) = plt.subplots(1, 2, figsize=(10.5, 3.5))

# Colors for HNEM and TFM from sequential colormaps
hnem_colors = plt.cm.Blues(np.linspace(0.45, 0.85, len(hnem_data)))
tfm_colors  = plt.cm.Reds(np.linspace(0.45, 0.85, len(tfm_data)))

# --- Pressure panel ---
ax_p.set_xlabel("Axial distance [m]")
ax_p.set_ylabel("Pressure [kPa]")

for entry, color in zip(hnem_data, hnem_colors):
    ax_p.plot(entry["x"], entry["p"] / 1e3,
              color=color, linestyle="--", linewidth=1.1, label=entry["label"])

for entry, color in zip(tfm_data, tfm_colors):
    ax_p.plot(entry["x"], entry["p"] / 1e3,
              color=color, linestyle="-", linewidth=1.2, label=entry["label"])

ax_p.plot(exp_x, exp_p / 1e3,
          "ko", markersize=3.5, zorder=5, label="Experiment")

ax_p.legend(fontsize=8)

# --- Velocity panel ---
ax_v.set_xlabel("Axial distance [m]")
ax_v.set_ylabel("Mixture velocity [m/s]")

for entry, color in zip(hnem_data, hnem_colors):
    ax_v.plot(entry["x"], entry["v"],
              color=color, linestyle="--", linewidth=1.1, label=entry["label"])

for entry, color in zip(tfm_data, tfm_colors):
    ax_v.plot(entry["x"], entry["v"],
              color=color, linestyle="-", linewidth=1.2, label=entry["label"])

x_outlet = hnem_data[0]["x"][-1]
ax_v.plot(x_outlet, velocity_experiment,
          "ko", markersize=5, zorder=5, label=f"Experiment ({velocity_experiment:.1f} m/s)")

ax_v.legend(fontsize=8)

fig.tight_layout(pad=1)

if SAVE_FIGS:
    base = os.path.join(_HERE, "comparison_big_hnem_tfm_exp")
    nzx.savefig_in_formats(fig, base, formats=[".png", ".svg"])

# ======================================================
# FIGURE 2: Weber numbers (TFM only)
# ======================================================
SIGMA_WATER = 0.072  # N/m

fig2, (ax_wd, ax_wb) = plt.subplots(1, 2, figsize=(10.5, 3.5))
ax_wd.set_xlabel("Axial distance [m]")
ax_wd.set_ylabel("Droplet Weber number $We_d$")
ax_wb.set_xlabel("Axial distance [m]")
ax_wb.set_ylabel("Bubble Weber number $We_b$")

for entry, color in zip(tfm_data, tfm_colors):
    if "d_droplet" not in entry:
        continue
    u_rel   = entry["v2"] - entry["v1"]
    We_drop = entry["rho2"] * u_rel**2 * entry["d_droplet"] / SIGMA_WATER
    We_bubb = entry["rho1"] * u_rel**2 * entry["d_bubble"]  / SIGMA_WATER
    ax_wd.plot(entry["x"], We_drop, color=color, linewidth=1.2, label=entry["label"])
    ax_wb.plot(entry["x"], We_bubb, color=color, linewidth=1.2, label=entry["label"])

ax_wd.legend(fontsize=8)
ax_wb.legend(fontsize=8)
fig2.tight_layout(pad=1)

plt.show()
