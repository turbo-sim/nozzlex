import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import pickle
import nozzlex as bpy

# ======================================================
# PART 1 — Load Data
# ======================================================
SAVE_FIGS = False

# ======================================================
# PART 2 — Load sensitivity .pkl files
# ======================================================

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
# ------1D HEM data--------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def extract_id(path):
    match = re.search(r"solution_(\d+\.?\d*)\.pkl", path)
    return float(match.group(1)) if match else None

main_folder_path = os.path.join("..", "..")

M_DOT_HEM = []
hem_folder = "hem_solutions"
hem_files = sorted(glob.glob(os.path.join(main_folder_path, hem_folder,"solution_*.pkl")), key=extract_id)
for file in hem_files:
    sol = load_pickle(file)
    # print(sol)
    m_dot_hem = sol["m_dot"][-1]
    M_DOT_HEM.append(m_dot_hem)

def extract_param(path):
    # solution_76.46_0.pkl or solution_76.46_0.95.pkl
    match = re.search(r"solution_([0-9\.]+)_([0-9\.]+)\.pkl", path)
    if not match:
        return None
    return float(match.group(2))   # sensitivity value

pkl_files = glob.glob("frictionSensitivity_newThroat/solution_*.pkl")

params = []
valid_files = []

for f in pkl_files:
    p = extract_param(f)
    if p is None:
        print(f"⚠ Skipping (no match): {f}")
        continue
    params.append(p)
    valid_files.append(f)

# Sort by sensitivity parameter
if params:
    params, valid_files = zip(*sorted(zip(params, valid_files)))
    # Colormap
    cmap = plt.get_cmap("viridis")
    colors = {p: cmap(i / max(len(params)-1, 1)) for i, p in enumerate(params)}
else:
    colors = {}


# ======================================================
# PART 3 — Plot comparison + pkl curves
# ======================================================

bpy.set_plot_options(minor_ticks=False, grid=False)
fig, ax1 = plt.subplots(figsize=(5.5*4, 3.0*4))

# -----------------------------
# Left Y-axis → PRESSURE
# -----------------------------
ax1.set_xlabel("Axial distance [m]")
ax1.set_ylabel("Pressure [kPa]")

# Plot Barotropic (Black Solid)
if not df_baro.empty:
    ax1.plot(df_baro['x'], df_baro['barotropic_pressure'], 
             label='Barotropic model', linestyle='-', linewidth=1.05, color='black')

# Plot Mixture (Gray Dashed)
if not df_mix.empty:
    ax1.plot(df_mix['x'], df_mix['mixture_pressure'], 
             label='Mixture model', linestyle='--', linewidth=1.05, color='gray')

# Add sensitivity curves (pressure)
for p, fname in zip(params, valid_files):
    sol = load_pickle(fname)
    ax1.plot(sol["x"], sol["p"] * 1e-5 * 100,  # kPa = bar * 100
             linestyle='-', linewidth=1.0,
             color=colors[p],
             label=f"$\\tau_{p:g}$")

ax1.set_ylim([-100, 2100])
ax1.legend(loc="center left", fontsize=8)


# -----------------------------
# Right Y-axis → VELOCITY
# -----------------------------
ax2 = ax1.twinx()
ax2.set_ylabel("Velocity [m/s]")

# Plot Barotropic (Black Solid)
if not df_baro.empty:
    ax2.plot(df_baro['x'], df_baro['barotropic_velocity'], 
             linestyle='-', linewidth=1.05, color='black')

# Plot Mixture (Gray Dashed)
if not df_mix.empty:
    ax2.plot(df_mix['x'], df_mix['mixture_velocity'], 
             linestyle='--', linewidth=1.05, color='gray')

# Sensitivity velocity curves
for p, fname in zip(params, valid_files):
    sol = load_pickle(fname)
    ax2.plot(sol["x"], sol["v"],
             linestyle='-', linewidth=1.0,
             color=colors[p])

ax2.set_ylim([-5, 165])

fig.tight_layout(pad=1)

if SAVE_FIGS:
    if not os.path.exists("plots"):
        os.makedirs("plots", exist_ok=True)
    base_filename = os.path.join(os.getcwd(), "comparison_velocity_pressure_with_sensitivity")
    bpy.savefig_in_formats(fig, base_filename, formats=[".pdf", ".svg"])

plt.show()
