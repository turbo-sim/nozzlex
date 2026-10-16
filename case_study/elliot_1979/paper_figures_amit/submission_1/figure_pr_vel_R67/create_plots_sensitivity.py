# # import os
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import matplotlib.colors as mcolors
# # import pandas as pd

# # import nozzlex as bpy


# # # Define the dictionary with filenames and corresponding y column names
# # SAVE_FIGS = True
# # files_and_vars = {
# #     'baro_axial_press.dat': 'barotropic_pressure',
# #     'baro_axial_vel.dat': 'barotropic_velocity',
# #     'mix_axial_press.dat': 'mixture_pressure',
# #     'mix_axial_vel.dat': 'mixture_velocity',
# # }

# # # Initialize an empty DataFrame
# # all_data = pd.DataFrame()

# # # Read the x data from the first file (assuming x is the same in all files)
# # first_file = next(iter(files_and_vars))  # Get the first filename
# # x_data = pd.read_csv(first_file, sep='\t', header=None, usecols=[0], names=['x'])
# # all_data['x'] = x_data['x']  # Set the x-column in all_data

# # # Loop through the dictionary, reading the y data from each file
# # for file, y_var_name in files_and_vars.items():
# #     y_data = pd.read_csv(file, sep='\t', header=None, usecols=[1], names=[y_var_name])
# #     all_data[y_var_name] = y_data  # Add the y-column to all_data

# # # Create figure
# # bpy.set_plot_options(minor_ticks=False, grid=False)
# # fig, ax1 = plt.subplots(figsize=(5.2, 3.0))

# # # Left Y-axis (Pressure)
# # ax1.set_xlabel("Axial distance [m]")
# # ax1.set_ylabel("Pressure [kPa]")
# # ax1.plot(all_data['x'], all_data['barotropic_pressure'], label='Barotropic model', linestyle='-', linewidth=1.05)
# # ax1.plot(all_data['x'], all_data['mixture_pressure'], label='Mixture model', linestyle='--', linewidth=1.05)
# # ax1.legend(loc="upper right", fontsize=9)
# # # ax1.set_xlim([0, 0.35])
# # ax1.set_ylim([-100, 2100])

# # # Right Y-axis (Velocity)
# # ax2 = ax1.twinx() 
# # ax2.set_ylabel('Velocity [m/s]') 
# # ax2.plot(all_data['x'], all_data['barotropic_velocity'], linestyle='-', linewidth=1.05)
# # ax2.plot(all_data['x'], all_data['mixture_velocity'], linestyle='--', linewidth=1.05)
# # ax2.set_ylim([-5, 165])

# # # Adjust pad
# # fig.tight_layout(pad=1)

# # # Save figure
# # if SAVE_FIGS:
# #     base_filename = os.path.join(os.getcwd(), f"comparison_velocity_pressure")
# #     bpy.savefig_in_formats(fig, base_filename, formats=[".png", ".svg"])

# # # Show figures
# # plt.show()
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import glob
# import re
# import pickle
# import nozzlex as bpy


# # ======================================================
# # PART 1 — Load .dat files (barotropic vs mixture model)
# # ======================================================
# SAVE_FIGS = True

# files_and_vars = {
#     'mix_axial_press.dat': 'mixture_pressure',
#     'mix_axial_vel.dat': 'mixture_velocity',
# }
# files_barotropic = {
#     'case_006_a_pressure.csv': 'barotropic_pressure',
#     'case_006_a_velocity.csv': 'barotropic_velocity',
# }

# # Load x once
# first_file = next(iter(files_and_vars))
# x_data = pd.read_csv(first_file, sep='\t', header=None, usecols=[0], names=['x'])

# all_data = pd.DataFrame()
# all_data['x'] = x_data['x']

# # Load y columns
# for file, colname in files_and_vars.items():
#     all_data[colname] = pd.read_csv(file, sep='\t', header=None, usecols=[1], names=[colname])



# # Load x once
# barotropic_file = next(iter(files_barotropic))
# x_data = pd.read_csv(barotropic_file, usecols=[0], names=['x'])

# all_data = pd.DataFrame()
# all_data['x'] = x_data['x']

# # Load y columns
# for file, colname in files_and_vars.items():
#     all_data[colname] = pd.read_csv(file, sep='\t', header=None, usecols=[1], names=[colname])


# # ======================================================
# # PART 2 — Load sensitivity .pkl files
# # ======================================================

# def load_pickle(path):
#     with open(path, "rb") as f:
#         return pickle.load(f)

# def extract_param(path):
#     # solution_76.46_0.pkl or solution_76.46_0.95.pkl
#     match = re.search(r"solution_([0-9\.]+)_([0-9\.]+)\.pkl", path)
#     if not match:
#         return None
#     return float(match.group(2))   # sensitivity value

# pkl_files = glob.glob("frictionSensitivity/solution_*.pkl")

# params = []
# valid_files = []

# for f in pkl_files:
#     p = extract_param(f)
#     if p is None:
#         print(f"⚠ Skipping (no match): {f}")
#         continue
#     params.append(p)
#     valid_files.append(f)

# # Sort by sensitivity parameter
# params, valid_files = zip(*sorted(zip(params, valid_files)))

# # Colormap
# cmap = plt.get_cmap("viridis")
# colors = {p: cmap(i / max(len(params)-1, 1)) for i, p in enumerate(params)}


# # ======================================================
# # PART 3 — Plot comparison + pkl curves
# # ======================================================

# bpy.set_plot_options(minor_ticks=False, grid=False)
# fig, ax1 = plt.subplots(figsize=(5.2, 3.0))

# # -----------------------------
# # Left Y-axis → PRESSURE
# # -----------------------------
# ax1.set_xlabel("Axial distance [m]")
# ax1.set_ylabel("Pressure [kPa]")

# # Main two curves
# ax1.plot(all_data['x'], all_data['barotropic_pressure'], label='Barotropic model', linestyle='-', linewidth=1.05, color='black')
# ax1.plot(all_data['x'], all_data['mixture_pressure'], label='Mixture model', linestyle='--', linewidth=1.05, color='gray')

# # Add sensitivity curves (pressure)
# for p, fname in zip(params, valid_files):
#     sol = load_pickle(fname)
#     ax1.plot(sol["x"], sol["p"] * 1e-5 * 100,  # kPa = bar * 100
#              linestyle='-', linewidth=1.0,
#              color=colors[p],
#              label=f"Sens {p:g}")

# ax1.set_ylim([-100, 2100])
# ax1.legend(loc="upper right", fontsize=8)


# # -----------------------------
# # Right Y-axis → VELOCITY
# # -----------------------------
# ax2 = ax1.twinx()
# ax2.set_ylabel("Velocity [m/s]")

# ax2.plot(all_data['x'], all_data['barotropic_velocity'], linestyle='-', linewidth=1.05, color='black')
# ax2.plot(all_data['x'], all_data['mixture_velocity'], linestyle='--', linewidth=1.05, color='gray')

# # Sensitivity velocity curves
# for p, fname in zip(params, valid_files):
#     sol = load_pickle(fname)
#     ax2.plot(sol["x"], sol["v"],
#              linestyle='-', linewidth=1.0,
#              color=colors[p])

# ax2.set_ylim([-5, 165])

# fig.tight_layout(pad=1)

# if SAVE_FIGS:
#     base_filename = os.path.join(os.getcwd(), "comparison_velocity_pressure_with_sensitivity")
#     bpy.savefig_in_formats(fig, base_filename, formats=[".png", ".svg"])

# plt.show()

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
SAVE_FIGS = True

# --- 1A. Load Mixture Model Data (.dat, tab-separated) ---
files_mixture = {
    'mix_axial_press.dat': 'mixture_pressure',
    'mix_axial_vel.dat':   'mixture_velocity',
}

# We assume these files share the exact same X-grid
try:
    # Read Pressure
    df_mix_p = pd.read_csv('mix_axial_press.dat', sep='\t', header=None, names=['x', 'mixture_pressure'])
    # Read Velocity
    df_mix_v = pd.read_csv('mix_axial_vel.dat', sep='\t', header=None, names=['x', 'mixture_velocity'])
    # Merge on X
    df_mix = pd.merge(df_mix_p, df_mix_v, on='x')
except FileNotFoundError:
    print("Warning: Mixture .dat files not found. Creating empty DataFrame for testing.")
    df_mix = pd.DataFrame({'x': [], 'mixture_pressure': [], 'mixture_velocity': []})

# --- 1B. Load Barotropic Model Data (.csv, comma-separated) ---
files_barotropic = {
    'case_006_a_pressure.csv': 'barotropic_pressure',
    'case_006_a_velocity.csv': 'barotropic_velocity',
}

df_baro_p = pd.read_csv('case_006_a_pressure.csv')


try:
    # Read Pressure (CSV has headers: x, pressure)
    df_baro_p = pd.read_csv('case_006_a_pressure.csv')
    df_baro_p.columns = ['x', 'barotropic_pressure'] # Ensure naming consistency

    # Read Velocity (CSV has headers: x, velocity)
    df_baro_v = pd.read_csv('case_006_a_velocity.csv') 
    df_baro_v.columns = ['x', 'barotropic_velocity'] # Ensure naming consistency

    # Merge on X
    df_baro = pd.merge(df_baro_p, df_baro_v, on='x')
except FileNotFoundError:
    print("Warning: Barotropic .csv files not found. Creating empty DataFrame for testing.")
    df_baro = pd.DataFrame({'x': [], 'barotropic_pressure': [], 'barotropic_velocity': []})


# ======================================================
# PART 2 — Load sensitivity .pkl files
# ======================================================

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def extract_param(path):
    # solution_76.46_0.pkl or solution_76.46_0.95.pkl
    match = re.search(r"solution_([0-9\.]+)_([0-9\.]+)\.pkl", path)
    if not match:
        return None
    return float(match.group(2))   # sensitivity value

pkl_files = glob.glob("frictionSensitivity/solution_*.pkl")
# pkl_files = "frictionSensitivity/solution_67.88_1.pkl"

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
fig, ax1 = plt.subplots(figsize=(5.2, 3.0))

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
             label=f"HEM ($f_w={p:g}$)")

ax1.set_ylim([-100, 2100])
ax1.legend(loc="upper right", fontsize=9)


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
    bpy.savefig_in_formats(fig, base_filename, formats=[".png", ".svg"])

plt.show()
