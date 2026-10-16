import os
import re
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nozzlex as bpy


'''
    plot mass flow rate. exp, 2D-CFD, 1D
'''

main_folder_path = os.path.join("..", "..")

# ------experimental data--------------
mass_flow_experiment = 3.657 # kg/s, reported in Amit et al.
index_experiment = 6

# ------CFD data--------------
cfd_folder = "output_data_cfd"
case_data = pd.read_excel(os.path.join(main_folder_path, "simulation_cases.xlsx"), sheet_name="Sheet1")
CFD_DATA = []
for i, row in case_data.iterrows():
    case_folder = f"case_00{row["case"]}_a"
    file_name = f"case_00{row["case"]}_a.out"
    fluent_out_path = os.path.join(main_folder_path, cfd_folder, case_folder, "simulation_latest", file_name)
    cfd_data = bpy.graphics.parse_fluent_out(fluent_out_path)
    cfd_data = -cfd_data["mass_flow_rate_outlet"].iloc[-1]
    CFD_DATA.append(cfd_data)

# ------1D HEM data--------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def extract_id(path):
    match = re.search(r"solution_(\d+\.?\d*)\.pkl", path)
    return float(match.group(1)) if match else None

M_DOT_HEM = []
hem_folder = "hem_solutions"
hem_files = sorted(glob.glob(os.path.join(main_folder_path, hem_folder,"solution_*.pkl")), key=extract_id)
for file in hem_files:
    sol = load_pickle(file)
    # print(sol)
    m_dot_hem = sol["m_dot"][-1]
    M_DOT_HEM.append(m_dot_hem)

M_DOT_HNEM = []
hnem_folder = "hnem_solutions"
hnem_files = sorted(glob.glob(os.path.join(main_folder_path, hnem_folder,"solution_*.pkl")), key=extract_id)
for file in hnem_files:
    sol = load_pickle(file)
    # print(sol)
    m_dot_hnem = sol["m_dot"][-1]
    M_DOT_HNEM.append(m_dot_hnem)

MIXTURE_RATIOS = case_data["mixture_ratio"]


# --- 4. Plotting ---
bpy.set_plot_options(minor_ticks=False, grid=False)
fig, ax1 = plt.subplots(figsize=(5.2, 3.0))

ax1.set_xlabel("Mixture ratio")
ax1.set_ylabel("Mass flow [kg/s]")
# ax1.set_ylabel("Exit velocity [m/s]")

# 0. Experimental Data

ax1.plot(MIXTURE_RATIOS.iloc[index_experiment], mass_flow_experiment,label='Experiment', 
             marker='o', markersize=4, linestyle='', linewidth=1.05, markeredgewidth=1.05)


# 1. CFD Data
ax1.plot(MIXTURE_RATIOS, CFD_DATA, label='Barotropic model', 
            marker='o', markersize=4, linestyle='-', linewidth=1.05, markeredgewidth=1.05)

# 2. HEM Solutions
ax1.plot(MIXTURE_RATIOS, M_DOT_HEM, label='HEM Solution', color='tab:red',
             marker='s', markersize=4, linestyle='--', linewidth=1.05, markeredgewidth=1.05)

# 3. HNEM Solutions
ax1.plot(MIXTURE_RATIOS, M_DOT_HNEM, label='HNEM Solution', color='tab:green', 
             marker='D', markersize=4, linestyle='--', linewidth=1.05, markeredgewidth=1.05)

ax1.legend(loc="lower right", fontsize=9)
# ax1.set_ylim([80, 180])

fig.tight_layout(pad=1)

# if SAVE_FIGS:
#     base_filename = os.path.join(os.getcwd(), f"comparison_velocity_mixture_ratio")
#     bpy.savefig_in_formats(fig, base_filename, formats=[".png", ".svg"])

plt.show()
