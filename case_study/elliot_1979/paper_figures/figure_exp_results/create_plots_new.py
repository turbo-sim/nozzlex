import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nozzlex as bpy

# --- 1. Define Calculation Function ---
def calculate_equivalent_velocity(v_exit, A_exit, m_mix, p_exit, p_amb=98600):
    pressure_contribute = (A_exit / m_mix) * (p_exit - p_amb)
    return v_exit + pressure_contribute

# --- 2. Define Helper Function to Load Data ---
def load_data_from_folder(folder_path, area_key="A"):
    """
    Scans a folder for solution_*.pkl files.
    area_key: The dictionary key for Area (defaults to "A", but can be changed to "area")
    """
    x_vals = []
    y_vals = []
    z_vals = []
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found -> {folder_path}")
        return [], [], []

    pkl_files = glob.glob(os.path.join(folder_path, "solution_*.pkl"))
    print(f"Found {len(pkl_files)} files in {os.path.basename(folder_path)}")

    for filepath in pkl_files:
        filename = os.path.basename(filepath)
        try:
            # Parse filename "solution_22.45.pkl" -> 22.45
            number_str = filename.split('_')[1].replace('.pkl', '')
            mixture_ratio = float(number_str)

            # --- SCALE ADJUSTMENT ---
            # mixture_ratio = mixture_ratio / 100.0 

            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            # Extract variables
            # We use the variable 'area_key' here to select "A" or "area"
            v_exit = data['v'][-1]
            A_exit = data[area_key][-1] 
            p_exit = data['p'][-1]
            m_mix  = data['m_dot'][-1]

            # Calculate Equivalent Velocity
            v_eq = calculate_equivalent_velocity(v_exit, A_exit, m_mix, p_exit)
            x_vals.append(mixture_ratio)
            y_vals.append(v_eq)
            z_vals.append(v_exit)
            
        except KeyError as e:
            print(f"Key Error in {filename}: Missing {e}. Check if keys match.")
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    # Sort
    if x_vals:
        x_vals, y_vals, z_vals = zip(*sorted(zip(x_vals, y_vals, z_vals)))
        
    return list(x_vals), list(y_vals), list(z_vals)

SAVE_FIGS = True

# --- 3. Load Data ---
# A. CSV Data
try:
    df = pd.read_csv('data.csv', sep=',')
except FileNotFoundError:
    print("Error: data.csv not found")
    df = pd.DataFrame({'R_barotropic':[], 'v_barotropic':[], 'R_mixture':[], 'v_mixture':[], 'R_experimental':[], 'v_experimental':[]})

# F. Barotropic data from .out files
main_folder_path = os.path.join("..", "..")
cfd_folder = "output_data_cfd"
case_data = pd.read_excel(os.path.join(main_folder_path, "simulation_cases.xlsx"), sheet_name="Sheet1")
barotropic_exit = []
barotropic_near_exit = []

for i, row in case_data.iterrows():
    case_folder = f"case_00{row["case"]}_a"
    file_name = f"case_00{row["case"]}_a.out"
    fluent_out_path = os.path.join(main_folder_path, cfd_folder, case_folder, "simulation_latest", file_name)
    cfd_data = bpy.graphics.parse_fluent_out(fluent_out_path)
    v_exit = cfd_data["velocity_outlet"].iloc[-1]
    v_near_exit = cfd_data["velocity_near_outlet"].iloc[-1]
    barotropic_exit.append(v_exit)
    barotropic_near_exit.append(v_near_exit)


# B. HEM Solutions (Folder 1) - Uses default "A"
path_hem = os.path.join('..', '..', 'hem_solutions')
hem_x, hem_y, hem_z = load_data_from_folder(path_hem, area_key="A") 
# x is the position | y is the v_eq | z is the v_exit

# C. HNEM Solutions (Folder 2) - Uses "area"
path_hnem = os.path.join('..', '..', 'hnem_solutions')
hnem_x, hnem_y, hnem_z = load_data_from_folder(path_hnem, area_key="area")


# --- 4. Plotting ---
bpy.set_plot_options(minor_ticks=False, grid=False)
fig, ax1 = plt.subplots(figsize=(5.2*2, 3.0*1.5))

ax1.set_xlabel("Mixture ratio")
ax1.set_ylabel("Exit velocity [m/s]")

# 1. Previous paper data
# Experiment
ax1.plot(df['R_experimental'], df['v_experimental'], label='Experimental data', 
             markersize=4, color='black', marker='v', linestyle='None', markeredgewidth=1.05)

# Mixture
ax1.plot(df['R_mixture'], df['v_mixture'], label='Mixture model', 
             marker='o', markersize=4, linestyle='-', linewidth=1.05, markeredgewidth=1.05)

# Barotropic
ax1.plot(df['R_barotropic'], barotropic_exit, label='Barotropic model v$_{exit}$', 
         marker='o', markersize=4, linestyle='-', linewidth=1.05, markeredgewidth=1.05)
ax1.plot(df['R_barotropic'], barotropic_near_exit, label='Barotropic model v_$_{near exit}$', 
         marker='o', markersize=4, linestyle='-', linewidth=1.05, markeredgewidth=1.05)
# ax1.plot(df['R_barotropic'], df['v_barotropic'], label='Barotropic model', marker='o', markersize=4, linestyle='-', linewidth=1.05, markeredgewidth=1.05)

# 2. HEM Solutions
if hem_x:
    ax1.plot(hem_x, hem_y, label='HEM v$_{eq.}$', color='tab:red',
             marker='s', markersize=4, linestyle='-', linewidth=1.05, markeredgewidth=1.05)
    ax1.plot(hem_x, hem_z, label='HEM v$_{exit}$', color='tab:red',
             marker='s', markersize=4, linestyle='--', linewidth=1.05, markeredgewidth=1.05)

# 3. HNEM Solutions
if hnem_x:
    ax1.plot(hnem_x, hnem_y, label='HNEM v$_{eq.}$', color='tab:green', 
             marker='D', markersize=4, linestyle='-', linewidth=1.05, markeredgewidth=1.05)
    ax1.plot(hnem_x, hnem_z, label='HNEM v$_{exit}$', color='tab:green', 
             marker='D', markersize=4, linestyle='--', linewidth=1.05, markeredgewidth=1.05)

ax1.legend(loc="upper right", fontsize=9)
ax1.set_ylim([80, 180])

fig.tight_layout(pad=1)

if SAVE_FIGS:
    base_filename = os.path.join(os.getcwd(), f"comparison_velocity_mixture_ratio_new")
    bpy.savefig_in_formats(fig, base_filename, formats=[".png", ".svg"])

plt.show()
