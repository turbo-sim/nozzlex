import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

import nozzlex as bpy

def calcualte_equivalent_veelocity(v_exit, A_exit, m_mix, p_exit, p_amb=101325):
    pressure_contribute = (A_exit / m_mix) * (p_exit - p_amb)
    return v_exit + pressure_contribute

# Define the dictionary with filenames and corresponding y column names
SAVE_FIGS = True

# Read the x data from the first file (assuming x is the same in all files)
df = pd.read_csv('data.csv', sep=',')

# Create figure
bpy.set_plot_options(minor_ticks=False, grid=False)
fig, ax1 = plt.subplots(figsize=(5.2, 3.0))

# Left Y-axis (Pressure)
ax1.set_xlabel("Mixture ratio")
ax1.set_ylabel("Exit velocity [m/s]")
ax1.plot(df['R_barotropic'], df['v_barotropic'], label='Barotropic model', marker='o', markersize=4, linestyle='-', linewidth=1.05, markeredgewidth=1.05)
ax1.plot(df['R_mixture'], df['v_mixture'], label='Mixture model', marker='o', markersize=4, linestyle='-', linewidth=1.05, markeredgewidth=1.05)
ax1.plot(df['R_experimental'], df['v_experimental'], label='Experimental data', markersize=4, color='black', marker='v', linestyle='None', markeredgewidth=1.05)


ax1.legend(loc="upper right", fontsize=9)
# ax1.set_xlim([0, 0.35])
ax1.set_ylim([80, 180])

# Adjust pad
fig.tight_layout(pad=1)

# Save figure
if SAVE_FIGS:
    base_filename = os.path.join(os.getcwd(), f"comparison_velocity_mixture_ratio")
    bpy.savefig_in_formats(fig, base_filename, formats=[".png", ".svg"])

# Show figures
plt.show()
