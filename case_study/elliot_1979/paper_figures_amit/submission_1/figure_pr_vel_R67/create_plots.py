import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

import nozzlex as bpy


# Define the dictionary with filenames and corresponding y column names
SAVE_FIGS = True
files_and_vars = {
    'baro_axial_press.dat': 'barotropic_pressure',
    'baro_axial_vel.dat': 'barotropic_velocity',
    'mix_axial_press.dat': 'mixture_pressure',
    'mix_axial_vel.dat': 'mixture_velocity',
}

# Initialize an empty DataFrame
all_data = pd.DataFrame()

# Read the x data from the first file (assuming x is the same in all files)
first_file = next(iter(files_and_vars))  # Get the first filename
x_data = pd.read_csv(first_file, sep='\t', header=None, usecols=[0], names=['x'])
all_data['x'] = x_data['x']  # Set the x-column in all_data

# Loop through the dictionary, reading the y data from each file
for file, y_var_name in files_and_vars.items():
    y_data = pd.read_csv(file, sep='\t', header=None, usecols=[1], names=[y_var_name])
    all_data[y_var_name] = y_data  # Add the y-column to all_data

# Create figure
bpy.set_plot_options(minor_ticks=False, grid=False)
fig, ax1 = plt.subplots(figsize=(5.2, 3.0))

# Left Y-axis (Pressure)
ax1.set_xlabel("Axial distance [m]")
ax1.set_ylabel("Pressure [kPa]")
ax1.plot(all_data['x'], all_data['barotropic_pressure'], label='Barotropic model', linestyle='-', linewidth=1.05)
ax1.plot(all_data['x'], all_data['mixture_pressure'], label='Mixture model', linestyle='--', linewidth=1.05)
ax1.legend(loc="upper right", fontsize=9)
# ax1.set_xlim([0, 0.35])
ax1.set_ylim([-100, 2100])

# Right Y-axis (Velocity)
ax2 = ax1.twinx() 
ax2.set_ylabel('Velocity [m/s]') 
ax2.plot(all_data['x'], all_data['barotropic_velocity'], linestyle='-', linewidth=1.05)
ax2.plot(all_data['x'], all_data['mixture_velocity'], linestyle='--', linewidth=1.05)
ax2.set_ylim([-5, 165])

# Adjust pad
fig.tight_layout(pad=1)

# Save figure
if SAVE_FIGS:
    base_filename = os.path.join(os.getcwd(), f"comparison_velocity_pressure")
    bpy.savefig_in_formats(fig, base_filename, formats=[".png", ".svg"])

# Show figures
plt.show()
