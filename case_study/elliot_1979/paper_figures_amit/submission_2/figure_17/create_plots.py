import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

import barotropy as bpy


# Define the dictionary with filenames and corresponding y column names
SAVE_FIGS = True

# Read the x data from the first file (assuming x is the same in all files)
df = pd.read_excel("torque_data.xlsx")
design_speed = 1652
# Create figure
bpy.set_plot_options(minor_ticks=False, grid=False)
fig, ax1 = plt.subplots(figsize=(5.2, 3.8))

# Calculate 10% of the y-values as error
y_err = 0.10 * df["torque_experimental"]

# Experimental data with error bars and caps
a = 1.00
w = 1.1
s = 3.0
ax1.errorbar(
    df["speed_experimental"]*design_speed,
    df["torque_experimental"],
    yerr=y_err,
    fmt="o",
    label="Experimental data",
    markersize=s,
    color="black",
    linestyle="None",
    linewidth=w,
    markeredgewidth=w,
    capsize=3,
)

# Left Y-axis (Pressure)
ax1.set_xlabel("Rotational speed [RPM]")
ax1.set_ylabel("Torque [N$\cdot$m]")
ax1.plot(
    df["speed_barotropic"]*design_speed*a,
    df["torque_barotropic"],
    label="Barotropic CFD",
    color=bpy.COLORS_MATLAB[0],
    marker="o",
    markersize=s,
    linestyle="-",
    linewidth=w,
    markeredgewidth=w,
)
ax1.plot(
    df["speed_barotropic"]*design_speed*a,
    df["torque_barotropic_corrected"],
    label="Barotropic corrected",
    color=bpy.COLORS_MATLAB[0],
    marker="o",
    markersize=s,
    linestyle=":",
    linewidth=w,
    markeredgewidth=w,
)
ax1.plot(
    df["speed_mixture"]*design_speed/a,
    df["torque_mixture"],
    label="Mixture CFD",
    color=bpy.COLORS_MATLAB[1],
    marker="x",
    markersize=s+0.5,
    linestyle="None",
    linewidth=w,
    markeredgewidth=w,
)
ax1.plot(
    df["speed_mixture"]*design_speed/a,
    df["torque_mixture_corrected"],
    label="Mixture corrected",
    color=bpy.COLORS_MATLAB[1],
    marker="+",
    markersize=s+1.5,
    linestyle="None",
    linewidth=w,
    markeredgewidth=w,
)



ax1.legend(loc="lower left", fontsize=9)
ax1.set_xlim([500, 2500])
ax1.set_ylim([20, 120])
# ax1.set_ylim([-10, 130])

# Adding a second x-axis for normalized speed
ax2 = ax1.twiny()  # Create a second x-axis sharing the same y-axis
ax2.set_xlabel("Normalized rotational speed relative to design speed")
ax2.set_xlim(np.array(ax1.get_xlim())/design_speed)

# Adjust pad
fig.tight_layout(pad=1)

# Save figure
if SAVE_FIGS:
    base_filename = os.path.join(os.getcwd(), f"comparison_torque_speed")
    bpy.savefig_in_formats(fig, base_filename, formats=[".png", ".svg"])

# Show figures
plt.show()
