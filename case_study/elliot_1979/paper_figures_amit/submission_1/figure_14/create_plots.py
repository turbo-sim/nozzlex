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

# Left Y-axis (Pressure)
ax1.set_xlabel("Rotational speed [RPM]")
ax1.set_ylabel("`Torque [N$\cdot$m]")
ax1.plot(
    df["speed_barotropic"]*design_speed,
    df["torque_barotropic"],
    label="Barotropic model",
    marker="o",
    markersize=4,
    linestyle="-",
    linewidth=1.05,
    markeredgewidth=1.05,
)
ax1.plot(
    df["speed_mixture"]*design_speed,
    df["torque_mixture"],
    label="Mixture model",
    marker="o",
    markersize=4,
    linestyle="-",
    linewidth=1.05,
    markeredgewidth=1.05,
)

# Calculate 10% of the y-values as error
y_err = 0.10 * df["torque_experimental"]

# Experimental data with error bars and caps
ax1.errorbar(
    df["speed_experimental"]*design_speed,
    df["torque_experimental"],
    yerr=y_err,
    fmt="o",
    label="Experimental data",
    markersize=4,
    color="black",
    linestyle="None",
    markeredgewidth=1.05,
    capsize=3,
)

ax1.legend(loc="upper right", fontsize=9)
ax1.set_xlim([500, 2500])
ax1.set_ylim([-10, 130])


# Adding a second x-axis for normalized speed
ax2 = ax1.twiny()  # Create a second x-axis sharing the same y-axis
ax2.set_xlabel("Normalized rotational speed relative to design speed")
# Set the x-limits of the new x-axis based on the main axis limits and design speed
ax2.set_xlim(np.array(ax1.get_xlim())/design_speed)
# Alternatively, if you want specific ticks or labels, set them here
# ax2.set_xticks([your_custom_ticks])  
# ax2.set_xticklabels([your_custom_tick_labels])


# # Adding a second x-axis at the top for normalized speed
# ax2 = ax1.twiny()
# ax2.set_xlabel("Normalized rotational speed [Relative to design speed]")
# ax2.set_xlim(np.array(ax1.get_xlim())/design_speed)

# # # Adding a second x-axis at the bottom for another scale or representation
# # # The new x-axis could be created using secondary_xaxis function or by manually positioning it
# # # Here we use the secondary_xaxis function for demonstration
# # def forward(x):
# #     return x/design_speed  # Convert to normalized speed

# # def inverse(x):
# #     return x*design_speed  # Convert back to original speed

# # secax = ax1.secondary_xaxis('bottom', functions=(forward, inverse))
# # secax.set_xlabel("Another x-axis scale [Normalized]")


# Adjust pad
fig.tight_layout(pad=1)

# Save figure
if SAVE_FIGS:
    base_filename = os.path.join(os.getcwd(), f"comparison_torque_speed")
    bpy.savefig_in_formats(fig, base_filename, formats=[".png", ".svg"])

# Show figures
plt.show()
