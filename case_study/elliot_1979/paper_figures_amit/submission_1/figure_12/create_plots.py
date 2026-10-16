import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

import barotropy as bpy


SAVE_FIGS = True
blades = [4, 5, 6, 7, 8, 9, 10]
blade_span = [0.50, 0.75]

# Create figure with 1 row and 2 columns for subplots
bpy.set_plot_options(minor_ticks=False, grid=False)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4))
colors = plt.get_cmap("magma")(np.linspace(0.2, 0.75, len(blades)//1))
linestyles = ['-', '--', ':']

# Plot settings for each subplot
for ax, span in zip([ax1, ax2], blade_span):
    ax.set_xlabel("Fraction of axial chord")
    ax.set_ylim([0, 300])
    ax.set_title(f'Surface pressure distribution at {span*100:.0f}% span')
    if ax == ax1:  # Add y-axis label only to the left subplot
        ax.set_ylabel(f"Static pressure [kPa]")

    for index, blade in enumerate(blades):
        filename = f"loading_blade{blade:d}_{span:0.2f}.dat"
        df = pd.read_csv(filename, sep='\t', header=None)
        df = pd.concat([df, df.iloc[0:1, :]], ignore_index=True)
        ax.plot(df[0], df[1]/1e3,
                label=f"Blade {blade}",
                linewidth=0.9, 
                color=colors[index // 1 % len(colors)],  # use integer division for color indexing
                linestyle=linestyles[index % len(linestyles)])


# Skip y axis of the right subplot
# ax2.set_yticklabels([])
# ax2.set_yticks([])

# Set legend outside the plot area
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc='right', bbox_to_anchor=(0.96, 0.55), fontsize=10)

# Adjust layout to make space for the outside legend
fig.tight_layout(pad=1, rect=[0, 0, 0.85, 1])

# Increase horizontal spacing between subplots
fig.subplots_adjust(wspace=0.2)  # Increase this value to increase spacing

# Save figure
if SAVE_FIGS:
    base_filename = os.path.join(os.getcwd(), f"blade_pressure_distribution")
    bpy.savefig_in_formats(fig, base_filename, formats=[".png", ".svg"])

# Show figures
plt.show()


# # Define the dictionary with filenames and corresponding y column names
# SAVE_FIGS = True
# blades = [4, 5, 6, 7, 8, 9, 10]
# blade_span = [0.50, 0.75]

# # Create figure
# bpy.set_plot_options(minor_ticks=False, grid=False)
# fig, ax1 = plt.subplots(figsize=(5.2, 3.8))
# colors = plt.get_cmap("magma")(np.linspace(0.2, 0.75, len(blades)))
# linestyles = ['-', '-.']

# # Left Y-axis (Pressure)
# ax1.set_xlabel("Fraxction of axial chord")
# ax1.set_ylabel("Static pressure [kPa]")

# for index, blade in enumerate(blades):
#     filename = f"loading_blade{blade:d}_{blade_span[0]:0.2f}.dat"
#     df = pd.read_csv(filename, sep='\t', header=None)
#     df = pd.concat([df, df.iloc[0:1, :]], ignore_index=True)
#     ax1.plot(df[0], df[1]/1e3, label=blade, linewidth=0.75, 
#                      color=colors[index % len(colors)],
#                      linestyle=linestyles[index % len(linestyles)])
# ax1.legend(loc="upper right", fontsize=9)

# # Adjust pad
# fig.tight_layout(pad=1)


