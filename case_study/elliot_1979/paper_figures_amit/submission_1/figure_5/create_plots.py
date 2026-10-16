import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

import barotropy as bpy


# Select cases to plot
SAVE_FIGS = True
bpy.set_plot_options(minor_ticks=False, grid=False)
cases = [1, 2, 3, 4, 5, 6, 7, 8]
values = [22.448, 29.257, 37.962, 48.852, 60.765, 67.884, 76.460, 94.485]
cases_dict = dict(zip(cases, values))
cases_dict = {k: v for k, v in cases_dict.items() if k in [1, 3, 6, 8]}

# Create figure
fig, ax1 = plt.subplots(figsize=(5.2, 3.0))#, constrained_layout=True)
ax1.set_xlabel("Axial distance [m]")
ax1.set_ylabel("Pressure [kPa]")
colors = plt.get_cmap("Blues")(np.linspace(0.35, 0.80, len(cases)+1))

# Right Y-axis (Velocity)
ax2 = ax1.twinx() 
ax2.set_ylabel('Void fraction [%]') 
ax2.set_ylim([0, 100])


# Create the inset axis with more control over its position
# bbox_to_anchor=(left, bottom, width, height)
axins = inset_axes(ax1, width="60%", height="100%", loc='upper right', 
                   bbox_to_anchor=(0.55, 0.3, 0.35, 0.35), bbox_transform=ax1.transAxes)


# Reduce edge width of the inset axes
axins.tick_params(axis='both', which='both', width=0.75, length=3)  # Adjust width and length as needed
for spine in axins.spines.values():
    spine.set_linewidth(0.75)  # Adjust the edge width as needed

# Specify the part of the graph you want to zoom in on
x1, x2, y1, y2 = 0.28, 0.32, 30, 170  # adjust these values to focus on the desired area
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

for index, R in cases_dict.items():
    df = bpy.parse_fluent_xy(f"case_{index:03}_a_pressure.xy")
    x = df['axis']['Position']
    y = df['axis']['Static Pressure'] / 1e3
    ax1.plot(x, y, linestyle='-', linewidth=1.05, label=rf"$R={R:0.2f}$",
                     color=colors[index % len(colors)],)

    axins.plot(x, y, linestyle='-', linewidth=1.05, color=colors[index % len(colors)])


    df = bpy.parse_fluent_xy(f"case_{index:03}_a_barotropic_void_fraction.xy")
    x = df['axis']['Position']
    y = df['axis']['barotropic_void_fraction'] * 100
    ax2.plot(x, y, linestyle='--', linewidth=1.05,
                     color=colors[index % len(colors)],)


# Create square around zoom axes
axins.tick_params(axis='both', which='major', labelsize=10)  # Adjust labelsize as needed
mark_inset(ax1, axins, loc1=1, loc2=1, fc="none", ec="0", lw=0.75)

# Add legend
ax1.legend(loc="lower left", fontsize=9, ncol=2, columnspacing=0.75, handlelength=1)

# Adjust pad
fig.tight_layout(pad=1)

# Save figure
if SAVE_FIGS:
    base_filename = os.path.join(os.getcwd(), f"comparison_pressure")
    bpy.savefig_in_formats(fig, base_filename, formats=[".png", ".svg"])

# Show figures
plt.show()

