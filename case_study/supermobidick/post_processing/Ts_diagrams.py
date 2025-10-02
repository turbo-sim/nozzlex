import os
import numpy as np
import matplotlib.pyplot as plt
import barotropy as bpy


# Create the folder to save figures
bpy.set_plot_options(grid=False)
colors = bpy.COLORS_MATLAB

fluid_name = "water"
fluid = bpy.Fluid(name=fluid_name, backend="HEOS")


# Create figure
fig, ax1 = plt.subplots(figsize=(6, 5))
ax1.set_xlabel("Entropy (J/(kg K)", fontsize = 14)
ax1.set_ylabel("Temperature (K)", fontsize = 14)

prop_x1, prop_y1 = "s", "T"

fluid = bpy.Fluid(name=fluid_name, backend="HEOS")

fluid.plot_phase_diagram(
    prop_x1,
    prop_y1,
    axes=ax1,
    N=50,
    plot_critical_point=True,
    plot_quality_isolines=True,
    plot_saturation_line=True,
    plot_spinodal_line=True,
    spinodal_line_color="gray",
    show_in_legend=True,

)

# Plot expansions
p_in = 120e5

# 120a
T_in_a = 578.95
p_out_a =6.76e6
s_out_a = 3299.83
state_in_a = fluid.get_state(bpy.PT_INPUTS, p_in, T_in_a)
state_out_a = fluid.get_state(bpy.PSmass_INPUTS, p_out_a, s_out_a)
plt.plot([state_in_a["s"], state_out_a["s"]], [state_in_a["T"], state_out_a["T"]], "-o", linewidth=2, label = "Case 120a")

# 120b
T_in_b = 592.55
p_out_b =7.19e6
s_out_b = 3441
state_in_b = fluid.get_state(bpy.PT_INPUTS, p_in, T_in_b)
state_out_b = fluid.get_state(bpy.PSmass_INPUTS, p_out_b, s_out_b)
plt.plot([state_in_b["s"], state_out_b["s"]], [state_in_b["T"], state_out_b["T"]], "-o", linewidth=2, label = "Case 120b")

ax1.legend(loc="best")
plt.tight_layout()
plt.xlim(3000, 4500)
plt.ylim(520, 660)

# Save the plot
# Define folder to save plots
save_folder = "images"
os.makedirs(save_folder, exist_ok=True)

# Define base filename
base_filename = os.path.join(save_folder, "Ts_diagram")

# Save in multiple formats
fig.savefig(base_filename + ".png", dpi=300)
fig.savefig(base_filename + ".svg")
fig.savefig(base_filename + ".eps")

# plt.show()
