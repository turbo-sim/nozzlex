import matplotlib.pyplot as plt
import numpy as np
import nozzlex.functions_old.functions_dem_nakagawa as function
import yaml
import time
import pandas as pd
import os
import barotropy as bpy


# ====================================================
# === 1. IMPORT SETTINGS                           ===
# ====================================================

with open("settings_nakagawa_dem.yaml", 'r') as file:
    case_data = yaml.safe_load(file)

# Upload fluid parameters from yaml file
fluid_name = case_data["fluid"]["fluid_name"]
fluid = bpy.Fluid(fluid_name)

# Upload nozzle parameters from yaml file
nozzle_type = case_data["nozzle_parameters"]["type"]
convergent_length = float(case_data["nozzle_parameters"]["convergent_length"])
divergent_length = float(case_data["nozzle_parameters"]["divergent_length"])
radius_in = float(case_data["nozzle_parameters"]["radius_in"])
radius_throat = float(case_data["nozzle_parameters"]["radius_throat"])
radius_out = float(case_data["nozzle_parameters"]["radius_out"])
roughness = float(case_data["nozzle_parameters"]["roughness"])
if nozzle_type == "Planar":
    width = float(case_data["nozzle_parameters"]["width"])
else:
    width = np.nan

# Upload nozzle parameters from yaml file
p_in = int(float(case_data["boundary_conditions"]["p_stagnation"]))
T_in = case_data["boundary_conditions"]["T_stagnation"]
critical_flow = case_data["boundary_conditions"]["critical_flow"]


# ====================================================
# === 2. PERFORM SIMULATION                        ===
# ====================================================

state_in = fluid.get_state(bpy.PT_INPUTS, p_in, T_in)

start_time = time.time()
flow_rate, pif_iterations, subsonic_solution, supersonic_solution = function.pipeline_steady_state_1D_autonomous(
    fluid_name=fluid_name, properties_in=state_in, temperature_in=T_in, pressure_in=p_in, convergent_length=convergent_length,
    divergent_length=divergent_length, roughness=roughness, radius_in=radius_in, radius_throat=radius_throat, radius_out=radius_out,
    nozzle_type=nozzle_type, width=width, critical_flow=critical_flow, include_friction=False, include_heat_transfer=False)

# Calculate the duration
end_time = time.time()
duration = end_time - start_time


# ====================================================
# === 3. PRINT RESULTS                             ===
# ====================================================

# os.system('cls')
function.print_dict(case_data)
print(" ")
# print("Mach at the inlet:                         ", f"{solution["mach_number"][0]:.4f}", "(-)")
# print("Mach at the throat:                        ", f"{solution["mach_number"][-1]:.4f}", "(-)")
# print("Critical lenght:                           ", f"{solution["distance"][-1]:.4f}", "(m)")
# print("Flow rate:                                 ", f"{flow_rate:.7f}", "(kg/s)")
print("PIF number of iterations:                  ", pif_iterations)
print("Computation time:                          ", f"{duration:.4f} seconds")


# ====================================================
# === 4. SAVE PLOT THERMODYNAMIC PROPERTIES        ===
# ====================================================

bpy.set_plot_options(grid=False)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(12, 9))

# First subplot - Normalized pressure
ax1 = axs[0, 0]
ax1.set_xlabel("Axis position [-]", fontsize=14)
ax1.set_ylabel("Normalized static pressure [-]", fontsize=14)
ax1.plot(
    subsonic_solution["distance"],
    subsonic_solution["pressure"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Critical branch HEM",
)
# ax1.plot(
#     supersonic_solution["distance"],
#     supersonic_solution["pressure"],
#     linewidth=1.00,
#     marker="o",
#     markersize=3.5,
#     markeredgewidth=1.00,
#     markerfacecolor="w",
#     label="Critical branch HEM",
# )
ax1.legend(loc="best")

# Second subplot - Velocity
ax2 = axs[0, 1]
ax2.set_xlabel("Axis position [-]", fontsize=14)
ax2.set_ylabel("Density", fontsize=14)
ax2.plot(
    subsonic_solution["distance"],
    subsonic_solution["quality"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Critical branch HEM",
)
# ax2.plot(
#     supersonic_solution["distance"],
#     supersonic_solution["quality"],
#     linewidth=1.00,
#     marker="o",
#     markersize=3.5,
#     markeredgewidth=1.00,
#     markerfacecolor="w",
#     label="Critical branch HEM",
# )
ax2.legend(loc="best")

# Third subplot - Mach
ax3 = axs[1, 0]
ax3.set_xlabel("Axis position [-]", fontsize=14)
ax3.set_ylabel("Mach [-]", fontsize=14)
ax3.plot(
    subsonic_solution["distance"],
    subsonic_solution["mach_number"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Critical flow",
)
# ax3.plot(
#     supersonic_solution["distance"],
#     supersonic_solution["mach_number"],
#     linewidth=1.00,
#     marker="o",
#     markersize=3.5,
#     markeredgewidth=1.00,
#     markerfacecolor="w",
#     label="Critical flow",
# )
ax3.legend(loc="best")

# Fourth subplot - Density
ax4 = axs[1, 1]
ax4.set_xlabel("Axis position [-]", fontsize=14)
ax4.set_ylabel("Void fraction[-]", fontsize=14)
ax4.plot(
    subsonic_solution["distance"],
    subsonic_solution["determinant_D"],  # Replace with density if available
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Critical flow",
)
# ax4.plot(
#     supersonic_solution["distance"],
#     supersonic_solution["determinant_D"],  # Replace with density if available
#     linewidth=1.00,
#     marker="o",
#     markersize=3.5,
#     markeredgewidth=1.00,
#     markerfacecolor="w",
#     label="Critical flow",
# )
ax4.legend(loc="best")

fig.tight_layout(pad=1.0)


# ====================================================
# === 5. SAVE SOLUTION AS CSV                      ===
# ====================================================

# # Create post_processing folder if not existing
# output_dir = os.path.join(os.path.dirname(__file__), "post_processing")
# os.makedirs(output_dir, exist_ok=True)

# # Convert solution dict into a DataFrame
# df = pd.DataFrame(solution)

# # Save CSV inside post_processing
# output_file = os.path.join(output_dir, "results_120a.csv")
# df.to_csv(output_file, index=False)

# print(f"Solution saved to {output_file}")


plt.show()