import matplotlib.pyplot as plt
import numpy as np
import nozzlex.functions_old.functions_dem_smd as function
import yaml
import time
import os
import shutil
import barotropy as bpy # Only used to plot the Ts and Ph diagrams, not for evaluating properties during computation
import pandas as pd
import os

# ====================================================
# === 1. IMPORT SETTINGS                           ===
# ====================================================

with open("settings_mobidick_dem.yaml", 'r') as file:
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
_, _, solution = function.pipeline_steady_state_1D_autonomous(
    fluid_name=fluid_name, properties_in=state_in, temperature_in=T_in, pressure_in=p_in, convergent_length=convergent_length,
    divergent_length=divergent_length, roughness=roughness, radius_in=radius_in, radius_throat=radius_throat, radius_out=radius_out,
    nozzle_type=nozzle_type, width=width, critical_flow=critical_flow, include_friction=True, include_heat_transfer=False)

# solution, solution_supersonic = function.pipeline_steady_state_1D_critical(
#     fluid_name=fluid_name, properties_in=state_in, temperature_in=T_in, pressure_in=p_in, convergent_length=convergent_length,
#     divergent_length=divergent_length, roughness=roughness, radius_in=radius_in, radius_throat=radius_throat, radius_out=radius_out,
#     nozzle_type=nozzle_type, mass_flow=52.3, include_friction=True, include_heat_transfer=False)

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
# print("PIF number of iterations:                  ", pif_iterations)
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
    solution["z"],
    solution["pressure"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Critical branch HEM",
)
ax1.legend(loc="best")

# Second subplot - Velocity
ax2 = axs[0, 1]
ax2.set_xlabel("Axis position [-]", fontsize=14)
ax2.set_ylabel("Velocity [m/s]", fontsize=14)
ax2.plot(
    solution["z"],
    solution["velocity"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Critical branch HEM",
)
ax2.legend(loc="best")

# Third subplot - Mach
ax3 = axs[1, 0]
ax3.set_xlabel("Axis position [-]", fontsize=14)
ax3.set_ylabel("Determinant [-]", fontsize=14)
ax3.plot(
    solution["z"],
    solution["diameter"],  # Replace with Mach if available
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Critical flow",
)
ax3.legend(loc="best")

# Fourth subplot - Density
ax4 = axs[1, 1]
ax4.set_xlabel("Axis position [-]", fontsize=14)
ax4.set_ylabel("Gamma [-]", fontsize=14)
ax4.plot(
    solution["z"],
    solution["gamma"],  # Replace with density if available
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Critical flow",
)
ax4.legend(loc="best")

# 1. Convert the dictionary to a pandas DataFrame
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 2. Construct the filename using p_in and T_in
# We use f-strings to plug the variables directly into the string
filename = f"p{p_in/1e5}_T{T_in}_SMD.csv"
filepath = os.path.join(output_folder, filename)

# 3. Save the solution dictionary to CSV via Pandas
df = pd.DataFrame(solution)
df.to_csv(filepath, index=False)

print(f"--- Data saved to: {filepath} ---")

fig.tight_layout(pad=1.0)
plt.show()