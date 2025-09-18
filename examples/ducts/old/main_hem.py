import matplotlib.pyplot as plt
from perfect_gas_prop import perfect_gas_prop
from real_gas_prop import real_gas_prop
import numpy as np
import yaml
import time
import os
import shutil
import barotropy as bpy # Only used to plot the Ts and Ph diagrams, not for evaluating properties during computation


# ====================================================
# === 1. IMPORT SETTINGS                           ===
# ====================================================
with open("settings.yaml", 'r') as file:
    case_data = yaml.safe_load(file)

import functions_hem_cp as function

# if case_data["fluid"]["look_up_table"] is False:
#     import functions_hem_cp as function
# else:
#     import functions_hem_lut as function

# Upload fluid parameters from yaml file
fluid_name = case_data["fluid"]["fluid_name"]
fluid = real_gas_prop.Fluid(fluid_name)


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

state_in = fluid.set_state(real_gas_prop.PT_INPUTS, p_in, T_in)

start_time = time.time()

supersonic_solution, possible_solution, impossible_solution, solution, flow_rate, pif_iterations = function.pipeline_steady_state_1D_autonomous(
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
bpy.print_dict(case_data)
print(" ")
print("Mach at the inlet:                         ", f"{solution["mach_number"][0]:.4f}", "(-)")
print("Mach at the throat:                        ", f"{solution["mach_number"][-1]:.4f}", "(-)")
print("Critical lenght:                           ", f"{solution["distance"][-1]:.4f}", "(m)")
print("Flow rate:                                 ", f"{flow_rate:.7f}", "(kg/s)")
print("PIF number of iterations:                  ", pif_iterations)
print("Computation time:                          ", f"{duration:.4f} seconds")


# ====================================================
# === 4. SAVE PLOT THERMODYNAMIC PROPERTIES        ===
# ====================================================
# Initialize x_values
x_values = np.linspace(0, convergent_length + divergent_length, num=50)

# Initialize empty lists to store results
areas = []
area_slopes = []
perimeters = []
radii = []

# # Prepare the plot
# figure, ax1 = plt.subplots(figsize=(6.0, 4.8))

# # Loop through each x position
# for x_i in x_values:
#     area, area_slope, perimeter, radius = function.get_linear_convergent_divergent(
#         x_i,
#         convergent_length=convergent_length,
#         divergent_length=divergent_length,
#         radius_in=radius_in,
#         radius_out=radius_out,
#         radius_throat=radius_throat,
#         width=width,
#         type=nozzle_type,
#     )
    
#     # Store the values
#     areas.append(area)
#     area_slopes.append(area_slope)
#     perimeters.append(perimeter)
#     radii.append(radius)

# # Now plot after collecting all the values
# ax1.plot(x_values, radii)
# # ax1.plot([0,27.35], [5, 0.12])
# # Labeling and showing the plot
# ax1.set_xlabel("Position [m]", fontsize=14)
# ax1.set_ylabel("Area Slope", fontsize=14)
# ax1.grid(True)

##################
colors = function.COLORS_MATLAB

bpy.set_plot_options(grid=False)
fig, axs = plt.subplots(2, 2, figsize=(12, 9))
# figure, ax1 = plt.subplots(figsize=(6.0, 4.8))
# First subplot - Normalized pressure
ax1 = axs[0, 0]
ax1.set_xlabel("Axis position [-]", fontsize=14)
ax1.set_ylabel("Normalized static pressure [-]", fontsize=14)
ax1.plot(
    impossible_solution["distance"],
    impossible_solution["pressure"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Impossible flow",
)
ax1.plot(
    solution["distance"],
    solution["pressure"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Critical flow",
)
ax1.plot(
    possible_solution["distance"],
    possible_solution["pressure"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Possible flow",
)
ax1.plot(
    supersonic_solution["distance"],
    supersonic_solution["pressure"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Supersonic branch",
)
ax1.legend(loc="best")
# figure.tight_layout(pad=1)

# figure, ax2 = plt.subplots(figsize=(6.0, 4.8))
# Second subplot - Velocity
ax2 = axs[0, 1]
ax2.set_xlabel("Axis position [-]", fontsize=14)
ax2.set_ylabel("Velocity [m/s]", fontsize=14)
ax2.plot(
    solution["distance"],
    solution["velocity"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Critical flow",
)
ax2.plot(
    impossible_solution["distance"],
    impossible_solution["velocity"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Impossible flow",
)
ax2.plot(
    possible_solution["distance"],
    possible_solution["velocity"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Possible flow",
)
ax2.plot(
    supersonic_solution["distance"],
    supersonic_solution["velocity"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Supersonic branch",
)
ax2.legend(loc="best")
# figure.tight_layout(pad=1)

# figure, ax3 = plt.subplots(figsize=(6.0, 4.8))
# Second subplot - Mach
ax3 = axs[1, 0]
ax3.set_xlabel("Axis position [-]", fontsize=14)
ax3.set_ylabel("Mach number [-]", fontsize=14)
ax3.plot(
    solution["distance"],
    solution["mach_number"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Critical flow",
)
ax3.plot(
    impossible_solution["distance"],
    impossible_solution["mach_number"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Impossible flow",
)
ax3.plot(
    possible_solution["distance"],
    possible_solution["mach_number"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Possible flow",
)
ax3.plot(
    supersonic_solution["distance"],
    supersonic_solution["mach_number"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Supersonic branch",
)
ax3.legend(loc="best")
# figure.tight_layout(pad=1)

figure, ax4 = plt.subplots(figsize=(6.0, 4.8))
# Second subplot - Density
ax4 = axs[1, 1]
ax4.set_xlabel("Axis position [-]", fontsize=14)
ax4.set_ylabel("Determinant [-]", fontsize=14)
ax4.plot(
    possible_solution["distance"],
    possible_solution["quality"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="equilibrium quality",
)
ax4.plot(
    supersonic_solution["distance"],
    supersonic_solution["entropy"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="equilibrium quality",
)
ax4.plot(
    supersonic_solution["distance"],
    supersonic_solution["entropy"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="quality",
)
ax4.legend(loc="best")
plt.show()
figure.tight_layout(pad=1)
fig.tight_layout(pad=2)
# plt.show()
plt.savefig(os.path.join("results", "properties.png"))

# ====================================================
# === 5. SAVE PLOT T-s AND P-h DIAGRAMS            ===
# ====================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0), gridspec_kw={"wspace": 0.25})
ax1.set_xlabel("Entropy (J/kg/K)")
ax1.set_ylabel("Temperature (K)")
ax2.set_xlabel("Enthalpy (J/kg)")
ax2.set_ylabel("Pressure (Pa)")

prop_x1, prop_y1 = "s","T"
prop_x2, prop_y2 = "h", "p"

fluid = bpy.Fluid(name=fluid_name, backend="HEOS")
fluid.plot_phase_diagram(
    prop_x1,
    prop_y1,
    axes=ax1,
    plot_critical_point=True,
    plot_quality_isolines=True,
    plot_pseudocritical_line=False,
    plot_spinodal_line=True,
)

# ax1.plot(solution["entropy"], 
#         solution["temperature"],
#         linewidth=1.00,
#         marker="o",
#         markersize=3.5,
#         markeredgewidth=1.00,
#         markerfacecolor="w",
#         label="Convergent",
#         )

# ax1.plot(supersonic_solution["entropy"], 
#         supersonic_solution["temperature"],
#         linewidth=1.00,
#         marker="o",
#         markersize=3.5,
#         markeredgewidth=1.00,
#         markerfacecolor="w",
#         label="Divergent",
#         )
ax1.legend(loc="best")

fluid.plot_phase_diagram(
    prop_x2,
    prop_y2,
    axes=ax2,
    plot_critical_point=True,
    plot_quality_isolines=True,
    plot_pseudocritical_line=False,
    plot_spinodal_line=False,
)

# ax2.plot(solution["enthalpy"], 
#         solution["pressure"],
#         linewidth=1.00,
#         marker="o",
#         markersize=3.5,
#         markeredgewidth=1.00,
#         markerfacecolor="w",
#         label="Convergent",
#         )

# ax2.plot(supersonic_solution["enthalpy"], 
#         supersonic_solution["pressure"],
#         linewidth=1.00,
#         marker="o",
#         markersize=3.5,
#         markeredgewidth=1.00,
#         markerfacecolor="w",
#         label="Divergent",
#         )
ax2.legend(loc="best")

fig.tight_layout(pad=2)

plt.savefig(os.path.join("results", "diagrams.png"))
plt.show()

# ====================================================
# === 6. SAVE PLOT NUMERICAL INTEGRATION ERROR     ===
# ====================================================

# The mass should always be conserved
# The total enthalpy is conserved if the heat transfer is zero
# The entropy is conserved if both heat transfer and friction are zero

figure, ax = plt.subplots(figsize=(6.0, 4.8))
m_error_sub = solution["mass_flow"] / solution["mass_flow"][0] - 1
h_error_sub = solution["total_enthalpy"] / solution["total_enthalpy"][0] - 1
s_error_sub = solution["entropy"] / solution["entropy"][0] - 1
ax.set_xlabel("Axis distance (m)")
ax.set_ylabel("Integration error")
ax.set_yscale("log")
ax.plot(solution["distance"], np.abs(m_error_sub), label="Mass flow error")
ax.plot(solution["distance"], np.abs(h_error_sub), label="Total enthalpy error")
# ax.plot(solution["distance"], np.abs(s_error), label="Entropy error")
m_error_sup = supersonic_solution["mass_flow"] / solution["mass_flow"][0] - 1
h_error_sup = supersonic_solution["total_enthalpy"] / solution["total_enthalpy"][0] - 1
s_error_sup = supersonic_solution["entropy"] / solution["entropy"][0] - 1
ax.plot(supersonic_solution["distance"], np.abs(m_error_sup), label="Mass flow error")
ax.plot(supersonic_solution["distance"], np.abs(h_error_sup), label="Total enthalpy error")
# ax.plot(solution["distance"], np.abs(s_error), label="Entropy error")
ax.legend(loc="best")
fig.tight_layout(pad=2)

plt.savefig(os.path.join("results", "error.png"))

# Copy yaml file in the result folder
shutil.copyfile("settings.yaml", 
                os.path.join("results", "settings.yaml"))


plt.show()
