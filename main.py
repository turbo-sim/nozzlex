import matplotlib.pyplot as plt
from perfect_gas_prop import perfect_gas_prop
from real_gas_prop import real_gas_prop
import numpy as np
import functions as function
import yaml
import time
import os


# Read YAML file
with open("settings.yaml", 'r') as file:
    case_data = yaml.safe_load(file)

# Upload fluid parameters from yaml file
fluid_name = case_data["fluid"]["fluid_name"]
fluid = real_gas_prop.Fluid(fluid_name)

# Upload nozzle parameters from yaml file
nozzle_type = case_data["nozzle_parameters"]["type"]
convergent_length = case_data["nozzle_parameters"]["convergent_length"]
divergent_length = case_data["nozzle_parameters"]["divergent_length"]
radius_in = case_data["nozzle_parameters"]["radius_in"]
radius_throat = case_data["nozzle_parameters"]["radius_throat"]
radius_out = case_data["nozzle_parameters"]["radius_out"]
roughness = float(case_data["nozzle_parameters"]["roughness"])

# Upload nozzle parameters from yaml file
p_in = int(float(case_data["boundary_conditions"]["p_stagnation"]))
T_in = case_data["boundary_conditions"]["T_stagnation"]
critical_flow = case_data["boundary_conditions"]["critical_flow"]

state_in = fluid.set_state(real_gas_prop.PT_INPUTS, p_in, T_in)

start_time = time.time()
supersonic_solution, possible_solution, impossible_solution, solution, flow_rate, pif_iterations = function.pipeline_steady_state_1D_autonomous(
    fluid_name=fluid_name, properties_in=state_in, temperature_in=T_in, pressure_in=p_in, convergent_length=convergent_length,
    divergent_length=divergent_length, roughness=roughness, radius_in=radius_in, radius_throat=radius_throat, radius_out=radius_out,
    nozzle_type=nozzle_type, critical_flow=critical_flow, include_friction=True, include_heat_transfer=False)

# solution, solution_supersonic = function.pipeline_steady_state_1D_critical(
#     fluid_name=fluid_name, properties_in=state_in, temperature_in=T_in, pressure_in=p_in, convergent_length=convergent_length,
#     divergent_length=divergent_length, roughness=roughness, radius_in=radius_in, radius_throat=radius_throat, radius_out=radius_out,
#     nozzle_type=nozzle_type, mass_flow=52.3, include_friction=True, include_heat_transfer=False)

# Calculate the duration
end_time = time.time()
duration = end_time - start_time

# Print simulation info
# os.system('cls')
function.print_dict(case_data)
print(" ")
print("Mach at the inlet:                         ", f"{solution["mach_number"][0]:.4f}", "(-)")
print("Mach at the throat:                        ", f"{solution["mach_number"][-1]:.4f}", "(-)")
print("Critical lenght:                           ", f"{solution["distance"][-1]:.4f}", "(m)")
print("Flow rate:                                 ", f"{flow_rate:.7f}", "(kg/s)")
print("PIF number of iterations:                  ", pif_iterations)
print("Computation time:                          ", f"{duration:.4f} seconds")

# Plot evolution of flow variables
colors = function.COLORS_MATLAB

function.set_plot_options(grid=False)
figure, ax1 = plt.subplots(figsize=(6.0, 4.8))
ax1.set_xlabel("Axis position [-]", fontsize=14)
ax1.set_ylabel("Normalized static pressure [-]", fontsize=14)
ax1.plot(
    impossible_solution["distance"],
    impossible_solution["pressure"] / impossible_solution["pressure"][0],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Impossible flow",
)
ax1.plot(
    solution["distance"],
    solution["pressure"] / impossible_solution["pressure"][0],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Critical flow",
)
ax1.plot(
    possible_solution["distance"],
    possible_solution["pressure"] / possible_solution["pressure"][0],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Possible flow",
)
ax1.plot(
    supersonic_solution["distance"],
    supersonic_solution["pressure"] / possible_solution["pressure"][0],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label="Supersonic branch",
)
ax1.legend(loc="best")
figure.tight_layout(pad=1)

figure, ax2 = plt.subplots(figsize=(6.0, 4.8))
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
figure.tight_layout(pad=1)

figure, ax3 = plt.subplots(figsize=(6.0, 4.8))
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
figure.tight_layout(pad=1)

# Plot numerical integration error
# The mass should always be conserved
# The total enthalpy is conserved if the heat transfer is zero
# The entropy is conserved if both heat transfer and friction are zero
m_error_sub = solution["mass_flow"] / solution["mass_flow"][0] - 1
h_error_sub = solution["total_enthalpy"] / solution["total_enthalpy"][0] - 1
s_error_sub = solution["entropy"] / solution["entropy"][0] - 1
figure, ax = plt.subplots(figsize=(6.0, 4.8))
ax.set_xlabel("Axis distance (m)")
ax.set_ylabel("Integration error")
ax.set_yscale("log")
ax.plot(solution["distance"], np.abs(m_error_sub), label="Mass flow error")
ax.plot(solution["distance"], np.abs(h_error_sub), label="Total enthalpy error")
# ax.plot(solution["distance"], np.abs(s_error), label="Entropy error")

# m_error_sup = solution_supersonic["mass_flow"] / solution_supersonic["mass_flow"][0] - 1
# h_error_sup = solution_supersonic["total_enthalpy"] / solution_supersonic["total_enthalpy"][0] - 1
# s_error_sup = solution_supersonic["entropy"] / solution_supersonic["entropy"][0] - 1
# ax.plot(solution_supersonic["distance"], np.abs(m_error_sup), label="Mass flow error")
# ax.plot(solution_supersonic["distance"], np.abs(h_error_sup), label="Total enthalpy error")
# # ax.plot(solution["distance"], np.abs(s_error), label="Entropy error")
# ax.legend(loc="best", fontsize=9)
# figure.tight_layout(pad=1)

# Plot the determinant of the matrix
# plt.figure()
# x = np.linspace(0, length, len(solution["determinant"]))
# plt.ylabel("Jacobian determinant")
# plt.xlabel("Axis position (m)")
# plt.plot(x, solution["determinant"])
# plt.tight_layout()


plt.show()
