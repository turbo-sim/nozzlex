import matplotlib.pyplot as plt
import perfect_gas_props 
import numpy as np
import functions_2 as functions 
import yaml
import time
import os


# Read YAML file
with open("settings.yaml", 'r') as file:
    case_data = yaml.safe_load(file)

# perfect_gas = case_data["fluid"]["perfect_gas"] As an example

p_in = int(float(case_data["boundary_conditions"]["p_stagnation"]))
T_in = case_data["boundary_conditions"]["T_stagnation"]
critical_flow = case_data["boundary_conditions"]["critical_flow"]
if critical_flow is True:
    M_in = None
    mass_flow_inlet = None
else:
    M_in = case_data["boundary_conditions"]["mach_inlet"]
    mass_flow_inlet = case_data["boundary_conditions"]["mass_flow_inlet"]


D_in = 0.1
A_ratio = 0.762
length = 1

properties = perfect_gas_props.perfect_gas_props("PT_INPUTS", p_in, T_in)
rho_in = properties["d"]
speed_sound_in = properties["a"]
mu_in = properties["mu"]
gamma_ratio = properties["gamma"]
R = properties["R"]

start_time = time.time()
solution, out, flow_rate, pif_iterations = functions.pipeline_steady_state_1D(p_in, T_in, D_in, properties, length, 1e-6, A_ratio, mass_flow=mass_flow_inlet, mach_in=M_in, critical_flow=critical_flow)

# Calculate the duration
end_time = time.time()
duration = end_time - start_time

# Print simulation info
os.system('cls')
print("Mach at the inlet:                         ", f"{solution["mach_number"][0]:.4f}", "(-)")
print("Mach at the throat:                        ", f"{solution["mach_number"][-1]:.4f}", "(-)")
print("Critical lenght:                           ", f"{solution["distance"][-1]:.4f}", "(m)")
print("Flow rate:                                 ", f"{flow_rate:.4f}", "(kg/s)")
print("PIF number of iterations:                  ", pif_iterations)
print("Computation time:                          ", f"{duration:.4f} seconds")



# Plot evolution of flow variables
functions.set_plot_options(grid=False)
figure, ax = plt.subplots(figsize=(6.0, 4.8))
ax.set_xlabel("Distance along pipe [m]")
ax.set_ylabel("Normalized flow variables")
ax.plot(
    solution["distance"],
    solution["pressure"] / solution["pressure"][0],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label=r"$p/p_{\mathrm{in}}$",
)
ax.plot(
    solution["distance"],
    solution["velocity"] / solution["velocity"][0],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label=r"$v/v_{\mathrm{in}}$",
)
# ax.plot(
#     solution["distance"],
#     solution["temperature"] / solution["temperature"][0],
#     linewidth=1.00,
#     marker="o",
#     markersize=3.5,
#     markeredgewidth=1.00,
#     markerfacecolor="w",
#     label=r"$T/T_{\mathrm{in}}$",
# )
ax.plot(
    solution["distance"],
    solution["mach_number"],
    linewidth=1.00,
    marker="o",
    markersize=3.5,
    markeredgewidth=1.00,
    markerfacecolor="w",
    label=r"$Ma=v/a$",
)
ax.legend(loc="best")
figure.tight_layout(pad=1)

plt.figure()
x = np.linspace(0, length, len(solution["determinant"]))
plt.ylabel("Jacobian determinant")
plt.xlabel("Axis position (m)")
plt.plot(x, solution["determinant"])
plt.tight_layout()

# Plot numerical integration error
# The mass should always be conserved
# The total enthalpy is conserved if the heat transfer is zero
# The entropy is conserved if both heat transfer and friction are zero
m_error = solution["mass_flow"] / solution["mass_flow"][0] - 1
h_error = solution["total_enthalpy"] / solution["total_enthalpy"][0] - 1
s_error = solution["entropy"] / solution["entropy"][0] - 1
figure, ax = plt.subplots(figsize=(6.0, 4.8))
ax.set_xlabel("Pipeline distance [km]")
ax.set_ylabel("Integration error")
ax.set_yscale("log")
ax.plot(solution["distance"], np.abs(m_error), label="Mass flow error")
ax.plot(solution["distance"], np.abs(h_error), label="Total enthalpy error")
# ax.plot(solution["distance"], np.abs(s_error), label="Entropy error")
ax.legend(loc="best", fontsize=9)
figure.tight_layout(pad=1)

plt.show()