# %%
import matplotlib.pyplot as plt
# from perfect_gas_prop import perfect_gas_prop
# from real_gas_prop import real_gas_prop
# import numpy as np
import functions_5eq_jax_autonomus as function
import yaml
import time
# import os
# import shutil
# import barotropy as bpy # Only used to plot the Ts and Ph diagrams, not for evaluating properties during computation
import jaxprop as jxp


# %%

# ====================================================
# === 1. IMPORT SETTINGS                           ===
# ====================================================

with open("settings_6eq.yaml", 'r') as file:
    case_data = yaml.safe_load(file)

# Upload fluid parameters from yaml file
fluid_name = ["water", "nitrogen"]

fluid1 = jxp.FluidJAX(fluid_name[0])
fluid2 = jxp.FluidJAX(fluid_name[1])


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

# Upload nozzle parameters from yaml file
p_in = int(float(case_data["boundary_conditions"]["p_stagnation"]))
T_in = case_data["boundary_conditions"]["T_stagnation"]
critical_flow = case_data["boundary_conditions"]["critical_flow"]

# ====================================================
# === 2. PERFORM SIMULATION                        ===
# ====================================================

throat_location = 141.63e-3

## Works with flashing only as of now
# TODO: extend to condensation

start_time = time.time()

supersonic_solution, solution, out_list, x = function.pipeline_steady_state_1D_two_component(
    fluid_name_1 = fluid_name[0],
    fluid_name_2 = fluid_name[1],
    pressure_in_1 = 2e6,
    pressure_in_2 = 2e6,
    temperature_in_1 = 22 + 273.15,
    temperature_in_2 = 22 + 273.15,
    properties_in_1 = None,
    properties_in_2 = None,
    area_1 = None,
    area_2 = None,
    m_dot_in_1 = 3.604,
    m_dot_in_2 = 0.053,
    mixture_ratio = 68, # m_dot_liquid / m_dot_gas
    nozzle_length = 0.30, # 90 mm
    convergent_length = None,
    divergent_length = None,
    roughness = 0.0,
    radius_in = None,
    radius_throat = None,
    radius_out = None,
    nozzle_type = None,
    width = None,
    number_of_points=10,
    source_terms= 1e4
)

print(x)

# print(raw_solution)
end_time = time.time()
duration = end_time - start_time

# to_save = ["distance", "velocity", "density", "pressure", "speed_of_sound", "mass_flow", "entropy", "mach_number", "quality", "stable_fraction"]
# function.save_all_to_csv(solution, filename="nakagawa_p61_T293_IDEM2.csv")


# # ====================================================
# # === 3. PRINT RESULTS                             ===
# # ====================================================

# # os.system('cls')
# function.print_dict(case_data)
# print(" ")
# # print("Mach at the inlet:                         ", f"{solution["mach_number"][0]:.4f}", "(-)")
# # print("Mach at the throat:                        ", f"{solution["mach_number"][-1]:.4f}", "(-)")
# # print("Critical lenght:                           ", f"{solution["distance"][-1]:.4f}", "(m)")
# # print("Flow rate:                                 ", f"{flow_rate:.7f}", "(kg/s)")
# print("Computation time:                          ", f"{duration:.4f} seconds")
out = solution

# Extract raw data from out_list
distance = solution["distance"]
# print(distance)

velocity = solution["velocity"]
                    
density_1 = solution["density_1"]
density_2 = solution["density_2"]

# delta_rho_water = density_1[-3] - density_1[0]
# # print("Delta rho water:",delta_rho_water)
# delta_rho_nitrogen = density_2[-3] - density_2[0]
# print("Delta rho nitrogen:",delta_rho_nitrogen)

alpha_1 = solution["alpha_1"]
alpha_2 = solution["alpha_2"]

h1 = solution["enthalpy_1"]
h2 = solution["enthalpy_2"]

area = solution["diameter"]
# d = [solution["distance"],supersonic_solution["distance"]]
# v = [solution["velocity"],supersonic_solution["velocity"]]
# plt.figure()
# plt.plot(d,v)


# print("\n")
# print(f"supersonic solution:{supersonic_solution}")
# print("\n")

fig, axs = plt.subplots(1, 3, figsize=(18, 3))
fig.suptitle("Simulation Results vs Distance")

axs[0].plot(solution["distance"], solution["velocity"], label="solution")
axs[0].plot(supersonic_solution["distance"], supersonic_solution["velocity"], label="supersonic_solution")
axs[0].set_xlabel("Distance")
axs[0].set_ylabel("Velocity")
axs[0].legend()
# plt.show()


# plt.figure()
# plt.title("diameter")
# plt.plot(distance, area, marker="o")

axs[1].plot(distance, solution["determinant"])
axs[1].plot(solution["distance"], solution["determinant"], label="solution")
axs[1].plot(supersonic_solution["distance"], supersonic_solution["determinant"], label="supersonic_solution")
axs[1].set_xlabel("Distance")
axs[1].set_ylabel("determinant")
axs[1].legend()

axs[2].plot(distance, solution["pressure"])
axs[2].plot(solution["distance"], solution["pressure"], label="solution")
axs[2].plot(supersonic_solution["distance"], supersonic_solution["pressure"], label="supersonic_solution")
axs[2].set_xlabel("Distance")
axs[2].set_ylabel("Pressure")
axs[2].legend()

# plt.figure()
# plt.title("Mach")
# plt.plot(distance, solution["mach_mix"])
# plt.show()


# plt.figure()
# plt.title("velocity")
# plt.plot(distance, solution["velocity"])

# Create figure with 4 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Simulation Results vs Distance")

# Plot velocities
axs[0, 0].plot(distance, solution["temperature_1"], label="T1", color="red")
axs[0, 0].plot(supersonic_solution["distance"], supersonic_solution["temperature_1"],linestyle="--", color="red")
axs[0, 0].plot(distance, solution["temperature_2"], label="T2", color="blue")
axs[0, 0].plot(supersonic_solution["distance"], supersonic_solution["temperature_2"],linestyle="--", color="blue")
axs[0, 0].set_xlabel("Distance [m]")
axs[0, 0].set_ylabel("T [K]")
axs[0, 0].set_title("Temperature")
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot densities
axs[0, 1].plot(distance, density_1/density_1[0], label="Density 1",color="red")
axs[0, 1].plot(supersonic_solution["distance"], supersonic_solution["density_1"]/density_1[0],linestyle="--", color="red")
axs[0, 1].plot(distance, density_2/density_2[0], label="Density 2", color="blue")
axs[0, 1].plot(supersonic_solution["distance"], supersonic_solution["density_2"]/density_2[0],linestyle="--", color="blue")
axs[0, 1].set_xlabel("Distance [m]")
axs[0, 1].set_ylabel("Density/Density_in")
axs[0, 1].set_title("Density")
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plot alphas
axs[1, 0].plot(distance, alpha_1, label="Alpha 1", color="red")
axs[1, 0].plot(supersonic_solution["distance"], supersonic_solution["alpha_1"],linestyle="--", color="red")
axs[1, 0].plot(distance, alpha_2, label="Alpha 2", color="blue")
axs[1, 0].plot(supersonic_solution["distance"], supersonic_solution["alpha_2"],linestyle="--", color="blue")
axs[1, 0].set_xlabel("Distance [m]")
axs[1, 0].set_ylabel("Void Fraction")
axs[1, 0].set_title("Void Fraction")
axs[1, 0].legend()
axs[1, 0].grid(True)

# Plot enthalpies

# ------------- normalized ------------- #
axs[1, 1].plot(distance, h1/h1[0], label="Enthalpy 1", color="red")
axs[1, 1].plot(supersonic_solution["distance"], supersonic_solution["enthalpy_1"]/h1[0],linestyle="--", color="red")
axs[1, 1].plot(distance, h2/h2[0], label="Enthalpy 2", color="blue")
axs[1, 1].plot(supersonic_solution["distance"], supersonic_solution["enthalpy_2"]/h2[0],linestyle="--", color="blue")

# ----------- non-normalized ----------- #
# axs[1, 1].plot(distance, h1, label="Enthalpy 1", color="red")
# axs[1, 1].plot(supersonic_solution["distance"], supersonic_solution["enthalpy_1"],linestyle="--", color="red")
# axs[1, 1].plot(distance, h2, label="Enthalpy 2", color="blue")
# axs[1, 1].plot(supersonic_solution["distance"], supersonic_solution["enthalpy_2"],linestyle="--", color="blue")

axs[1, 1].set_xlabel("Distance [m]")
axs[1, 1].set_ylabel("Enthalpy/Enthalpy_in")
axs[1, 1].set_title("Enthalpy")
axs[1, 1].legend()
axs[1, 1].grid(True)



# # Create figure with 4 subplots
# fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# fig.suptitle("Simulation Results vs Distance")

# # ---------------- TEMPERATURE ----------------
# axs[0, 0].plot(distance, solution["temperature_1"], label="T1")
# axs[0, 0].plot(distance, solution["temperature_2"], label="T2")
# axs[0, 0].set_xlabel("Distance [m]")
# axs[0, 0].set_ylabel("T [K]")
# axs[0, 0].set_title("Temperature")
# axs[0, 0].legend()
# axs[0, 0].grid(True)

# # ---------------- DENSITY (two y-axes) ----------------
# ax_density_left = axs[0, 1]               # left y-axis
# ax_density_right = ax_density_left.twinx()  # right y-axis

# ax_density_left.plot(distance, solution["density_1"], color="tab:blue", label="Density 1")
# ax_density_right.plot(distance, solution["density_2"], color="tab:orange", label="Density 2")

# ax_density_left.set_xlabel("Distance [m]")
# ax_density_left.set_ylabel("Density 1 [kg/m³]", color="tab:blue")
# ax_density_right.set_ylabel("Density 2 [kg/m³]", color="tab:orange")
# ax_density_left.set_title("Density")
# ax_density_left.tick_params(axis='y', labelcolor='tab:blue')
# ax_density_right.tick_params(axis='y', labelcolor='tab:orange')

# ax_density_left.grid(True)

# # ---------------- VOID FRACTION ----------------
# axs[1, 0].plot(distance, alpha_1, label="Alpha 1")
# axs[1, 0].plot(distance, alpha_2, label="Alpha 2")
# axs[1, 0].set_xlabel("Distance [m]")
# axs[1, 0].set_ylabel("Void Fraction")
# axs[1, 0].set_title("Void Fraction")
# axs[1, 0].legend()
# axs[1, 0].grid(True)

# # ---------------- ENTHALPY (two y-axes) ----------------
# ax_enthalpy_left = axs[1, 1]
# ax_enthalpy_right = ax_enthalpy_left.twinx()

# line3, = ax_enthalpy_left.plot(distance, h1, color="tab:red", label="Enthalpy 1")
# line4, = ax_enthalpy_right.plot(distance, h2, color="tab:green", label="Enthalpy 2")

# ax_enthalpy_left.set_xlabel("Distance [m]")
# ax_enthalpy_left.set_ylabel("Enthalpy 1 [J/kg]", color="tab:red")
# ax_enthalpy_right.set_ylabel("Enthalpy 2 [J/kg]", color="tab:green")
# ax_enthalpy_left.set_title("Enthalpy")
# ax_enthalpy_left.tick_params(axis='y', labelcolor='tab:red')
# ax_enthalpy_right.tick_params(axis='y', labelcolor='tab:green')
# axs[1, 1].legend()
# axs[1, 1].grid(True)

# # Optional independent y-limits
# ax_enthalpy_left.set_ylim(1e5, 2.5e5)
# ax_enthalpy_right.set_ylim(5e4, 1.5e5)

# # Combined legend
# lines_enthalpy = [line3, line4]
# labels_enthalpy = [l.get_label() for l in lines_enthalpy]
# ax_enthalpy_left.legend(lines_enthalpy, labels_enthalpy, loc="best")
# ax_enthalpy_left.grid(True)

print(f"Last position is {distance[-1]} and throath position is {throat_location}")
print(f"Difference is: {throat_location - distance[-1]}")
print(f"Throat velocity is {solution["velocity"][-1]}")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
# # ====================================================
# # === 4. SAVE PLOT THERMODYNAMIC PROPERTIES        ===
# # ====================================================
# # Initialize x_values
# x_values = np.linspace(0, convergent_length + divergent_length, num=50)

# # Initialize empty lists to store results
# areas = []
# area_slopes = []
# perimeters = []
# radii = []

# Prepare the plot

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
# ax1.plot(raw_solution.t, raw_solution.y)
# # ax1.plot([0,27.35], [5, 0.12])
# # Labeling and showing the plot
# ax1.set_xlabel("Position [m]", fontsize=14)
# ax1.set_ylabel("Area Slope", fontsize=14)
# ax1.grid(True)

# ##################
# colors = function.COLORS_MATLAB

# function.set_plot_options(grid=False)
# fig, axs = plt.subplots(2, 2, figsize=(12, 9))
# # figure, ax1 = plt.subplots(figsize=(6.0, 4.8))
# # First subplot - Normalized pressure
# ax1 = axs[0, 0]
# ax1.set_xlabel("Axis position [-]", fontsize=14)
# ax1.set_ylabel("Normalized static pressure [-]", fontsize=14)
# ax1.plot(
#     solution["distance"],
#     solution["pressure"],
#     linewidth=1.00,
#     marker="o",
#     markersize=3.5,
#     markeredgewidth=1.00,
#     markerfacecolor="w",
#     label="Critical branch HEM",
# )
# # ax1.plot(
# #     impossible_solution["distance"],
# #     impossible_solution["pressure"],
# #     linewidth=1.00,
# #     marker="o",
# #     markersize=3.5,
# #     markeredgewidth=1.00,
# #     markerfacecolor="w",
# #     label="Impossible branch HEM",
# # )
# # ax1.plot(
# #     possible_solution["distance"],
# #     possible_solution["pressure"],
# #     linewidth=1.00,
# #     marker="o",
# #     markersize=3.5,
# #     markeredgewidth=1.00,
# #     markerfacecolor="w",
# #     label="Possible branch HEM",
# # )
# # ax1.plot(
# #     supersonic_solution["distance"],
# #     supersonic_solution["pressure"],
# #     linewidth=1.00,
# #     marker="o",
# #     markersize=3.5,
# #     markeredgewidth=1.00,
# #     markerfacecolor="w",
# #     label="Supersonic branch DEM",
# # )
# ax1.legend(loc="best")
# # figure.tight_layout(pad=1)

# # figure, ax2 = plt.subplots(figsize=(6.0, 4.8))
# # Second subplot - Velocity
# ax2 = axs[0, 1]
# ax2.set_xlabel("Axis position [-]", fontsize=14)
# ax2.set_ylabel("Velocity [m/s]", fontsize=14)
# ax2.plot(
#     solution["distance"],
#     solution["alpha_liquid"],
#     linewidth=1.00,
#     marker="o",
#     markersize=3.5,
#     markeredgewidth=1.00,
#     markerfacecolor="w",
#     label="Critical branch HEM",
# )
# # ax2.plot(
# #     impossible_solution["distance"],
# #     impossible_solution["velocity"],
# #     linewidth=1.00,
# #     marker="o",
# #     markersize=3.5,
# #     markeredgewidth=1.00,
# #     markerfacecolor="w",
# #     label="Impossible branch HEM",
# # )
# # ax2.plot(
# #     possible_solution["distance"],
# #     possible_solution["velocity"],
# #     linewidth=1.00,
# #     marker="o",
# #     markersize=3.5,
# #     markeredgewidth=1.00,
# #     markerfacecolor="w",
# #     label="Possible branch HEM",
# # )
# # ax2.plot(
# #     supersonic_solution["distance"],
# #     supersonic_solution["velocity"],
# #     linewidth=1.00,
# #     marker="o",
# #     markersize=3.5,
# #     markeredgewidth=1.00,
# #     markerfacecolor="w",
# #     label="Supersonic branch DEM",
# # )
# ax2.legend(loc="best")
# figure.tight_layout(pad=1)

# # # figure, ax3 = plt.subplots(figsize=(6.0, 4.8))
# # # Second subplot - Mach
# # ax3 = axs[1, 0]
# # ax3.set_xlabel("Axis position [-]", fontsize=14)
# # ax3.set_ylabel("Mach number [-]", fontsize=14)
# # ax3.plot(
# #     solution["distance"],
# #     solution["stable_fraction"],
# #     linewidth=1.00,
# #     marker="o",
# #     markersize=3.5,
# #     markeredgewidth=1.00,
# #     markerfacecolor="w",
# #     label="Critical flow",
# # )
# # # ax3.plot(
# # #     impossible_solution["distance"],
# # #     impossible_solution["mach_number"],
# # #     linewidth=1.00,
# # #     marker="o",
# # #     markersize=3.5,
# # #     markeredgewidth=1.00,
# # #     markerfacecolor="w",
# # #     label="Impossible flow",
# # # )
# # # ax3.plot(
# # #     possible_solution["distance"],
# # #     possible_solution["mach_number"],
# # #     linewidth=1.00,
# # #     marker="o",
# # #     markersize=3.5,
# # #     markeredgewidth=1.00,
# # #     markerfacecolor="w",
# # #     label="Possible flow",
# # # )
# # # ax3.plot(
# # #     supersonic_solution["distance"],
# # #     supersonic_solution["stable_fraction"],
# # #     linewidth=1.00,
# # #     marker="o",
# # #     markersize=3.5,
# # #     markeredgewidth=1.00,
# # #     markerfacecolor="w",
# # #     label="Supersonic branch",
# # # )
# # ax3.legend(loc="best")
# # figure.tight_layout(pad=1)

# # figure, ax4 = plt.subplots(figsize=(6.0, 4.8))
# # # Second subplot - Density
# # ax4 = axs[1, 1]
# # ax4.set_xlabel("Axis position [-]", fontsize=14)
# # ax4.set_ylabel("Normalized density [-]", fontsize=14)
# # ax4.plot(
# #     solution["distance"],
# #     solution["determinant"],
# #     linewidth=1.00,
# #     marker="o",
# #     markersize=3.5,
# #     markeredgewidth=1.00,
# #     markerfacecolor="w",
# #     label="Critical flow",
# # )
# # # ax4.plot(
# # #     impossible_solution["distance"],
# # #     impossible_solution["density"],
# # #     linewidth=1.00,
# # #     marker="o",
# # #     markersize=3.5,
# # #     markeredgewidth=1.00,
# # #     markerfacecolor="w",
# # #     label="Impossible flow",
# # # )
# # # ax4.plot(
# # #     possible_solution["distance"],
# # #     possible_solution["density"],
# # #     linewidth=1.00,
# # #     marker="o",
# # #     markersize=3.5,
# # #     markeredgewidth=1.00,
# # #     markerfacecolor="w",
# # #     label="Possible flow",
# # # )
# # # ax4.plot(
# # #     supersonic_solution["distance"],
# # #     supersonic_solution["determinant"],
# # #     linewidth=1.00,
# # #     marker="o",
# # #     markersize=3.5,
# # #     markeredgewidth=1.00,
# # #     markerfacecolor="w",
# # #     label="Supersonic branch",
# # # )
# # # ax4.set_ylim(-3, 6)
# # # ax4.set_xlim(0.015, 0.035)
# # # ax4.legend(loc="best")

# # figure.tight_layout(pad=1)
# # fig.tight_layout(pad=2)
# # # plt.show()
# # plt.savefig(os.path.join("results", "properties.png"))

# # # ====================================================
# # # === 5. SAVE PLOT T-s AND P-h DIAGRAMS            ===
# # # ====================================================

# # # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0), gridspec_kw={"wspace": 0.25})
# # # ax1.set_xlabel("Entropy (J/kg/K)")
# # # ax1.set_ylabel("Temperature (K)")
# # # ax2.set_xlabel("Enthalpy (J/kg)")
# # # ax2.set_ylabel("Pressure (Pa)")

# # # prop_x1, prop_y1 = "s","T"
# # # prop_x2, prop_y2 = "h", "p"

# # # fluid = bpy.Fluid(name=fluid_name, backend="HEOS")
# # # fluid.plot_phase_diagram(
# # #     prop_x1,
# # #     prop_y1,
# # #     axes=ax1,
# # #     plot_critical_point=True,
# # #     plot_quality_isolines=True,
# # #     plot_pseudocritical_line=False,
# # #     plot_spinodal_line=True,
# # # )

# # # ax1.plot(solution["entropy"], 
# # #         solution["temperature"],
# # #         linewidth=1.00,
# # #         marker="o",
# # #         markersize=3.5,
# # #         markeredgewidth=1.00,
# # #         markerfacecolor="w",
# # #         label="Convergent",
# # #         )

# # # ax1.plot(supersonic_solution["entropy"], 
# # #         supersonic_solution["temperature"],
# # #         linewidth=1.00,
# # #         marker="o",
# # #         markersize=3.5,
# # #         markeredgewidth=1.00,
# # #         markerfacecolor="w",
# # #         label="Divergent",
# # #         )
# # # ax1.legend(loc="best")

# # # fluid.plot_phase_diagram(
# # #     prop_x2,
# # #     prop_y2,
# # #     axes=ax2,
# # #     plot_critical_point=True,
# # #     plot_quality_isolines=True,
# # #     plot_pseudocritical_line=False,
# # #     plot_spinodal_line=True,
# # # )

# # # ax2.plot(solution["enthalpy"], 
# # #         solution["pressure"],
# # #         linewidth=1.00,
# # #         marker="o",
# # #         markersize=3.5,
# # #         markeredgewidth=1.00,
# # #         markerfacecolor="w",
# # #         label="Convergent",
# # #         )

# # # ax2.plot(supersonic_solution["enthalpy"], 
# # #         supersonic_solution["pressure"],
# # #         linewidth=1.00,
# # #         marker="o",
# # #         markersize=3.5,
# # #         markeredgewidth=1.00,
# # #         markerfacecolor="w",
# # #         label="Divergent",
# # #         )
# # # ax2.legend(loc="best")

# # # fig.tight_layout(pad=2)

# # # plt.savefig(os.path.join("results", "diagrams.png"))

# # # ====================================================
# # # === 6. SAVE PLOT NUMERICAL INTEGRATION ERROR     ===
# # # ====================================================

# # # The mass should always be conserved
# # # The total enthalpy is conserved if the heat transfer is zero
# # # The entropy is conserved if both heat transfer and friction are zero


# plt.show()

# %%
