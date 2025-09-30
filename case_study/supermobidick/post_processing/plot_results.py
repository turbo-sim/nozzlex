import os
import pandas as pd
import matplotlib.pyplot as plt
import nozzlex as nx

# Apply your plotting options
nx.set_plot_options(grid=False)

# Read simulation CSV
simulation_results = pd.read_csv("results_120a.csv")
simulation_results.columns = simulation_results.columns.str.strip().str.lower()

# Read experimental CSV
experimental_data = pd.read_csv("SMD_120a.csv")
experimental_data.columns = experimental_data.columns.str.strip().str.lower()

# Define variables to plot
variables = [
    ("pressure", "Pressure [Pa]"),
    ("quality", "Quality [-]"),
    ("gamma", "Gamma [-]"),
    ("mach_number", "Mach number [-]"),
    ("density", "Density [kg/m³]"),
    ("mass_flow", "Mass flow [kg/s]"),
]

# Create subplots (3 rows × 2 cols = 6 plots)
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
axes = axes.flatten()

colors = plt.cm.tab10.colors  # color cycle

for i, (var, ylabel) in enumerate(variables):
    ax = axes[i]

    # Plot simulation results
    if i == 0:
        sim_label = "DEM"  # Use "DEM" as legend label for first subplot
    else:
        sim_label = None  # No legend for other subplots

    ax.plot(
        simulation_results["distance"],
        simulation_results[var],
        color=colors[0],
        linewidth=2,
        label=sim_label
    )

    # Overlay experimental data on the first subplot only
    if i == 0:
        ax.plot(
            experimental_data["x"],
            experimental_data["y"]*1e5, 
            marker="s",
            linestyle="None",
            color=colors[3],
            markerfacecolor=colors[3],
            markersize=4,
            label="Experimental data"
        )

    ax.set_xlabel("Distance [m]")
    ax.set_ylabel(ylabel)

    # Legend only on the first subplot
    if i == 0:
        ax.legend()
    else:
        ax.legend().set_visible(False)

# Hide last subplot if fewer variables
if len(variables) < len(axes):
    fig.delaxes(axes[-1])

plt.tight_layout()

# Define folder to save plots
save_folder = "images"
os.makedirs(save_folder, exist_ok=True)

# Define base filename
base_filename = os.path.join(save_folder, "120a")

# Save in multiple formats
fig.savefig(base_filename + ".png", dpi=300)
fig.savefig(base_filename + ".svg")
fig.savefig(base_filename + ".eps")

plt.show()
