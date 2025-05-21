import pandas as pd
import matplotlib.pyplot as plt
import functions as function

function.set_plot_options()
colors = function.COLORS_MATLAB

# Filenames
file1 = "nakagawa_p61_T293_IDEM1.csv"
file2 = "nakagawa_p61_T293_IDEM2.csv"

# Load CSVs
df1 = pd.read_csv(f"output/{file1}")
df2 = pd.read_csv(f"output/{file2}")

# Quantities to plot
quantities = ["pressure", "velocity", "quality", "stable_fraction"]
labels = ["Pressure (Pa)", "Velocity (m/s)", "Quality (-)", "Stable Fraction (-)"]

# Set up subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()  # Flatten 2x2 grid to 1D for easy indexing

for i, (key, label) in enumerate(zip(quantities, labels)):
    ax = axes[i]
    ax.plot(df1["distance"], df1[key], label="IDEM1", linestyle='-', linewidth = 1.5,  color = colors[2])
    ax.plot(df2["distance"], df2[key], label="IDEM2", linestyle='--', linewidth = 1.5, color = colors[6])
    # ax.set_title(label)
    ax.set_xlabel("Distance (m)", fontsize=14)
    ax.set_ylabel(label, fontsize=14, labelpad = 1)
    ax.legend(loc="best")
    ax.grid(True)

plt.tight_layout(pad=2.0)
plt.show()
