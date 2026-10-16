# import pickle
# import matplotlib.pyplot as plt
# import glob
# import re
# import numpy as np
# import jaxprop as jxp

# # --------------------------------------------------
# # Load pickle files
# # --------------------------------------------------
# def load_pickle(path):
#     with open(path, "rb") as f:
#         return pickle.load(f)

# # --------------------------------------------------
# # Extract numeric ID from filename: solution_67.88.pkl → 67.88
# # --------------------------------------------------
# def extract_id(path):
#     match = re.search(r"solution_(\d+\.?\d*)\.pkl", path)
#     return float(match.group(1)) if match else None

# # --------------------------------------------------
# # Collect files
# # --------------------------------------------------
# hem_files = sorted(glob.glob("hem_solutions/solution_*.pkl"), key=extract_id)
# hnem_files = sorted(glob.glob("hnem_solutions/new/solution_new_*.pkl"), key=extract_id)

# hem_dict = {extract_id(f): f for f in hem_files}
# hnem_dict = {extract_id(f): f for f in hnem_files}

# common_ids = sorted(set(hem_dict.keys()) & set(hnem_dict.keys()))
# print("Matched simulation IDs:", common_ids)

# # --------------------------------------------------
# # Prepare colormap
# # --------------------------------------------------
# cmap = plt.get_cmap("viridis")
# colors = {sim_id: cmap(i / max(len(common_ids) - 1, 1))
#           for i, sim_id in enumerate(common_ids)}

# # --------------------------------------------------
# # Create figures
# # --------------------------------------------------
# fig_vel, ax_vel = plt.subplots(figsize=(6,4))
# fig_p, ax_p = plt.subplots(figsize=(6,4))
# fig_ma, ax_ma = plt.subplots(figsize=(6,4))
# fig_h, ax_h = plt.subplots(figsize=(6,4))

# # --------------------------------------------------
# # Plot each pair using SAME color for HEM + HNEM
# # --------------------------------------------------
# for sim_id in common_ids:

#     sol_hem = load_pickle(hem_dict[sim_id])
#     sol_hnem = load_pickle(hnem_dict[sim_id])

#     x1 = sol_hem["x"]
#     x2 = sol_hnem["x"]   

#     color = colors[sim_id]
#     label = f"{sim_id:.2f}"

#     # ---- Velocity ----
#     ax_vel.plot(x1, sol_hem["v"], "--", color=color, label=f"HEM R={label}")
#     ax_vel.plot(x2, sol_hnem["v"], "-",  color=color, label=f"HNEM")

#     # ---- Pressure ----
#     ax_p.plot(x1, sol_hem["p"] * 1e-5, "--", color=color, label=f"HEM R={label}")
#     ax_p.plot(x2, sol_hnem["p"] * 1e-5, "-",  color=color, label=f"HNEM")

#     # ---- Mach Number ----
#     ax_ma.plot(x1, sol_hem["Ma"], "--", color=color, label=f"HEM R={label}")
#     ax_ma.plot(x2, sol_hnem["Ma_mix"], "-", color=color, label=f"HNEM")

#     ax_h.plot(x2, sol_hnem["T1"], "--", color=color,) # label=f"Water")
#     ax_h.plot(x2, sol_hnem["T2"], "-", color=color,label=f"$R={label}$") # label=f"Nitrogen")

# # --------------------------------------------------
# # Formatting helper
# # --------------------------------------------------

# jxp.set_plot_options()

# ax_h.plot(0, 295, "--", color="k", label=f"Water")
# ax_h.plot(0, 295, "-", color="k", label=f"Nitrogen")

# def finalize(ax, title, ylabel):
#     ax.set_xlabel("x")
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)
#     ax.grid(True)
#     ax.legend(fontsize=7)

# finalize(ax_vel, "Velocity Distribution", "Velocity")
# finalize(ax_p, "Pressure Distribution", "Pressure [bar]")
# finalize(ax_ma, "Mach Number Distribution", "Mach Number")
# finalize(ax_h, "Temperature Distribution", "T")



# fig_vel.tight_layout()
# fig_p.tight_layout()
# fig_ma.tight_layout()
# fig_h.tight_layout()

# plt.show()


import pickle
import matplotlib.pyplot as plt
import glob
import re
import numpy as np
import jaxprop as jxp

# --------------------------------------------------
# Load pickle files
# --------------------------------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# --------------------------------------------------
# Extract numeric ID from filename: solution_67.88.pkl → 67.88
# --------------------------------------------------
def extract_id(path):
    match = re.search(r"solution_(\d+(?:\.\d+)?)\.pkl$", path)
    return float(match.group(1)) if match else None

# --------------------------------------------------
# Collect files
# --------------------------------------------------
hem_files = glob.glob("hem_solutions/solution_*.pkl")
hnem_files = glob.glob("hnem_solutions/new/solution_*.pkl")

# Filter out anything that doesn't match the pattern
hem_files = [f for f in hem_files if extract_id(f) is not None]
hnem_files = [f for f in hnem_files if extract_id(f) is not None]

hem_files = sorted(hem_files, key=extract_id)
hnem_files = sorted(hnem_files, key=extract_id)

# Build ID → file maps
hem_dict = {extract_id(f): f for f in hem_files}
hnem_dict = {extract_id(f): f for f in hnem_files}

# Find common simulation IDs
common_ids = sorted(set(hem_dict.keys()) & set(hnem_dict.keys()))

print("HEM IDs :", sorted(hem_dict.keys()))
print("HNEM IDs:", sorted(hnem_dict.keys()))
print("Matched simulation IDs:", common_ids)

# --------------------------------------------------
# Stop early if nothing matches (prevents silent failure)
# --------------------------------------------------
if not common_ids:
    raise RuntimeError("No matching HEM / HNEM simulations found. Check file paths and names.")

# --------------------------------------------------
# Prepare colormap
# --------------------------------------------------
cmap = plt.get_cmap("viridis")
colors = {
    sim_id: cmap(i / max(len(common_ids) - 1, 1))
    for i, sim_id in enumerate(common_ids)
}

# --------------------------------------------------
# Create figures
# --------------------------------------------------
fig_vel, ax_vel = plt.subplots(figsize=(6, 4))
fig_p, ax_p = plt.subplots(figsize=(6, 4))
fig_ma, ax_ma = plt.subplots(figsize=(6, 4))
fig_h, ax_h = plt.subplots(figsize=(6, 4))

# --------------------------------------------------
# Plot each pair using SAME color for HEM + HNEM
# --------------------------------------------------
for sim_id in common_ids:

    sol_hem = load_pickle(hem_dict[sim_id])
    sol_hnem = load_pickle(hnem_dict[sim_id])

    x1 = sol_hem["x"]
    x2 = sol_hnem["x"]

    color = colors[sim_id]
    label = f"{sim_id:.2f}"

    # ---- Velocity ----
    ax_vel.plot(x1, sol_hem["v"], "--", color=color, label=f"HEM R={label}")
    ax_vel.plot(x2, sol_hnem["v"], "-", color=color, label="HNEM")

    # ---- Pressure ----
    ax_p.plot(x1, sol_hem["p"] * 1e-5, "--", color=color, label=f"HEM R={label}")
    ax_p.plot(x2, sol_hnem["p"] * 1e-5, "-", color=color, label="HNEM")

    # ---- Mach Number ----
    ax_ma.plot(x1, sol_hem["Ma"], "--", color=color, label=f"HEM R={label}")
    ax_ma.plot(x2, sol_hnem["Ma_mix"], "-", color=color, label="HNEM")

    # ---- Temperature ----
    ax_h.plot(x2, sol_hnem["T1"], "--", color=color)
    ax_h.plot(x2, sol_hnem["T2"], "-", color=color, label=f"$R={label}$")

# --------------------------------------------------
# Formatting
# --------------------------------------------------
jxp.set_plot_options()

ax_h.plot(0, 295, "--", color="k", label="Water")
ax_h.plot(0, 295, "-", color="k", label="Nitrogen")

def finalize(ax, title, ylabel):
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(fontsize=7)

finalize(ax_vel, "Velocity Distribution", "Velocity")
finalize(ax_p, "Pressure Distribution", "Pressure [bar]")
finalize(ax_ma, "Mach Number Distribution", "Mach Number")
finalize(ax_h, "Temperature Distribution", "T")

fig_vel.tight_layout()
fig_p.tight_layout()
fig_ma.tight_layout()
fig_h.tight_layout()

plt.show()
