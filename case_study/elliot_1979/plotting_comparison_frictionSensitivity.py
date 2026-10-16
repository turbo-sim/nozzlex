# import pickle
# import matplotlib.pyplot as plt
# import glob
# import re

# # --------------------------------------------------
# # Load pickle files
# # --------------------------------------------------
# def load_pickle(path):
#     with open(path, "rb") as f:
#         return pickle.load(f)

# # --------------------------------------------------
# # Extract parameter from filename:
# # solution_76.46_0.95.pkl → 0.95
# # --------------------------------------------------
# def extract_param(path):
#     match = re.search(r"solution_\d+\.\d+_\d+\.pkl", path)
#     return float(match.group(1)) if match else None

# # --------------------------------------------------
# # Gather files
# # --------------------------------------------------
# all_files = glob.glob("hem_solutions/frictionSensitivity/solution_*.pkl")

# # Filter out files that fail parsing
# valid_files = []
# params = []

# for f in all_files:
#     p = extract_param(f)
#     if p is None:
#         print(f"⚠️ Warning: Skipping file (pattern not matched): {f}")
#         continue
#     valid_files.append(f)
#     params.append(p)

# # Sort files by extracted parameter
# params, valid_files = zip(*sorted(zip(params, valid_files)))

# print("Found parameters:", params)

# # --------------------------------------------------
# # Prepare colormap
# # --------------------------------------------------
# cmap = plt.get_cmap("viridis")
# colors = {p: cmap(i / max(len(params)-1, 1)) for i, p in enumerate(params)}

# # --------------------------------------------------
# # Create figures
# # --------------------------------------------------
# fig_vel, ax_vel = plt.subplots(figsize=(6,4))
# fig_p, ax_p = plt.subplots(figsize=(6,4))
# fig_ma, ax_ma = plt.subplots(figsize=(6,4))

# # --------------------------------------------------
# # Plot each file using solid line and unique color
# # --------------------------------------------------
# for p, fname in zip(params, valid_files):

#     sol = load_pickle(fname)
#     x = sol["x"]
#     color = colors[p]
#     label = f"{p:.2f}"

#     ax_vel.plot(x, sol["v"], "-", color=color, label=label)
#     ax_p.plot(x, sol["p"] * 1e-5, "-", color=color, label=label)
#     ax_ma.plot(x, sol["Ma_mix"], "-", color=color, label=label)

# # --------------------------------------------------
# # Formatting helper
# # --------------------------------------------------
# def finalize(ax, title, ylabel):
#     ax.set_xlabel("x")
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)
#     ax.grid(True)
#     ax.legend(fontsize=7, title="Parameter")

# finalize(ax_vel, "Velocity Distribution", "Velocity")
# finalize(ax_p, "Pressure Distribution", "Pressure [bar]")
# finalize(ax_ma, "Mach Number Distribution", "Mach Number")

# fig_vel.tight_layout()
# fig_p.tight_layout()
# fig_ma.tight_layout()

# plt.show()


import pickle
import matplotlib.pyplot as plt
import glob
import re

# --------------------------------------------------
# Load pickle files
# --------------------------------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# --------------------------------------------------
# Extract parameter from filename:
# solution_76.46_0.pkl → 0
# solution_76.46_0.95.pkl → 0.95
# --------------------------------------------------
def extract_param(path):
    match = re.search(r"solution_([0-9\.]+)_([0-9\.]+)\.pkl", path)
    if not match:
        return None
    return float(match.group(2))      # <-- parameter is the SECOND number

# --------------------------------------------------
# Gather files
# --------------------------------------------------
all_files = glob.glob("hem_solutions/frictionSensitivity/solution_*.pkl")

valid_files = []
params = []

for f in all_files:
    p = extract_param(f)
    if p is None:
        print(f"⚠️ Warning: Skipping file (pattern not matched): {f}")
        continue
    valid_files.append(f)
    params.append(p)

# Sort by parameter
params, valid_files = zip(*sorted(zip(params, valid_files)))

print("Found parameters:", params)

# --------------------------------------------------
# Prepare colormap
# --------------------------------------------------
cmap = plt.get_cmap("viridis")
colors = {p: cmap(i / max(len(params)+1, 1)) for i, p in enumerate(params)}

# --------------------------------------------------
# Create figures
# --------------------------------------------------
fig_vel, ax_vel = plt.subplots(figsize=(6,4))
fig_p, ax_p = plt.subplots(figsize=(6,4))
fig_ma, ax_ma = plt.subplots(figsize=(6,4))

# --------------------------------------------------
# Plot each file using solid line and unique color
# --------------------------------------------------
for p, fname in zip(params, valid_files):

    sol = load_pickle(fname)
    x = sol["x"]
    color = colors[p]
    label = f"{p:g}"

    ax_vel.plot(x, sol["v"], "-", color=color, label=label)
    ax_p.plot(x, sol["p"] * 1e-5, "-", color=color, label=label)
    ax_ma.plot(x, sol["Ma"], "-", color=color, label=label)

# --------------------------------------------------
# Formatting helper
# --------------------------------------------------
def finalize(ax, title, ylabel):
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(fontsize=7, title="Parameter")

finalize(ax_vel, "Velocity Distribution", "Velocity")
finalize(ax_p, "Pressure Distribution", "Pressure [bar]")
finalize(ax_ma, "Mach Number Distribution", "Mach Number")

fig_vel.tight_layout()
fig_p.tight_layout()
fig_ma.tight_layout()

plt.show()
