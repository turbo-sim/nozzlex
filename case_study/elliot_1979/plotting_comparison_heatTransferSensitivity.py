import argparse
import os
import pickle
import matplotlib
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
    # Support filenames like solution_1.0.pkl or solution_76.46_0.95.pkl
    name = path.split("/")[-1]
    nums = re.findall(r"([0-9]+(?:\.[0-9]+)?)", name)
    if not nums:
        return None
    # return the last numeric group (most likely the parameter)
    return float(nums[-1])

# --------------------------------------------------
# Gather files
# --------------------------------------------------
all_files = glob.glob("hnem_solutions/heatTransferSensitivity/solution_*.pkl")

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
if len(params) > 1:
    colors = {p: cmap(i / (len(params)-1)) for i, p in enumerate(params)}
else:
    colors = {p: cmap(0.0) for p in params}

# --------------------------------------------------
# Create figures
# --------------------------------------------------
fig_vel, ax_vel = plt.subplots(figsize=(6,4))
fig_p, ax_p = plt.subplots(figsize=(6,4))
fig_ma, ax_ma = plt.subplots(figsize=(6,4))
fig_T, ax_T = plt.subplots(figsize=(6,4))
fig_s, ax_s = plt.subplots(figsize=(6,4))

# --------------------------------------------------
# Plot each file using solid line and unique color
# --------------------------------------------------
for p, fname in zip(params, valid_files):

    sol = load_pickle(fname)
    x = sol["x"]
    color = colors[p]
    label = f"{p:.2f}"

    ax_vel.plot(x, sol["v"], "-", color=color, label=label)
    ax_p.plot(x, sol["p"] * 1e-5, "-", color=color, label=label)
    # heat transfer sensitivity pickles contain 'Ma_mix'
    ma_key = "Ma_mix" if "Ma_mix" in sol else ("Ma" if "Ma" in sol else None)
    if ma_key is not None:
        ax_ma.plot(x, sol[ma_key], "-", color=color, label=label)
    else:
        # fallback: compute Ma from speed_of_sound and v if available
        if "speed_of_sound" in sol and "v" in sol:
            ax_ma.plot(x, (sol["v"]/sol["speed_of_sound"]), "-", color=color, label=label)

    # Plot temperatures T1 and T2 (overlay in same axes)
    if "T1" in sol or "T2" in sol:
        if "T1" in sol:
            ax_T.plot(x, sol["T1"], "-", color=color, label=f"{label} T1")
        if "T2" in sol:
            ax_T.plot(x, sol["T2"], "--", color=color, label=f"{label} T2")

    # Plot entropies s1 and s2 (overlay)
    if "s1" in sol or "s2" in sol:
        if "s1" in sol:
            ax_s.plot(x, sol["s1"]/sol["s1"][0], "-", color=color, label=f"{label} s1")
        if "s2" in sol:
            ax_s.plot(x, sol["s2"]/sol["s2"][0], "--", color=color, label=f"{label} s2")

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
finalize(ax_T, "Temperature Distribution", "Temperature [K]")
finalize(ax_s, "Entropy Distribution", "Entropy [J/kg/K]")

fig_vel.tight_layout()
fig_p.tight_layout()
fig_ma.tight_layout()

# plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot heat transfer sensitivity pickles")
    parser.add_argument("--save", action="store_true", help="Save figures to disk instead of showing")
    args = parser.parse_args()

    # If running headless (Agg) or --save requested, write figures to files
    backend = matplotlib.get_backend()
    out_dir = os.path.join("figures", "heatTransferSensitivity")
    if args.save or backend.lower().startswith("agg"):
        os.makedirs(out_dir, exist_ok=True)
        fig_vel.savefig(os.path.join(out_dir, "velocity.png"), dpi=200)
        fig_p.savefig(os.path.join(out_dir, "pressure.png"), dpi=200)
        fig_ma.savefig(os.path.join(out_dir, "mach.png"), dpi=200)
        fig_T.savefig(os.path.join(out_dir, "temperature.png"), dpi=200)
        fig_s.savefig(os.path.join(out_dir, "entropy.png"), dpi=200)
        print("Saved figures to", out_dir)
    else:
        plt.show()
