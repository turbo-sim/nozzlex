import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load pickle files
# --------------------------------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

CASE = 6
sol1 = load_pickle("hem_solutions\\solution_67.88.pkl")
sol2 = load_pickle("hnem_solutions\\solution_67.88.pkl")
# sol2 = load_pickle("hnem_solutions\\solution_60.76_richardson.pkl")
# sol2 = load_pickle("hnem_solutions\\heatTransferSensitivity\\solution_1.0_correctedRMcorrelation.pkl")

# Experimental reference values for the selected case
_cases = pd.read_excel("simulation_cases.xlsx", sheet_name="Sheet1")
_row = _cases[_cases["case"] == CASE].iloc[0]
exp_exit_velocity = float(_row["experimental_exit_velocity"])
p_out             = float(_row["p_out"])

# TFM (two-fluid model) — update filename after running diffrax_nozzle_two_fluid_jump.py
TFM_FILE = "tfm_solutions/solution_elliot_20.0bar_mr67.88_nb1.00e+10_nd1.00e+10.pkl"

try:
    sol_tfm_raw = load_pickle(TFM_FILE)
    _HAVE_TFM = True
except FileNotFoundError:
    sol_tfm_raw = None
    _HAVE_TFM = False
    print(f"TFM file not found: {TFM_FILE}  — skipping TFM curves")

def _concat_tfm(sol_tfm_raw):
    """Merge pass1/pass2/pass3 dicts into a single dict of 1-D arrays."""
    passes = [sol_tfm_raw.get("pass1"), sol_tfm_raw.get("pass2"), sol_tfm_raw.get("pass3")]
    passes = [p for p in passes if p is not None]
    keys = passes[0].keys()
    merged = {}
    for k in keys:
        arrays = []
        for p in passes:
            if k in p:
                arrays.append(np.asarray(p[k]).ravel())
        if arrays:
            merged[k] = np.concatenate(arrays)
    # mixture velocity = Ma_mix * a_mix  (v_mix is only stored in pass1)
    merged["v_mix"] = merged["Ma_mix"] * merged["a_mix"]
    return merged

if _HAVE_TFM:
    sol_tfm = _concat_tfm(sol_tfm_raw)

print(sol2)
x1 = sol1["x"]
x2 = sol2["x"]

# --------------------------------------------------
# Plot 1: Velocity
# --------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(x1, sol1["v"],  ":",  label="HEM")
plt.plot(x2, sol2["v"],  "-",  label="HNEM")
if _HAVE_TFM:
    plt.plot(sol_tfm["x"], sol_tfm["v_mix"], "--", label="TFM mixture")
plt.axhline(exp_exit_velocity, color="k", linestyle="--", linewidth=0.9, label=f"Exp. exit velocity ({exp_exit_velocity:.1f} m/s)")
plt.xlabel("x")
plt.ylabel("Velocity")
plt.title("Velocity Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()

# --------------------------------------------------
# Plot 2: Pressure
# --------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(x1, sol1["p"] * 1e-5, ":",  label="HEM")
plt.plot(x2, sol2["p"] * 1e-5, "-",  label="HNEM")
if _HAVE_TFM:
    plt.plot(sol_tfm["x"], sol_tfm["p"] * 1e-5, "--", label="TFM")
plt.axhline(p_out * 1e-5, color="k", linestyle="--", linewidth=0.9, label=f"$p_{{out}}$ ({p_out*1e-5:.2f} bar)")
plt.xlabel("x")
plt.ylabel("Pressure [bar]")
plt.title("Pressure Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()

# --------------------------------------------------
# Plot 3: Mach number
# --------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(x1, sol1["Ma"],       ":",  label="HEM")
plt.plot(x2, sol2["Ma_mix"],   "-",  label="HNEM")
if _HAVE_TFM:
    plt.plot(sol_tfm["x"], sol_tfm["Ma_mix"], "--", label="TFM")
plt.xlabel("x")
plt.ylabel("Mach Number")
plt.title("Mach Number Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
