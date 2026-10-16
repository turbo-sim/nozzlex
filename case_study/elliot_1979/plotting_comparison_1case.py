import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

# --------------------------------------------------
# Load pickle files
# --------------------------------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

CASE = 6
sol1 = load_pickle("hem_solutions\\solution_60.76.pkl")
sol2 = load_pickle("hnem_solutions\\solution_60.76.pkl")
# sol2 = load_pickle("hnem_solutions\\solution_60.76_richardson.pkl")
# sol2 = load_pickle("hnem_solutions\\heatTransferSensitivity\\solution_1.0_correctedRMcorrelation.pkl")

# Experimental reference values for the selected case
_cases = pd.read_excel("simulation_cases.xlsx", sheet_name="Sheet1")
_row = _cases[_cases["case"] == CASE].iloc[0]
exp_exit_velocity = float(_row["experimental_exit_velocity"])
p_out             = float(_row["p_out"])

# TFM (two-fluid model) — update filename after running diffrax_nozzle_two_fluid_jump.py
TFM_FILE = "tfm_solutions\\solution_elliot_subsonic_20.0bar_mr67.88_nb1.00e+13_nd1.00e+12.pkl"
TFM_FILE = "tfm_solutions/solution_case1_20.0bar_mr22.45_nb1.00e+10_nd1.00e+10.pkl"
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

m_flow_exp = 3.657   # kg/s
x_exit     = 0.311713  # m

x1 = sol1["x"]
x2 = sol2["x"]

# --------------------------------------------------
# Console: mass flow comparison
# --------------------------------------------------
print(f"\n--- Mass flow rate (HEM/HNEM at x_exit={x_exit} m, TFM last point) ---")
print(f"  Experimental: {m_flow_exp:.4f} kg/s")
print(f"  {'Model':<10} {'m_dot [kg/s]':>14} {'err [kg/s]':>12} {'err [%]':>10}")
print(f"  {'-'*48}")
for name, sol, key_mdot, use_last in [
    ("HEM",  sol1,     "m_dot", False),
    ("HNEM", sol2,     "m_dot", False),
    *([("TFM", sol_tfm, "m_dot", True)] if _HAVE_TFM else []),
]:
    if key_mdot not in sol:
        print(f"  {name:<10} {'(no m_dot key)':>14}")
        continue
    m_dot = float(sol[key_mdot][-1]) if use_last else float(np.interp(x_exit, sol["x"], sol[key_mdot]))
    err   = m_dot - m_flow_exp
    print(f"  {name:<10} {m_dot:>14.4f} {err:>+12.4f} {err/m_flow_exp*100:>+10.2f}%")
print()

# --------------------------------------------------
# Console: rocket-corrected exit velocity
# --------------------------------------------------
print(f"--- Equivalent exit velocity (HEM/HNEM at x_exit={x_exit} m, TFM last point, p_out={p_out*1e-5:.4f} bar) ---")
print(f"  Experimental: {exp_exit_velocity:.4f} m/s")
print(f"  {'Model':<10} {'v [m/s]':>10} {'p_ex [bar]':>11} {'v_eq [m/s]':>12} {'err [m/s]':>11} {'err [%]':>9}")
print(f"  {'-'*65}")
for name, sol, key_v, use_last in [
    ("HEM",  sol1,     "v",     False),
    ("HNEM", sol2,     "v",     False),
    *([("TFM", sol_tfm, "v_mix", True)] if _HAVE_TFM else []),
]:
    if use_last:
        v     = float(sol[key_v][-1])
        p_ex  = float(sol["p"][-1])
        A     = float(sol["area"][-1])  if "area"  in sol else None
        m_dot = float(sol["m_dot"][-1]) if "m_dot" in sol else None
    else:
        v     = float(np.interp(x_exit, sol["x"], sol[key_v]))
        p_ex  = float(np.interp(x_exit, sol["x"], sol["p"]))
        A     = float(np.interp(x_exit, sol["x"], sol["area"]))  if "area"  in sol else None
        m_dot = float(np.interp(x_exit, sol["x"], sol["m_dot"])) if "m_dot" in sol else None
    v_eq = v + (p_ex - p_out) * A / m_dot if (A is not None and m_dot is not None) else v
    err  = v_eq - exp_exit_velocity
    print(f"  {name:<10} {v:>10.4f} {p_ex*1e-5:>11.4f} {v_eq:>12.4f} {err:>+11.4f} {err/exp_exit_velocity*100:>+9.2f}%")
print()

# --------------------------------------------------
# Plot
# --------------------------------------------------
C_HEM  = "#4878CF"
C_HNEM = "#6ACC65"
C_TFM  = "#D65F5F"
C_EXP  = "black"
LW = 1.5

fig = plt.figure(figsize=(7, 9))
gs  = gridspec.GridSpec(3, 1, hspace=0.08)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1], sharex=ax0)
ax2 = fig.add_subplot(gs[2], sharex=ax0)

# --- velocity ---
ax0.plot(x1, sol1["v"],       color=C_HEM,  lw=LW, ls=":",  label="HEM")
ax0.plot(x2, sol2["v"],       color=C_HNEM, lw=LW, ls="-",  label="HNEM")
if _HAVE_TFM:
    ax0.plot(sol_tfm["x"], sol_tfm["v_mix"], color=C_TFM, lw=LW, ls="--", label="TFM")
ax0.axhline(exp_exit_velocity, color=C_EXP, lw=0.9, ls="--",
            label=f"Exp. ({exp_exit_velocity:.1f} m/s)")
ax0.set_ylabel("Velocity (m/s)")
ax0.legend(fontsize=8, framealpha=0.9)
ax0.grid(True, lw=0.4, alpha=0.5)
plt.setp(ax0.get_xticklabels(), visible=False)

# --- pressure ---
ax1.plot(x1, sol1["p"] * 1e-5,       color=C_HEM,  lw=LW, ls=":")
ax1.plot(x2, sol2["p"] * 1e-5,       color=C_HNEM, lw=LW, ls="-")
if _HAVE_TFM:
    ax1.plot(sol_tfm["x"], sol_tfm["p"] * 1e-5, color=C_TFM, lw=LW, ls="--")
ax1.axhline(p_out * 1e-5, color=C_EXP, lw=0.9, ls="--",
            label=f"$p_{{out}}$ ({p_out*1e-5:.2f} bar)")
ax1.set_ylabel("Pressure (bar)")
ax1.legend(fontsize=8, framealpha=0.9)
ax1.grid(True, lw=0.4, alpha=0.5)
plt.setp(ax1.get_xticklabels(), visible=False)

# --- Mach number ---
ax2.plot(x1, sol1["Ma"],       color=C_HEM,  lw=LW, ls=":")
ax2.plot(x2, sol2["Ma_mix"],   color=C_HNEM, lw=LW, ls="-")
if _HAVE_TFM:
    ax2.plot(sol_tfm["x"], sol_tfm["Ma_mix"], color=C_TFM, lw=LW, ls="--")
ax2.axhline(1.0, color="gray", lw=0.8, ls=":", label="Ma = 1")
ax2.set_ylabel("Mach number")
ax2.set_xlabel("Axial coordinate $x$ (m)")
ax2.legend(fontsize=8, framealpha=0.9)
ax2.grid(True, lw=0.4, alpha=0.5)

fig.suptitle(f"Elliot case {CASE} — HEM / HNEM / TFM comparison", fontsize=10, y=1.01)
fig.tight_layout()
plt.show()
