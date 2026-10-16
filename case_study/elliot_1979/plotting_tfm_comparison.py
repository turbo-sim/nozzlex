import pickle
import re
import jax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

from nozzlex.homogeneous_nonequilibrium import get_nozzle_elliot
from nozzlex.two_fluid import interfacial_area

# --------------------------------------------------
# Experimental case reference  (set to None to skip)
# --------------------------------------------------
CASE = 6   # row in simulation_cases.xlsx; set to None to skip experimental lines
SIGMA_WATER = 0.072  # N/m, surface tension of water

# --------------------------------------------------
# Files to compare — edit this list freely
# --------------------------------------------------
# TFM_FILES = [
#     ("tfm_solutions/solution_case5_20.0bar_mr60.76_nb1.00e+08_nd1.00e+08.pkl", "Nb=1e8, Nd=1e8"),
#     ("tfm_solutions/solution_case5_20.0bar_mr60.76_nb1.00e+09_nd1.00e+09.pkl", "Nb=1e9, Nd=1e9"),
#     ("tfm_solutions/solution_case5_20.0bar_mr60.76_nb1.00e+10_nd1.00e+10.pkl", "Nb=1e10, Nd=1e10"),
#     ("tfm_solutions/solution_case5_20.0bar_mr60.76_nb1.00e+11_nd1.00e+11.pkl", "Nb=1e11, Nd=1e11"),
# ]

# TFM_FILES = [
#     ("tfm_solutions/solution_elliot_20.0bar_mr67.88_nb1.00e+08_nd1.00e+08.pkl", "Nb=1e8, Nd=1e8"),
#     ("tfm_solutions/solution_elliot_20.0bar_mr67.88_nb1.00e+09_nd1.00e+09.pkl", "Nb=1e9, Nd=1e9"),
#     ("tfm_solutions/solution_elliot_20.0bar_mr67.88_nb1.00e+10_nd1.00e+10.pkl", "Nb=1e10, Nd=1e10"),
#     ("tfm_solutions/solution_elliot_20.0bar_mr67.88_nb1.00e+11_nd1.00e+11.pkl", "Nb=1e11, Nd=1e11"),
#     ("tfm_solutions/solution_elliot_20.0bar_mr67.88_nb1.00e+12_nd1.00e+12.pkl", "Nb=1e12, Nd=1e12"),
# ]

# TFM_FILES = [
#     # ("tfm_solutions/solution_elliot_subsonic_20.0bar_mr67.88_nb1.00e+08_nd1.00e+08.pkl", "Nb=1e8, Nd=1e8"),
#     # ("tfm_solutions/solution_elliot_subsonic_20.0bar_mr67.88_nb1.00e+09_nd1.00e+09.pkl", "Nb=1e9, Nd=1e9"),
#     # ("tfm_solutions/solution_elliot_subsonic_20.0bar_mr67.88_nb1.00e+10_nd1.00e+10.pkl", "Nb=1e10, Nd=1e10"),
#     ("tfm_solutions/solution_elliot_subsonic_20.0bar_mr67.88_nb1.00e+11_nd1.00e+11.pkl", "Nb=1e11, Nd=1e11"),
#     ("tfm_solutions/solution_elliot_subsonic_20.0bar_mr67.88_nb1.00e+12_nd1.00e+12.pkl", "Nb=1e12, Nd=1e12"),
#     ("tfm_solutions/solution_elliot_subsonic_20.0bar_mr67.88_nb1.00e+13_nd1.00e+11.pkl", "Nb=1e13, Nd=1e11"),
#     ("tfm_solutions/solution_elliot_subsonic_20.0bar_mr68_nb1.00e+13_nd1.00e+12.pkl", "Nb=1e13, Nd=1e12"),
#     ("tfm_solutions/solution_elliot_subsonic_20.0bar_mr67.88_nb1.00e+13_nd1.00e+12.pkl", "Nb=1e13, Nd=1e12"),
# ]

TFM_FILES = [

    # Nb = 1e8
    # ("tfm_solutions/solution_elliot_subsonic_20.0bar_mr68_nb1.00e+08_nd1.00e+12.pkl", "Nb=1e8, Nd=1e12"),
    
    # Nb = 1e10
    # ("tfm_solutions/solution_elliot_subsonic_20.0bar_mr68_nb1.00e+10_nd1.00e+08.pkl", "Nb=1e10, Nd=1e08"),
    # ("tfm_solutions/solution_elliot_subsonic_20.0bar_mr68_nb1.00e+10_nd1.00e+10.pkl", "Nb=1e10, Nd=1e10"),
    ("tfm_solutions/solution_elliot_subsonic_20.0bar_mr68_nb1.00e+10_nd1.00e+12.pkl", "Nb=1e10, Nd=1e12"),
    
    # Nb = 1e13
    # ("tfm_solutions/solution_elliot_subsonic_20.0bar_mr68_nb1.00e+13_nd1.00e+08.pkl", "Nb=1e13, Nd=1e08"),
    # ("tfm_solutions/solution_elliot_subsonic_20.0bar_mr68_nb1.00e+13_nd1.00e+10.pkl", "Nb=1e13, Nd=1e10"),
    ("tfm_solutions/solution_elliot_subsonic_20.0bar_mr68_nb1.00e+13_nd1.00e+12.pkl", "Nb=1e13, Nd=1e12"),
    
    # Nb = 1e14
    # ("tfm_solutions/solution_elliot_subsonic_20.0bar_mr68_nb1.00e+14_nd1.00e+11.pkl", "Nb=1e14, Nd=1e11"),
    ("tfm_solutions/solution_elliot_subsonic_20.0bar_mr68_nb1.00e+14_nd1.00e+12.pkl", "Nb=1e14, Nd=1e12"),

]

m_flow_exp = 3.657 #kg/s

x_exit = 0.311713 #m



# --------------------------------------------------
# Helpers
# --------------------------------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def parse_nd_from_label(label):
    """Extract Nd value from label string like 'Nb=1e8, Nd=1e8'."""
    m = re.search(r"Nd=([0-9.e+E]+)", label)
    return float(m.group(1)) if m else None

def parse_nb_from_label(label):
    """Extract Nd value from label string like 'Nb=1e8, Nd=1e8'."""
    m = re.search(r"Nb=([0-9.e+E]+)", label)
    return float(m.group(1)) if m else None

def concat_passes(raw):
    """Merge pass1/pass2/pass3 into a single dict of 1-D numpy arrays."""
    passes = [raw.get("pass1"), raw.get("pass2"), raw.get("pass3")]
    passes = [p for p in passes if p is not None]
    merged = {}
    for k in passes[0].keys():
        arrays = [np.asarray(p[k]).ravel() for p in passes if k in p]
        if arrays:
            merged[k] = np.concatenate(arrays)
    merged["v_mix"] = merged["Ma_mix"] * merged["a_mix"]
    return merged

# --------------------------------------------------
# Load experimental reference values
# --------------------------------------------------
exp_exit_velocity = None
p_out             = None
if CASE is not None:
    _cases = pd.read_excel("simulation_cases.xlsx", sheet_name="Sheet1")
    _row   = _cases[_cases["case"] == CASE].iloc[0]
    exp_exit_velocity = float(_row["experimental_exit_velocity"])
    p_out             = float(_row["p_out"])

# --------------------------------------------------
# Load all solutions
# --------------------------------------------------
solutions = []
for path, label in TFM_FILES:
    try:
        sol = concat_passes(load_pickle(path))
        solutions.append((label, sol))
    except FileNotFoundError:
        print(f"Not found (skipped): {path}")

if not solutions:
    raise RuntimeError("No TFM files loaded — check TFM_FILES paths.")

# --------------------------------------------------
# Mass flow rate comparison at x_exit
# --------------------------------------------------
# --- mass flow: interpolate at x_exit ---
# print(f"\n--- Mass flow rate comparison at x_exit = {x_exit} m ---")
# print(f"  Experimental: {m_flow_exp:.4f} kg/s")
# print(f"  {'Label':<28} {'m_dot [kg/s]':>14} {'err [kg/s]':>12} {'err [%]':>10}")
# print(f"  {'-'*66}")
# for label, sol in solutions:
#     m_dot_at_exit = float(np.interp(x_exit, sol["x"], sol["m_dot"]))
#     err_abs = m_dot_at_exit - m_flow_exp
#     err_rel = err_abs / m_flow_exp * 100
#     print(f"  {label:<28} {m_dot_at_exit:>14.4f} {err_abs:>+12.4f} {err_rel:>+10.2f}%")
# print()

# --- mass flow: last point of solution arrays ---
print(f"\n--- Mass flow rate comparison (outlet last point) ---")
print(f"  Experimental: {m_flow_exp:.4f} kg/s")
print(f"  {'Label':<28} {'x_out [m]':>10} {'m_dot [kg/s]':>14} {'err [kg/s]':>12} {'err [%]':>10}")
print(f"  {'-'*78}")
for label, sol in solutions:
    x_out         = float(sol["x"][-1])
    m_dot_at_exit = float(sol["m_dot"][-1])
    err_abs = m_dot_at_exit - m_flow_exp
    err_rel = err_abs / m_flow_exp * 100
    print(f"  {label:<28} {x_out:>10.6f} {m_dot_at_exit:>14.4f} {err_abs:>+12.4f} {err_rel:>+10.2f}%")
print()

# --------------------------------------------------
# Equivalent velocity  v_eq = v_mix + (p_exit - p_out)*A_exit / m_dot
# --------------------------------------------------
if p_out is not None and exp_exit_velocity is not None:
    # --- version: interpolate at x_exit ---
    # print(f"--- Equivalent velocity at x_exit = {x_exit} m  (p_out = {p_out*1e-5:.4f} bar) ---")
    # print(f"  Experimental exit velocity: {exp_exit_velocity:.4f} m/s")
    # print(f"  {'Label':<28} {'v_mix [m/s]':>12} {'p_ex [bar]':>11} {'v_eq [m/s]':>12} {'err [m/s]':>11} {'err [%]':>9}")
    # print(f"  {'-'*85}")
    # for label, sol in solutions:
    #     v_mix_exit  = float(np.interp(x_exit, sol["x"], sol["v_mix"]))
    #     p_exit      = float(np.interp(x_exit, sol["x"], sol["p"]))
    #     A_exit      = float(np.interp(x_exit, sol["x"], sol["area"]))
    #     m_dot_exit  = float(np.interp(x_exit, sol["x"], sol["m_dot"]))
    #     v_eq        = v_mix_exit + (p_exit - p_out) * A_exit / m_dot_exit
    #     err_abs     = v_eq - exp_exit_velocity
    #     err_rel     = err_abs / exp_exit_velocity * 100
    #     print(f"  {label:<28} {v_mix_exit:>12.4f} {p_exit*1e-5:>11.4f} {v_eq:>12.4f} {err_abs:>+11.4f} {err_rel:>+9.2f}%")
    # print()

    # --- version: last point of the solution arrays ---
    print(f"--- Equivalent velocity (outlet last point, p_out = {p_out*1e-5:.4f} bar) ---")
    print(f"  Experimental exit velocity: {exp_exit_velocity:.4f} m/s")
    print(f"  {'Label':<28} {'x_out [m]':>10} {'v_mix [m/s]':>12} {'p_ex [bar]':>11} {'v_eq [m/s]':>12} {'err [m/s]':>11} {'err [%]':>9}")
    print(f"  {'-'*95}")
    for label, sol in solutions:
        v_mix_exit  = float(sol["v_mix"][-1])
        p_exit      = float(sol["p"][-1])
        A_exit      = float(sol["area"][-1])
        m_dot_exit  = float(sol["m_dot"][-1])
        x_out       = float(sol["x"][-1])
        v_eq        = v_mix_exit + (p_exit - p_out) * A_exit / m_dot_exit
        err_abs     = v_eq - exp_exit_velocity
        err_rel     = err_abs / exp_exit_velocity * 100
        print(f"  {label:<28} {x_out:>10.6f} {v_mix_exit:>12.4f} {p_exit*1e-5:>11.4f} {v_eq:>12.4f} {err_abs:>+11.4f} {err_rel:>+9.2f}%")
    print()
else:
    print("  (p_out or exp_exit_velocity not available — set CASE to compute v_eq)\n")

# --------------------------------------------------
# Plot
# --------------------------------------------------
colors = plt.cm.tab10.colors

fig = plt.figure(figsize=(7, 12))
gs  = gridspec.GridSpec(5, 1)
ax0 = fig.add_subplot(gs[0])                   # mixture velocity
ax1 = fig.add_subplot(gs[1], sharex=ax0)       # v1 and v2
ax2 = fig.add_subplot(gs[2], sharex=ax0)       # volume fraction alpha1
ax3 = fig.add_subplot(gs[3], sharex=ax0)       # pressure
ax4 = fig.add_subplot(gs[4], sharex=ax0)       # Weber number

for idx, (label, sol) in enumerate(solutions):
    c = colors[idx % len(colors)]
    x = sol["x"]
    ax0.plot(x, sol["v_mix"],  color=c, label=label)
    ax1.plot(x, sol["Ma_mix"],     color=c, linestyle="-",  label=f"{label} (f1)")
    # ax1.plot(x, sol["v2"],     color=c, linestyle="--", label=f"{label} (f2)")
    ax2.plot(x, sol["alpha1"], color=c, label=label)
    ax3.plot(x, sol["p"] * 1e-5, color=c, label=label)

    nd = parse_nd_from_label(label)
    nb = parse_nb_from_label(label)
    print(f"Label:{label}")
    print(f"Nb:{nb:.2e} | Nd:{nd:.2e}")
    print(60*'-')

    if nd is not None:
        alpha_drop = np.asarray(sol["alpha1"])          # fluid1 = water droplets
        rho_gas    = np.asarray(sol["rho2"])            # fluid2 = nitrogen gas
        v_drop     = np.asarray(sol["v1"])
        v_gas      = np.asarray(sol["v2"])
        # d_drop     = (6.0 * alpha_drop / (np.pi * nd)) ** (1.0 / 3.0)
        _, d_drop     = jax.vmap(lambda a: interfacial_area(a, Nb=nb, Nd=nd))(sol["alpha2"])
        We         = rho_gas * (v_gas - v_drop) ** 2 * d_drop / SIGMA_WATER
        ax4.plot(x, We, color=c, label=label)

if exp_exit_velocity is not None:
    ax0.axhline(exp_exit_velocity, color="k", linestyle="--", linewidth=0.9,
                label=f"Exp. exit vel. ({exp_exit_velocity:.1f} m/s)")
if p_out is not None:
    ax3.axhline(p_out * 1e-5, color="k", linestyle="--", linewidth=0.9,
                label=f"$p_{{out}}$ ({p_out*1e-5:.2f} bar)")

ax0.set_ylabel("$v_{mix}$ (m/s)")
ax0.legend(fontsize=7)

ax1.set_ylabel("$Ma_{mix}$ []")
ax1.plot([], [], "k-",  label="fluid 1")
ax1.plot([], [], "k--", label="fluid 2")
ax1.legend(fontsize=7, ncol=2)

ax2.set_ylabel("Volume fraction $\\alpha_1$")
ax2.legend(fontsize=7)

ax3.set_ylabel("Pressure (bar)")
ax3.legend(fontsize=7)

ax4.set_ylabel("Weber number $We$")
ax4.set_xlabel("Axial coordinate $x$ (m)")
ax4.axhline(12.0, color="gray", linestyle=":", linewidth=0.9, label="We = 12")
ax4.legend(fontsize=7)

fig.tight_layout()
plt.show()
