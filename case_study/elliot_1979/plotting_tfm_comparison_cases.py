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
# Configuration: (case_number, pkl_path, label)
# --------------------------------------------------
SIGMA_WATER = 0.072   # N/m
M_FLOW_EXP_CASE6 = 3.657   # kg/s — experimental mass flow available only for case 6

CASES_FILES = [
    (1, "tfm_solutions/solution_elliot_subsonic_20.0bar_mr22.45_nb1.00e+13_nd1.00e+11.pkl", "Case 1 (mr=22.45)"),
    (6, "tfm_solutions/solution_elliot_subsonic_20.0bar_mr67.88_nb1.00e+13_nd1.00e+11.pkl", "Case 6 (mr=67.88)"),
    (8, "tfm_solutions/solution_elliot_subsonic_20.0bar_mr94.48_nb1.00e+13_nd1.00e+11.pkl", "Case 8 (mr=94.48)"),
]

# CASES_FILES = [
#     # (1, "tfm_solutions/solution_elliot_subsonic_20.0bar_mr22.45_nb1.00e+13_nd1.00e+11.pkl", "Case 1 (mr=22.45)"),
#     (6, "tfm_solutions/solution_elliot_subsonic_transport_20.0bar_mr67.88_nb1.00e+13_nd1.00e+10.pkl", "Case 6 (mr=67.88)"),
#     (6, "tfm_solutions/solution_elliot_subsonic_transport_20.0bar_mr67.88_we_b6_we_d6_nb1.00e+13_nd1.00e+13.pkl", "Case 6, we lower (mr=67.88)"),
#     (6, "tfm_solutions/solution_elliot_subsonic_transport_low_timescale_20.0bar_mr67.88_we_b6_we_d6_nb1.00e+13_nd1.00e+13.pkl", "Case 6, timescale (mr=67.88)"),
#     # (8, "tfm_solutions/solution_elliot_subsonic_20.0bar_mr94.48_nb1.00e+13_nd1.00e+11.pkl", "Case 8 (mr=94.48)"),
# ]

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def parse_nd_from_path(path):
    m = re.search(r"_nd([^_]+)\.pkl", path)
    return float(m.group(1)) if m else None

def parse_nb_from_path(path):
    m = re.search(r"_nb([^_]+)_", path)
    return float(m.group(1)) if m else None

def concat_passes(raw):
    passes = [raw.get("pass1"), raw.get("pass2"), raw.get("pass3")]
    passes = [p for p in passes if p is not None]
    merged = {}
    for k in passes[0].keys():
        arrays = [np.asarray(p[k]).ravel() for p in passes if k in p]
        if arrays:
            merged[k] = np.concatenate(arrays)
    return merged

# --------------------------------------------------
# Load experimental reference values per case
# --------------------------------------------------
_cases_df = pd.read_excel("simulation_cases.xlsx", sheet_name="Sheet1")

def get_exp(case_num):
    row = _cases_df[_cases_df["case"] == case_num].iloc[0]
    return float(row["experimental_exit_velocity"]), float(row["p_out"])

# --------------------------------------------------
# Load solutions
# --------------------------------------------------
solutions = []   # list of (case_num, path, label, sol, exp_v, p_out)
for case_num, path, label in CASES_FILES:
    try:
        sol = concat_passes(load_pickle(path))
        exp_v, p_out = get_exp(case_num)
        solutions.append((case_num, path, label, sol, exp_v, p_out))
    except FileNotFoundError:
        print(f"Not found (skipped): {path}")

if not solutions:
    raise RuntimeError("No TFM files loaded — check CASES_FILES paths.")

# --------------------------------------------------
# Print: mass flow comparison (case 6 only)
# --------------------------------------------------
print(f"\n--- Mass flow rate (outlet last point) — case 6 only ---")
print(f"  Experimental: {M_FLOW_EXP_CASE6:.4f} kg/s")
for case_num, path, label, sol, exp_v, p_out in solutions:
    if case_num == 6:
        m_dot = float(sol["m_dot"][-1])
        err   = m_dot - M_FLOW_EXP_CASE6
        print(f"  {label:<28}  m_dot = {m_dot:.4f} kg/s  err = {err:+.4f} kg/s ({err/M_FLOW_EXP_CASE6*100:+.2f}%)")
print()

# --------------------------------------------------
# Print: equivalent velocity comparison (all cases)
# --------------------------------------------------
print(f"--- Equivalent exit velocity (outlet last point) ---")
print(f"  {'Case':<28} {'v_exp [m/s]':>12} {'v_mix [m/s]':>12} {'p_ex [bar]':>11} {'v_eq [m/s]':>12} {'err [m/s]':>11} {'err [%]':>9}")
print(f"  {'-'*95}")
for case_num, path, label, sol, exp_v, p_out in solutions:
    v_mix  = float(sol["v_mix"][-1])
    p_exit = float(sol["p"][-1])
    A_exit = float(sol["area"][-1])
    m_dot  = float(sol["m_dot"][-1])
    v_eq   = v_mix + (p_exit - p_out) * A_exit / m_dot
    err    = v_eq - exp_v
    print(f"  {label:<28} {exp_v:>12.4f} {v_mix:>12.4f} {p_exit*1e-5:>11.4f} {v_eq:>12.4f} {err:>+11.4f} {err/exp_v*100:>+9.2f}%")
print()

# --------------------------------------------------
# Plot
# --------------------------------------------------
colors = plt.cm.tab10.colors

fig = plt.figure(figsize=(7, 10))
gs  = gridspec.GridSpec(4, 1)
ax0 = fig.add_subplot(gs[0])                   # mixture velocity
ax1 = fig.add_subplot(gs[1], sharex=ax0)       # volume fraction alpha1
ax2 = fig.add_subplot(gs[2], sharex=ax0)       # pressure
ax3 = fig.add_subplot(gs[3], sharex=ax0)       # Weber number

for idx, (case_num, path, label, sol, exp_v, p_out) in enumerate(solutions):
    c = colors[idx % len(colors)]
    x = sol["x"]

    ax0.plot(x, sol["v_mix"],      color=c, label=label)
    ax0.axhline(exp_v, color=c, linestyle="--", linewidth=0.9,
                label=f"Exp. {label.split()[0]+label.split()[1]} ({exp_v:.1f} m/s)")

    ax1.plot(x, sol["alpha1"], color=c, label=label)
    ax2.plot(x, sol["p"] * 1e-5, color=c, label=label)
    ax2.axhline(p_out * 1e-5, color=c, linestyle=":", linewidth=0.9)

    print(f"Label:{label}")
    print(f"Path:{path}")
    print(60*'-')

    try:
        d_drop = np.asarray(sol["d_droplet"])
        d_bubb = np.asarray(sol["d_bubble"])
        rho_gas = np.asarray(sol["rho2"])
        v_drop  = np.asarray(sol["v1"])
        v_gas   = np.asarray(sol["v2"])
        u_rel   = v_gas - v_drop
        We_drop = rho_gas * u_rel**2 * d_drop / SIGMA_WATER
        rho_liq = np.asarray(sol["rho1"])
        We_bubb = rho_liq * u_rel**2 * d_bubb / SIGMA_WATER
        ax3.plot(x, We_drop, color=c,            label=f"{label} (drop)")
        ax3.plot(x, We_bubb, color=c, linestyle="--", label=f"{label} (bubb)")
    except KeyError:
        nd = parse_nd_from_path(path)
        nb = parse_nb_from_path(path)
        if nd is not None and nb is not None:
            rho_gas = np.asarray(sol["rho2"])
            v_drop  = np.asarray(sol["v1"])
            v_gas   = np.asarray(sol["v2"])
            _, d_drop = jax.vmap(lambda a: interfacial_area(a, Nb=nb, Nd=nd))(sol["alpha2"])
            We = rho_gas * (v_gas - v_drop) ** 2 * d_drop / SIGMA_WATER
            ax3.plot(x, We, color=c, label=label)

ax0.set_ylabel("$v_{mix}$ (m/s)")
ax0.legend(fontsize=7)

ax1.set_ylabel("Volume fraction $\\alpha_1$")
ax1.legend(fontsize=7)

ax2.set_ylabel("Pressure (bar)")
ax2.legend(fontsize=7)

ax3.set_ylabel("Weber number $We$")
ax3.set_xlabel("Axial coordinate $x$ (m)")
ax3.axhline(12.0, color="gray", linestyle=":", linewidth=0.9, label="We = 12")
ax3.legend(fontsize=7)

fig.tight_layout()
plt.show()
