"""
Run the SPACE-MARCHING solvers from the TIME-MARCHING inlet state, then compare.

Motivation
----------
`compare_tfm_vs_hyperbolic_vs_timemarching.py` overlays the three solutions as
they stand, but the two families were not started from the same inlet:

    space marching (as set up in the demos)   time marching (FVM yaml)
      alpha_v,in = 1e-6                         alpha_v,in ~ 1e-3
      u_in       = 8.05 m/s                     u_in       ~ 7.31 m/s
      p_in       = 120.00 bar (stagnation)      p_in       ~ 119.81 bar (static)

So part of the disagreement downstream is just a different starting point.  This
script removes that ambiguity: it reads the inlet state straight out of the
time-marching CSV, feeds it to BOTH space-marching formulations (TFM 9-variable
and hyperbolic 6-variable), runs them, and produces the same five-panel
comparison figure.

The vapor phase is consistent by construction: the space-marching solvers build
the inlet vapor from `get_state_saturation(p_in)`, and the FVM ghost vapor is
pinned to saturation too (`pin_ghost: true`).  The script prints a side-by-side
check of the reconstructed inlet state against the CSV so you can see this.

Everything you are likely to change lives in the CONFIG block below.
"""

import os
import sys
import time
import dataclasses

import numpy as np
import pandas as pd
import jax
import equinox as eqx

jax.config.update("jax_enable_x64", True)

import jaxprop as jxp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nozzlex.two_fluid.nozzle_model_solver import replace_param
from nozzlex.homogeneous_nonequilibrium.nozzle_model_core import (
    get_nozzle_smd_divergent,
    get_nozzle_smd_divergent_smooth,
)

# Reuse the case builder (fluid tables + solver settings) and the plotting code
# instead of duplicating them.
from compare_tfm_vs_hyperbolic_single_solution import build_case, run_and_postprocess
import diffrax_nozzle_tfm_sinlge_solution as tfm_mod
import diffrax_nozzle_hyperbolic_single_solution as hyp_mod
from compare_tfm_vs_hyperbolic_vs_timemarching import load_all, make_comparison_figure

# =============================================================================
# CONFIG -- edit here
# =============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

CASE = "a"  # only selects the fluid tables / defaults from build_case()

# --- Which time-marching run to compare against -------------------------------
# The FVM exports are `solution.csv` inside each nozzle-hyperbolic run directory;
# copy one here and add an entry.  `length` MUST equal that run's grid.x2.
TIME_MARCHING_RUNS = {
    # N250, x2 = 0.6 m, p_out = 117.69 bar, nc_form=cell, CFL=5, ic=stagnation.
    # Weak case: alpha_v only creeps 1.0e-3 -> 1.3e-3, no real flashing.
    # (run smd_supersonic_N250_muscl_van_albada_20260806_103328)
    "short": dict(csv=os.path.join(HERE, "solution_time_marching.csv"), length=0.6),
    # N600, x2 = 1.2 m, p_out = 80 bar, nc_form=wb, CFL=2, ic=restart.
    # Real flashing: alpha_v 1.0e-3 -> 0.69 at x~0.6, then recompression.
    # (run smd_supersonic_N600_muscl_van_albada_20260725_173528)
    "long": dict(csv=os.path.join(HERE, "solution_time_marching_long.csv"), length=1.2),
}
RUN = "long"

TIME_CSV = TIME_MARCHING_RUNS[RUN]["csv"]
INLET_ROW = 0  # which row of TIME_CSV to take the inlet state from

# Column names in TIME_CSV (change here if the FVM export ever renames them)
COL_X, COL_P = "x", "p"
COL_ALPHA_G = "alpha_g"
COL_UG, COL_UL = "u_g", "u_l"
COL_TG, COL_TL = "T_g", "T_l"
COL_HG, COL_HL = "h_g", "h_l"

# --- Which inlet quantities to actually take from the CSV ---------------------
# Set to False to keep the demo value from build_case() for that quantity --
# handy for isolating which inlet difference drives the downstream disagreement.
USE_CSV_PRESSURE = True
USE_CSV_TEMPERATURE = True
USE_CSV_VOID_FRACTION = True
USE_CSV_VELOCITY = True

# --- Closure: build_case() and the FVM yaml DISAGREE on these -----------------
#                      nozzlex demo / build_case      FVM yaml
#   bubble_number        1e12                          1e11
#   droplet_number       1e10                          1e10
#   heat_transfer        0.0  (interphase HTC OFF)     always ON, no switch
# Matched to the FVM yaml below.  Set to None to keep the demo value.
BUBBLE_NUMBER = 1.0e11
DROPLET_NUMBER = 1.0e10
# Residual mismatch you cannot close from here: the FVM closure ALWAYS applies
# an interphase HTC (yaml caveat 2), while nozzlex multiplies its own HTC by
# this factor -- the demo runs it at 0.0.  1.0 is "closer" but not the same
# correlation, so the demo value is kept as the default.
HEAT_TRANSFER = 0.0

# --- Geometry / domain --------------------------------------------------------
# The FVM yamls use `get_nozzle_smd_divergent_smooth`, so match it here (the two
# variants differ only over a 2 cm C1 blend just after the throat).
# NOTE: both variants extrapolate the LINEAR divergent past its design end at
# z_div_end = 0.9001 m, so a 1.2 m domain runs out to A/A* ~ 30 -- beyond the
# physical nozzle.  The FVM does the same, so the two stay consistent.
GEOMETRY = get_nozzle_smd_divergent_smooth  # or get_nozzle_smd_divergent
LENGTH = TIME_MARCHING_RUNS[RUN]["length"]  # m -- must match grid.x2 of that run

# --- Solver budget ------------------------------------------------------------
NSTEPS = 100_000
N_OUTPUT_POINTS = 1000

# --- Output -------------------------------------------------------------------
TAG = f"fvminlet_{RUN}"  # keeps these results apart from the demo-inlet ones
TFM_OUT_CSV = os.path.join(RESULTS_DIR, f"smd_tfm_case_{CASE}_{TAG}.csv")
HYP_OUT_CSV = os.path.join(RESULTS_DIR, f"smd_hyperbolic_case_{CASE}_{TAG}.csv")
FIG_PATH = os.path.join(RESULTS_DIR, f"compare_smd_three_way_case_{CASE}_{TAG}.png")
SHOW = True

# From the FVM inlet the vapor barely grows (alpha_v ~ 1e-3 throughout), so the
# flat alpha_l = 1 curve would hide all of the structure -> plot the vapor only.
PLOT_ALPHA_LIQUID = False
ALPHA_YSCALE = "linear"  # "log" if you switch to a case that really flashes

# --- Curve styling for the comparison figure ----------------------------------
# Same schema as compare_tfm_vs_hyperbolic_vs_timemarching.py: colour =
# formulation, dashes = liquid phase, symbols = the time-marching reference.
SPACE_MARCH_COLS = dict(
    x="x", p="p", v_g="v1", v_l="v2", T_g="T1", T_l="T2",
    alpha_g="alpha1", alpha_l="alpha2", D="D",
)
TIME_MARCH_COLS = dict(
    x=COL_X, p=COL_P, v_g=COL_UG, v_l=COL_UL, T_g=COL_TG, T_l=COL_TL,
    alpha_g=COL_ALPHA_G,
    alpha_l=lambda df: 1.0 - df[COL_ALPHA_G].to_numpy(),
    D=None,  # det(A) is a space-marching diagnostic only
)


def solution_specs():
    return [
        dict(key="tfm", label="TFM (9-var, FVM inlet)", path=TFM_OUT_CSV,
             color="tab:blue", linestyle="-", cols=SPACE_MARCH_COLS),
        dict(key="hyp", label="Hyperbolic (6-var, FVM inlet)", path=HYP_OUT_CSV,
             color="tab:orange", linestyle="-", cols=SPACE_MARCH_COLS),
        dict(key="time", label="Time marching (FVM)", path=TIME_CSV,
             color="black", linestyle="none", marker="o", markevery=8,
             markersize=3.0, cols=TIME_MARCH_COLS),
    ]


# =============================================================================
# Build the inlet-matched parameters
# =============================================================================
def params_from_timemarching_inlet():
    """build_case() parameters with the inlet overwritten by the CSV row."""
    df = pd.read_csv(TIME_CSV)
    row = df.iloc[INLET_ROW]

    params_model, params_bvp, params_ivp = build_case(case=CASE)

    # Geometry / domain first -- the demo defaults may differ from the FVM run.
    params_model = replace_param(params_model, "length", LENGTH)
    # `geometry` is a STATIC eqx field, so it is not a pytree leaf: swap it with
    # dataclasses.replace rather than replace_param/tree_at.
    params_model = dataclasses.replace(params_model, geometry=GEOMETRY)

    # Closure knobs (see the CONFIG block for why these differ from build_case).
    if BUBBLE_NUMBER is not None:
        params_model = replace_param(params_model, "bubble_number", BUBBLE_NUMBER)
    if DROPLET_NUMBER is not None:
        params_model = replace_param(params_model, "droplet_number", DROPLET_NUMBER)
    if HEAT_TRANSFER is not None:
        params_model = replace_param(params_model, "heat_transfer", HEAT_TRANSFER)

    if USE_CSV_PRESSURE:
        # The solvers use p0_in directly as the inlet STATIC pressure (the inlet
        # velocity is a few m/s, so the distinction is ~0.03 % anyway).
        params_model = replace_param(params_model, "p0_in", float(row[COL_P]))
    if USE_CSV_TEMPERATURE:
        # T02_in fixes the liquid via (p, T); the vapor comes from saturation at
        # p_in, so T01_in does not enter the inlet state -- set for bookkeeping.
        params_model = replace_param(params_model, "T02_in", float(row[COL_TL]))
        params_model = replace_param(params_model, "T01_in", float(row[COL_TG]))
    if USE_CSV_VOID_FRACTION:
        alpha_v = float(row[COL_ALPHA_G])
        params_model = replace_param(params_model, "alphav_in", alpha_v)
        params_model = replace_param(params_model, "alphal_in", 1.0 - alpha_v)
    if USE_CSV_VELOCITY:
        params_model = replace_param(params_model, "u1_in", float(row[COL_UG]))
        params_model = replace_param(params_model, "u2_in", float(row[COL_UL]))

    return params_model, params_bvp, params_ivp, row


def report_inlet(params_model, row):
    """Print the CSV inlet next to the state the space-march actually builds."""
    fluid_vapor, fluid_liquid = params_model.fluid1, params_model.fluid2
    p_in = params_model.p0_in
    st_v = fluid_vapor.get_state_saturation(p_in)
    st_l = fluid_liquid.get_state(jxp.PT_INPUTS, p_in, params_model.T02_in)

    rows = [
        ("p        [bar]", float(row[COL_P]) * 1e-5, float(p_in) * 1e-5),
        ("alpha_v  [-]  ", float(row[COL_ALPHA_G]), float(params_model.alphav_in)),
        ("u_vapor  [m/s]", float(row[COL_UG]), float(params_model.u1_in)),
        ("u_liquid [m/s]", float(row[COL_UL]), float(params_model.u2_in)),
        ("T_vapor  [K]  ", float(row[COL_TG]), float(st_v["temperature"])),
        ("T_liquid [K]  ", float(row[COL_TL]), float(st_l["temperature"])),
        ("h_vapor  [kJ/kg]", float(row[COL_HG]) * 1e-3, float(st_v["enthalpy"]) * 1e-3),
        ("h_liquid [kJ/kg]", float(row[COL_HL]) * 1e-3, float(st_l["enthalpy"]) * 1e-3),
    ]
    print("\nInlet state (row {} of the time-marching CSV)".format(INLET_ROW))
    print("-" * 70)
    print(f"  {'quantity':<17s}{'time marching':>16s}{'space marching':>17s}{'rel. diff':>12s}")
    for name, a, b in rows:
        rel = abs(b - a) / max(abs(a), 1e-30)
        print(f"  {name:<17s}{a:>16.6g}{b:>17.6g}{rel:>12.2e}")


def march(label, solve_fn, ode_full_fn, params_model, params_bvp, params_ivp, out_csv):
    print("\n" + "-" * 70)
    print(f"Solving {label} from the time-marching inlet...")
    print("-" * 70)
    t0 = time.perf_counter()
    sol, out = run_and_postprocess(
        solve_fn, ode_full_fn, params_model, params_bvp, params_ivp,
        n_points=N_OUTPUT_POINTS,
    )
    x_end = float(sol.ts[0])
    L = float(params_model.length)
    print(
        f"Done in {(time.perf_counter() - t0):.1f} s | status: {sol.result} | "
        f"steps: {int(sol.stats['num_steps'])} | "
        f"x_end: {x_end:.5f} / {L:.5f} m ({100 * x_end / L:.1f} %)"
    )
    if x_end < 0.99 * L:
        print(
            f"*** {label} did NOT reach the outlet: it stopped at x = {x_end:.5f} m "
            f"({sol.result}). Its curve in the figure is truncated -- not a plotting "
            f"artifact. ***"
        )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame({k: np.asarray(v) for k, v in out.items()}).to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")
    return sol, out


if __name__ == "__main__":
    tfm_mod.nsteps = NSTEPS
    hyp_mod.nsteps = NSTEPS

    params_model, params_bvp, params_ivp, row = params_from_timemarching_inlet()
    report_inlet(params_model, row)

    march("TFM (9-variable)",
          tfm_mod.transonic_nozzle_one_component_non_autonomous, tfm_mod.ode_full,
          params_model, params_bvp, params_ivp, TFM_OUT_CSV)

    march("hyperbolic (6-variable)",
          hyp_mod.transonic_nozzle_hyperbolic_non_autonomous, hyp_mod.ode_full,
          params_model, params_bvp, params_ivp, HYP_OUT_CSV)

    sols = load_all(solution_specs())
    make_comparison_figure(
        sols,
        title=(f"SMD nozzle (case {CASE}): space marching started from the "
               f"time-marching inlet"),
        fig_path=FIG_PATH,
        show=SHOW,
        plot_alpha_liquid=PLOT_ALPHA_LIQUID,
        alpha_yscale=ALPHA_YSCALE,
    )
