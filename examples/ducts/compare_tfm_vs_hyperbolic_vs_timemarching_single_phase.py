"""
Compare THREE solutions of the SMD divergent-nozzle flashing case:

  1. TFM (9-variable) space-marching   -- diffrax_nozzle_tfm_sinlge_solution.py
  2. Hyperbolic (6-variable) space-marching
                                       -- diffrax_nozzle_hyperbolic_single_solution.py
  3. Time-marching FVM (two-fluid)     -- nozzle-hyperbolic solver, exported to CSV

Same quantities as `compare_tfm_vs_hyperbolic_single_solution.py` (pressure,
velocities, det(A), temperatures, volume fractions), only with the time-marching
solution overlaid.

This script does NOT solve anything: it reads pre-computed CSVs.  The two
space-marching CSVs are written by `compare_tfm_vs_hyperbolic_single_solution.py`
(run that first if they are missing).

Everything you are likely to want to change -- file paths, labels, colours and
the CSV column names of each source -- lives in the CONFIG block right below.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

# =============================================================================
# CONFIG -- edit here
# =============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

CASE = "a"  # SMD inlet case: "a" (T0 = 578.95 K) or "b" (T0 = 592.55 K)

# --- Where the CSVs live ------------------------------------------------------
TFM_CSV = os.path.join(RESULTS_DIR, f"compare_smd_tfm_case_{CASE}.csv")
HYP_CSV = os.path.join(RESULTS_DIR, f"compare_smd_hyperbolic_case_{CASE}.csv")
TIME_CSV = os.path.join(HERE, "solution_time_marching.csv")

# --- Where the figure goes ----------------------------------------------------
FIG_PATH = os.path.join(RESULTS_DIR, f"compare_smd_tfm_vs_hyp_vs_time_case_{CASE}.png")
FIG_DPI = 150
FIG_SIZE = (6.5, 12)
SHOW = True

# --- The three solutions ------------------------------------------------------
# `cols` maps the canonical quantity names used by the plotting code to the
# column names of each CSV.  A value can be
#   * a string   -> column name in that CSV
#   * a callable -> f(dataframe) -> 1D array  (for derived quantities)
#   * None       -> quantity not available; that curve is simply skipped
#
# Phase convention: "g" = vapor / gas (phase 1), "l" = liquid (phase 2).
# Style convention: colour = formulation, and within one formulation the gas uses
# `linestyle` / filled markers and the liquid uses `linestyle_liquid` / open
# markers.  The time-marching solution is drawn as symbols only so that the
# space-marching line underneath stays visible where the two agree.
SOLUTIONS = [
    dict(
        key="tfm",
        label="TFM (9-var, space march)",
        path=TFM_CSV,
        color="tab:blue",
        linestyle="-",
        cols=dict(
            x="x",
            p="p",
            v_g="v1",
            v_l="v2",
            T_g="T1",
            T_l="T2",
            alpha_g="alpha1",
            alpha_l="alpha2",
            D="D",
        ),
    ),
    dict(
        key="hyp",
        label="Hyperbolic (6-var, space march)",
        path=HYP_CSV,
        color="tab:orange",
        linestyle="-",
        cols=dict(
            x="x",
            p="p",
            v_g="v1",
            v_l="v2",
            T_g="T1",
            T_l="T2",
            alpha_g="alpha1",
            alpha_l="alpha2",
            D="D",
        ),
    ),
    dict(
        key="time",
        label="Time marching (FVM)",
        path=TIME_CSV,
        color="black",
        linestyle="none",  # symbols only -> the space-marching lines stay visible
        marker="o",
        markevery=8,
        markersize=3.0,
        cols=dict(
            x="x",
            p="p",
            v_g="u_g",
            v_l="u_l",
            T_g="T_g",
            T_l="T_l",
            alpha_g="alpha_g",
            alpha_l=lambda df: 1.0 - df["alpha_g"].to_numpy(),
            D=None,  # det(A) is a space-marching diagnostic; not exported by the FVM
        ),
    ),
]

# Defaults applied to every source unless it overrides them in its own dict.
LIQUID_LINESTYLE = "--"  # gas keeps the source's `linestyle`
LINEWIDTH = 1.4
MARKERSIZE = 3.0
MARKEVERY = 8


# =============================================================================
# Loading helpers
# =============================================================================
def load_solution(spec):
    """Read one CSV and resolve it into a dict of canonical name -> array."""
    if not os.path.isfile(spec["path"]):
        raise FileNotFoundError(
            f"Missing CSV for '{spec['label']}':\n  {spec['path']}\n"
            "Run compare_tfm_vs_hyperbolic_single_solution.py first (it writes the "
            "two space-marching CSVs), or fix the path in the CONFIG block."
        )
    df = pd.read_csv(spec["path"])

    data = {}
    for name, source in spec["cols"].items():
        if source is None:
            data[name] = None
        elif callable(source):
            data[name] = np.asarray(source(df))
        elif source in df.columns:
            data[name] = df[source].to_numpy()
        else:
            print(
                f"  [warn] '{spec['label']}': column '{source}' for '{name}' "
                f"not found in {os.path.basename(spec['path'])} -- skipped"
            )
            data[name] = None

    out = dict(spec)
    out["data"] = data
    out["n_points"] = len(df)
    return out


def plot_curve(ax, sol, key, scale=1.0, liquid=False, label=None):
    """Plot one quantity of one solution, silently skipping missing data."""
    d = sol["data"]
    y = d.get(key)
    if y is None or d.get("x") is None:
        return False

    if liquid:
        linestyle = sol.get("linestyle_liquid", LIQUID_LINESTYLE)
        if sol.get("linestyle") == "none":  # symbol-only source: dashes carry no info
            linestyle = "none"
    else:
        linestyle = sol.get("linestyle", "-")

    ax.plot(
        d["x"],
        y * scale,
        color=sol["color"],
        linestyle=linestyle,
        linewidth=sol.get("linewidth", LINEWIDTH),
        marker=sol.get("marker", None),
        markevery=sol.get("markevery", MARKEVERY),
        markersize=sol.get("markersize", MARKERSIZE),
        # open symbols for the liquid phase, filled for the gas phase
        markerfacecolor="none" if liquid else sol["color"],
        markeredgewidth=0.9,
        label=label if label is not None else sol["label"],
    )
    return True


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("\nLoading solutions")
    print("-" * 70)
    sols = []
    for spec in SOLUTIONS:
        sol = load_solution(spec)
        sols.append(sol)
        x = sol["data"]["x"]
        print(
            f"  {sol['label']:<34s} {sol['n_points']:>5d} pts | "
            f"x = {x[0]:.4f} -> {x[-1]:.4f} m | "
            f"p_exit = {sol['data']['p'][-1] * 1e-5:.3f} bar"
        )

    # --- Figure: same 5-row layout as the two-way comparison ------------------
    fig = plt.figure(figsize=FIG_SIZE)
    gs = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 3, 3, 3])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    axs = [ax0, ax1, ax2, ax3, ax4]

    # 1) Pressure
    axs[0].set_ylabel("Pressure (bar)")
    for sol in sols:
        plot_curve(axs[0], sol, "p", scale=1e-5)
    axs[0].legend(fontsize=7)

    # 2) Velocities
    axs[1].set_ylabel("Velocity (m/s)")
    for sol in sols:
        plot_curve(axs[1], sol, "v_g", label=f"vapor, {sol['label']}")
        plot_curve(axs[1], sol, "v_l", liquid=True, label=f"liquid, {sol['label']}")
    axs[1].legend(fontsize=5, ncol=2)

    # 3) Determinant of the flux matrix (space-marching diagnostic only)
    axs[2].set_ylabel("Determinant D")
    n_det = sum(plot_curve(axs[2], sol, "D") for sol in sols)
    axs[2].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    if n_det < len(sols):
        axs[2].text(
            0.99,
            0.05,
            "det(A) not available for the time-marching solution",
            transform=axs[2].transAxes,
            ha="right",
            va="bottom",
            fontsize=6,
            color="gray",
        )
    axs[2].legend(fontsize=7)

    # 4) Temperatures
    axs[3].set_ylabel("Temperature (K)")
    for sol in sols:
        plot_curve(axs[3], sol, "T_g", label=f"vap, {sol['label']}")
        plot_curve(axs[3], sol, "T_l", liquid=True, label=f"liq, {sol['label']}")
    axs[3].legend(fontsize=5, ncol=2)

    # 5) Volume fractions
    axs[4].set_ylabel("Volume fraction")
    for sol in sols:
        plot_curve(axs[4], sol, "alpha_g", label=f"v, {sol['label']}")
        plot_curve(axs[4], sol, "alpha_l", liquid=True, label=f"l, {sol['label']}")
    axs[4].set_xlabel("Axial coordinate x (m)")
    axs[4].legend(fontsize=5, ncol=2)

    fig.suptitle(f"SMD nozzle (case {CASE}): space marching (TFM, hyperbolic) vs. time marching")
    fig.tight_layout(pad=1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=FIG_DPI)
    print(f"\nSaved comparison figure to {FIG_PATH}")

    if SHOW:
        plt.show()