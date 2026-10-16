"""
Compare the TFM (9-variable) and hyperbolic (6-variable) two-fluid formulations
on the SMD divergent-nozzle single-solution case (case "b"):

  - TFM        : diffrax_nozzle_tfm_sinlge_solution.py
  - Hyperbolic : diffrax_nozzle_hyperbolic_single_solution.py

Both files guard their setup/solve/plot code behind `if __name__ ==
"__main__"`, so importing them here only pulls in their function definitions.
We rebuild the exact same case (fluids, NozzleParams, BVPSettings,
IVPSettings — including the *same* IVP tolerances/solver/step budget on both
sides) once, feed it to both non-autonomous solvers, and compare pressure,
velocity, det(A), temperature and volume fraction along x.
"""

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jaxprop as jxp

from nozzlex.two_fluid.nozzle_model_solver import (
    NozzleParams,
    BVPSettings,
    IVPSettings,
)
from nozzlex.homogeneous_nonequilibrium.nozzle_model_core import (
    get_nozzle_smd_divergent,
)

import diffrax_nozzle_tfm_sinlge_solution as tfm_mod
import diffrax_nozzle_hyperbolic_single_solution as hyp_mod

# Both files read a module-level `nsteps` that is normally only set inside
# their own `if __name__ == "__main__":` block. Since we import them as
# plain modules that block never runs, so we set it ourselves here — same
# budget on both sides so the comparison is apples-to-apples.
NSTEPS = 100_000
tfm_mod.nsteps = NSTEPS
hyp_mod.nsteps = NSTEPS


# -----------------------------------------------------------------------------
# Case setup (identical to the `__main__` block of both files)
# -----------------------------------------------------------------------------
def build_case(case="b"):
    here = os.path.dirname(os.path.abspath(__file__))
    table_dir = os.path.join(here, "fluid_tables")
    fluid_name = "water"
    metastable_phase = ["liquid", "vapor"]
    hl_table = [300e3, 1800e3]   # J/kg
    pl_table = [1e5, 13e6]       # Pa
    N_point_liquid_table = 120
    hv_table = [2200e3, 3400e3]  # J/kg
    pv_table = [5e4, 13e6]       # Pa
    N_point_vapor_table = 150
    N_p_sat = 150
    gradient_method = "forward"

    P_0_in = 1.2e7
    T_in_a = 578.95
    T_in_b = 592.55
    T_0_in = T_in_a if case == "a" else T_in_b

    fluid_liquid = jxp.FluidBicubic(
        fluid_name=fluid_name, backend="HEOS",
        h_min=hl_table[0], h_max=hl_table[1],
        p_min=pl_table[0], p_max=pl_table[1],
        N_h=N_point_liquid_table, N_p=N_point_liquid_table,
        table_dir=table_dir, gradient_method=gradient_method,
        metastable_phase=metastable_phase[0], N_p_sat=N_p_sat,
    )

    fluid_vapor = jxp.FluidBicubic(
        fluid_name=fluid_name, backend="HEOS",
        h_min=hv_table[0], h_max=hv_table[1],
        p_min=pv_table[0], p_max=pv_table[1],
        N_h=N_point_vapor_table, N_p=N_point_vapor_table,
        table_dir=table_dir, gradient_method=gradient_method,
        metastable_phase=metastable_phase[1], N_p_sat=N_p_sat,
    )

    # u1_in = 6.148760057985783
    u1_in = 8.05

    params_model = NozzleParams(
        p0_in=P_0_in,
        T01_in=T_0_in,
        T02_in=T_0_in,
        roughness=0e-6,
        Ma_low=0.95,
        Ma_high=1.05,
        heat_transfer=0.0,
        wall_friction=1.0,
        fluid1=fluid_vapor,
        fluid2=fluid_liquid,
        bubble_number=jnp.array(1e12),
        droplet_number=1e10,
        geometry=get_nozzle_smd_divergent,
        length=0.6,
        alphav_in=1e-6,
        alphal_in=1.0 - 1e-6,
        u1_in=u1_in,
        u2_in=u1_in,
    )

    params_bvp = BVPSettings(
        solve_mode="mach_crit",
        num_points=60,
        rtol=1e-8,
        atol=1e-8,
        max_steps=500,
        jac_mode="bwd",
        verbose=False,
        method="Bisection",
        warmup_method="Dogleg",
        warmup_steps=0,
    )

    # Same solver/tolerances/step budget for both formulations.
    params_ivp = IVPSettings(
        solver_name="ImplicitEuler",
        adjoint_name="DirectAdjoint",
        number_of_points=60,
        rtol=1e-6,
        atol=1e-6,
    )

    return params_model, params_bvp, params_ivp


# -----------------------------------------------------------------------------
# Run one formulation and post-process into a dict of 1D arrays vs. x
# -----------------------------------------------------------------------------
def run_and_postprocess(solve_fn, ode_full_fn, params_model, params_bvp, params_ivp, n_points=1000):
    sol, Y_ref, params_model = solve_fn(params_model, params_bvp, params_ivp)

    x_end = float(sol.ts[0])
    # Start just above 0 -- sol.evaluate() is only defined on [t0, t1] and the
    # solver's t0 = 1e-9, so evaluating exactly at x = 0 raises/produces NaN.
    x_grid = jnp.linspace(1e-9, x_end, n_points)
    rows = [ode_full_fn(x, sol.evaluate(x), params_model) for x in x_grid]

    keys = rows[0].keys()
    out = {
        k: jnp.stack([r[k] for r in rows])
        for k in keys
        if jnp.isscalar(rows[0][k]) or rows[0][k].shape == ()
    }
    return sol, out


if __name__ == "__main__":
    case = "a"
    params_model, params_bvp, params_ivp = build_case(case=case)

    print("\n" + "-" * 60)
    print("Solving TFM (9-variable) formulation...")
    print("-" * 60)
    t0 = time.perf_counter()
    sol_tfm, out_tfm = run_and_postprocess(
        tfm_mod.transonic_nozzle_one_component_non_autonomous,
        tfm_mod.ode_full,
        params_model, params_bvp, params_ivp,
    )
    x_end_tfm = float(sol_tfm.ts[0])
    print(f"Done in {(time.perf_counter() - t0):.1f} s | status: {sol_tfm.result} | "
          f"steps: {int(sol_tfm.stats['num_steps'])} | "
          f"x_end: {x_end_tfm:.5f} / {float(params_model.length):.5f} m "
          f"({100 * x_end_tfm / float(params_model.length):.1f}%)")

    print("\n" + "-" * 60)
    print("Solving hyperbolic (6-variable) formulation...")
    print("-" * 60)
    t0 = time.perf_counter()
    sol_hyp, out_hyp = run_and_postprocess(
        hyp_mod.transonic_nozzle_hyperbolic_non_autonomous,
        hyp_mod.ode_full,
        params_model, params_bvp, params_ivp,
    )
    x_end_hyp = float(sol_hyp.ts[0])
    print(f"Done in {(time.perf_counter() - t0):.1f} s | status: {sol_hyp.result} | "
          f"steps: {int(sol_hyp.stats['num_steps'])} | "
          f"x_end: {x_end_hyp:.5f} / {float(params_model.length):.5f} m "
          f"({100 * x_end_hyp / float(params_model.length):.1f}%)")

    if x_end_hyp < 0.99 * float(params_model.length):
        print(
            f"\n*** NOTE: the hyperbolic run did NOT reach the outlet — it stopped at "
            f"x={x_end_hyp:.5f} m because {sol_hyp.result}. Its curve below is truncated; "
            f"this is not a plotting artifact. ***"
        )

    # --- Save both solutions to CSV ---
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    tfm_csv = os.path.join(results_dir, f"compare_smd_tfm_case_{case}.csv")
    hyp_csv = os.path.join(results_dir, f"compare_smd_hyperbolic_case_{case}.csv")
    pd.DataFrame({k: np.asarray(v) for k, v in out_tfm.items()}).to_csv(tfm_csv, index=False)
    pd.DataFrame({k: np.asarray(v) for k, v in out_hyp.items()}).to_csv(hyp_csv, index=False)
    print(f"\nSaved TFM solution to        {tfm_csv}")
    print(f"Saved hyperbolic solution to  {hyp_csv}")

    # --- Comparison plot (same 5-row layout as the single-solution demos) ---
    c1, c2 = "tab:blue", "tab:orange"

    fig = plt.figure(figsize=(6.5, 12))
    gs = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 3, 3, 3])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    axs = [ax0, ax1, ax2, ax3, ax4]

    axs[0].set_ylabel("Pressure (bar)")
    axs[0].plot(out_tfm["x"], out_tfm["p"] * 1e-5, color=c1, label="TFM (9-var)")
    axs[0].plot(out_hyp["x"], out_hyp["p"] * 1e-5, color=c2, label="Hyperbolic (6-var)")
    axs[0].legend(fontsize=7)

    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].plot(out_tfm["x"], out_tfm["v1"], color=c1, label="vapor, TFM")
    axs[1].plot(out_tfm["x"], out_tfm["v2"], color=c1, linestyle="--", label="liquid, TFM")
    axs[1].plot(out_hyp["x"], out_hyp["v1"], color=c2, label="vapor, Hyperbolic")
    axs[1].plot(out_hyp["x"], out_hyp["v2"], color=c2, linestyle="--", label="liquid, Hyperbolic")
    axs[1].legend(fontsize=6, ncol=2)

    axs[2].set_ylabel("Determinant D")
    axs[2].plot(out_tfm["x"], out_tfm["D"], color=c1, label="TFM")
    axs[2].plot(out_hyp["x"], out_hyp["D"], color=c2, label="Hyperbolic")
    axs[2].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    axs[2].legend(fontsize=7)

    axs[3].set_ylabel("Temperature (K)")
    axs[3].plot(out_tfm["x"], out_tfm["T1"], color=c1, label="vap, TFM")
    axs[3].plot(out_tfm["x"], out_tfm["T2"], color=c1, linestyle="--", label="liq, TFM")
    axs[3].plot(out_hyp["x"], out_hyp["T1"], color=c2, label="vap, Hyperbolic")
    axs[3].plot(out_hyp["x"], out_hyp["T2"], color=c2, linestyle="--", label="liq, Hyperbolic")
    axs[3].legend(fontsize=6, ncol=2)

    axs[4].set_ylabel("Volume fraction")
    axs[4].plot(out_tfm["x"], out_tfm["alpha1"], color=c1, label="v, TFM")
    axs[4].plot(out_tfm["x"], out_tfm["alpha2"], color=c1, linestyle="--", label="l, TFM")
    axs[4].plot(out_hyp["x"], out_hyp["alpha1"], color=c2, label="v, Hyperbolic")
    axs[4].plot(out_hyp["x"], out_hyp["alpha2"], color=c2, linestyle="--", label="l, Hyperbolic")
    axs[4].set_xlabel("Axial coordinate x (m)")
    axs[4].legend(fontsize=6, ncol=2)

    fig.suptitle(f"SMD nozzle (case {case}): TFM vs. hyperbolic formulation")
    fig.tight_layout(pad=1)

    fig_path = os.path.join(results_dir, f"compare_smd_tfm_vs_hyperbolic_case_{case}.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Saved comparison figure to    {fig_path}")

    plt.show()
