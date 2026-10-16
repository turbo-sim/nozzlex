"""
Compare the two two-fluid pipe formulations on the same inlet/outlet case:

  - "TFM"        : demo_two_fluid_pipe.py            (9-variable state,
                    [x, alpha_v, alpha_l, rho_v, rho_l, u_v, u_l, p, h_v, h_l])
  - "Hyperbolic" : demo_two_fluid_hyperbolic_pipe.py  (6-variable primitive state,
                    [x, p, u_v, u_l, alpha_v, h_v, h_l])

Both demo scripts guard their setup/solve/plot code behind `if __name__ ==
"__main__"`, so importing them here only pulls in their function definitions
(no fluid tables get built, nothing gets plotted). We rebuild the exact same
case (fluids, NozzleParams, BVPSettings, IVPSettings) once and feed it to both
solvers, then compare pressure, temperature, volume fraction and mixture Mach
number along x.
"""

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jaxprop as jxp

from nozzlex.two_fluid.nozzle_model_solver import (
    NozzleParams,
    BVPSettings,
    IVPSettings,
    replace_param,
)
from nozzlex.two_fluid.nozzle_model_core import (
    get_pipe,
    nozzle_two_fluid_one_component_autonomous,
    nozzle_two_fluid_hyperbolic_autonomous,
)

import demo_two_fluid_pipe as tfm_mod
import demo_two_fluid_hyperbolic_pipe as hyp_mod

# Both demo modules read a module-level `nsteps` that is normally only set
# inside their own `if __name__ == "__main__":` block. Since we import them
# as plain modules that block never runs, so we have to set it ourselves or
# `transonic_nozzle_one_component_lax` raises NameError on `max_steps=nsteps`.
tfm_mod.nsteps = 20000
hyp_mod.nsteps = 20000


# -----------------------------------------------------------------------------
# Case setup (identical to the `__main__` block of both demo scripts)
# -----------------------------------------------------------------------------
def build_case(case="a"):
    here = os.path.dirname(os.path.abspath(__file__))
    table_dir = os.path.join(here, "fluid_tables")
    fluid_name = "water"
    metastable_phase = ["liquid", "vapor"]
    hl_table = [1000e3, 1800e3]  # J/kg
    pl_table = [5e6, 13e6]       # Pa
    hv_table = [2200e3, 3400e3]
    pv_table = [5e6, 13e6]       # Pa
    N_point_liquid_table = 100
    N_point_vapor_table = 100
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

    params_model = NozzleParams(
        p0_in=P_0_in,
        T01_in=T_0_in,
        T02_in=T_0_in,
        length=1.0,
        roughness=0e-6,
        Ma_low=0.95,
        Ma_high=1.05,
        heat_transfer=0.0,
        wall_friction=0.0,
        fluid1=fluid_vapor,
        fluid2=fluid_liquid,
        geometry=get_pipe,
        bubble_number=1e10,
        droplet_number=1e8,
        u1_in=8.0,
        u2_in=8.0,
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

    params_ivp = IVPSettings(
        solver_name="ImplicitEuler",  # Tsit5 | Kvaerno3 | Dopri5 | ImplicitEuler
        adjoint_name="DirectAdjoint",
        number_of_points=60,
        rtol=1e-6,
        atol=1e-6,
    )

    return params_model, params_bvp, params_ivp


# -----------------------------------------------------------------------------
# Run one formulation and post-process into a dict of 1D arrays vs. x
# -----------------------------------------------------------------------------
def run_and_postprocess(solve_fn, autonomous_fn, params_model, params_bvp, params_ivp, n_points=200):
    sol, Y_ref, params_model = solve_fn(params_model, params_bvp, params_ivp)

    ts = jnp.linspace(sol.t0, sol.ts[0], n_points)
    ys_hat = [sol.evaluate(t) for t in ts]
    ys = [y_hat * Y_ref for y_hat in ys_hat]
    rows = [autonomous_fn(0.0, y, params_model) for y in ys]

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
    params_model = replace_param(params_model, "Ma_in", jnp.float64(0.01))

    print("\n" + "-" * 60)
    print("Solving TFM (9-variable) formulation...")
    print("-" * 60)
    t0 = time.perf_counter()
    sol_tfm, out_tfm = run_and_postprocess(
        tfm_mod.transonic_nozzle_one_component_lax,
        nozzle_two_fluid_one_component_autonomous,
        params_model, params_bvp, params_ivp,
    )
    print(f"Done in {(time.perf_counter() - t0) * 1e3:.1f} ms | "
          f"status: {sol_tfm.result} | steps: {int(sol_tfm.stats['num_steps'])}")

    print("\n" + "-" * 60)
    print("Solving hyperbolic (6-variable) formulation...")
    print("-" * 60)
    t0 = time.perf_counter()
    sol_hyp, out_hyp = run_and_postprocess(
        hyp_mod.transonic_nozzle_one_component_lax,
        nozzle_two_fluid_hyperbolic_autonomous,
        params_model, params_bvp, params_ivp,
    )
    print(f"Done in {(time.perf_counter() - t0) * 1e3:.1f} ms | "
          f"status: {sol_hyp.result} | steps: {int(sol_hyp.stats['num_steps'])}")

    # --- Save both solutions to CSV ---
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    tfm_csv = os.path.join(results_dir, f"compare_tfm_case_{case}.csv")
    hyp_csv = os.path.join(results_dir, f"compare_hyperbolic_case_{case}.csv")
    pd.DataFrame({k: np.asarray(v) for k, v in out_tfm.items()}).to_csv(tfm_csv, index=False)
    pd.DataFrame({k: np.asarray(v) for k, v in out_hyp.items()}).to_csv(hyp_csv, index=False)
    print(f"\nSaved TFM solution to        {tfm_csv}")
    print(f"Saved hyperbolic solution to  {hyp_csv}")

    # --- Numeric summary at the outlet ---
    print("\n--- Outlet comparison ---")
    for label, out in [("TFM", out_tfm), ("Hyperbolic", out_hyp)]:
        print(
            f"{label:12s}: x={float(out['x'][-1]):6.4f} m  "
            f"p={float(out['p'][-1]) / 1e5:8.3f} bar  "
            f"T1={float(out['T1'][-1]):7.2f} K  T2={float(out['T2'][-1]):7.2f} K  "
            f"alpha1={float(out['alpha1'][-1]):7.5f}  "
            f"Ma_mix={float(out['Ma_mix'][-1]):7.5f}"
        )

    # --- Comparison plot ---
    fig, axs = plt.subplots(4, 1, figsize=(6.5, 11), sharex=True)

    axs[0].plot(out_tfm["x"], out_tfm["p"] * 1e-5, "-o", ms=3, color="C0", label="TFM (9-var)")
    axs[0].plot(out_hyp["x"], out_hyp["p"] * 1e-5, "-s", ms=3, color="C1", label="Hyperbolic (6-var)")
    axs[0].set_ylabel("Pressure (bar)")
    axs[0].legend(fontsize=8)

    axs[1].plot(out_tfm["x"], out_tfm["T1"], "-o", ms=3, color="C0", label="vapor, TFM")
    axs[1].plot(out_tfm["x"], out_tfm["T2"], "-o", ms=3, color="C2", label="liquid, TFM")
    axs[1].plot(out_hyp["x"], out_hyp["T1"], "-s", ms=3, color="C1", label="vapor, Hyperbolic")
    axs[1].plot(out_hyp["x"], out_hyp["T2"], "-s", ms=3, color="C3", label="liquid, Hyperbolic")
    axs[1].set_ylabel("Temperature (K)")
    axs[1].legend(fontsize=7, ncol=2)

    axs[2].plot(out_tfm["x"], out_tfm["alpha1"], "-o", ms=3, color="C0", label="vapor, TFM")
    axs[2].plot(out_tfm["x"], out_tfm["alpha2"], "-o", ms=3, color="C2", label="liquid, TFM")
    axs[2].plot(out_hyp["x"], out_hyp["alpha1"], "-s", ms=3, color="C1", label="vapor, Hyperbolic")
    axs[2].plot(out_hyp["x"], out_hyp["alpha2"], "-s", ms=3, color="C3", label="liquid, Hyperbolic")
    axs[2].set_ylabel("Volume fraction")
    axs[2].legend(fontsize=7, ncol=2)

    axs[3].plot(out_tfm["x"], out_tfm["Ma_mix"], "-o", ms=3, color="C0", label="TFM (9-var)")
    axs[3].plot(out_hyp["x"], out_hyp["Ma_mix"], "-s", ms=3, color="C1", label="Hyperbolic (6-var)")
    axs[3].set_ylabel("Mach number (mix)")
    axs[3].set_xlabel("x (m)")
    axs[3].legend(fontsize=8)

    fig.suptitle(f"Two-fluid pipe: TFM vs. hyperbolic formulation (case {case})")
    fig.tight_layout()

    fig_path = os.path.join(results_dir, f"compare_tfm_vs_hyperbolic_case_{case}.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Saved comparison figure to    {fig_path}")

    plt.show()
