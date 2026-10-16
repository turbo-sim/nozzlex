# Hyperbolic (6-variable primitive-state) counterpart of
# diffrax_nozzle_tfm_sinlge_solution.py, for the same SMD-nozzle case (case "b").
#
# State reduction vs. the TFM (9-var) file:
#   TFM:        [x, alpha_v, alpha_l, rho_v, rho_l, u_v, u_l, p, h_v, h_l]
#   Hyperbolic: [x, p, u_v, u_l, alpha_v, h_v, h_l]
# rho_v, rho_l, alpha_l are not carried as states here — rho_v/rho_l follow
# from the EOS via (p, h_v)/(p, h_l), and alpha_l = 1 - alpha_v.
#
# Only the non-autonomous solver path (the one actually exercised in
# __main__ of the TFM file) is ported — the commented-out autonomous/
# two-pass variants were left out to keep this file focused.

import os
import time
import sys
import jax
import jax.numpy as jnp
import diffrax as dfx
import optimistix as optx
import jaxprop as jxp
import matplotlib.pyplot as plt
import pandas as pd

jax.config.update("jax_enable_x64", True)

from matplotlib import gridspec

jxp.set_plot_options(grid=False)

from nozzlex.two_fluid.nozzle_model_solver import (
    NozzleParams,
    BVPSettings,
    IVPSettings,
    replace_param,
    compute_inlet_from_outlet_pressure_hyperbolic,
)

from nozzlex.two_fluid.nozzle_model_core import (
    nozzle_two_fluid_hyperbolic_autonomous,
    nozzle_two_fluid_hyperbolic_core,
)

from nozzlex.homogeneous_nonequilibrium.nozzle_model_core import (
    get_nozzle_smd,
    get_nozzle_smd_divergent,
    get_nozzle_smd_divergent_smooth,
)


# -----------------------------------------------------------------------------
# Non-autonomous form: x is the independent variable, dy/dx = A^{-1}·b
# -----------------------------------------------------------------------------
def transonic_nozzle_hyperbolic_non_autonomous(
    params_model: NozzleParams,
    params_bvp: BVPSettings,
    params_ivp: IVPSettings,
):
    """
    Non-autonomous hyperbolic solver: integrates dy/dx = A^{-1}·b with x ∈ [0, L].
    Mirrors transonic_nozzle_one_component_non_autonomous in
    diffrax_nozzle_tfm_sinlge_solution.py, but for the 6-variable primitive state.
    """
    fluid_vapor  = params_model.fluid1
    fluid_liquid = params_model.fluid2

    state_vapor_in  = fluid_vapor.get_state_saturation(params_model.p0_in)
    state_liquid_in = fluid_liquid.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T02_in)

    p_in, rho_l_in, rho_v_in, h_l_in, h_v_in = (
        state_liquid_in["pressure"],
        state_liquid_in["density"],
        state_vapor_in["density"],
        state_liquid_in["enthalpy"],
        state_vapor_in["enthalpy"],
    )

    jax.debug.print("p_in:{} | rho_l_in:{} | h_l_in:{}", p_in, rho_l_in, h_l_in)

    alpha_v_in = params_model.alphav_in
    uv_in = params_model.u1_in
    ul_in = params_model.u2_in

    # State: [p, uv, ul, alphav, hv, hl] — 6 vars, x excluded (it is the ODE's t)
    y_inlet = jnp.array([p_in, uv_in, ul_in, alpha_v_in, h_v_in, h_l_in])


    # --- Row/col scaling from inlet A_mat (same recipe as the TFM solver) ---
    Y_inlet_full  = jnp.concatenate([jnp.array([0.0]), y_inlet])
    params_model  = replace_param(params_model, "row_scales", jnp.ones(6))
    params_model  = replace_param(params_model, "col_scales", jnp.ones(6))
    params_model  = replace_param(params_model, "det_ref",    jnp.float64(1.0))
    out_inlet     = nozzle_two_fluid_hyperbolic_autonomous(0.0, Y_inlet_full, params_model)
    A_mat_inlet   = out_inlet["A_mat"]
    col_scales_in = jnp.maximum(jnp.max(jnp.abs(A_mat_inlet), axis=0), 1e-30)
    A_col_scaled  = A_mat_inlet / col_scales_in[None, :]
    row_scales_in = jnp.maximum(jnp.max(jnp.abs(A_col_scaled), axis=1), 1e-30)
    params_model  = replace_param(params_model, "row_scales", row_scales_in)
    params_model  = replace_param(params_model, "col_scales", col_scales_in)
    out_inlet_scaled = nozzle_two_fluid_hyperbolic_autonomous(0.0, Y_inlet_full, params_model)
    # NOTE: det_ref only rescales `rhs_autonomous`; this solver marches on
    # `out["rhs"]` (= N/det_D), which is independent of det_ref. So this is a
    # no-op here -- it is kept only so the params match the autonomous solver.
    det_ref_val = jnp.sign(out_inlet_scaled["D"]) * jnp.maximum(
        jnp.abs(out_inlet_scaled["D"]), 1e-10
    )
    params_model = replace_param(params_model, "det_ref", det_ref_val)


    def ode_full(t, y, args):
        """Reconstruct full ODE output dict from physical state and position x."""
        Y_full = jnp.concatenate([jnp.array([t]), y])
        return nozzle_two_fluid_hyperbolic_autonomous(0.0, Y_full, args)

    def eval_rhs(t, y, args):
        """Non-autonomous RHS: dy/dx = N / det_D (physical units, no extra scaling)."""
        out = ode_full(t, y, args)
        return out["rhs"]

    x0 = 1e-9
    x1 = params_model.length

    solver  = jxp.make_diffrax_solver(params_ivp.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_ivp.adjoint_name)
    term    = dfx.ODETerm(eval_rhs)
    ctrl    = dfx.PIDController(rtol=params_ivp.rtol, atol=params_ivp.atol)

    def eval_end_of_domain_event(t, y, args, **kwargs):
        x = t
        out   = ode_full(x, y, args)
        p_min = jnp.maximum(args.fluid1.p_min, args.fluid2.p_min)
        h1_min, h1_max = args.fluid1.h_min, args.fluid1.h_max
        h2_min, h2_max = args.fluid2.h_min, args.fluid2.h_max
        conditions = jnp.stack([
            x / x1,
            1.0 - x / x1,
            0.97 - out["Ma_mix"],
            out["p"] - p_min - 100,
            out["h1"] - h1_min - 10,
            h1_max - out["h1"] + 10,
            out["h2"] - h2_min - 10,
            h2_max - out["h2"] + 10,
        ])
        return jnp.min(conditions)

    event = dfx.Event(
        cond_fn=eval_end_of_domain_event,
        root_finder=optx.Newton(rtol=1e-2, atol=1e-2),
    )

    saveat = dfx.SaveAt(t1=True, dense=True, fn=ode_full)
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=x0,
        t1=x1,
        dt0=1e-9,
        y0=y_inlet,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=saveat,
        event=event,
        max_steps=nsteps,
        throw=False,
    )

    Y_ref = jnp.array([
        params_model.length,   # x ~ L
        p_in,                  # p  ~ initial pressure
        uv_in,                 # u  ~ initial velocity
        ul_in,                 # u  ~ initial velocity
        1.0,                   # alphav already O(1)
        h_v_in,                # hv ~ initial vapor enthalpy
        h_l_in,                # hl ~ initial liquid enthalpy
    ])

    return sol, Y_ref, params_model


def sol_from_out_pressure(params_model, params_ivp):

    u1_subsonic = compute_inlet_from_outlet_pressure_hyperbolic(1.0, 10.0, 8000000.0, params_model, params_ivp)
    return u1_subsonic
# -----------------------------------------------------------------------------
# SMD nozzle example — same case as diffrax_nozzle_tfm_sinlge_solution.py
# -----------------------------------------------------------------------------
class Tee:
    """Write to both a file and the original stream."""
    def __init__(self, stream, filepath):
        self.stream = stream
        self.file = open(filepath, "w", encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def ode_full(x, y, args):
    """Reconstruct full ODE output dict from physical state and position x."""
    return nozzle_two_fluid_hyperbolic_core(x, y, args)


if __name__ == "__main__":

    log_file = "smd_hyperbolic_single_solution_nb13_nd12.txt"
    tee_out = Tee(sys.stdout, log_file)
    tee_err = Tee(sys.stderr, log_file)
    sys.stdout = tee_out
    sys.stderr = tee_err

    nsteps = 100_000
    case = "a"

    table_dir = "fluid_tables"
    fluid_name = "water"
    metastable_phase = ["liquid", "vapor"]
    backend = "HEOS"
    hl_table = [300e3, 1800e3]  # J/kg
    pl_table = [1e5, 13e6]      # Pa
    N_point_liquid_table = 120
    hv_table = [2200e3, 3400e3]
    pv_table = [5e4, 13e6]      # Pa
    N_point_vapor_table = 150
    N_p_sat = 150
    gradient_method = "forward"

    P_0_in = 1.2e7

    T_in_a = 578.95
    p_out_a = 6.76e6

    T_in_b = 592.55
    p_out_b = 7.19e6

    if case == "a":
        T_0_in = T_in_a
    else:
        T_0_in = T_in_b

    fluid_liquid = jxp.FluidBicubic(
        fluid_name=fluid_name,
        backend="HEOS",
        h_min=hl_table[0],
        h_max=hl_table[1],
        p_min=pl_table[0],
        p_max=pl_table[1],
        N_h=N_point_liquid_table,
        N_p=N_point_liquid_table,
        table_dir=table_dir,
        gradient_method=gradient_method,
        metastable_phase=metastable_phase[0],
        N_p_sat=N_p_sat,
    )

    fluid_vapor = jxp.FluidBicubic(
        fluid_name=fluid_name,
        backend="HEOS",
        h_min=hv_table[0],
        h_max=hv_table[1],
        p_min=pv_table[0],
        p_max=pv_table[1],
        N_h=N_point_vapor_table,
        N_p=N_point_vapor_table,
        table_dir=table_dir,
        gradient_method=gradient_method,
        metastable_phase=metastable_phase[1],
        N_p_sat=N_p_sat,
    )

    u1_in = 6.148760057985783

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
        bubble_number=jnp.array(1e11),
        droplet_number=1e10,
        geometry=get_nozzle_smd_divergent,
        length=1.2,
        alphav_in=1e-6,
        alphal_in=1.0 - 1e-6,
        u1_in=u1_in,
        u2_in=u1_in * 1.000001,
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
        solver_name="Kvaerno5",  # Tsit5 | Kvaerno3 | Dopri5 | ImplicitEuler
        adjoint_name="DirectAdjoint",
        number_of_points=60,
        rtol=1e-8,
        atol=1e-8,
    )

    # Solve the problem
    print("\n" + "-" * 60)
    print(f"Evaluating transonic solution (non-autonomous hyperbolic + {params_ivp.solver_name})")
    print("-" * 60)

    t_start = time.perf_counter()
    print(params_model)

    u1_in = sol_from_out_pressure(params_model, params_ivp)
    params_model = replace_param(params_model, "u1_in", u1_in)
    params_model = replace_param(params_model, "u2_in", u1_in*0.99999)

    sol, Y_ref, params_model = transonic_nozzle_hyperbolic_non_autonomous(params_model, params_bvp, params_ivp)
    dt_ms = (time.perf_counter() - t_start) * 1e3
    print(f"\nTotal wall time: {dt_ms:.1f} ms")

    # Post-processing: x is the independent variable directly
    n_plot = 1000
    # start at the solver's t0 (1e-9), not 0.0: sol.evaluate() is only defined on
    # [t0, t1], so asking for x = 0 returns NaN and poisons the mdot check.
    x_grid = jnp.linspace(1e-9, float(sol.ts[0]), n_plot)
    out1_list = [ode_full(x, sol.evaluate(x), params_model) for x in x_grid]
    out1 = {
        k: jnp.stack([d[k] for d in out1_list])
        for k in out1_list[0] if jnp.asarray(out1_list[0][k]).shape == ()
    }
    mdot_err1 = (out1["m_dot"].max() - out1["m_dot"].min()) / out1["m_dot"][0]
    print(f"mdot error = {float(mdot_err1):.3e}")

    # --- Save solution to CSV ---
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"smd_hyperbolic_single_solution_case_{case}.csv")
    pd.DataFrame({k: jnp.asarray(v) for k, v in out1.items()}).to_csv(csv_path, index=False)
    print(f"Saved solution to {csv_path}")

    c1 = "tab:blue"

    fig = plt.figure(figsize=(5, 10))
    gs  = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 3, 3, 3])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    axs = [ax0, ax1, ax2, ax3, ax4]

    # --- row 1: pressure (bar) ---
    axs[0].set_ylabel("Pressure (bar)")
    axs[0].plot(out1["x"], out1["p"] * 1e-5, color=c1)

    # --- row 2: velocity ---
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].plot(out1["x"], out1["v1"], color=c1, label="vapor")
    axs[1].plot(out1["x"], out1["v2"], color=c1, linestyle="--", label="liquid")
    axs[1].legend(fontsize=7)

    # --- row 3: determinant ---
    axs[2].set_ylabel("Determinant")
    axs[2].plot(out1["x"], out1["D"], color=c1)
    axs[2].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)

    # --- row 4: temperature ---
    axs[3].set_ylabel("Temperature (K)")
    axs[3].plot(out1["x"], out1["T1"], color=c1, label="vap")
    axs[3].plot(out1["x"], out1["T2"], color=c1, linestyle="--", label="liq")
    axs[3].legend(fontsize=7)

    # --- row 5: volume fraction ---
    axs[4].set_ylabel("Vol. Fraction")
    axs[4].plot(out1["x"], out1["alpha1"], color=c1, label="alpha_v")
    axs[4].plot(out1["x"], out1["alpha2"], color=c1, linestyle="--", label="alpha_l")
    axs[4].set_xlabel("Axial coordinate x (m)")
    axs[4].legend(fontsize=7)
    fig.tight_layout(pad=1)

    # --- p-h diagram ---
    fluid_ph = jxp.Fluid(fluid_name, backend)
    fig_ph, ax_ph = fluid_ph.plot_phase_diagram(
        x_prop="h", y_prop="p", x_scale="linear", y_scale="log",
        plot_critical_point=True, plot_quality_isolines=True,
    )
    ax_ph.plot(out1["h1"], out1["p"], "-o", color=c1, markersize=3, label="vapor")
    ax_ph.plot(out1["h2"], out1["p"], "-s", color=c1, markersize=3, label="liquid")
    ax_ph.set_xlabel("Enthalpy (J/kg)")
    ax_ph.set_ylabel("Pressure (Pa)")
    ax_ph.legend(fontsize=7)
    fig_ph.tight_layout()

    plt.show()
