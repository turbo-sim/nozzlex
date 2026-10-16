
# the T of the nitrgogen is suspicious. it is growing higher than the inlet, higher than ambient. there must be an issue. this can be related to bot the thermal source term or the drag which also appears in the energy eq. 
# check the eq first and then move on

import os
import pickle as pkl
import time
import jax
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
import optimistix as optx
import jaxprop as jxp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

from matplotlib import gridspec

jxp.set_plot_options(grid=False)

from nozzlex.two_fluid.nozzle_model_core import (
    nozzle_two_fluid_core,
    nozzle_two_fluid_autonomous,
    get_nozzle_elliot,
    get_nozzle_elliot_fast,
)

from nozzlex.two_fluid.nozzle_model_solver import (
    NozzleParams,
    BVPSettings,
    IVPSettings,
    nozzle_two_fluid_velocity_crit,
    replace_param,
    compute_critical_inlet_two_fluid,
    eval_end_of_domain_event,
)
# v1 solves the ode system using the space marching in non-autonomous form
# v2 solves the autonomous system with events for the bounds of domain ends
# v3 uses the non-autonomous system and determines the inlet critical condition with collocation method


# -----------------------------------------------------------------------------
# Main API to the converging-diverging nozzle model
# -----------------------------------------------------------------------------
# @eqx.filter_jit
def transonic_nozzle_one_component_autonomus(
    params_model: NozzleParams,
    params_bvp: BVPSettings,
    params_ivp: IVPSettings,
):
    """
    Transonic converging-diverging nozzle.
    First uses the root finding to find the critical (sonic) state,
    then integrates the ODE system with a blended RHS near the critical location.
    """
    # jax.debug.print("params_model={pm}", pm=params_model)
    # jax.debug.print("params_solver={ps}", ps=params_ivp)


    fluid1 = params_model.fluid1
    fluid2 = params_model.fluid2

    state1_in = fluid1.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T01_in)
    state2_in = fluid2.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T02_in)

    p_in, rho1_in, rho2_in, h1_in, h2_in = (
        state1_in["p"],
        state1_in["rho"],
        state2_in["rho"],
        state1_in["h"],
        state2_in["h"]
    )

    q_in = (1)/(1 + params_model.mixture_ratio)
    G1, a1 = state1_in["G"], state1_in["a"]
    # area nitrogen
    diameter_tube = 4.57e-3
    diameter_tube_ext = 6.35e-3
    Area_1 = 61 * (jnp.pi * diameter_tube * diameter_tube / 4)
    Area_ext = (71.15e-3)**2 * jnp.pi / 4
    Area_2 = Area_ext - 61 * jnp.pi * (diameter_tube_ext ** 2) / 4
    Area_tot = Area_1 + Area_2
    alpha1_in = Area_1 / Area_tot
    alpha1_in = 0.6365890590354819
    alpha2_in = 1.0 - alpha1_in
    # alpha2_in = 1 / (1 + ((1 - q_in) / q_in) * (rho2_in / rho1_in))
    # alpha1_in = 1 - alpha2_in
    jax.debug.print("alpha1:{a1} | alpha2:{a2}", a1=alpha1_in, a2=alpha2_in)
    # h = st["h"]
    G2, a2 = state2_in["G"], state2_in["a"]

    # a_in_mix = get_speed_of_sound_mixture(G1, alpha1_in, a1, rho1_in, G2, alpha2_in, a2, rho2_in)

    # u_in = params_model.Ma_in * a_in_mix
    u1_in = params_model.u1_in
    # u2_in = rho1_in * u1_in * alpha1_in / (rho2_in * alpha2_in * params_model.mixture_ratio)
        
    u2_in = u1_in * (rho1_in / rho2_in) * (alpha1_in / alpha2_in) * (1 / params_model.mixture_ratio)

    # u2_in = u1_in # no slip at the inlet



    # # If no slip is assumed:
    # u2_in = u1_in # no slip at the inlet
    # alpha2_in = 1 / (1 + params_model.mixture_ratio * (rho2_in / rho1_in))
    # alpha1_in = 1 - alpha2_in

    jax.debug.print("alpha1:{a1} |  alpha2:{a2}",a2=alpha2_in, a1=alpha1_in)

    y_variable = jnp.array([alpha1_in, alpha2_in, rho1_in, rho2_in, u1_in, u2_in, p_in, h1_in, h2_in])

    # --- Fixed row/column scales for A_mat (equilibration from inlet A_mat) ---
    # Evaluate the core once at inlet to get the actual A_mat entries
    # Use dummy row/col scales of 1.0 to get the unscaled matrix
    params_model = replace_param(params_model, "row_scales", jnp.ones(9))
    params_model = replace_param(params_model, "col_scales", jnp.ones(9))
    x_in_tmp = jnp.array([1e-9])
    y_inlet_tmp = jnp.concatenate([x_in_tmp, y_variable])
    out_inlet = nozzle_two_fluid_autonomous(0.0, y_inlet_tmp, params_model)
    A_mat_inlet = out_inlet["A_mat"]
    # Column scaling: max absolute value in each column
    col_scales_in = jnp.maximum(jnp.max(jnp.abs(A_mat_inlet), axis=0), 1e-30)
    # Row scaling: max absolute value in each row after column scaling
    A_col_scaled = A_mat_inlet / col_scales_in[None, :]
    row_scales_in = jnp.maximum(jnp.max(jnp.abs(A_col_scaled), axis=1), 1e-30)
    jax.debug.print("col_scales: {c}", c=col_scales_in)
    jax.debug.print("row_scales: {r}", r=row_scales_in)
    params_model = replace_param(params_model, "row_scales", row_scales_in)
    params_model = replace_param(params_model, "col_scales", col_scales_in)
    # Compute |det(A_scaled)| at inlet to normalize the autonomous RHS to O(1)
    out_inlet_scaled = nozzle_two_fluid_autonomous(0.0, y_inlet_tmp, params_model)
    det_ref_val = jnp.maximum(jnp.abs(out_inlet_scaled["D"]), 1e-10)
    jax.debug.print("det_ref: {d}", d=det_ref_val)
    params_model = replace_param(params_model, "det_ref", det_ref_val)

    x_in = jnp.array([1e-9])  # start slightly after inlet
    y_inlet = jnp.concatenate([x_in, y_variable])

    # --- State-variable scaling ---
    # Reference scales so that y_hat = Y / Y_ref is O(1) for all components.
    # Improves Jacobian conditioning for the implicit solver.
    #   Y = [x,    α1,  α2,  ρ1,      ρ2,      u1,    u2,    p,    h1,     h2    ]
    Y_ref = jnp.array([
        params_model.length,  # x ~ L
        1.0,                  # α1 already O(1)
        1.0,                  # α2 already O(1)
        rho1_in,              # ρ1 ~ initial liquid density
        rho2_in,              # ρ2 ~ initial gas density
        u1_in,                # u1 ~ initial liquid velocity
        u2_in,                # u2 ~ initial gas velocity
        p_in,                 # p  ~ initial pressure
        h1_in,                # h1 ~ initial liquid enthalpy
        h2_in,                # h2 ~ initial gas enthalpy
    ])

    # --- Scaled ODE wrappers ---
    def ode_full_scaled(t, y, args):
        """Evaluate full ODE output from scaled state."""
        y_phys = y * Y_ref
        return nozzle_two_fluid_autonomous(0.0, y_phys, args)

    def ode_rhs_scaled(t, y, args):
        """RHS in scaled coordinates: dy_hat/dτ = f(y_hat * Y_ref) / Y_ref."""
        out = ode_full_scaled(t, y, args)
        return out["rhs_autonomous"] / Y_ref

    def eval_end_of_domain_event_scaled(t, y, args, **kwargs):
        """Event function: unscale before evaluating physical conditions."""
        y_phys = y * Y_ref
        x = y_phys[0]
        L = args.length
        out = nozzle_two_fluid_autonomous(0.0, y_phys, args)
        det = jnp.min(out.get("D"))
        M = det + 10
        # M = 10

        p = y_phys[7]
        p_min = jnp.maximum(args.fluid1.p_min, args.fluid2.p_min)
        h1_end, h2_end =  y_phys[8], y_phys[9]
        h1_min, h1_max = fluid1.h_min, fluid1.h_max
        h2_min, h2_max = fluid2.h_min, fluid2.h_max
 

        conditions = jnp.stack([
            x / L,                  # 0: x >= 0
            1 - x / L,              # 1: x <= L
            M,                      # 2: determinant condition
            p - p_min - 10,         # 3: p above table min
            h1_end - h1_min - 10,
            h1_max - h1_end + 10,
            h2_end - h2_min - 10,
            h2_max - h2_end + 10,

        ])
        min_val = jnp.min(conditions)
        min_idx = jnp.argmin(conditions)
        # jax.debug.print("event: min_idx={i} min_val={v} x={x}", i=min_idx, v=min_val, x=x)
        return min_val

    y_inlet_hat = y_inlet / Y_ref

    t0, t1 = 1e-9, 1e9
    solver = jxp.make_diffrax_solver(params_ivp.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_ivp.adjoint_name)
    term = dfx.ODETerm(ode_rhs_scaled)
    ctrl = dfx.PIDController(
        rtol=params_ivp.rtol,
        atol=params_ivp.atol,
        dtmax=1e-3,      # hard cap on pseudo-time step size
    )
    event = dfx.Event(
        cond_fn=eval_end_of_domain_event_scaled,
        root_finder=optx.Bisection(rtol=1e-3, atol=1e-3),
    )

    # --- Diagnostic: evaluate RHS at inlet to check magnitudes ---
    rhs_at_inlet = ode_rhs_scaled(t0, y_inlet_hat, params_model)
    jax.debug.print("=== INLET DIAGNOSTICS ===")
    jax.debug.print("y_inlet_hat (scaled): {y}", y=y_inlet_hat)
    jax.debug.print("rhs_scaled at inlet:  {r}", r=rhs_at_inlet)
    jax.debug.print("rhs_scaled magnitudes: max={mx:.4e}, min={mn:.4e}",
                     mx=jnp.max(jnp.abs(rhs_at_inlet)),
                     mn=jnp.min(jnp.abs(rhs_at_inlet)))
    out_inlet_diag = ode_full_scaled(t0, y_inlet_hat, params_model)
    jax.debug.print("rhs_autonomous (unscaled): {r}", r=out_inlet_diag["rhs_autonomous"])
    jax.debug.print("det_D = {d:.4e}, b_vec = {b}", d=out_inlet_diag["D"], b=out_inlet_diag["b_vec"])

    # --- first solve (find domain end) ---
    saveat = dfx.SaveAt(t1=True, dense=True)
    sol1 = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=1e-9,
        y0=y_inlet_hat,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=saveat,
        event=event,
        max_steps=5_000,
        throw=False,
    )

    # --- Solver stats ---
    print(f"\n--- Solver stats ---")
    print(f"  result: {sol1.result}")
    print(f"  τ_end = {float(sol1.ts[0]):.6e}")
    print(f"  num_steps = {sol1.stats['num_steps']}")
    print(f"  num_accepted = {sol1.stats['num_accepted_steps']}")
    print(f"  num_rejected = {sol1.stats['num_rejected_steps']}")

    # # --- Diagnose which event condition stopped the integration ---
    # y_end = sol.evaluate(sol.ts[0]) * Y_ref
    # x_end, h1_end, h2_end, p_end = y_end[0], y_end[8], y_end[9], y_end[7]
    # out_end = nozzle_two_fluid_autonomous(0.0, y_end, params_model)
    # det_end = jnp.min(out_end.get("D"))
    # # h1_min, h1_max = fluid1.h_min, fluid1.h_max
    # # h2_min, h2_max = fluid2.h_min, fluid2.h_max
    # # p_min_d = jnp.maximum(fluid1.p_min, fluid2.p_min)
    # # p_max_d = jnp.minimum(fluid1.p_max, fluid2.p_max)
    # # cond_names = [
    # #     "x/L >= 0", "x/L <= L", "det + 10",
    # #     "h1 > h1_min", "h1 < h1_max",
    # #     "p > p_min", "p < p_max",
    # # ]
    # cond_vals = jnp.stack([
    #     x_end / params_model.length,
    #     1 - x_end / params_model.length,
    #     det_end + 10,
    #     # h1_end - h1_min - 10,
    #     # h1_max - h1_end + 10,
    #     # h2_end - h2_min - 10,
    #     # h2_max - h2_end + 10,
    #     # p_end - p_min_d - 10,
    #     # p_max_d - p_end + 10,
    # ])
    # print("\n--- Event diagnostics at termination ---")
    # print(f"  x_end = {float(x_end):.6f} m  (L = {params_model.length})")
    # print(f"  p_end = {float(p_end):.1f} Pa")
    # # print(f"  h1_end = {float(h1_end):.1f} J/kg  (table: {float(h1_min):.1f} – {float(h1_max):.1f})")
    # # print(f"  h2_end = {float(h2_end):.1f} J/kg  (table: {float(h2_min):.1f} – {float(h2_max):.1f})")
    # print(f"  det_end = {float(det_end):.4e}")
    # # for name, val in zip(cond_names, cond_vals):
    #     # marker = " <-- TRIGGERED" if float(val) <= 0 else ""
    #     # print(f"  {name:15s} = {float(val):+.4e}{marker}")

    # lb = jnp.array(t0)
    # ub = jnp.array(sol.ts[0])


    # # Initial guess
    # x0 = jnp.array(0.5 * (lb + ub))

    # # Use a stable optimizer
    # solver = optx.BFGS(rtol=1e-3, atol=1e-3)

    # # Run optimization (unscale before evaluating physics)
    # res = optx.minimise(
    #     lambda t, *args: nozzle_two_fluid_autonomous(
    #         0.0, sol.evaluate(t) * Y_ref, params_model
    #     )["D"],
    #     solver,
    #     y0=x0,  # A guess inside the bracket
    # )

    # t_max = res.value
    # yy_max = sol.evaluate(t_max) * Y_ref
    # f_max = nozzle_two_fluid_autonomous(0.0, yy_max, params_model)[
    #     "D"
    # ]
    # print(f_max)

    return sol1, Y_ref, params_model

# -----------------------------------------------------------------------------
# Converging-diverging nozzle example
# -----------------------------------------------------------------------------

table_dir = "fluid_tables"
fluid_name = ["water","nitrogen"]
backend = "HEOS"
h1_table = [0, 150e3]  # J/kg
h2_table = [90e3, 350e3]
p_table = [3e4, 3e6]   # Pa
N_h = 130
N_p = 130


@eqx.filter_jit
def _eval_model_batch(Y, params):
    return jax.lax.map(lambda y: nozzle_two_fluid_autonomous(0.0, y, params), Y)

if __name__ == "__main__":

    # # Define model parameters
    # fluid_name = "air"
    # fluid = jxp.perfect_gas.get_constants(fluid_name, T_ref=300, P_ref=101325)

    # -- 1. Find critical state with continuation --

    # ---------------------------
    # Build fluids
    # ---------------------------
    params_model = NozzleParams(
        p0_in=20e5,  # Pa
        T01_in = 22 + 273.15, 
        T02_in = 22 + 273.15,
        length=0.3, # m
        roughness=0e-6, # m
        Ma_low=0.95,
        Ma_high=1.2,
        Ma_in=0.001,
        u1_in=1e-12,
        heat_transfer=1.0,
        wall_friction=1.0,
        fluid1=jxp.FluidBicubic(
            fluid_name=fluid_name[0],
            backend="HEOS",
            h_min=h1_table[0],
            h_max=h1_table[1],
            p_min=p_table[0],
            p_max=p_table[1],
            N_h=N_h,
            N_p=N_p,
            table_dir=table_dir,
        ),
        fluid2=jxp.FluidBicubic(
            fluid_name=fluid_name[1],
            backend="HEOS",
            h_min=h2_table[0],
            h_max=h2_table[1],
            p_min=p_table[0],
            p_max=p_table[1],
            N_h=N_h,
            N_p=N_p,
            table_dir=table_dir,
        ),
        # fluid1=jxp.FluidJAX(
        #     fluid_name[0],
        #     backend="HEOS",
        # ),
        # fluid2=jxp.FluidJAX(
        #     fluid_name[1],
        #     backend="HEOS",
        # ),
        geometry=get_nozzle_elliot,
        # geometry=symmetric_nozzle_geometry,
        mixture_ratio=68,
        bubble_number=1e13,
        droplet_number=1e11,
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
        solver_name="Kvaerno3", # Tsit5 | Kvaerno3 | Dopri5 | ImplicitEuler
        adjoint_name="DirectAdjoint",
        number_of_points=60,
        rtol=1e-7,
        atol=1e-7,
    )

    # L_elliot = 0.2 # m
    # params_model = replace_param(params_model, "length", L_elliot)


    # Solve the problem
    print("\n" + "-" * 60)
    print("Evaluating transonic solution")
    print("-" * 60)
    # input_array = jnp.linspace(0.001,0.005,5)
    
    input_array = jnp.array([1.0]) # 3.306200761348009, 3062005937099457
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(input_array)))  # Generate colors
    solution_list = []
    out_list = []
    for i, p0_in in enumerate(input_array):
        t0 = time.perf_counter()
        params_model = replace_param(params_model, "u1_in", p0_in)
        
        # sol = transonic_nozzle_one_component(params_model, params_bvp, params_ivp)

        sol, Y_ref, params_model = transonic_nozzle_one_component_autonomus(params_model, params_bvp, params_ivp)

        # Relative error diagnostics
        dt_ms = (time.perf_counter() - t0) * 1e3

        # --- Post-processing: unscale dense output ---
        # 1. Fine τ grid to discover the x(τ) mapping
        n_fine = 20000
        ts = jnp.linspace(sol.t0, sol.ts[0], n_fine)
        # y_hat_fine = jax.vmap(sol.evaluate)(ts_fine)
        # x_of_tau = y_hat_fine[:, 0] * Y_ref[0]  # unscale x

        # 2. Build uniform x grid and interpolate back to τ
        # n_plot = 1000
        # x_uniform = jnp.linspace(float(x_of_tau[0]), float(x_of_tau[-1]), n_plot)
        # ts = jnp.interp(x_uniform, x_of_tau, ts_fine)

        # 3. Evaluate dense output at uniform-x points and unscale
        y_hat_uniform = jax.vmap(sol.evaluate)(ts)
        out2 = y_hat_uniform * Y_ref[None, :]  # unscale all components

        out_autonomus = out2
        sol2 = _eval_model_batch(out_autonomus, params_model)
        out = sol2

        mdot_error = (out["m_dot"].max() - out["m_dot"].min()) / out["m_dot"][0]
        h01_error = (out["h1"].max() - out["h1"].min()) / out["h1"][0]
        h02_error = (out["h2"].max() - out["h2"].min()) / out["h2"][0]
        s_error = (out["s1"].max() - out["s1"].min()) / out["s1"][0]

        print(
            f"Solution {i} | Solver status {sol.result} | "
            f"steps {int(sol.stats['num_steps']):3d} | "
            f"mdot error {mdot_error:0.2e} | h01 error {h01_error:0.2e} | "
            f"s_error {s_error:0.2e} | time {dt_ms:7.2f} ms"
        )
 
        solution_list.append(sol)
        out_list.append(out)

    # Create the figure
    fig = plt.figure(figsize=(5, 10))
    gs = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 3, 3, 1])
    # xg = solution_list[0].ys["x"]
    # rg = solution_list[0].ys["diameter"] / 2.0
    xg = out_list[0]["x"]
    rg = out_list[0]["diameter"] / 2.0
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    axs = [ax0, ax1, ax2, ax3, ax4]

    # --- row 1: pressure (bar): solid p0, dashed p ---
    axs[0].set_ylabel("Ma_mix (-)")
    for color, val, out in zip(colors, input_array, out_list):
        x = out["x"]
        # axs[0].plot(x, out["p"] * 1e-5, linestyle="-", color=color,
        #             label=rf"$p_{{0,\mathrm{{in}}}} = {val:0.3f}$")
        axs[0].plot(x, out["Ma_mix"], linestyle="-", color=color,
                    label=rf"$p_{{0,\mathrm{{in}}}} = {val:0.3f}$")    
        # axs[0].legend(loc="upper right", fontsize=7)

    # --- row 2: velocity ---
    axs[1].set_ylabel("Velocity (m/s)")
    for color, val, out in zip(colors, input_array, out_list):
        axs[1].plot(out["x"], out["v1"], color=color, label="water")
        axs[1].plot(out["x"], out["v2"], color=color, linestyle="--", label="nitrogen")
        axs[1].legend(loc="upper right", fontsize=7)

    # Static and stagnation enthalpy
    axs[2].set_ylabel("Diameter [m]")
    for color, val, out in zip(colors, input_array, out_list):
        axs[2].plot(out["x"], out["d_bubble"], color=color, label="bubble")
        axs[2].plot(out["x"], out["d_droplet"], color=color, linestyle="--", label="droplet")

    # Determinant
    axs[3].set_ylabel("Pressure")
    for color, val, out in zip(colors, input_array, out_list):
        axs[3].plot(out["x"], out["p"], color=color)

    # --- row 3: nozzle geometry ---
    axs[4].fill_between(xg, -rg, +rg, color="lightgrey")  # shaded nozzle
    x_closed = jnp.concatenate([xg, xg[::-1], xg[:1]])
    y_closed = jnp.concatenate([rg, -rg[::-1], rg[:1]])
    axs[4].plot(x_closed, y_closed, "k", linewidth=1.2)
    # r_abs_max = float(jnp.max(jnp.abs(rg)))
    # ax3.set_ylim(-1.5 * r_abs_max, 1.5 * r_abs_max)
    # axs[4].set_aspect("equal", adjustable="box")
    axs[4].set_xlabel("Axial coordinate x (m)")
    fig.tight_layout(pad=1)

    # fig = plt.figure(figsize=(5, 10))
    # plt.fill_between(xg, -rg, +rg, color="lightgrey")  # shaded nozzle
    # x_closed = jnp.concatenate([xg, xg[::-1], xg[:1]])
    # y_closed = jnp.concatenate([rg, -rg[::-1], rg[:1]])
    # plt.plot(x_closed, y_closed, "k", linewidth=1.2)
    # r_abs_max = float(jnp.max(jnp.abs(rg)))
    # ax3.set_ylim(-1.5 * r_abs_max, 1.5 * r_abs_max)
    # axs[4].set_aspect("equal", adjustable="box")
    # plt.set_xlabel("Axial coordinate x (m)")

    # print(out["eigenvalues"])

    # # --- p-h diagram for both fluids ---
    # fig_ph, (ax_ph1, ax_ph2) = plt.subplots(1, 2, figsize=(10, 5))

    # # Water saturation curve (liquid side)
    # import numpy as np
    # from CoolProp import CoolProp as CP
    # T_triple_w = CP.PropsSI("Ttriple", "Water")
    # T_crit_w = CP.PropsSI("Tcrit", "Water")
    # T_sat_w = np.linspace(T_triple_w + 1, T_crit_w - 1, 300)
    # h_sat_l_w = [CP.PropsSI("H", "T", T, "Q", 0, "Water") for T in T_sat_w]
    # p_sat_w = [CP.PropsSI("P", "T", T, "Q", 0, "Water") for T in T_sat_w]

    # ax_ph1.plot(np.array(h_sat_l_w) * 1e-3, np.array(p_sat_w) * 1e-5,
    #             "k--", linewidth=0.8, label="sat. liquid")
    # ax_ph1.set_title("Water (phase 1)")
    # ax_ph1.set_xlabel("Enthalpy (kJ/kg)")
    # ax_ph1.set_ylabel("Pressure (bar)")
    # for color, out in zip(colors, out_list):
    #     ax_ph1.plot(out["h1"] * 1e-3, out["p"] * 1e-5, color=color,
    #                 marker="o", markersize=2, linewidth=1.5)
    # ax_ph1.legend(fontsize=7)

    # # Nitrogen saturation curve (vapor side)
    # T_triple_n = CP.PropsSI("Ttriple", "Nitrogen")
    # T_crit_n = CP.PropsSI("Tcrit", "Nitrogen")
    # T_sat_n = np.linspace(T_triple_n + 1, T_crit_n - 1, 300)
    # h_sat_v_n = [CP.PropsSI("H", "T", T, "Q", 1, "Nitrogen") for T in T_sat_n]
    # p_sat_n = [CP.PropsSI("P", "T", T, "Q", 1, "Nitrogen") for T in T_sat_n]

    # ax_ph2.plot(np.array(h_sat_v_n) * 1e-3, np.array(p_sat_n) * 1e-5,
    #             "k--", linewidth=0.8, label="sat. vapor")
    # ax_ph2.set_title("Nitrogen (phase 2)")
    # ax_ph2.set_xlabel("Enthalpy (kJ/kg)")
    # ax_ph2.set_ylabel("Pressure (bar)")
    # for color, out in zip(colors, out_list):
    #     ax_ph2.plot(out["h2"] * 1e-3, out["p"] * 1e-5, color=color,
    #                 marker="o", markersize=2, linewidth=1.5)
    # ax_ph2.legend(fontsize=7)

    # fig_ph.tight_layout(pad=1)
    save_data = {"pass1": out}
    fname = (
        f"solution_elliot_subsonic"
        f"_{float(params_model.p0_in)/1e5:.1f}bar"
        f"_mr{float(params_model.mixture_ratio):.4g}"
        f"_nb{float(params_model.bubble_number):.2e}"
        f"_nd{float(params_model.droplet_number):.2e}"
        f"_u1_{float(params_model.u1_in):.1f}"
        f".pkl"
    )
    with open(os.path.join("tfm_solutions", fname), "wb") as f:
        pkl.dump(save_data, f)

    # Show figures
    plt.show()

    # # --- Differentiability check: mdot vs. p0_in ---
    # def mdot_vs_p0(p0_in):
    #     num_points = 50
    #     local_params = replace_param(params_model, "p0_in", p0_in)
    #     sol = transonic_nozzle_single_phase(local_params, params_bvp, params_ivp, fluid)
    #     return sol.ys["m_dot"][0]

    # # Base point
    # p0_in = 101325.0

    # # JAX derivative
    # mdot_val = mdot_vs_p0(p0_in)
    # mdot_grad = jax.grad(mdot_vs_p0)(p0_in)

    # # Finite difference derivative
    # h = 1.0  # small perturbation in Pa
    # fd_grad = (mdot_vs_p0(p0_in + h) - mdot_vs_p0(p0_in - h)) / (2 * h)

    # # Print results
    # print()
    # print(f" mdot(p0_in={p0_in:.3f}) = {mdot_val:.6e}")
    # print(f" JAX   d(mdot)/d(p0_in) = {mdot_grad:.6e}")
    # print(f" FD    d(mdot)/d(p0_in) = {fd_grad:.6e}")
    # print(f" Relative diff = {abs((mdot_grad - fd_grad) / fd_grad):.3e}")
