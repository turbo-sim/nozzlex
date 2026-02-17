# import os
# os.environ["JAX_TRACEBACK_FILTERING"] = "off"


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

from nozzlex.homogeneous_nonequilibrium.nozzle_model_core_classic import (
    nozzle_homogeneous_non_equilibrium_classic,
    get_speed_of_sound_mixture,
    symmetric_nozzle_geometry,
    get_nozzle_elliot_wrong
)

from nozzlex.homogeneous_nonequilibrium.nozzle_model_core import get_nozzle_elliot

from nozzlex.homogeneous_nonequilibrium.nozzle_model_solver_classic import (
    # nozzle_homogeneous_non_equilibrium_autonomous,
    eval_end_of_domain_event,
    # get_nozzle_elliot,
    NozzleParams,
    BVPSettings,
    IVPSettings,
    replace_param,
    # compute_static_state,
    eval_ode_rhs,
    eval_ode_full,
)

# v1 solves the ode system using the space marching in non-autonomous form
# v2 solves the autonomous system with events for the bounds of domain ends
# v3 uses the non-autonomous system and determines the inlet critical condition with collocation method


# -----------------------------------------------------------------------------
# Main API to the converging-diverging nozzle model
# -----------------------------------------------------------------------------
# @eqx.filter_jit
def transonic_nozzle_one_component(
    params_model: NozzleParams,
    params_bvp: BVPSettings,
    params_ivp: IVPSettings,
):
    """
    Transonic converging-diverging nozzle.
    First uses the root finding to find the critical (sonic) state,
    then integrates the ODE system with a blended RHS near the critical location.
    """

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
    alpha2 = 1 / (1 + ((1 - q_in) / q_in) * (rho1_in / rho2_in))
    alpha1 = 1 - alpha2

    # h = st["h"]
    G2, a2 = state2_in["G"], state2_in["a"]

    a_in_mix = get_speed_of_sound_mixture(G1, alpha1, a1, rho1_in, G2, alpha2, a2, rho2_in)

    u_in_crit = params_model.Ma_in * a_in_mix
    q_in = (1)/(1 + params_model.mixture_ratio)
    alpha2_in = 1 / (1 + ((1 - q_in) / q_in) * (rho2_in / rho1_in))
    alpha1_in = 1 - alpha2_in

    # --- Define helpers with closure over args_base and fluid ---
    def ode_full_subsonic(t, y, args):
        return nozzle_homogeneous_non_equilibrium_classic(t, y, args)

    def ode_rhs_subsonic(t, y, args):
        return ode_full_subsonic(t, y, args)["rhs"]

    # --- 2. First pass: inlet → Ma_low ---
    solver = jxp.make_diffrax_solver(params_ivp.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_ivp.adjoint_name)
    ctrl = dfx.PIDController(rtol=params_ivp.rtol, atol=params_ivp.atol,dtmax=1e-4)
    y_inlet = jnp.array([alpha1_in, alpha2_in, rho1_in, rho2_in, u_in_crit, p_in, h1_in, h2_in])
    # event1 = dfx.Event(
    #     cond_fn=eval_end_of_domain_event,
    #     root_finder=optx.Bisection(rtol=1e-6, atol=1e-6),
    # )
    # term1 = dfx.ODETerm(eval_ode_rhs)
    # save1 = dfx.SaveAt(t1=True, fn=ode_full_subsonic)
    # sol1 = dfx.diffeqsolve(
    #     term1,
    #     solver,
    #     t0=1e-9,
    #     t1=1,
    #     dt0=1e-8,
    #     y0=y_inlet,
    #     args=params_model,
    #     saveat=save1,
    #     stepsize_controller=ctrl,
    #     adjoint=adjoint,
    #     event=event1,
    #     max_steps=20_000,
    # )

    t0, t1 = 1e-9, params_model.length
    solver = jxp.make_diffrax_solver(params_ivp.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_ivp.adjoint_name)
    term = dfx.ODETerm(eval_ode_rhs)
    ctrl = dfx.PIDController(rtol=params_ivp.rtol, atol=params_ivp.atol) #,dtmin=1e-40) # , dtmin=1e-9) #,dtmax=1e-4)

    # --- event: stop at nozzle exit ---
    event = dfx.Event(
        cond_fn=eval_end_of_domain_event,
        root_finder=optx.Newton(rtol=1e-0, atol=1e-0),
    )



    # --- first solve (find domain end) ---
    saveat = dfx.SaveAt(t1 = True, dense = True, fn=eval_ode_full)
    # saveat = dfx.SaveAt(dense = True, fn=eval_ode_full)
    sol1 = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=1e-4,
        y0=y_inlet,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=saveat,
        event=event,
        max_steps=200_000,
        # throw=False,
    )
    jax.debug.print("last_x={t1}", t1=sol1.ts)

    # TODO: It is not stopping at 1-Ma=0!!! And is prinjting all the solutions!!!
    # TODO: it seems works. Implement distance method

    # ts = jnp.linspace(t0, sol1.ts[-1], 1000)
    # saveat = dfx.SaveAt(ts=ts, fn=eval_ode_full)
    # saveat = dfx.SaveAt(t1=True, dense=True, fn=eval_ode_full)
    # sol1 = dfx.diffeqsolve(
    #     term,
    #     solver,
    #     t0=t0,
    #     t1=sol1.ts[0]*1.01,
    #     dt0=1e-9,
    #     y0=y_inlet,
    #     args=params_model,
    #     stepsize_controller=ctrl,
    #     adjoint=adjoint,
    #     saveat=saveat,
    #     event=event,
    #     max_steps=200_000,
    #     # throw=False,
    # )

    # jax.debug.print("last_x={t1}", t1=sol1.ts[0])


#     # --- 3. Linearization close to critical point ---
#     x_crit = sol1.ys["x"][-1]
#     y_crit = jnp.array([sol1.ys["p"][-1], sol1.ys["v"][-1], sol1.ys["h"][-1]])
#     y_crit = jnp.array([alpha1_in, alpha2_in, rho1_in, rho2_in, u_in_crit, p_in, h1_in, h2_in])
#     y_crit = jnp.array([sol1.ys["alpha1"][-1],
#                         sol1.ys["alpha2"][-1],
#                         sol1.ys["rho1"][-1],
#                         sol1.ys["rho2"][-1],
#                         sol1.ys["u"][-1],
#                         sol1.ys["p"][-1],
#                         sol1.ys["h1"][-1],
#                         sol1.ys["h2"][-1]])
#     f_star, Jy, Jt = _linearize_rhs_at(x_crit, y_crit, params_model)



# #  TODO from here
#     def ode_full_transonic(t, y, args):
#         base = nozzle_homogeneous_non_equilibrium_core(t, y, args)
#         u = base["u"]
#         rhs_true = base["rhs"]
#         rhs_lin = f_star + Jy @ (y - y_crit) + Jt * (t - x_crit)

#         # Start blending slightly after sonic conditions, finish at Ma_high
#         u_start = u * 1.001
#         u_end = args.u_high
#         s = (u - u_start) / (u_end - u_start)
#         w = _smoothstep(jnp.clip(s, 0.0, 1.0))  # smoothly 0→1 in that narrow range
#         rhs_blend = (1.0 - w) * rhs_lin + w * rhs_true
#         in_window = (u >= args.u_low) & (u <= args.u_high)
#         rhs_blend = jnp.where(in_window, rhs_blend, rhs_true)

#         return {**base, "rhs": rhs_true, "rhs_blend": rhs_blend}

#     def ode_rhs_transonic(t, y, args):
#         return ode_full_transonic(t, y, args)["rhs_blend"]

#     # --- 4. Second pass ---
#     # TODO, perhaps it is more robust to do 1 pass for each segment instead of attempting a single pass?
#     ts2 = jnp.linspace(1e-9, params_model.length, params_ivp.number_of_points)
#     term2 = dfx.ODETerm(ode_rhs_transonic)
#     save2 = dfx.SaveAt(ts=ts2, t1=True, fn=ode_full_transonic)
#     sol2 = dfx.diffeqsolve(
#         term2,
#         solver,
#         t0=1e-9,
#         t1=params_model.length,
#         dt0=1e-9,
#         y0=y_inlet,
#         args=params_model,
#         saveat=save2,
#         stepsize_controller=ctrl,
#         adjoint=adjoint,
#         max_steps=20_000,
#     )
    # return sol2
    return sol1


def _velocity_event_cond(t, y, args, **kwargs):
    """Event function: zero when M^2 - Ma_low^2 = 0."""
    # _, _, _, _, u, _, _, _ = y
    out = nozzle_homogeneous_non_equilibrium_classic(t,y, args)
    # a = args.fluid.get_state(jxp.HmassP_INPUTS, h, p)["a"]
    # Ma_sqr = (v / a)**2
    # u_sqr = u ** 2
    # return u_sqr - args.u_low**2
    return out["Ma_mix"] - params_model.Ma_low

# def eval_end_of_domain_event(t, y, args, **kwargs):
#     x = y[0]
#     L = args.length
#     jax.debug.print("y={y}",y=y)
#     # Determinant condition
#     out = nozzle_homogeneous_non_equilibrium_autonomous(0.0, y, args)
#     Ma_mix = jnp.min(out.get("Ma_mix"))
    
#     M = 5 - Ma_mix
#     # jax.debug.print("x={x}, L-x={leng},D={D}",x=x, leng=L-x, D = out.get("D"))

#     # Stack all event conditions
#     conditions = jnp.stack([x/L, 1 - x/L, M]) # D])
#     output = jnp.min(conditions)
#     # jax.debug.print("out={output}",output=output)
#     return output


def _linearize_rhs_at(x_star, y_star, model):
    """Return Taylor expansion coefficients of RHS around (x_star, y_star)."""
    rhs_fn = lambda xx, yy: nozzle_homogeneous_non_equilibrium_classic(xx, yy, model)["rhs"]
    f_star = rhs_fn(x_star, y_star)  # RHS at expansion point
    Jy = jax.jacrev(lambda yy: rhs_fn(x_star, yy))(y_star)  # ∂f/∂y
    Jt = jax.jacrev(lambda xx: rhs_fn(xx, y_star))(x_star)  # ∂f/∂t
    return f_star, Jy, Jt


def _smoothstep(x):
    x = jnp.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

    # return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


# -----------------------------------------------------------------------------
# Converging-diverging nozzle example
# -----------------------------------------------------------------------------

table_dir = "fluid_tables"
fluid_name = ["water","nitrogen"]
backend = "HEOS"
h1_table = [80e3, 100e3]  # J/kg
h2_table = [290e3, 310e3]
p_table = [3e4, 3e6]   # Pa
N_h = 120
N_p = 120

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
        length=0.25,  # m
        roughness=0e-6,  # m
        Ma_low=0.95,
        Ma_high=1.05,
        heat_transfer=0.0,
        wall_friction=0.0,
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
        mixture_ratio=68
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
        solver_name="Kvaerno3",
        adjoint_name="DirectAdjoint",
        number_of_points=60,
        rtol=1e-3,
        atol=1e-3,
        )

    # L_elliot = 0.25 # m
    # params_model = replace_param(params_model, "length", L_elliot)


    # Solve the problem
    print("\n" + "-" * 60)
    print("Evaluating transonic solution")
    print("-" * 60)
    # input_array = jnp.linspace(0.001,0.005,5)
    input_array = jnp.array([0.005])
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(input_array)))  # Generate colors
    solution_list = []
    out_list = []
    for i, p0_in in enumerate(input_array):
        t0 = time.perf_counter()
        params_model = replace_param(params_model, "Ma_in", p0_in)
        sol = transonic_nozzle_one_component(params_model, params_bvp, params_ivp)

        # Relative error diagnostics
        dt_ms = (time.perf_counter() - t0) * 1e3
        out = sol.ys
        out2 = []
        sol2 = []
        ts = jnp.linspace(sol.t0, sol.ts[0], 1)
        # for t in ts:
        #     out2.append(sol.evaluate(t))  # vmap instead of for loop
        #     sol2.append(nozzle_homogeneous_non_equilibrium_classic(t, out2[-1], params_model))
        # # print(sol2)

        # Vectorize the first step: sol.evaluate(t) for all t in ts
        evaluate_all = jax.vmap(sol.evaluate)
        out2 = evaluate_all(ts)

        # Vectorize the second step: apply the model to each (t, out)
        model_all = jax.vmap(lambda t, y: nozzle_homogeneous_non_equilibrium_classic(t, y, params_model))
        sol2 = model_all(ts, out2)
        out = sol2
        # print(sol2)

        # keys = sol2[0].keys()
        # # Build a dict of stacked JAX arrays (like a JAX-native DataFrame)
        # jax_table = {
        #     k: jnp.stack([d[k] for d in sol2]) 
        #     for k in keys if jnp.isscalar(sol2[0][k]) or sol2[0][k].shape == ()
        # }
        # out = jax_table
        mdot_error = (out["m_dot"].max() - out["m_dot"].min()) / out["m_dot"][0]
        h01_error = (out["h1"].max() - out["h1"].min()) / out["h1"][0]
        h02_error = (out["h2"].max() - out["h2"].min()) / out["h2"][0]
        s_error = (out["s1"].max() - out["s1"].min()) / out["s1"][0]

        print(
            f"Solution {i} | Solver status {sol.result._value:2d} | "
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
    axs[0].set_ylabel("Pressure (bar)")
    for color, val, out in zip(colors, input_array, out_list):
        x = out["x"]
        axs[0].plot(x, out["p"] * 1e-5, linestyle="--", color=color)
        axs[0].plot(
            x,
            out["p"] * 1e-5,
            linestyle="-",
            color=color,
            marker="o",
            markersize="3",
            label = rf"$p_{{0,\mathrm{{in}}}} = {val:0.3f}$"
        )
    # axs[0].legend(loc="upper right", fontsize=7)

    # --- row 2: Mach number ---
    axs[1].set_ylabel("Velocity (m/s)")
    for color, val, out in zip(colors, input_array, out_list):
        axs[1].plot(
            out["x"],
            out["v"],
            color=color,
            marker="o",
            markersize="3",
        )

    # Static and stagnation enthalpy
    axs[2].set_ylabel("Mixture Mach (-)")
    for color, val, out in zip(colors, input_array, out_list):
        # axs[2].plot(sol.ys["x"], sol.ys["h"], color=r["color"], markersize=3, marker"o")
        axs[2].plot(out["x"], out["Ma_mix"], color=color, markersize=3, marker="o")

    # Entropy
    axs[3].set_ylabel("Entropy 2 (J/kg/K)")
    for color, val, out in zip(colors, input_array, out_list):
        axs[3].plot(out["x"],out["s2"], color=color, markersize=3, marker="o")

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
