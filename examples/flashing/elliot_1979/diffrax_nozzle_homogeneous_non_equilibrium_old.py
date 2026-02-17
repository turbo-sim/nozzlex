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

from nozzlex.homogeneous_nonequilibrium.nozzle_model_core import (
    nozzle_homogeneous_non_equilibrium_core,
    nozzle_homogeneous_non_equilibrium_autonomous,
    get_nozzle_elliot,
    get_speed_of_sound_mixture,
    get_nozzle_elliot_wrong
)

from nozzlex.homogeneous_nonequilibrium.nozzle_model_solver import (
    nozzle_homogeneuous_non_equilibrium_mach_out,
    NozzleParams,
    BVPSettings,
    IVPSettings,
    replace_param,
    compute_critical_inlet,
    eval_ode_full,
    eval_ode_rhs,
    eval_end_of_domain_event,
    eval_ode_mach,
    eval_end_of_domain_event_classic
)

# v1 solves the ode system using the space marching in non-autonomous form
# v2 solves the autonomous system with events for the bounds of domain ends
# v3 uses the non-autonomous system and determines the inlet critical condition with collocation method


# -----------------------------------------------------------------------------
# Main API to the converging-diverging nozzle model
# -----------------------------------------------------------------------------
@eqx.filter_jit
def transonic_nozzle_two_components(
    params_model: NozzleParams,
    params_bvp: BVPSettings,
    params_ivp: IVPSettings,
):
    """
    Transonic converging-diverging nozzle.
    First uses the root finding to find the critical (sonic) state,
    then integrates the ODE system with a blended RHS near the critical location.
    """
    

    Ma_in_cr = compute_critical_inlet(Ma_lower=0.001, Ma_upper=0.05, params_model=params_model, params_solver=params_ivp)
    # Ma_in_cr = 0.031535542812347404
    # Ma_in_cr = Ma_in_cr * 0.995
    jax.debug.print("{M}", M=Ma_in_cr)

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
    G1, a1 = state1_in["G"], state1_in["a"]
    q_in = (1)/(1 + params_model.mixture_ratio)
    alpha2_in = 1 / (1 + ((1 - q_in) / q_in) * (rho2_in / rho1_in))
    alpha1_in = 1 - alpha2_in

    # h = st["h"]
    G2, a2 = state2_in["G"], state2_in["a"]

    a_in_mix = get_speed_of_sound_mixture(G1, alpha1_in, a1, rho1_in, G2, alpha2_in, a2, rho2_in)

    u_in_crit = Ma_in_cr * a_in_mix

    x_in = 1e-9

    y_inlet = jnp.array([x_in, alpha1_in, alpha2_in, rho1_in, rho2_in, u_in_crit, p_in, h1_in, h2_in])


    # --- 2. First pass: inlet → Ma_low ---
    t0, t1 = 1e-9, 1e9
    solver = jxp.make_diffrax_solver(params_ivp.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_ivp.adjoint_name)
    term = dfx.ODETerm(eval_ode_rhs)
    ctrl = dfx.PIDController(rtol=params_ivp.rtol, atol=params_ivp.atol)

    event1 = dfx.Event(
        cond_fn=_velocity_event_cond,
        root_finder=optx.Bisection(rtol=1e-9, atol=1e-9),
    )

    saveat = dfx.SaveAt(t1 = True,dense=True, fn=eval_ode_full)
    sol1 = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=1e-9,
        y0=y_inlet,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=saveat,
        event=event1,
        max_steps=200_000,
        # throw=False,
    )


    # --- 3. Linearization close to critical point ---
    # TODO check that x_crit and y_crit make sense
    x_crit = sol1.ys["x"][0]
    # jax.debug.print("t={t}",t=sol1.ys)
    y_crit = jnp.array([sol1.ys["alpha1"][0],
                        sol1.ys["alpha2"][0],
                        sol1.ys["rho1"][0],
                        sol1.ys["rho2"][0],
                        sol1.ys["v"][0],
                        sol1.ys["p"][0],
                        sol1.ys["h1"][0],
                        sol1.ys["h2"][0]])
    

    f_star, Jy, Jx = _linearize_rhs_at(x_crit, y_crit, params_model)


    # ------------------------------------------------------------------------------------------------

    # TODO from here
    # @eqx.filter_jit
    def ode_full_transonic(t, y, args):
        base = nozzle_homogeneous_non_equilibrium_core(t, y, args)
        Ma = base["Ma_mix"]
        rhs_true = base["rhs"]
        rhs_lin = f_star + Jy @ (y - y_crit) + Jx * (t - x_crit)

        # Start blending slightly after sonic conditions, finish at Ma_high
        Ma_start = 1.001
        Ma_end = args.Ma_high
        s = (Ma - Ma_start) / (Ma_end - Ma_start)
        w = _smoothstep(jnp.clip(s, 0.0, 1.0))  # smoothly 0→1 in that narrow range
        rhs_blend = (1.0 - w) * rhs_lin + w * rhs_true
        in_window = (Ma >= args.Ma_low) & (Ma <= args.Ma_high)
        rhs_blend = jnp.where(in_window, rhs_blend, rhs_true)

        return {**base, "rhs_blend": rhs_blend}
 
 
    
    # def ode_full_transonic_test(t, y, args):
    #     jax.debug.print("x={x} | y={y}", x=t, y=y)
    #     base = nozzle_homogeneous_non_equilibrium_core(t, y, args)
    #     Ma = base["Ma_mix"]
    #     rhs_true = base["rhs"]
    #     rhs_lin = f_star + Jy @ (y - y_crit) + Jx * (t - x_crit)

    #     # Start blending slightly after sonic conditions, finish at Ma_high
    #     Ma_start = 1.001
    #     Ma_end = args.Ma_high
    #     s = (Ma - Ma_start) / (Ma_end - Ma_start)
    #     w = _smoothstep(jnp.clip(s, 0.0, 1.0))  # smoothly 0→1 in that narrow range
    #     rhs_blend = (1.0 - w) * rhs_lin + w * rhs_true
    #     # in_window = (Ma >= args.Ma_low) & (Ma <= args.Ma_high)
    #     # blend_window = (Ma >= Ma_start) & (Ma <= args.Ma_high)
    #     in_window = (Ma >= args.Ma_low) & (Ma <= args.Ma_high)
    #     blend_window = (Ma >= Ma_start) & (Ma <= args.Ma_high)
    #     # jax.debug.print("x={x} | Ma={M} | w={w} | in_wind={iw} | bl_wind={bw}", x=x, M=Ma, w=w, iw=in_window, bw=blend_window)
    #     # jax.debug.print("Ma={M} | w={w} | in_wind={iw}", M=Ma, w=w, iw=in_window)
    #     rhs_blend = jnp.where(in_window, jnp.where(blend_window, rhs_blend, rhs_lin), rhs_true)

    #     return {**base, "rhs": rhs_true, "rhs_blend": rhs_blend}


    def ode_rhs_transonic(t, y, args):
        return ode_full_transonic(t, y, args)["rhs_blend"]



    # --- 4. Second pass ---
    y_inlet = jnp.array([alpha1_in, alpha2_in, rho1_in, rho2_in, u_in_crit, p_in, h1_in, h2_in])

     
    x0, x1 = 1e-9, 100.0

    solver = jxp.make_diffrax_solver(params_ivp.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_ivp.adjoint_name)

    ctrl = dfx.PIDController(rtol=params_ivp.rtol, atol=params_ivp.atol)

    event1 = dfx.Event(
        cond_fn=eval_end_of_domain_event_classic,
        root_finder=optx.Bisection(rtol=1e-6, atol=1e-6),
    )

    saveat = dfx.SaveAt(t1 = True,dense=True, fn=ode_full_transonic)
    term = dfx.ODETerm(ode_rhs_transonic)
    # term = dfx.ODETerm(ode_rhs_transonic_test)
    # saveat = dfx.SaveAt(t1 = True,dense=True, fn=ode_full_transonic_test)
    sol2 = dfx.diffeqsolve(
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
        event=event1,
        max_steps=200_000,
        # throw=False,
    )

    xs = jnp.linspace(sol2.t0, sol2.ts[0], 1000)

    evaluate_all = jax.vmap(sol2.evaluate)
    ys = evaluate_all(xs)

    # Vectorize the second step: apply the model to each (t, out)
    model_all = jax.vmap(lambda x, y: nozzle_homogeneous_non_equilibrium_core(x, y, params_model))
    sol_out = model_all(xs, ys)

    return sol_out, sol2


def _velocity_event_cond(t, y, args, **kwargs):
    """Event function: zero when M^2 - Ma_low^2 = 0."""
    # _, _, _, _, u, _, _, _ = y
    mach_out = nozzle_homogeneous_non_equilibrium_autonomous(t, y, args)["Ma_mix"]
    # a = args.fluid.get_state(jxp.HmassP_INPUTS, h, p)["a"]
    # Ma_sqr = (v / a)**2
    # u_sqr = u ** 2
    # return u_sqr - args.u_low**2
    return mach_out - (params_model.Ma_low)



def _linearize_rhs_at(x_star, y_star, model):
    """Return Taylor expansion coefficients of RHS around (x_star, y_star)."""
    # rhs_fn = lambda xx, yy: nozzle_homogeneous_non_equilibrium_autonomous(xx, yy, model)["rhs_autonomous"]
    rhs_fn = lambda xx, yy: nozzle_homogeneous_non_equilibrium_core(xx, yy, model)["rhs"]
    f_star = rhs_fn(x_star, y_star)  # RHS at expansion point
    Jy = jax.jacrev(lambda yy: rhs_fn(x_star, yy))(y_star)  # ∂f/∂y
    Jx = jax.jacrev(lambda xx: rhs_fn(xx, y_star))(x_star)  # ∂f/∂x
    return f_star, Jy, Jx


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

    # Define model parameters

    # ---------------------------
    # Build fluids
    # ---------------------------
    params_model = NozzleParams(
        p0_in=20e5,  # Pa
        T01_in = 22 + 273.15, 
        T02_in = 22 + 273.15,
        length=0.250,  # m
        roughness=0e-6,  # m
        Ma_low=0.95,
        Ma_high=1.05,
        Ma_in=0.002,
        heat_transfer=0.0,
        wall_friction=0.0,
        mixture_ratio=68,
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
        solver_name="Kvaerno3",  #"Kvaerno3",
        adjoint_name="DirectAdjoint",
        number_of_points=60,
        rtol=1e-8,
        atol=1e-8,
    )

    # Solve the problem
    print("\n" + "-" * 60)
    print("Evaluating transonic solution")
    print("-" * 60)
    input_array = jnp.asarray([70])
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(input_array)))  # Generate colors
    solution_list = []

    # Solution speeds up 3 times after first compilation

    for i, r_in in enumerate(input_array):
        t0 = time.perf_counter()

        # Solving the nozzle flowfield
        params_model = replace_param(params_model, "mixture_ratio", r_in)
        out, sol = transonic_nozzle_two_components(params_model, params_bvp, params_ivp)

        # Relative error diagnostics
        dt_ms = (time.perf_counter() - t0) * 1e3
        mdot_error = (out["m_dot"][0]- out["m_dot"][-1]) / out["m_dot"][-1]

        print(
            f"Solution {i+1} | Solver status {sol.result._value:2d} | "
            f"steps {int(sol.stats['num_steps']):3d} | "
            f"mdot error {mdot_error:0.2e} | time {dt_ms:7.2f} ms"
        )

        solution_list.append(out)

    # Create the figure
    fig = plt.figure(figsize=(8, 10))
    
    gs = gridspec.GridSpec(3, 2,) # height_ratios=[3, 3, 3, 3, 3, 3, ])
    # xg = solution_list[0].ys["x"]
    # rg = solution_list[0].ys["diameter"] / 2.0
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    ax5 = fig.add_subplot(gs[5], sharex=ax0)
    axs = [ax0, ax1, ax2, ax3, ax4, ax5]

    # ax0.plot(out["x"], out["diameter"])
    # ax1.plot(out["x"], out["Ma_mix"])

    
    # --- plot 1: pressure (bar): p ---
    axs[0].set_ylabel("Pressure (bar)")
    for color, val, sol in zip(colors, input_array, solution_list):
        x = sol["x"]
        # axs[0].plot(x, sol["p0"] * 1e-5, linestyle="--", color=color)
        axs[0].plot(
            x,
            sol["p"] * 1e-5,
            linestyle="-",
            color=color,
            label = rf"$R_{{0,\mathrm{{in}}}} = {val}$"
        )
    axs[0].legend(loc="upper right", fontsize=7)

    # --- plot 2: Mach number ---
    axs[1].set_ylabel("Mach mix (-)")
    for color, val, sol in zip(colors, input_array, solution_list):
        axs[1].plot(
            sol["x"],
            sol["Ma_mix"],
            color=color,
        )

    # --- plot 3: Temperature ---
    axs[2].set_ylabel("Temperature (K)")

    for color, val, sol in zip(colors, input_array, solution_list):
        axs[2].plot(sol["x"], sol["T1"], color=color, linestyle="-")
        axs[2].plot(sol["x"], sol["T2"], color=color, linestyle="--")

    # --- plot 4: Velocity ---
    axs[3].set_ylabel("Velocity (m/s)")

    # # Create a twin y-axis that shares the same x-axis
    # twinx2 = axs[2].twinx()
    # twinx2.set_ylabel("Enthalpy 2 (J/kg)", color='tab:red')

    for color, val, sol in zip(colors, input_array, solution_list):
        # axs[2].plot(sol.ys["x"], sol.ys["h"], color=r["color"], markersize=3, marker"o")
        # twinx2.plot(sol["x"], sol["h2"], color=color, linestyle="--")
        axs[3].plot(sol["x"], sol["v"], color=color, linestyle="-")


    # --- plot 5: void fraction ---
    axs[4].set_ylabel("Void fraction (-)")

    for color, val, sol in zip(colors, input_array, solution_list):
        axs[4].plot(sol["x"], sol["alpha1"], color=color, linestyle="-")
        axs[4].plot(sol["x"], sol["alpha2"], color=color, linestyle="--")

    # --- plot 6: Geometry ---
    axs[5].set_ylabel("Radius (m)")
    axs[5].plot(sol["x"], sol["diameter"]/2, color="k", linestyle="-")
    axs[5].set_ylim(0.0)

    fig.tight_layout(pad=1)

    plt.show()
