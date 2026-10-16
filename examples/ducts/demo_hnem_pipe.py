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

from nozzlex.homogeneous_nonequilibrium.nozzle_model_solver import (
    NozzleParams,
    BVPSettings,
    IVPSettings,
    replace_param,
)

from nozzlex.homogeneous_nonequilibrium.nozzle_model_core import (
    get_pipe,
    nozzle_homogeneous_non_equilibrium_one_component_core,
    nozzle_homogeneous_non_equilibrium_one_component_autonomous,
    get_speed_of_sound_mixture,
)

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

    fluid_vapor = params_model.fluid1
    fluid_liquid = params_model.fluid2

    # print(params_model.p0_in)
    print(params_model.T01_in)
    # ff

    # state_liquid_in = fluid_liquid.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T01_in)
    # state_liquid_in = fluid_eq.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T01_in)
    # h_l_in = state_liquid_in["h"]
    # state_liquid_in = fluid_liquid.get_state(jxp.HmassP_INPUTS, h_l_in, params_model.p0_in)
    # state_vapor_in = fluid_vapor.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T02_in)
    state_vapor_in = fluid_vapor.get_state_saturation(params_model.p0_in)
    state_liquid_in = fluid_liquid.get_state_saturation(params_model.p0_in)
    hin_l = state_liquid_in["enthalpy"]
    hin_v = state_vapor_in["enthalpy"]
    G = state_liquid_in["gruneisen"]
    # print(G)
    state_liquid_in = fluid_liquid.get_state(jxp.HmassP_INPUTS, hin_l+100e3, params_model.p0_in)
    # state_vapor_in = fluid_vapor.get_state(jxp.HmassP_INPUTS, hin_v, params_model.p0_in)

    G = state_liquid_in["T"]
    print(G)
    
    p_in, rho_l_in, rho_v_in, h_l_in, h_v_in = (
        state_liquid_in["pressure"],
        state_liquid_in["density"],
        state_vapor_in["density"],
        state_liquid_in["enthalpy"],
        state_vapor_in["enthalpy"]
    )

    jax.debug.print("p_in:{} | rho_l_in:{} | h_l_in:{}",p_in, rho_l_in, h_l_in)


    Gl, al = state_liquid_in["gruneisen"], state_liquid_in["speed_of_sound"]
    alpha_v_in = 0.1
    alpha_l_in = 1.0 - alpha_v_in

    # h = st["h"]
    Gv, av = state_vapor_in["gruneisen"], state_vapor_in["speed_of_sound"]

    a_in_mix = get_speed_of_sound_mixture(Gl, alpha_v_in, al, rho_l_in, Gv, alpha_l_in, av, rho_v_in)

    u_in_crit = params_model.Ma_in * a_in_mix




    # --- Define helpers with closure over args_base and fluid ---
    def ode_full_autonomous(t, y, args):
        return nozzle_homogeneous_non_equilibrium_one_component_autonomous(t, y, args)

    def eval_ode_rhs_autonomous(t, y, args):
        return ode_full_autonomous(t, y, args)["rhs_autonomous"]

    # --- 2. First pass: inlet → Ma_low ---
    # solver = jxp.make_diffrax_solver(params_ivp.solver_name)
    # adjoint = jxp.make_diffrax_adjoint(params_ivp.adjoint_name)
    # ctrl = dfx.PIDController(rtol=params_ivp.rtol, atol=params_ivp.atol,dtmax=1e-4)
    y_inlet = jnp.array([1e-9, alpha_v_in, alpha_l_in, rho_v_in, rho_l_in, u_in_crit, p_in, h_v_in, h_l_in])


    t0, t1 = 1e-9, 1e9
    # t1, t0 = 0.0, 1.0 # not working!
    solver = jxp.make_diffrax_solver(params_ivp.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_ivp.adjoint_name)
    term = dfx.ODETerm(eval_ode_rhs_autonomous)
    ctrl = dfx.PIDController(rtol=params_ivp.rtol, atol=params_ivp.atol) # , dtmin=1e-9) #,dtmax=1e-4)

    event = dfx.Event(
        cond_fn=eval_end_of_domain_event,
        root_finder=optx.Newton(rtol=1e-6, atol=1e-6),
    )

    # --- first solve (find domain end) ---
    saveat = dfx.SaveAt(t1 = True, dense=True, fn=ode_full_autonomous)
    # saveat = dfx.SaveAt(dense = True, fn=eval_ode_full)
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
        event=event,
        max_steps=nsteps,
        throw=False,
    )

    return sol1



def transonic_nozzle_one_component_classic(
    params_model: NozzleParams,
    params_bvp: BVPSettings,
    params_ivp: IVPSettings,
):
    """
    Transonic converging-diverging nozzle.
    First uses the root finding to find the critical (sonic) state,
    then integrates the ODE system with a blended RHS near the critical location.
    """

    fluid_vapor = params_model.fluid1
    fluid_liquid = params_model.fluid2

    print(params_model.p0_in)
    print(params_model.T01_in)

    state_liquid_in = fluid_liquid.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T01_in)
    # state_liquid_in = fluid_liquid.get_state(jxp.HmassP_INPUTS, -85150.37327238737, params_model.p0_in)
    # state_vapor_in = fluid_vapor.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T02_in)
    state_vapor_in = fluid_vapor.get_state_saturation(params_model.p0_in)


    p_in, rho_l_in, rho_v_in, h_l_in, h_v_in = (
        state_liquid_in["p"],
        state_liquid_in["rho"],
        state_vapor_in["density"],
        state_liquid_in["h"],
        state_vapor_in["enthalpy"]
    )


    Gl, al = state_liquid_in["G"], state_liquid_in["a"]
    alpha_v_in = 1e-6
    alpha_l_in = 1.0 - alpha_v_in

    # h = st["h"]
    Gv, av = state_vapor_in["gruneisen"], state_vapor_in["speed_of_sound"]

    a_in_mix = get_speed_of_sound_mixture(Gl, alpha_v_in, al, rho_l_in, Gv, alpha_l_in, av, rho_v_in)

    u_in_crit = params_model.Ma_in * a_in_mix




    # --- Define helpers with closure over args_base and fluid ---
    def ode_full_classic(t, y, args):
        return nozzle_homogeneous_non_equilibrium_one_component_core(t, y, args)

    def eval_ode_rhs_classic(t, y, args):
        return ode_full_classic(t, y, args)["rhs"]

    # --- 2. First pass: inlet → Ma_low ---
    # solver = jxp.make_diffrax_solver(params_ivp.solver_name)
    # adjoint = jxp.make_diffrax_adjoint(params_ivp.adjoint_name)
    # ctrl = dfx.PIDController(rtol=params_ivp.rtol, atol=params_ivp.atol,dtmax=1e-4)
    y_inlet = jnp.array([alpha_v_in, alpha_l_in, rho_v_in, rho_l_in, u_in_crit, p_in, h_v_in, h_l_in])


    t0, t1 = 1e-9, 1.0
    # t1, t0 = 0.0, 1.0 # not working!
    solver = jxp.make_diffrax_solver(params_ivp.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_ivp.adjoint_name)
    term = dfx.ODETerm(eval_ode_rhs_classic)
    ctrl = dfx.PIDController(rtol=params_ivp.rtol, atol=params_ivp.atol) # , dtmin=1e-9) #,dtmax=1e-4)

    event = dfx.Event(
        cond_fn=eval_end_of_domain_event_classic,
        root_finder=optx.Newton(rtol=1e-6, atol=1e-6),
    )

    # --- first solve (find domain end) ---
    saveat = dfx.SaveAt(t1 = True, dense=True, fn=ode_full_classic)
    # saveat = dfx.SaveAt(dense = True, fn=eval_ode_full)
    sol1 = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=1e-12,
        y0=y_inlet,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=saveat,
        event=event,
        max_steps=nsteps,
        throw=False,
    )

    return sol1


def eval_end_of_domain_event(t, y, args, **kwargs):
    x = y[0]
    L = args.length

    # Determinant condition
    out = nozzle_homogeneous_non_equilibrium_one_component_autonomous(0.0, y, args)
    Ma_mix = jnp.min(out.get("Ma_mix"))

    M = 150.0 - Ma_mix
    jax.debug.print("x={x}, L-x={leng},M={D}",x=x, leng=L-x, D = out.get("Ma_mix"))
    # M = 5
    # jax.debug.print("Not considering mach as event")
    # Stack all event conditions
    # L = 0.15
    conditions = jnp.stack([x / L, 1 - x / L, M])  # D])
    output = jnp.min(conditions)

    # jax.debug.print("out={output}",output=output)
    return output



def eval_end_of_domain_event_classic(t, y, args, **kwargs):
    L = args.length
    # Determinant condition
    x=t
    out = nozzle_homogeneous_non_equilibrium_one_component_core(t, y, args)
    Ma_mix = jnp.min(out.get("Ma_mix"))

    M = 5.0 - Ma_mix
    # jax.debug.print("x={x}, L-x={leng},M={D}",x=x, leng=L-x, D = out.get("Ma_mix"))
    # M = 5
    # jax.debug.print("Not considering mach as event")
    # Stack all event conditions
    # L = 0.15
    conditions = jnp.stack([x / L, 1 - x / L, M])  # D])
    output = jnp.min(conditions)
    # jax.debug.print("out={output}",output=output)
    return output




# def _linearize_rhs_at(x_star, y_star, model):
#     """Return Taylor expansion coefficients of RHS around (x_star, y_star)."""
#     rhs_fn = lambda xx, yy: nozzle_homogeneous_non_equilibrium_core(xx, yy, model)["rhs"]
#     f_star = rhs_fn(x_star, y_star)  # RHS at expansion point
#     Jy = jax.jacrev(lambda yy: rhs_fn(x_star, yy))(y_star)  # ∂f/∂y
#     Jt = jax.jacrev(lambda xx: rhs_fn(xx, y_star))(x_star)  # ∂f/∂t
#     return f_star, Jy, Jt


# def _smoothstep(x):
#     x = jnp.clip(x, 0.0, 1.0)
#     return x * x * (3.0 - 2.0 * x)

    # return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)

# TODO check the mass transfer enthalpy. there is the temperature increasing. no sense

# -----------------------------------------------------------------------------
# Converging-diverging nozzle example
# -----------------------------------------------------------------------------



if __name__ == "__main__":

    nsteps=1000
    case = "a"


    table_dir = "fluid_tables"
    fluid_name = "water"
    metastable_phase = ["liquid", "vapor"]
    backend = "HEOS"
    hl_table = [1000e3, 1800e3]  # J/kg
    pl_table = [5e6, 13e6]   # Pa
    hv_table = [2200e3, 3400e3]
    pv_table = [5e6, 13e6]   # Pa
    N_point_liquid_table = 100
    N_point_vapor_table = 100
    N_p_sat = 150
    gradient_method = "forward"
    
    P_0_in = 1.2e7

    T_in_a = 578.95
    p_out_a =6.76e6

    T_in_b = 592.55
    p_out_b =7.19e6

    if case == "a":
        T_0_in = T_in_a
    else:
        T_0_in = T_in_b    


    # fluid_eq = jxp.FluidBicubic(
    #             fluid_name=fluid_name,
    #             backend="HEOS",
    #             h_min=h1_table[0],
    #             h_max=h1_table[1],
    #             p_min=p_table[0],
    #             p_max=p_table[1],
    #             N_h=N_p_l,
    #             N_p=N_p_l,
    #             table_dir=table_dir,
    #             gradient_method=gradient_method,
            # )


    # fluid_ph = jxp.Fluid(fluid_name, backend)
    # hl_in = fluid_eq.get_state(jxp.PT_INPUTS, 7.8e5, 94.7)["h"]
    # print(hl_in)

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

    # # Define model parameters
    # fluid = jxp.Fluid(fluid_name)
    # h = fluid_vapor.get_state(jxp.PT_INPUTS, P_0_in, T_0_in)
    # print(h, P_0_in, T_0_in)
    # gg

    # fluid_name = "air"
    # fluid = jxp.perfect_gas.get_constants(fluid_name, T_ref=300, P_ref=101325)

    # -- 1. Find critical state with continuation --

    # ---------------------------
    # Build fluids
    # ---------------------------
    params_model = NozzleParams(
        p0_in=P_0_in,  # Pa
        T01_in = T_0_in, # K
        T02_in = T_0_in, # K
        length=5.0,  # m
        roughness=0e-6,  # m
        Ma_low=0.95,
        Ma_high=1.05,
        heat_transfer=0.0,
        wall_friction=0.0, # 0 -> no friction | 1 -> tau_w = tau_w | n -> tau_w = n * tau_w
        fluid1 = fluid_vapor,
        fluid2 = fluid_liquid,
        # fluid1=jxp.FluidJAX(
        #     fluid_name[0],
        #     backend="HEOS",
        # ),
        # fluid2=jxp.FluidJAX(
        #     fluid_name[1],
        #     backend="HEOS",
        # ),
        geometry=get_pipe,
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
        solver_name="Dopri5", # Kvaerno5 | Dopri5
        adjoint_name="DirectAdjoint",
        number_of_points=60,
        rtol=1e-6,
        atol=1e-6,
    )

    # Solve the problem
    print("\n" + "-" * 60)
    print("Evaluating transonic solution")
    print("-" * 60)
    print("-" * 60)
    # input_array = jnp.linspace(0.001,0.005,5)
    input_array = jnp.array([0.01])
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(input_array)))  # Generate colors
    solution_list = []
    out_list = []
    for i, p0_in in enumerate(input_array):
        t0 = time.perf_counter()
        params_model = replace_param(params_model, "Ma_in", p0_in)
        # sol = transonic_nozzle_one_component(params_model, params_bvp, params_ivp)
        # sol = transonic_nozzle_one_component_classic(params_model, params_bvp, params_ivp)
        sol = transonic_nozzle_one_component(params_model, params_bvp, params_ivp)
        # ff 
        # Relative error diagnostics
        dt_ms = (time.perf_counter() - t0) * 1e3
        out = sol.ys
        out2 = []
        sol2 = []
        ts = jnp.linspace(sol.t0, sol.ts[0], 100)
        # for t in ts:
        #     out2.append(sol.evaluate(t))  # vmap instead of for loop
        #     sol2.append(nozzle_homogeneous_non_equilibrium_autonomous(0.0, out2[-1], params_model))
        # # print(sol2)
        # keys = sol2[0].keys()
        # # Build a dict of stacked JAX arrays (like a JAX-native DataFrame)
        # jax_table = {
        #     k: jnp.stack([d[k] for d in sol2]) 
        #     for k in keys if jnp.isscalar(sol2[0][k]) or sol2[0][k].shape == ()
        # }
        # out = jax_table

        # Vectorize the first step: sol.evaluate(t) for all t in ts
        evaluate_all = jax.vmap(sol.evaluate)
        out2 = evaluate_all(ts)

        # Vectorize the second step: apply the model to each (t, out)
        model_all = jax.vmap(lambda y: nozzle_homogeneous_non_equilibrium_one_component_autonomous(0.0, y, params_model))
        
        # if solving rhs directly uncomment next line
        # out_autonomus = jnp.concatenate((ts[:, None], out2), axis=1)
        
        # if solving in autonomous form uncomment next line
        out_autonomus  = out2

        sol2 = model_all(out_autonomus)
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
    gs = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 3, 3, 3])
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
    axs[1].set_ylabel("Speed of Sound (m/s)")
    for color, val, out in zip(colors, input_array, out_list):
        axs[1].plot(
            out["x"],
            out["a_mix"],
            color=color,
            marker="o",
            markersize="3",
        )

    # Static and stagnation enthalpy
    axs[2].set_ylabel("Mass source term")
    for color, val, out in zip(colors, input_array, out_list):
        # axs[2].plot(sol.ys["x"], sol.ys["h"], color=r["color"], markersize=3, marker"o")
        axs[2].plot(out["x"], out["m_dot_lv"], color=color, markersize=3, marker="o")

    # Entropy
    # axs[3].set_ylabel("Determinant")
    # for color, val, out in zip(colors, input_array, out_list):
    #     axs[3].plot(out["x"],out["D"], color=color, markersize=3, marker="o")
    axs[3].set_ylabel("Temperature")
    for color, val, out in zip(colors, input_array, out_list):
        axs[3].plot(out["x"],out["T1"], color="red", label="vap")
        axs[3].plot(out["x"],out["T2"], color="blue", label="liq")

    # Entropy
    axs[4].set_ylabel("Vol. Fraction")
    for color, val, out in zip(colors, input_array, out_list):
        axs[4].plot(out["x"],out["alpha1"], markersize=3, marker="o",  color="red")
        axs[4].plot(out["x"],out["alpha2"], markersize=3, marker="o",  color="blue")


    # # --- row 3: nozzle geometry ---
    # axs[4].fill_between(xg, -rg, +rg, color="lightgrey")  # shaded nozzle
    # x_closed = jnp.concatenate([xg, xg[::-1], xg[:1]])
    # y_closed = jnp.concatenate([rg, -rg[::-1], rg[:1]])
    # axs[4].plot(x_closed, y_closed, "k", linewidth=1.2)
    # # r_abs_max = float(jnp.max(jnp.abs(rg)))
    # # ax3.set_ylim(-1.5 * r_abs_max, 1.5 * r_abs_max)
    # # axs[4].set_aspect("equal", adjustable="box")
    # axs[4].set_xlabel("Axial coordinate x (m)")
    # fig.tight_layout(pad=1)

    fig = plt.figure(figsize=(5, 10))
    plt.fill_between(xg, -rg, +rg, color="lightgrey")  # shaded nozzle
    x_closed = jnp.concatenate([xg, xg[::-1], xg[:1]])
    y_closed = jnp.concatenate([rg, -rg[::-1], rg[:1]])
    plt.plot(x_closed, y_closed, "k", linewidth=1.2)
    # r_abs_max = float(jnp.max(jnp.abs(rg)))
    # ax3.set_ylim(-1.5 * r_abs_max, 1.5 * r_abs_max)
    # axs[4].set_aspect("equal", adjustable="box")
    # plt.set_xlabel("Axial coordinate x (m)")

    # --- p-h diagram ---
    fluid_ph = jxp.Fluid(fluid_name, backend)
    fig_ph, ax_ph = fluid_ph.plot_phase_diagram(
        x_prop="h", y_prop="p", x_scale="linear", y_scale="log",
        plot_critical_point=True, plot_quality_isolines=True,
    )
    for color, val, out in zip(colors, input_array, out_list):
        ax_ph.plot(out["h1"], out["p"], "-o", color=color, markersize=3, label=f"vapor (h1)")
        ax_ph.plot(out["h2"], out["p"], "-s", color=color, markersize=3, label=f"liquid (h2)")
    ax_ph.set_xlabel("Enthalpy (kJ/kg)")
    ax_ph.set_ylabel("Pressure (Pa)")
    ax_ph.legend(fontsize=7)
    fig_ph.tight_layout()

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
