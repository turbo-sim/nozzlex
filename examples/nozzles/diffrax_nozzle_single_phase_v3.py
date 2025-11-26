import time
import pandas as pd
import jax
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
import optimistix as optx
import jaxprop as jxp
import matplotlib.pyplot as plt

from matplotlib import gridspec

jxp.set_plot_options(grid=False)

from nozzlex.functions import (
    nozzle_single_phase_core,
    symmetric_nozzle_geometry,
    linear_convergent_divergent_nozzle,
    NozzleParams,
    BVPSettings,
    IVPSettings,
    replace_param,
    solve_nozzle_model_collocation,
    initialize_flowfield,
    compute_critical_inlet,
    compute_static_state,
)

# v1 solves the ode system using the space marching in non-autonomous form
# v2 solves the autonomous system with events for the bounds of domain ends
# v3 uses the non-autonomous system and determines the inlet critical condition with collocation method


# -----------------------------------------------------------------------------
# Main API to the converging-diverging nozzle model
# -----------------------------------------------------------------------------
@eqx.filter_jit
def transonic_nozzle_single_phase(
    params_model: NozzleParams,
    params_bvp: BVPSettings,
    params_ivp: IVPSettings,
):
    """
    Transonic converging-diverging nozzle.
    First uses the root finding to find the critical (sonic) state,
    then integrates the ODE system with a blended RHS near the critical location.
    """

    Ma_in_cr = compute_critical_inlet(Ma_lower=0.001, Ma_upper=0.2, params_model=params_model, params_solver=params_ivp)

    state_in = compute_static_state(
        params_model.p0_in,
        params_model.h0_in,
        Ma_in_cr,
        params_model.fluid,
    )
    p_in, h_in, a_in = state_in["p"], state_in["h"], state_in["a"]
    v_in = Ma_in_cr * a_in

    # --- Define helpers with closure over args_base and fluid ---
    def ode_full_subsonic(t, y, args):
        return nozzle_single_phase_core(t, y, args)

    def ode_rhs_subsonic(t, y, args):
        return ode_full_subsonic(t, y, args)["rhs"]

    # --- 2. First pass: inlet → Ma_low ---
    solver = jxp.make_diffrax_solver(params_ivp.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_ivp.adjoint_name)
    ctrl = dfx.PIDController(rtol=params_ivp.rtol, atol=params_ivp.atol)
    y_inlet = jnp.array([p_in, v_in, h_in])
    event1 = dfx.Event(
        cond_fn=_mach_event_cond,
        root_finder=optx.Bisection(rtol=1e-9, atol=1e-9),
    )
    term1 = dfx.ODETerm(ode_rhs_subsonic)
    save1 = dfx.SaveAt(t1=True, fn=ode_full_subsonic)
    sol1 = dfx.diffeqsolve(
        term1,
        solver,
        t0=1e-9,
        t1=1000,
        dt0=1e-9,
        y0=y_inlet,
        args=params_model,
        saveat=save1,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        event=event1,
        max_steps=20_000,
    )

    # --- 3. Linearization close to critical point ---
    x_crit = sol1.ys["x"][-1]
    y_crit = jnp.array([sol1.ys["p"][-1], sol1.ys["v"][-1], sol1.ys["h"][-1]])
    f_star, Jy, Jt = _linearize_rhs_at(x_crit, y_crit, params_model)

    # def ode_full_transonic(t, y, args):
    #     base = nozzle_single_phase_core(t, y, args)
    #     M = base["Ma"]
    #     rhs_true = base["rhs"]
    #     rhs_lin = f_star + Jy @ (y - y_crit) + Jt * (t - x_crit)

    #     # Start blending slightly after sonic conditions, finish at Ma_high
    #     Ma_start = 1.000005
    #     Ma_end = args.Ma_high
    #     s = (M - Ma_start) / (Ma_end - Ma_start)
    #     w = _smoothstep(jnp.clip(s, 0.0, 1.0))  # smoothly 0→1 in that narrow range
    #     rhs_blend = (1.0 - w) * rhs_lin + w * rhs_true
    #     in_window = (M >= args.Ma_low) & (M <= args.Ma_high)
    #     rhs_blend = jnp.where(in_window, rhs_blend, rhs_true)

    #     return {**base, "rhs": rhs_true, "rhs_blend": rhs_blend}
    
    def ode_full_transonic(t, y, args):
        # jax.debug.print("x={x} | y={y}", x=t, y=y)
        base = nozzle_single_phase_core(t, y, args)
        Ma = base["Ma"]
        rhs_true = base["rhs"]
        rhs_lin = f_star + Jy @ (y - y_crit) + Jt * (t - x_crit)
 
        # Start blending slightly after sonic conditions, finish at Ma_high
        Ma_start = 1.00001
        Ma_end = args.Ma_high
        s = (Ma - Ma_start) / (Ma_end - Ma_start)
        w = _smoothstep(jnp.clip(s, 0.0, 1.0))  # smoothly 0→1 in that narrow range
        rhs_blend = (1.0 - w) * rhs_lin + w * rhs_true
        # in_window = (Ma >= args.Ma_low) & (Ma <= args.Ma_high)
        # blend_window = (Ma >= Ma_start) & (Ma <= args.Ma_high)
        in_window = (Ma >= args.Ma_low) & (Ma <= args.Ma_high)
        blend_window = (Ma >= Ma_start) & (Ma <= args.Ma_high)
        # jax.debug.print("x={x} | Ma={M} | w={w} | in_wind={iw} | bl_wind={bw}", x=x, M=Ma, w=w, iw=in_window, bw=blend_window)
        # jax.debug.print("Ma={M} | w={w} | in_wind={iw}", M=Ma, w=w, iw=in_window)
        rhs_blend = jnp.where(in_window, jnp.where(blend_window, rhs_blend, rhs_lin), rhs_true)
 
        return {**base, "rhs": rhs_true, "rhs_blend": rhs_blend}
 
 

    def ode_rhs_transonic(t, y, args):
        return ode_full_transonic(t, y, args)["rhs_blend"]

    # --- 4. Second pass ---

    event_low_pressure = dfx.Event(
        cond_fn=_low_pressure_event_cond,
        root_finder=optx.Bisection(rtol=1e-6, atol=1e-6),
    )

    # TODO, perhaps it is more robust to do 1 pass for each segment instead of attempting a single pass?
    ts2 = jnp.linspace(1e-9, params_model.length, params_ivp.number_of_points)
    term2 = dfx.ODETerm(ode_rhs_transonic)
    save2 = dfx.SaveAt(ts=ts2, t1=True, fn=ode_full_transonic)
    sol2 = dfx.diffeqsolve(
        term2,
        solver,
        t0=1e-9,
        t1=params_model.length,
        dt0=1e-9,
        y0=y_inlet,
        args=params_model,
        saveat=save2,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        max_steps=20_000,
        event=event_low_pressure,
    )

    return sol2


def _mach_event_cond(t, y, args, **kwargs):
    """Event function: zero when M^2 - Ma_low^2 = 0."""
    p, v, h = y
    a = args.fluid.get_state(jxp.HmassP_INPUTS, h, p)["a"]
    Ma_sqr = (v / a)**2
    return Ma_sqr - args.Ma_low**2

def _low_pressure_event_cond(t, y, args, **kwargs):
    """Event function: avoid too long pressures"""
    p, v, h = y
    cond = p - args.p_termination
    # jax.debug.print("t = {t:.3e}, p = {p:.6e}, cond = {cond:.6e}", t=t, p=p, cond=cond)

    return cond

def _linearize_rhs_at(x_star, y_star, model):
    """Return Taylor expansion coefficients of RHS around (x_star, y_star)."""
    rhs_fn = lambda xx, yy: nozzle_single_phase_core(xx, yy, model)["rhs"]
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

outdir = "fluid_tables"
fluid_name = "CO2"
backend = "HEOS"
h_min = 10e3  # J/kg
h_max = 550e3  # J/kg
p_min = 6e5    # Pa
p_max = 150e5   # Pa
N_h = 50
N_p = 50

if __name__ == "__main__":

    # # Define model parameters
    # fluid_name = "air"
    # fluid = jxp.perfect_gas.get_constants(fluid_name, T_ref=300, P_ref=101325)

    # -- 1. Find critical state with continuation --

    params_model = NozzleParams(
        p0_in=9288492.359,  # Pa 
        h0_in=310309.245,
        # D_in=0.050,  # m
        # length=5.00,  # m
        L_convergent=0.02735,
        height_in=0.005,
        height_throat=0.00012,
        height_out=0.00027,
        width=0.003,
        roughness=0e-6,  # m
        T_wall=300.0,  # K
        Ma_low=0.95,
        Ma_high=1.0005,
        heat_transfer=0.0,
        wall_friction=1.0,  # 1.0 if friction has to be considered, 0.0 if not
        p_termination=p_min,
        # fluid=jxp.FluidPerfectGas("CO2", T_ref=300, p_ref=101325),
        # fluid=jxp.FluidJAX(name="CO2", backend="HEOS"),
        fluid = jxp.FluidBicubic(fluid_name=fluid_name,
                                 backend="HEOS",
                                 h_max=h_max,
                                 h_min=h_min,
                                 p_min=p_min,
                                 p_max=p_max,
                                 N_h=N_h,
                                 N_p=N_p),
        geometry=linear_convergent_divergent_nozzle,
        two_phase_friction="Beattie",
    )

    params_bvp = BVPSettings(
        solve_mode="mach_crit",
        num_points=60,
        rtol=1e-8,
        atol=1e-8,
        max_steps=500,
        jac_mode="bwd",
        verbose=False,
        method="Newton",
        warmup_method="Dogleg",
        warmup_steps=0,
    )

    params_ivp = IVPSettings(
        solver_name="Dopri5",
        adjoint_name="DirectAdjoint",
        number_of_points=400,
        rtol=1e-8,
        atol=1e-8,
    )

    L_nakagawa = 0.0835
    params_model = replace_param(params_model, "length", L_nakagawa)
   
    # Extract columns as arrays (optional but convenient)
    df = pd.read_csv("temp_2.csv")

    L_convergent = df["L_convergent"].values
    Height_in = df["height_in"].values
    Height_throat = df["height_throat"].values
    Height_out = df["height_out"].values
    Width = df["width"].values
    Roughness = df["roughness"].values
    P0_in = df["p0_in"].values
    H0_in = df["h0_in"].values


    # Solve the problem
    print("\n" + "-" * 60)
    print("Evaluating transonic solution")
    print("-" * 60)
    # input_array = jnp.linspace(0, 2e-6, 11)
    # input_array_p = jnp.linspace(80e5, 95e5, 5) 
    # input_array_h = jnp.linspace(290e3, 300e3, 5)
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, (150-54)))  # Generate colors
    solution_list = []
    # failed: 53
    for i in range(150, 350):
        t0 = time.perf_counter()
        params_model = replace_param(params_model, "L_convergent", L_convergent[i])
        params_model = replace_param(params_model, "height_in", Height_in[i])
        params_model = replace_param(params_model, "height_throat", Height_throat[i])
        params_model = replace_param(params_model, "height_out", Height_out[i])
        params_model = replace_param(params_model, "width", Width[i])
        params_model = replace_param(params_model, "p0_in", P0_in[i])
        params_model = replace_param(params_model, "h0_in", H0_in[i])

        sol = transonic_nozzle_single_phase(params_model, params_bvp, params_ivp)

        # Relative error diagnostics
        dt_ms = (time.perf_counter() - t0) * 1e3
        out = sol.ys
        mdot_error = (out["m_dot"].max() - out["m_dot"].min()) / out["m_dot"][0]
        h0_error = (out["h0"].max() - out["h0"].min()) / out["h0"][0]
        s_error = (out["s"].max() - out["s"].min()) / out["s"][0]

        print(
            f"Solution {i} | Solver status {sol.result._value:2d} | "
            f"steps {int(sol.stats['num_steps']):3d} | "
            f"mdot error {mdot_error:0.2e} | h0 error {h0_error:0.2e} | "
            f"s_error {s_error:0.2e} | time {dt_ms:7.2f} ms"
        )

        solution_list.append(sol)

    # Create the figure
    fig = plt.figure(figsize=(5, 10))
    gs = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 3, 3, 1])
    xg = solution_list[0].ys["x"]
    rg = solution_list[0].ys["diameter"] / 2.0
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    axs = [ax0, ax1, ax2, ax3, ax4]

    # --- row 1: pressure (bar): solid p0, dashed p ---
    axs[0].set_ylabel("Pressure (bar)")
    for color, sol in zip(colors, solution_list):
        x = sol.ys["x"]
        # axs[0].plot(x, sol.ys["p0"] * 1e-5, linestyle="--", color=color)
        axs[0].plot(
            x,
            sol.ys["p"] * 1e-5,
            linestyle="-",
            color=color,
            marker="o",
            markersize="1",
            # label = rf"$P_0 = {val:0.3e}$"
        )
    axs[0].legend(loc="upper right", fontsize=7)

    # --- row 2: Mach number ---
    axs[1].set_ylabel("Mach number (-)")
    for color, sol in zip(colors, solution_list):
        axs[1].plot(
            sol.ys["x"],
            sol.ys["Ma"],
            color=color,
            marker="o",
            markersize="1",
        )

    # Static and stagnation enthalpy
    axs[2].set_ylabel("Quality (-)")
    for color, sol in zip(colors, solution_list):
        # var = 4 * sol.ys["A"] / sol.ys["Q"]
        axs[2].plot(sol.ys["x"], sol.ys["Q"], color=color, markersize=3, marker="o")
        # axs[2].plot(sol.ys["x"], var, color=color, markersize=1, marker="o")

    # Entropy
    axs[3].set_ylabel("Enthalpy (-)")
    for color, sol in zip(colors, solution_list):
        axs[3].plot(sol.ys["x"], sol.ys["Re"], color=color, markersize=1, marker="o")

    # --- row 3: nozzle geometry ---
    axs[4].fill_between(xg, -rg, +rg, color="lightgrey")  # shaded nozzle
    x_closed = jnp.concatenate([xg, xg[::-1], xg[:1]])
    y_closed = jnp.concatenate([rg, -rg[::-1], rg[:1]])
    axs[4].plot(x_closed, y_closed, "k", linewidth=1.2)
    # r_abs_max = float(jnp.max(jnp.abs(rg)))
    # ax3.set_ylim(-1.5 * r_abs_max, 1.5 * r_abs_max)
    axs[4].set_aspect("equal", adjustable="box")
    axs[4].set_xlabel("Axial coordinate x (m)")
    fig.tight_layout(pad=1)


    fluid = jxp.Fluid(name="CO2", backend="HEOS")
    fig, ax = fluid.plot_phase_diagram(x_prop="h", y_prop="p", plot_quality_isolines=True)
    for color, sol in zip(colors, solution_list):
        ax.plot(
            sol.ys["h"],
            sol.ys["p"],
            color=color,
            marker="o",
            markersize="1",
        )

    # Show figures
    plt.show()