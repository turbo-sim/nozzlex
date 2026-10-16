import time
import jax
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
import optimistix as optx
import jaxprop as jxp
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle as pkl

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

from nozzlex.homogeneous_nonequilibrium import(
    # get_nozzle_elliot_segments,
    get_nozzle_elliot_big
)

# -----------------------------------------------------------------------------
# Main API to the converging-diverging nozzle model
# -----------------------------------------------------------------------------
# @eqx.filter_jit
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

    Ma_in_cr = compute_critical_inlet(Ma_lower=0.001, Ma_upper=0.1, params_model=params_model, params_solver=params_ivp)
    # Ma_in_cr = 0.03132522678375244
    jax.debug.print("Critical mach computed. M_cr:{Mc}", Mc=Ma_in_cr)
    # jax.debug.print("Warning!!!!! New value of the throat used")
    state_in = compute_static_state(
        params_model.p0_in,
        params_model.h0_in,
        Ma_in_cr,params_model.fluid,
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
        root_finder=optx.Bisection(rtol=1e-6, atol=1e-6),
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
    # jax.debug.print("critical x:{xc} | critical y:{yc}", xc=x_crit, yc=y_crit)
    
    f_star, Jy, Jx = _linearize_rhs_at(x_crit, y_crit, params_model)

    # def ode_full_transonic(t, y, args):
    #     base = nozzle_single_phase_core(t, y, args)
    #     M = base["Ma"]
    #     rhs_true = base["rhs"]
    #     rhs_lin = f_star + Jy @ (y - y_crit) + Jx * (t - x_crit)

    #     # Start blending slightly after sonic conditions, finish at Ma_high
    #     Ma_start = 1.001
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
        rhs_lin = f_star + Jy @ (y - y_crit) + Jx * (t - x_crit)

        # Start blending slightly after sonic conditions, finish at Ma_high
        Ma_start = 1.0001
        Ma_end = args.Ma_high
        s = (Ma - Ma_start) / (Ma_end - Ma_start)
        w = _smoothstep(jnp.clip(s, 0.0, 1.0))  # smoothly 0→1 in that narrow range
        rhs_blend = (1.0 - w) * rhs_lin + w * rhs_true

        in_window = (Ma >= args.Ma_low) & (Ma <= args.Ma_high)
        blend_window = (Ma >= Ma_start) & (Ma <= args.Ma_high)
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
    term2 = dfx.ODETerm(ode_rhs_transonic)
    save2 = dfx.SaveAt(dense=True, t1=True, fn=ode_full_transonic)
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
        max_steps=200_000,
        event=event_low_pressure,
    )

    xs = jnp.linspace(sol2.t0, sol2.ts[0], 3000)

    evaluate_all = jax.vmap(sol2.evaluate)
    ys = evaluate_all(xs)

    # Vectorize the second step: apply the model to each (t, out)
    model_all = jax.vmap(lambda x, y: nozzle_single_phase_core(x, y, params_model))
    out2 = model_all(xs, ys)

    return sol2, out2


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

    return cond

def _linearize_rhs_at(x_star, y_star, model):
    """Return Taylor expansion coefficients of RHS around (x_star, y_star)."""
    rhs_fn = lambda xx, yy: nozzle_single_phase_core(xx, yy, model)["rhs"]
    f_star = rhs_fn(x_star, y_star)  # RHS at expansion point
    # jax.debug.print("fcrit:{fs} | x_star:{xs} | y_star:{ys}", fs=f_star, xs=x_star, ys=y_star)
    Jx = jax.jacfwd(lambda xx: rhs_fn(xx, y_star))(x_star)  # ∂f/∂t
    Jy = jax.jacfwd(lambda yy: rhs_fn(x_star, yy))(y_star)  # ∂f/∂y
    return f_star, Jy, Jx


def _smoothstep(x):
    x = jnp.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

    # return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


# -----------------------------------------------------------------------------
# Converging-diverging nozzle example
# -----------------------------------------------------------------------------



if __name__ == "__main__":

    # # Define model parameters
    # fluid_name = "air"
    # fluid = jxp.perfect_gas.get_constants(fluid_name, T_ref=300, P_ref=101325)


    START_CASE = [1]

    data = pd.read_excel("simulation_case_big.xlsx", sheet_name="Sheet1")

    # data = pd.read_excel("simulation_cases_big.xlsx", sheet_name="Sheet1")


    if isinstance(START_CASE, str) and START_CASE.lower() == "all":
        # START_CASE is the string "all"
        case_data = data

    elif isinstance(START_CASE, list):
        # START_CASE is a list of case numbers
        case_data = data[data["case"].isin(START_CASE)]

    elif isinstance(START_CASE, int):
        # Use START_CASE and END_CASE to build range
        CASES = list(range(START_CASE, END_CASE + 1))
        case_data = pd.DataFrame()  # initialize an empty DataFrame with the same columns
        for CASE in CASES:
            case_temporary = data[data["case"] == CASE]  # or .isin([CASE]) if needed
            case_data = pd.concat([case_data, case_temporary], ignore_index=True)

    else:
        raise ValueError("Unsupported type for START_CASE")

    # Loop over cases
    print("Cases scheduled for simulation:")
    print(", ".join(case_data["case"].astype(str)))

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
        solver_name="Kvaerno5",
        adjoint_name="DirectAdjoint",
        number_of_points=2000,
        rtol=1e-8,
        atol=1e-8,
    )

    table_dir = "fluid_tables"
    fluid_name = "water_nitrogen_mixture"
    backend = "HEOS"
    p_table = [2e4, 3e6]
    N_h = 120
    N_p = 120
    T_table = [275, 300]

    fluid1 = jxp.FluidJAX("water")
    h1_min = fluid1.get_state(jxp.PT_INPUTS, p_table[0], T_table[0])["enthalpy"]
    h1_max = fluid1.get_state(jxp.PT_INPUTS, p_table[1], T_table[1])["enthalpy"]

    fluid2 = jxp.FluidJAX("nitrogen")
    h2_min = fluid2.get_state(jxp.PT_INPUTS, p_table[0], T_table[0])["enthalpy"]
    h2_max = fluid2.get_state(jxp.PT_INPUTS, p_table[1], T_table[1])["enthalpy"]

    # Solve the problem
    print("\n" + "-" * 60)
    print("Evaluating transonic solution")
    print("-" * 60)
    # input_array = jnp.asarray([20, 21, 22]) * 1e5

    FLUID = {"fluid_object": None,
            "h0_in": None}

    FLUID = pd.DataFrame(columns=FLUID.keys())

    for i, row in case_data.iterrows():
        x1 = row["mixture_ratio"] / (row["mixture_ratio"] + 1)
        x2 =  1 - x1
        print(f"x1:{x1} | x2:{x2}")
        hmix_min = (x1 * h1_min + x2 * h2_min)*0.8
        hmix_max = (x1 * h1_max + x2 * h2_max)*1.2
        # print(f"hmin:{hmix_min} | hmax:{hmix_max}")

        fluid=jxp.FluidBicubic(
                fluid_name=fluid_name,
                backend="HEOS",
                h_min=hmix_min,
                h_max=hmix_max,
                p_min=p_table[0],
                p_max=p_table[1],
                N_h=N_h,
                N_p=N_p,
                table_dir=table_dir,
                mixture_ratio=row["mixture_ratio"],
            )
        
        T0_in = float(row["T0_in"])
        p0_in = float(row["p0_in"])
        print(T0_in, p0_in)
        
        h0_in = fluid.get_state(jxp.PT_INPUTS, p0_in, T0_in)
        h0_in = h0_in["enthalpy"]

        FLUID.loc[len(FLUID)] = {"fluid_object": fluid,
                           "h0_in": h0_in}

    # print(FLUID)

    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(case_data["T0_in"])))  # Generate colors
    solution_list = []
    # print(case_data)
    for i, row in case_data.iterrows():
        t0 = time.perf_counter()
        # print(FLUID)
        # print("i:",i)
        params_model = NozzleParams(
            p0_in=p0_in,  # Pa
            h0_in= FLUID["h0_in"].iloc[0],
            length=1.3082,  # m
            roughness=0e-6,  # m
            Ma_low=0.98,
            Ma_high=1.01,
            # Ma_in=0.002,
            wall_friction=1.0,
            heat_transfer=1.0,
            fluid=FLUID["fluid_object"].iloc[0],
            geometry=get_nozzle_elliot_big,
        )

        sol, out = transonic_nozzle_single_phase(params_model, params_bvp, params_ivp)

        # Relative error diagnostics
        dt_ms = (time.perf_counter() - t0) * 1e3
        mdot_error = (out["m_dot"].max() - out["m_dot"].min()) / out["m_dot"][0]
        h0_error = (out["h0"].max() - out["h0"].min()) / out["h0"][0]
        s_error = (out["s"].max() - out["s"].min()) / out["s"][0]

        print(
            f"Solution {i} | Solver status {sol.result._value:2d} | "
            f"steps {int(sol.stats['num_steps']):3d} | "
            f"mdot error {mdot_error:0.2e} | h0 error {h0_error:0.2e} | "
            f"s_error {s_error:0.2e} | time {dt_ms:7.2f} ms"
        )

        with open(os.path.join(f"hem_solutions", f"solution_{row['mixture_ratio']:.2f}.pkl"), "wb") as f:
            pkl.dump(out, f)
        solution_list.append(out)

    # Create the figure
    fig = plt.figure(figsize=(5, 10))
    gs = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 3, 3, 1])
    xg = solution_list[0]["x"]
    rg = solution_list[0]["diameter"] / 2.0
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    axs = [ax0, ax1, ax2, ax3, ax4]

    # --- row 1: pressure (bar): solid p0, dashed p ---
    axs[0].set_ylabel("Pressure (bar)")
    for color, val, sol in zip(colors, case_data["mixture_ratio"], solution_list):
        x = sol["x"]
        # axs[0].plot(x, sol.ys["p0"] * 1e-5, linestyle="--", color=color)
        axs[0].plot(
            x,
            sol["p"] * 1e-5,
            linestyle="-",
            color=color,
            # marker="o",
            markersize="3",
            label = rf"$p_{{0,\mathrm{{in}}}} = {val:.3f}$"
        )
    axs[0].legend(loc="upper right", fontsize=7)

    # --- row 2: Mach number ---
    axs[1].set_ylabel("Mach number (-)")
    for color, val, sol in zip(colors, case_data["mixture_ratio"], solution_list):
        axs[1].plot(
            sol["x"],
            sol["Ma"],
            color=color,
            # marker="o",
            markersize="3",
        )

    # Static and stagnation enthalpy
    axs[2].set_ylabel("Enthalpy (J/kg)")
    for color, val, sol in zip(colors, case_data["mixture_ratio"], solution_list):
        # axs[2].plot(sol.ys["x"], sol.ys["h"], color=r["color"], markersize=3, marker"o")
        axs[2].plot(sol["x"], sol["h"], color=color,)# markersize=3, marker="o")

    # Entropy
    axs[3].set_ylabel("Quality")
    for color, val, sol in zip(colors, case_data["mixture_ratio"], solution_list):
        axs[3].plot(sol["x"], sol["Q"], color=color,)# markersize=3, marker="o")

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



    # fluid = jxp.Fluid(name="CO2", backend="HEOS")
    # fig, ax = fluid.plot_phase_diagram(plot_quality_isolines=True)
    # ax.plot(sol.ys["s"], sol.ys["T"], "ko")

    # Show figures
    plt.show()
