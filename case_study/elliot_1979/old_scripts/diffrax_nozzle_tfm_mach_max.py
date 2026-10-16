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

from nozzlex.two_fluid.nozzle_model_core import (
    nozzle_two_fluid_core,
    get_nozzle_elliot,
    compute_inlet_state_hne,
    get_speed_of_sound_mixture,
    nozzle_two_fluid_autonomous
)

from nozzlex.two_fluid.nozzle_model_solver import (
    NozzleParams,
    BVPSettings,
    IVPSettings,
    replace_param,
    nozzle_det_min_two_fluid,
    compute_critical_inlet_det_two_fluid,
)

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
        root_finder=optx.Newton(rtol=1e-3, atol=1e-3),
    )

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
        # throw=False,
    )

    # --- Solver stats ---
    print(f"\n--- Solver stats ---")
    print(f"  result: {sol1.result}")
    print(f"  τ_end = {float(sol1.ts[0]):.6e}")
    print(f"  num_steps = {sol1.stats['num_steps']}")
    print(f"  num_accepted = {sol1.stats['num_accepted_steps']}")
    print(f"  num_rejected = {sol1.stats['num_rejected_steps']}")

    return sol1, Y_ref, params_model

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
    
    min_det = nozzle_det_min_two_fluid(params_model, params_ivp)
    jax.debug.print("min det:{M}", M=min_det)

    return min_det

def transonic_nozzle_zero_function(
    params_model: NozzleParams,
    params_bvp: BVPSettings,
    params_ivp: IVPSettings,
):
    """
    Transonic converging-diverging nozzle.
    First uses the root finding to find the critical (sonic) state,
    then integrates the ODE system with a blended RHS near the critical location.
    """
    
    # min_det = nozzle_homogeneuous_non_equilibrium_mach_out(params_model, params_ivp)
    # jax.debug.print("{M}", M=min_det)

    # Test critical mach function
    u1_crit = compute_critical_inlet_det_two_fluid(u1_lower=2.0, u1_upper=3.5, params_model=params_model, params_solver=params_ivp)
        
    # Ma_in_cr = jnp.min(min_det["Ma_mix"])
    # maximum_macxh_number = jnp.max(min_det["Ma_mix"])
    # jax.debug.print("min_det={M}", M=min_det)

    return u1_crit

@eqx.filter_jit
def _eval_model_batch(Y, params):
    return jax.lax.map(lambda y: nozzle_two_fluid_autonomous(0.0, y, params), Y)


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
        length=0.31,  # m
        roughness=0e-6,  # m
        Ma_low=0.95,
        Ma_high=1.05,
        Ma_in=0.001,
        # heat_transfer=0.0,
        # wall_friction=0.0,
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
        mixture_ratio=68,
        bubble_number=1e8,
        droplet_number=1e8,
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
        rtol=1e-7,
        atol=1e-7,
    )

    # Solve the problem
    print("\n" + "-" * 60)
    print("Evaluating transonic solution")
    print("-" * 60)
    input_array = jnp.asarray([3.306200761348009])
    # input_array = jnp.linspace(2.0, 3.5, 2)
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(input_array)))  # Generate colors
    solution_list = []
    out_list = []
    for i, Ma_in in enumerate(input_array):
        t0 = time.perf_counter()  
        params_model = replace_param(params_model, "u1_in", Ma_in)
        min_det = transonic_nozzle_one_component(params_model, params_bvp, params_ivp)
        solution_list.append(min_det)

    u_crit = transonic_nozzle_zero_function(params_model, params_bvp, params_ivp)
    params_model = replace_param(params_model, "u1_in", u_crit)
    
    sol, Y_ref, params_model = transonic_nozzle_one_component_autonomus(params_model, params_bvp, params_ivp)

    # --- Post-processing: unscale dense output ---
    n_fine = 20000
    ts = jnp.linspace(sol.t0, sol.ts[0], n_fine)

    y_hat_uniform = jax.vmap(sol.evaluate)(ts)
    out2 = y_hat_uniform * Y_ref[None, :]  # unscale all components

    out_autonomus = out2
    sol2 = _eval_model_batch(out_autonomus, params_model)
    out = sol2
    out_list.append(out)

    mdot_error = (out["m_dot"].max() - out["m_dot"].min()) / out["m_dot"][0]
    h01_error = (out["h1"].max() - out["h1"].min()) / out["h1"][0]
    h02_error = (out["h2"].max() - out["h2"].min()) / out["h2"][0]
    s_error = (out["s1"].max() - out["s1"].min()) / out["s1"][0]

    print(
        f"Solution {i} | Solver status {sol.result} | "
        f"steps {int(sol.stats['num_steps']):3d} | "
        f"mdot error {mdot_error:0.2e} | h01 error {h01_error:0.2e} | "
        f"s_error {s_error:0.2e}"
    )

    # Create the figure
    fig = plt.figure(figsize=(5, 10))
    gs = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 3, 3, 1])
    xg = out_list[0]["x"]
    rg = out_list[0]["diameter"] / 2.0
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    axs = [ax0, ax1, ax2, ax3, ax4]

    # --- row 1: pressure (bar): solid p0, dashed p ---
    axs[0].set_ylabel("Mach number")
    for color, val, o in zip(colors, input_array, out_list):
        axs[0].plot(o["x"], o["Ma_mix"], linestyle="-", color=color,
                    label=rf"$u_{{1,\mathrm{{in}}}} = {val:0.3f}$")

    # --- row 2: velocity ---
    axs[1].set_ylabel("Velocity (m/s)")
    for color, val, o in zip(colors, input_array, out_list):
        axs[1].plot(o["x"], o["v1"], color=color, label="water")
        axs[1].plot(o["x"], o["v2"], color=color, linestyle="--", label="nitrogen")
        axs[1].legend(loc="upper right", fontsize=7)

    # Static and stagnation enthalpy
    axs[2].set_ylabel("Temperature [K]")
    for color, val, o in zip(colors, input_array, out_list):
        axs[2].plot(o["x"], o["T1"], color=color)
        axs[2].plot(o["x"], o["T2"], color=color, linestyle="--")

    # Determinant
    axs[3].set_ylabel("Determinant")
    for color, val, o in zip(colors, input_array, out_list):
        axs[3].plot(o["x"], o["D"], color=color)

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


    # Create the figure
    fig = plt.figure(figsize=(5, 10))
    plt.plot(input_array, solution_list, "-o", color="k")
    plt.plot(u_crit, 0, "^", color="r")

    # Show figures
    plt.show()
