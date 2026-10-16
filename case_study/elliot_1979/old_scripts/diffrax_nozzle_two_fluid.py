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
    compute_critical_inlet_det_two_fluid
)

def _smoothstep(x):
    x = jnp.clip(x, 0.0, 1.0)
    # return x * x * (3.0 - 2.0 * x)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)

def transonic_nozzle_one_component_autonomus(
    params_model: NozzleParams,
    params_bvp: BVPSettings,
    params_ivp: IVPSettings,
):
    """
    Two-pass blended transonic solver for the two-fluid model.

    Pass 1 (autonomous, τ-space):
        Integrate from inlet until Ma_mix = Ma_low (just before the throat
        singularity).

    Linearise at the critical point:
        Compute f*, Jy, Jx of the non-autonomous RHS at (x_crit, y_crit).

    Pass 2 (non-autonomous, x-space):
        Integrate from x_crit to the outlet with a blended RHS that smoothly
        transitions from the linearised approximation (near the throat where
        det≈0) to the true RHS (away from the throat).

    Returns
    -------
    sol1    : diffrax solution (autonomous, dense)
    sol2    : diffrax solution (non-autonomous, dense)
    Y_ref   : 10-component state-variable scale (includes x scale)
    Y_ref_9 : 9-component scale for Pass-2 state (excludes x)
    params_model : updated with row/col/det_ref scales
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

    # u_in = params_model.Ma_in * a_in_mix
    u1_in = params_model.u1_in
    # u2_in = rho1_in * u1_in * alpha1_in / (rho2_in * alpha2_in * params_model.mixture_ratio)
        
    u2_in = u1_in * (rho1_in / rho2_in) * (alpha1_in / alpha2_in) * (1 / params_model.mixture_ratio)

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
    def ode_rhs_scaled(t, y, args):
        y_phys = y * Y_ref
        out = nozzle_two_fluid_autonomous(0.0, y_phys, args)
        return out["rhs_autonomous"] / Y_ref

    def eval_pass1_state(_t, y, _args):
        """SaveAt fn: unscale and return named physical state dict."""
        x, alpha1, alpha2, rho1, rho2, u1, u2, p, h1, h2 = y * Y_ref
        return {
            "x": x, "alpha1": alpha1, "alpha2": alpha2,
            "rho1": rho1, "rho2": rho2,
            "u1": u1, "u2": u2,
            "p": p, "h1": h1, "h2": h2,
        }

    y_inlet_hat = y_inlet / Y_ref

    solver  = jxp.make_diffrax_solver(params_ivp.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_ivp.adjoint_name)
    ctrl = dfx.PIDController(
        rtol=params_ivp.rtol,
        atol=params_ivp.atol,
        dtmax=1e-3,
    )

    # ------------------------------------------------------------------ Pass 1
    # Autonomous form: integrate inlet → Ma_low (just before throat singularity)
    # ------------------------------------------------------------------
    def mach_low_event(t, y, args, **_kwargs):
        y_phys = y * Y_ref
        Ma = nozzle_two_fluid_autonomous(0.0, y_phys, args)["Ma_mix"]
        return Ma - args.Ma_low

    sol1 = dfx.diffeqsolve(
        dfx.ODETerm(ode_rhs_scaled),
        solver,
        t0=1e-9, t1=1e9, dt0=1e-9,
        y0=y_inlet_hat,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=dfx.SaveAt(t1=True,dense=True, fn=eval_pass1_state),
        event=dfx.Event(
            cond_fn=mach_low_event,
            root_finder=optx.Newton(rtol=1e-7, atol=1e-7),
        ),
        max_steps=10_000,
        # throw=False,
    )

    print(f"\n--- Pass 1 (autonomous → Ma_low) ---")
    print(f"  result:       {sol1.result}")
    print(f"  τ_end       = {float(sol1.ts[0]):.6e}")
    print(f"  num_steps   = {sol1.stats['num_steps']}")

    # ------------------------------------------------------------------ Linearise at throat
    Y_ref_9 = Y_ref[1:]                             # scale for 9-component state (no x)
    x_crit  = sol1.ys["x"][-1]
    y_crit_9 = jnp.array([
        sol1.ys["alpha1"][-1],
        sol1.ys["alpha2"][-1],
        sol1.ys["rho1"][-1],
        sol1.ys["rho2"][-1],
        sol1.ys["u1"][-1],
        sol1.ys["u2"][-1],
        sol1.ys["p"][-1],
        sol1.ys["h1"][-1],
        sol1.ys["h2"][-1],
    ])

    def rhs_nonaut(x_val, y_val):
        """Non-autonomous RHS dy/dx from the autonomous form."""
        out     = nozzle_two_fluid_core(x_val, y_val, params_model)
        rhs_aut = out["rhs_autonomous"]
        return rhs_aut[1:] / rhs_aut[0]            # dy/dx = N / D  (9 components)

    f_star = rhs_nonaut(x_crit, y_crit_9)
    Jy     = jax.jacfwd(lambda y: rhs_nonaut(x_crit, y))(y_crit_9)   # 9×9
    Jx     = jax.jacfwd(lambda x: rhs_nonaut(x, y_crit_9))(x_crit)   # 9

    print(f"  x_crit      = {float(x_crit):.6f} m")

    # ------------------------------------------------------------------ Pass 2 (unscaled)
    # Non-autonomous x-form, state in physical units (no Y_ref_9 scaling).
    # rhs_true = rhs_autonomous[1:] / rhs_autonomous[0]  (Cramer's rule, dy/dx)
    # ------------------------------------------------------------------
    def ode_rhs_pass2(t, y, args):
        out      = nozzle_two_fluid_core(t, y, args)
        rhs_aut  = out["rhs_autonomous"]
        rhs_true = rhs_aut[1:] / rhs_aut[0]
        Ma       = out["Ma_mix"]

        # Linearised RHS (valid near the throat, y is physical)
        rhs_lin   = f_star + Jy @ (y - y_crit_9) + Jx * (t - x_crit)

        Ma_start  = 1.01
        s         = (Ma - Ma_start) / (args.Ma_high - Ma_start)
        w         = _smoothstep(jnp.clip(s, 0.0, 1.0))
        rhs_blend = (1.0 - w) * rhs_lin + w * rhs_true

        in_window    = (Ma >= args.Ma_low) & (Ma <= args.Ma_high)
        blend_window = (Ma >= Ma_start)    & (Ma <= args.Ma_high)
        return jnp.where(
            in_window,
            jnp.where(blend_window, rhs_blend, rhs_lin),
            rhs_true,
        )

    def event_outlet(t, y, args, **_kwargs):
        out    = nozzle_two_fluid_core(t, y, args)
        p_min  = jnp.maximum(args.fluid1.p_min, args.fluid2.p_min)
        h1_min, h1_max = args.fluid1.h_min, args.fluid1.h_max
        h2_min, h2_max = args.fluid2.h_min, args.fluid2.h_max
        return jnp.min(jnp.stack([
            t / args.length,
            1 - t / args.length,
            out["p"] - p_min - 100,
            out["h1"] - h1_min - 10,
            h1_max - out["h1"] + 10,
            out["h2"] - h2_min - 10,
            h2_max - out["h2"] + 10,
        ]))

    ctrl2 = dfx.PIDController(
        rtol=params_ivp.rtol,
        atol=params_ivp.atol,
        dtmax=1e-3,
    )

    sol2 = dfx.diffeqsolve(
        dfx.ODETerm(ode_rhs_pass2),
        solver,
        t0=1e-9, t1=params_model.length * 10,
        dt0=1e-9,
        y0=y_variable,
        args=params_model,
        stepsize_controller=ctrl2,
        adjoint=adjoint,
        saveat=dfx.SaveAt(t1=True, dense=True),
        event=dfx.Event(
            cond_fn=event_outlet,
            root_finder=optx.Newton(rtol=1e-7, atol=1e-7),
        ),
        max_steps=200_000,
        # throw=False,
    )

    print(f"\n--- Pass 2 (non-autonomous blended, inlet → outlet) ---")
    print(f"  result:       {sol2.result}")
    print(f"  x_end       = {float(sol2.ts[0]):.6f} m")
    print(f"  num_steps   = {sol2.stats['num_steps']}")

    return sol1, sol2, Y_ref, Y_ref_9, params_model


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
    u1_crit = compute_critical_inlet_det_two_fluid(u1_lower=3.0, u1_upper=3.5, params_model=params_model, params_solver=params_ivp)
        
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
        Ma_low=0.97,
        Ma_high=1.02,
        Ma_in=0.001,
        heat_transfer=1.0,
        wall_friction=1.0,
        bubble_number=1e8,
        droplet_number=1e8,
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
        solver_name="Kvaerno3",  # Tsit5 | Kvaerno3 | Dopri5 | ImplicitEuler
        adjoint_name="DirectAdjoint",
        number_of_points=60,
        rtol=1e-7,
        atol=1e-7,
    )

    # Solve the problem
    print("\n" + "-" * 60)
    print("Evaluating transonic solution")
    print("-" * 60)
    # input_array = jnp.asarray([3.0])
    # input_array = jnp.linspace(3.0, 3.5, 2)
    # colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(input_array)))  # Generate colors
    solution_list = []
    out_list = []
    # for i, Ma_in in enumerate(input_array):
    #     t0 = time.perf_counter()  
    #     params_model = replace_param(params_model, "u1_in", Ma_in)
    #     min_det = transonic_nozzle_one_component(params_model, params_bvp, params_ivp)
    #     solution_list.append(min_det)

    # u_crit = transonic_nozzle_zero_function(params_model, params_bvp, params_ivp)
    u_crit = 3.4845928251743317 # 3.3148438930611475*1.001
    print(f"\nu1_crit = {float(u_crit)} m/s")
    params_model = replace_param(params_model, "u1_in", u_crit)

    sol1, sol2, Y_ref, Y_ref_9, params_model = transonic_nozzle_one_component_autonomus(
        params_model, params_bvp, params_ivp
    )

    # ------------------------------------------------------------------ Post-processing
    # Pass 2 covers inlet → outlet in x-space: evaluate dense solution
    n_pts = 3000
    xs_all = jnp.linspace(float(sol2.t0), float(sol2.ts[0]), n_pts)
    Y_all  = jax.vmap(sol2.evaluate)(xs_all)  # (n_pts, 9) already physical (unscaled Pass 2)

    # Use a simple loop to get all diagnostic outputs (avoids vmap shape issues)
    out_list_pts = [
        nozzle_two_fluid_core(float(x), Y_all[j], params_model)
        for j, x in enumerate(xs_all)
    ]
    out = {k: jnp.stack([d[k] for d in out_list_pts])
           for k in out_list_pts[0] if jnp.asarray(out_list_pts[0][k]).shape == ()}

    out_list.append(out)

    mdot_error = (out["m_dot"].max() - out["m_dot"].min()) / out["m_dot"][0]
    print(f"mdot error = {float(mdot_error):.3e}")

    # ------------------------------------------------------------------ Plots
    fig = plt.figure(figsize=(5, 10))
    gs  = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 3, 3, 1])
    xg  = out["x"]
    rg  = out["diameter"] / 2.0
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    axs = [ax0, ax1, ax2, ax3, ax4]

    color = "tab:blue"
    # Mach
    axs[0].set_ylabel("Mach number")
    axs[0].plot(out["x"], out["Ma_mix"], color=color)
    axs[0].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)

    # Velocity
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].plot(out["x"], out["v1"], color=color, label="fluid 1")
    axs[1].plot(out["x"], out["v2"], color=color, linestyle="--", label="fluid 2")
    axs[1].legend(fontsize=7)

    # Temperature
    axs[2].set_ylabel("Temperature [K]")
    axs[2].plot(out["x"], out["T1"], color=color)
    axs[2].plot(out["x"], out["T2"], color=color, linestyle="--")

    # Determinant
    axs[3].set_ylabel("Determinant")
    axs[3].plot(out["x"], out["D"], color=color)
    axs[3].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)

    # Nozzle geometry
    axs[4].fill_between(xg, -rg, +rg, color="lightgrey")
    x_closed = jnp.concatenate([xg, xg[::-1], xg[:1]])
    y_closed = jnp.concatenate([rg, -rg[::-1], rg[:1]])
    axs[4].plot(x_closed, y_closed, "k", linewidth=1.2)
    axs[4].set_xlabel("Axial coordinate x (m)")
    fig.tight_layout(pad=1)

    # ------------------------------------------------------------------ Sol1 plot (autonomous, inlet → Ma_low)
    n_sub = 2000
    ts_sub = jnp.linspace(sol1.t0, sol1.ts[0], n_sub)
    Y_sub  = jax.vmap(sol1.evaluate)(ts_sub) * Y_ref[None, :]  # (n_sub, 10) physical

    @eqx.filter_jit
    def _batch_autonomous(Y, params):
        return jax.lax.map(lambda y: nozzle_two_fluid_autonomous(0.0, y, params), Y)

    out_sub = _batch_autonomous(Y_sub, params_model)

    fig_sub = plt.figure(figsize=(5, 10))
    gs_sub  = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 3, 3, 1])
    xg_sub  = out_sub["x"]
    rg_sub  = out_sub["diameter"] / 2.0
    sb0 = fig_sub.add_subplot(gs_sub[0])
    sb1 = fig_sub.add_subplot(gs_sub[1], sharex=sb0)
    sb2 = fig_sub.add_subplot(gs_sub[2], sharex=sb0)
    sb3 = fig_sub.add_subplot(gs_sub[3], sharex=sb0)
    sb4 = fig_sub.add_subplot(gs_sub[4], sharex=sb0)
    sbs = [sb0, sb1, sb2, sb3, sb4]

    color_sub = "tab:orange"
    sbs[0].set_ylabel("Mach number")
    sbs[0].plot(out_sub["x"], out_sub["Ma_mix"], color=color_sub)
    sbs[0].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    sbs[0].set_title("Pass 1 — autonomous (inlet → Ma_low)", fontsize=9)

    sbs[1].set_ylabel("Velocity (m/s)")
    sbs[1].plot(out_sub["x"], out_sub["v1"], color=color_sub, label="fluid 1")
    sbs[1].plot(out_sub["x"], out_sub["v2"], color=color_sub, linestyle="--", label="fluid 2")
    sbs[1].legend(fontsize=7)

    sbs[2].set_ylabel("Temperature [K]")
    sbs[2].plot(out_sub["x"], out_sub["T1"], color=color_sub)
    sbs[2].plot(out_sub["x"], out_sub["T2"], color=color_sub, linestyle="--")

    sbs[3].set_ylabel("Determinant")
    sbs[3].plot(out_sub["x"], out_sub["D"], color=color_sub)
    sbs[3].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)

    sbs[4].fill_between(xg_sub, -rg_sub, +rg_sub, color="lightgrey")
    x_cl = jnp.concatenate([xg_sub, xg_sub[::-1], xg_sub[:1]])
    y_cl = jnp.concatenate([rg_sub, -rg_sub[::-1], rg_sub[:1]])
    sbs[4].plot(x_cl, y_cl, "k", linewidth=1.2)
    sbs[4].set_xlabel("Axial coordinate x (m)")
    fig_sub.tight_layout(pad=1)

    # # det_min sweep plot
    # fig2, ax_det = plt.subplots(figsize=(6, 4))
    # ax_det.plot(input_array, solution_list, "-o", color="k")
    # ax_det.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    # ax_det.axvline(float(u_crit), color="tab:red", linestyle="--",
    #                label=rf"$u_{{1,\mathrm{{crit}}}} = {float(u_crit):.3f}$ m/s")
    # ax_det.set_xlabel(r"$u_{1,\mathrm{in}}$ (m/s)")
    # ax_det.set_ylabel(r"$\min(D)$")
    # ax_det.legend()
    # fig2.tight_layout(pad=1)

    # Show figures
    plt.show()
