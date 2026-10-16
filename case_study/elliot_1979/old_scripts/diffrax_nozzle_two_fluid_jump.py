# import os
# os.environ["JAX_TRACEBACK_FILTERING"] = "off"

# TODO: the solver is not working with other mixture ratios. Need to check. Basically the second pass is too shor. It might need different interfacial area

import os
import sys
import pickle as pkl
import time
import jax
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
import optimistix as optx
import jaxprop as jxp
import matplotlib.pyplot as plt
import pandas as pd

jax.config.update("jax_enable_x64", True)

from matplotlib import gridspec

jxp.set_plot_options(grid=False)

from nozzlex.two_fluid.nozzle_model_core import (
    nozzle_two_fluid_core,
    get_nozzle_elliot,
    compute_inlet_state_hne,
    get_speed_of_sound_mixture,
    nozzle_two_fluid_autonomous,
    # nozzle_two_fluid_autonomous_negative,
    # nozzle_two_fluid_core_negative,
)

from nozzlex.two_fluid.nozzle_model_solver import (
    NozzleParams,
    BVPSettings,
    IVPSettings,
    replace_param,
    nozzle_det_min_two_fluid,
    compute_critical_inlet_det_two_fluid,
    compute_inlet_from_outlet_pressure,
    nozzle_subsonic_solution_final
)

def transonic_nozzle_one_component_third_pass_non_autonomous(
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

    # TODO: CHECK AGAING THE CORRECTNESS OF THE ALPHA AT THE INLET
    # area nitrogen
    # diameter_tube = 4.57e-3
    # diameter_tube_ext = 6.35e-3
    # Area_1 = 61 * (jnp.pi * diameter_tube * diameter_tube / 4)
    # Area_ext = (71.15e-3)**2 * jnp.pi / 4
    # Area_2 = Area_ext - 61 * jnp.pi * (diameter_tube_ext ** 2) / 4
    # Area_tot = Area_1 + Area_2
    # alpha1_in = Area_1 / Area_tot
    alpha1_in = params_model.alphal_in
    alpha2_in = 1.0 - alpha1_in
    # alpha2_in = 1 / (1 + ((1 - q_in) / q_in) * (rho2_in / rho1_in))
    # alpha1_in = 1 - alpha2_in
    jax.debug.print("alpha1:{a1} | alpha2:{a2}", a1=alpha1_in, a2=alpha2_in)

    # u_in = params_model.Ma_in * a_in_mix
    u1_in = params_model.u1_in
    # u2_in = rho1_in * u1_in * alpha1_in / (rho2_in * alpha2_in * params_model.mixture_ratio)
        
    u2_in = u1_in * (rho1_in / rho2_in) * (alpha1_in / alpha2_in) * (1.0 / params_model.mixture_ratio)

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
    
    def rhs_nonaut(t, y, args):
        """Non-autonomous RHS dy/dx from the autonomous form."""
        out     = nozzle_two_fluid_core(t, y, args)
        rhs_aut = out["rhs_autonomous"]
        return rhs_aut[1:] / rhs_aut[0]            # dy/dx = N / D  (9 components)

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

    def ode_full(t, y, args):
        y_phys = y * Y_ref
        return nozzle_two_fluid_autonomous(0.0, y_phys, args)


    y_inlet_hat = y_inlet / Y_ref

    solver  = jxp.make_diffrax_solver(params_ivp.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_ivp.adjoint_name)
    ctrl = dfx.PIDController(
        rtol=params_ivp.rtol,
        atol=params_ivp.atol,
        dtmax=1e-3,
    )

    # ------------------------------------------------------------------ Pass 1
    # Non-autonomous form: integrate inlet → Ma_low (just before throat singularity)
    # ------------------------------------------------------------------
    def mach_low_event_nonautonomous(t, y, args, **_kwargs):
        Ma = nozzle_two_fluid_core(t, y, args)["Ma_mix"]
        return Ma**2 - args.Ma_low**2

    # sol1 = dfx.diffeqsolve(
    #     dfx.ODETerm(ode_rhs_scaled),
    #     solver,
    #     t0=1e-9, t1=1e9, dt0=1e-9,
    #     y0=y_inlet_hat,
    #     args=params_model,
    #     stepsize_controller=ctrl,
    #     adjoint=adjoint,
    #     saveat=dfx.SaveAt(t1=True,dense=True, fn=eval_pass1_state),
    #     # saveat=dfx.SaveAt(t1=True,dense=True, fn=ode_full),
    #     event=dfx.Event(
    #         cond_fn=mach_low_event,
    #         root_finder=optx.Newton(rtol=1e-7, atol=1e-7),
    #     ),
    #     max_steps=10_000,
    #     # throw=False,
    # )

    sol1 = dfx.diffeqsolve(
        dfx.ODETerm(rhs_nonaut),
        solver,
        t0=1e-9, t1=1e9, dt0=1e-9,
        y0=y_variable,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=dfx.SaveAt(t1=True,dense=True),
        # saveat=dfx.SaveAt(t1=True,dense=True, fn=ode_full),
        event=dfx.Event(
            cond_fn=mach_low_event_nonautonomous,
            root_finder=optx.Newton(rtol=1e-8, atol=1e-8),
        ),
        max_steps=10_000,
        # throw=False,)
    )

    print(f"\n--- Pass 1 (non-autonomous → Ma_low) ---")
    print(f"  result:       {sol1.result}")
    print(f"  x_end       = {float(sol1.ts[0]):.6e}")
    print(f"  num_steps   = {sol1.stats['num_steps']}")

    # ------------------------------------------------------------------ Linearise at throat
    Y_ref_9 = Y_ref[1:]                             # scale for 9-component state (no x)
    x_crit  = sol1.ts[-1]                           # t == x in non-autonomous form
    y_crit_9 = sol1.ys[-1]

    final_sol = nozzle_two_fluid_core(x_crit, y_crit_9, params_model)

    p_end = final_sol["p"]
    p_exp = 98600 #Pa

    # --- Check if Pass 1 already reached the outlet ---
    if float(x_crit) / float(params_model.length) >= 0.99:
        print("\nPass 1 reached the outlet — skipping Passes 2 and 3.")
        if p_end < p_exp:
            print("Adapted expantion has lower pressure than the experiments.")
            print("Finding the inlet that gives the correct outlet.")
            u1_sub = subsonic_nozzle_outlet_pressure_function(params_model, params_ivp, u1_in, p_exp)
            # u1_sub = 2.907255667720719
            params_model = replace_param(params_model, "u1_in", u1_sub)
            sol1, Y_ref, params_model = nozzle_subsonic_solution_final(params_model, params_ivp)
            print(f"Solver status {sol1.result}")
        return sol1, None, None, Y_ref, Y_ref_9, params_model

    f_star = rhs_nonaut(x_crit, y_crit_9, params_model)
    Jy     = jax.jacfwd(lambda y: rhs_nonaut(x_crit, y, params_model))(y_crit_9)   # 9×9
    Jx     = jax.jacfwd(lambda x: rhs_nonaut(x, y_crit_9, params_model))(x_crit)   # 9

    print(f"  x_crit      = {float(x_crit):.6f} m")

    # --- Find t0_div such that Ma_mix on the linearised trajectory = Ma_high ---
    det_crit = nozzle_two_fluid_core(x_crit, y_crit_9, params_model)["D"]

    def _mach_high_residual(t_val, __args):
        """Zero when Ma_mix on the linearised trajectory equals Ma_high."""
        y_lin = y_crit_9 + f_star * (t_val - x_crit)
        return nozzle_two_fluid_core(t_val, y_lin, params_model)["Ma_mix"] - params_model.Ma_high

    sol_t0 = optx.root_find(
        _mach_high_residual,
        optx.Bisection(rtol=1e-6, atol=1e-6),
        x_crit * 1.001,
        # throw=False,
    )
    t0_div = sol_t0.value
    y0_pass2 = y_crit_9 + f_star * (t0_div - x_crit)

    ca1, ca2, cr1, cr2, cu1, cu2, cp, ch1, ch2 = y_crit_9
    a1,  a2,  r1,  r2,  u1_,  u2_,  p_,  h1_,  h2_  = y0_pass2
    names = ["alpha1", "alpha2", "rho1  ", "rho2  ", "u1    ", "u2    ", "p     ", "h1    ", "h2    "]
    units = ["      ", "      ", "kg/m3 ", "kg/m3 ", "m/s   ", "m/s   ", "Pa    ", "J/kg  ", "J/kg  "]
    crit  = [ca1, ca2, cr1, cr2, cu1, cu2, cp, ch1, ch2]
    p2    = [a1,  a2,  r1,  r2,  u1_, u2_, p_, h1_, h2_]
    print(f"\n  {'var':<8}  {'y_crit_9':>14}  {'y0_pass2':>14}  {'diff (p2-crit)':>14}  unit")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*14}  ----")
    for name, unit, vc, vp in zip(names, units, crit, p2):
        diff = float(vp) - float(vc)
        print(f"  {name}  {float(vc):14.6e}  {float(vp):14.6e}  {diff:14.6e}  {unit}")

    p_new = nozzle_two_fluid_core(t0_div,y0_pass2, params_model)["p"]
    det_new = nozzle_two_fluid_core(t0_div,y0_pass2, params_model)["D"]
    print(f"  det_crit    = {float(det_crit):.3e}")
    print(f"  t0_div      = {float(t0_div):.6f} m  (det_new = {float(det_new):.3e})")
    print(f"  p_inlet_pass_2    = {float(p_new):.3e}")
    # y0_pass2 = jnp.concatenate([jnp.atleast_1d(t0_div), y0_pass2])
    # y0_pass2_hat = y0_pass2 / Y_ref

    # ------------------------------------------------------------------ Pass 2 (scaled)
    # Autonomous τ-form (negated RHS: supersonic D<0).
    # Pass 2 now stops at Ma_mix = Ma_2_stop (second sonic approach) instead of
    # running all the way to the outlet. A second linearisation then jumps across
    # this second sonic point, and Pass 3 resumes in the subsonic (D>0) regime.
    # ------------------------------------------------------------------
    Ma_2_stop  = 1.001   # stop Pass 2 when Ma drops back down to this value
    Ma_3_start = 0.999   # linearise Pass 2 trajectory until Ma_lin = this; Pass 3 starts here

    # def ode_rhs_pass2_scaled(t, y, args):
    #     y_phys = y * Y_ref
    #     out = nozzle_two_fluid_autonomous(t, y_phys, args)
    #     # Negate RHS: in the supersonic region D = det(A) < 0, so dx/dτ = D < 0
    #     # and x decreases. Negating gives dx/dτ = -D > 0 while dy_i/dx is unchanged.
    #     return -out["rhs_autonomous"] / Y_ref

    # --- Old Pass-2 event (outlet + table bounds) kept for reference ---
    # def event_outlet(t, y, args, **_kwargs):
    #     y_phys = y * Y_ref
    #     x = y_phys[0]
    #     out    = nozzle_two_fluid_autonomous(t, y_phys, args)
    #     p_min  = jnp.maximum(args.fluid1.p_min, args.fluid2.p_min)
    #     h1_min, h1_max = args.fluid1.h_min, args.fluid1.h_max
    #     h2_min, h2_max = args.fluid2.h_min, args.fluid2.h_max
    #     return jnp.min(jnp.stack([
    #         x / args.length,
    #         1 - x / args.length,
    #         out["p"] - p_min - 100,
    #         out["h1"] - h1_min - 10,
    #         h1_max - out["h1"] + 10,
    #         out["h2"] - h2_min - 10,
    #         h2_max - out["h2"] + 10,
    #     ]))

    def event_pass2(t, y, args, **_kwargs):
        """Stop Pass 2 at Ma = Ma_2_stop (second sonic) or outlet/bounds safety."""
        y_phys = y
        x = t
        out    = nozzle_two_fluid_core(t, y_phys, args)
        p_min  = jnp.maximum(args.fluid1.p_min, args.fluid2.p_min)
        h1_min, h1_max = args.fluid1.h_min, args.fluid1.h_max
        h2_min, h2_max = args.fluid2.h_min, args.fluid2.h_max
        return jnp.min(jnp.stack([
            out["Ma_mix"] - Ma_2_stop,
            1 - x / args.length,
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
        dfx.ODETerm(rhs_nonaut),
        solver,
        t0=t0_div, t1=1e9,
        dt0=1e-9,
        y0=y0_pass2,
        args=params_model,
        stepsize_controller=ctrl2,
        adjoint=adjoint,
        saveat=dfx.SaveAt(t1=True, dense=True),
        event=dfx.Event(
            cond_fn=event_pass2,
            root_finder=optx.Newton(rtol=1e-6, atol=1e-6),
        ),
        max_steps=10_000,
        # throw=False,
    )

    y2_final= sol2.evaluate(sol2.ts[0])
    x2_end_val = float(sol2.ts[0])
    print(f"\n--- Pass 2 (autonomous negated, Ma_high → Ma={Ma_2_stop}) ---")
    print(f"  result:       {sol2.result}")
    print(f"  x_start     = {float(t0_div):.6f} m")
    print(f"  x_end       = {x2_end_val:.6f} m")
    print(f"  num_steps   = {sol2.stats['num_steps']}")

    # Diagnose which event condition triggered the end of Pass 2
    _out_fin = nozzle_two_fluid_core(x2_end_val, y2_final, params_model)
    _x_f     = x2_end_val
    _p_min   = float(jnp.maximum(params_model.fluid1.p_min, params_model.fluid2.p_min))
    _conds = {
        f"Ma → {Ma_2_stop} (✓ jump 2)":  float(_out_fin["Ma_mix"] - Ma_2_stop),
        "x → L  (outlet)":              float(1.0 - _x_f / params_model.length),
        "p → p_min":                    float(_out_fin["p"] - _p_min - 100),
        "h1 → h1_min":                  float(_out_fin["h1"] - params_model.fluid1.h_min - 10),
        "h1 → h1_max":                  float(params_model.fluid1.h_max - _out_fin["h1"] + 10),
        "h2 → h2_min":                  float(_out_fin["h2"] - params_model.fluid2.h_min - 10),
        "h2 → h2_max":                  float(params_model.fluid2.h_max - _out_fin["h2"] + 10),
    }
    print(f"\n  Pass-2 event conditions at final state (closest to 0 triggered):")
    for name, val in _conds.items():
        marker = " ← TRIGGERED" if abs(val) < 1e-1 else ""
        print(f"    {name:<28s} = {val:+.4e}{marker}")

    # --- Check if Pass 2 reached the outlet ---
    if abs(float(1.0 - x2_end_val / params_model.length)) < 1e-2:
        print("\nPass 2 reached the outlet — skipping Pass 3.")
        return sol1, sol2, None, Y_ref, Y_ref_9, params_model

    # ------------------------------------------------------------------ Linearise at Pass-2 endpoint (second jump)
    x_crit_2   = x2_end_val
    y_crit_9_2 = y2_final

    def rhs_nonaut_2(t, y, args):
        out     = nozzle_two_fluid_core(t, y, args)
        rhs_aut = out["rhs_autonomous"]
        return rhs_aut[1:] / rhs_aut[0]

    f_star_2   = rhs_nonaut_2(x_crit_2, y_crit_9_2, params_model)
    det_crit_2 = nozzle_two_fluid_core(x_crit_2, y_crit_9_2, params_model)["D"]

    def _mach_low_residual_2(t_val, __args):
        """Zero when Ma_mix on the Pass-2-linearised trajectory equals Ma_3_start."""
        y_lin = y_crit_9_2 + f_star_2 * (t_val - x_crit_2)
        return nozzle_two_fluid_core(t_val, y_lin, params_model)["Ma_mix"] - Ma_3_start

    sol_t0_3 = optx.root_find(
        _mach_low_residual_2,
        optx.Newton(rtol=1e-6, atol=1e-6),
        x_crit_2 * 1.001,
        # throw=False,
    )
    t0_div_3   = sol_t0_3.value
    y0_pass3_9 = y_crit_9_2 + f_star_2 * (t0_div_3 - x_crit_2)

    print(f"\n  det_crit_2    = {float(det_crit_2):.3e}")
    print(f"  x_crit_2      = {float(x_crit_2):.6f} m")
    print(f"  t0_div_3      = {float(t0_div_3):.6f} m  (Ma_lin → {Ma_3_start})")

    ca1, ca2, cr1, cr2, cu1, cu2, cp, ch1, ch2 = y_crit_9_2
    a1,  a2,  r1,  r2,  u1_,  u2_,  p_,  h1_,  h2_  = y0_pass3_9
    names = ["alpha1", "alpha2", "rho1  ", "rho2  ", "u1    ", "u2    ", "p     ", "h1    ", "h2    "]
    units = ["      ", "      ", "kg/m3 ", "kg/m3 ", "m/s   ", "m/s   ", "Pa    ", "J/kg  ", "J/kg  "]
    crit  = [ca1, ca2, cr1, cr2, cu1, cu2, cp, ch1, ch2]
    p3    = [a1,  a2,  r1,  r2,  u1_, u2_, p_, h1_, h2_]
    print(f"\n  {'var':<8}  {'y_end_pass2':>14}  {'y0_pass3':>14}  {'diff (p3-p2)':>14}  unit")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*14}  ----")
    for name, unit, vc, vp in zip(names, units, crit, p3):
        diff = float(vp) - float(vc)
        print(f"  {name}  {float(vc):14.6e}  {float(vp):14.6e}  {diff:14.6e}  {unit}")

    # ------------------------------------------------------------------ Pass 3 (scaled, subsonic)
    # Autonomous τ-form, NOT negated since flow is subsonic (D > 0 ⇒ dx/dτ > 0).
    # Event: outlet or table bounds.
    # ------------------------------------------------------------------

    def event_pass3(t, y, args, **_kwargs):
        y_phys = y
        x = t
        out    = nozzle_two_fluid_core(t, y_phys, args)
        p_min  = jnp.maximum(args.fluid1.p_min, args.fluid2.p_min)
        h1_min, h1_max = args.fluid1.h_min, args.fluid1.h_max
        h2_min, h2_max = args.fluid2.h_min, args.fluid2.h_max
        return jnp.min(jnp.stack([
            x / args.length,
            1 - x / args.length,
            out["p"] - p_min - 100,
            out["h1"] - h1_min - 10,
            h1_max - out["h1"] + 10,
            out["h2"] - h2_min - 10,
            h2_max - out["h2"] + 10,
        ]))

    sol3 = dfx.diffeqsolve(
        dfx.ODETerm(rhs_nonaut_2),
        solver,
        t0=t0_div_3, t1=1e9,
        dt0=1e-9,
        y0=y0_pass3_9,
        args=params_model,
        stepsize_controller=ctrl2,
        adjoint=adjoint,
        saveat=dfx.SaveAt(t1=True, dense=True),
        event=dfx.Event(
            cond_fn=event_pass3,
            root_finder=optx.Bisection(rtol=1e-6, atol=1e-6),
        ),
        max_steps=10_000,
        # throw=False,
    )

    print(f"\n--- Pass 3 (autonomous subsonic, Ma={Ma_3_start} → outlet) ---")
    print(f"  result:       {sol3.result}")
    print(f"  x_start     = {float(t0_div_3):.6f} m")
    print(f"  x_end       = {float(sol3.ts[0]):.6f} m")
    print(f"  num_steps   = {sol3.stats['num_steps']}")

    # Diagnose which event condition triggered the end of Pass 3
    _out_fin3 = nozzle_two_fluid_core(sol3.ts[0], sol3.evaluate(sol3.ts[0]), params_model)
    _x_f3     = sol3.ts[0]
    _conds3 = {
        "x → 0  (inlet hit)":    float(_x_f3 / params_model.length),
        "x → L  (outlet ✓)":     float(1.0 - _x_f3 / params_model.length),
        "p → p_min":             float(_out_fin3["p"] - _p_min - 100),
        "h1 → h1_min":           float(_out_fin3["h1"] - params_model.fluid1.h_min - 10),
        "h1 → h1_max":           float(params_model.fluid1.h_max - _out_fin3["h1"] + 10),
        "h2 → h2_min":           float(_out_fin3["h2"] - params_model.fluid2.h_min - 10),
        "h2 → h2_max":           float(params_model.fluid2.h_max - _out_fin3["h2"] + 10),
    }
    print(f"\n  Pass-3 event conditions at final state (closest to 0 triggered):")
    for name, val in _conds3.items():
        marker = " ← TRIGGERED" if abs(val) < 1e-1 else ""
        print(f"    {name:<28s} = {val:+.4e}{marker}")

    return sol1, sol2, sol3, Y_ref, Y_ref_9, params_model

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
    u1_crit = compute_critical_inlet_det_two_fluid(u1_lower=1.0, u1_upper=15.0, params_model=params_model, params_solver=params_ivp)

    # Ma_in_cr = jnp.min(min_det["Ma_mix"])
    # maximum_macxh_number = jnp.max(min_det["Ma_mix"])
    # jax.debug.print("min_det={M}", M=min_det)

    return u1_crit

def subsonic_nozzle_outlet_pressure_function(
    params_model: NozzleParams,
    params_ivp: IVPSettings,
    u1_crit,
    p_out_exp,
):
    """
    Transonic converging-diverging nozzle.
    First uses the root finding to find the critical (sonic) state,
    then integrates the ODE system with a blended RHS near the critical location.
    """
    
    # Test critical mach function
    u1_in = compute_inlet_from_outlet_pressure(u1_lower=1.0, u1_upper=u1_crit, p_out=p_out_exp, params_model=params_model, params_solver=params_ivp)

    # Ma_in_cr = jnp.min(min_det["Ma_mix"])
    # maximum_macxh_number = jnp.max(min_det["Ma_mix"])
    # jax.debug.print("min_det={M}", M=min_det)

    return u1_in

@eqx.filter_jit
def _eval_model_batch(Y, params):
    return jax.lax.map(lambda y: nozzle_two_fluid_autonomous(0.0, y, params), Y)

class Tee:
    """Write to both a file and the original stream."""
    def __init__(self, stream, filepath):
        self.stream = stream
        # Explicitly set the encoding to utf-8 here
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


# -----------------------------------------------------------------------------
# Converging-diverging nozzle example
# -----------------------------------------------------------------------------
table_dir = "fluid_tables"
fluid_name = ["water","nitrogen"]
backend = "HEOS"
h1_table = [1e3, 140e3]  # J/kg
h2_table = [90e3, 350e3]
p_table = [1e4, 3e6]   # Pa
N_h = 145
N_p = 145

if __name__ == "__main__":

    # log_file = "three_jump_non_autonomous_two_fluid.txt"
    # tee_out = Tee(sys.stdout, log_file)
    # tee_err = Tee(sys.stderr, log_file)
    # sys.stdout = tee_out
    # sys.stderr = tee_err

    
    # START_CASE = "all"
    START_CASE = [6]
    # END_CASE = 8

    data = pd.read_excel("simulation_cases.xlsx", sheet_name="Sheet1")

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


    # # Define model parameters
    # fluid_name = "air"
    # fluid = jxp.perfect_gas.get_constants(fluid_name, T_ref=300, P_ref=101325)

    # -- 1. Find critical state with continuation --

    # ---------------------------
    # Build fluids
    # ---------------------------
    params_model = NozzleParams(
        p0_in=case_data["p0_in"].iloc[0],  # Pa
        T01_in = case_data["T0_in"].iloc[0], 
        T02_in = case_data["T0_in"].iloc[0],
        length=0.311713,  # m
        roughness=0e-6,  # m
        Ma_low=0.99,
        Ma_high=1.002,
        Ma_in=0.001,
        heat_transfer=1.0,
        wall_friction=1.0,
        bubble_number=1e13,
        droplet_number=1e10,
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
        # mixture_ratio=40,
        alphal_in=0.6365890590354819,
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
        rtol=1e-12,
        atol=1e-12,
    )

    # Solve the problem
    print("\n" + "-" * 60)
    print("Evaluating transonic solution")
    print("-" * 60)
    # input_array = jnp.asarray([3.0])
    # input_array = jnp.linspace(3.0, 3.5, 2)
    # colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(input_array)))  # Generate colors
    @eqx.filter_jit
    def _batch_autonomous(Y, params):
        return jax.lax.map(lambda y: nozzle_two_fluid_autonomous(0.0, y, params), Y)

    results_list = []
    n_cases = len(case_data)
    colors = plt.cm.tab10.colors  # up to 10 distinct colors

    for i, row in enumerate(case_data.itertuples(index=True)):
        t0 = time.perf_counter()
        params_model = replace_param(params_model, "mixture_ratio", row.mixture_ratio)

        u_crit = transonic_nozzle_zero_function(params_model, params_bvp, params_ivp)
        # u_crit = 2.707151942566637
        # u_crit = 2.9072556836627497
        # u_crit =  2.882963281962361
        print(f"\nu1_crit = {float(u_crit)} m/s")
        params_model = replace_param(params_model, "u1_in", u_crit)

        sol1, sol2, sol3, Y_ref, Y_ref_9, params_model = transonic_nozzle_one_component_third_pass_non_autonomous(
            params_model, params_bvp, params_ivp
        )


        # --- Old two-pass call (Pass 1 + Pass 2 to outlet). Kept for reference. ---
        # sol1, sol2, Y_ref, Y_ref_9, params_model = transonic_nozzle_one_component_full_autonomus(
        #     params_model, params_bvp, params_ivp
        # )

        # --- New three-pass call (Pass 1 → jump 1 → Pass 2 → jump 2 → Pass 3 → outlet) ---
        # sol1, sol2, sol3, Y_ref, Y_ref_9, params_model = transonic_nozzle_one_component_third_pass(
        #     params_model, params_bvp, params_ivp
        # )

        # sol1, sol2, sol3, Y_ref, Y_ref_9, params_model = transonic_nozzle_one_component_third_pass_non_autonomous(
        #     params_model, params_bvp, params_ivp
        # )

        print("Solution done. Start generation of data for plotting")

        # sol1, sol2, Y_ref, Y_ref_9, params_model = transonic_nozzle_one_component_autonomus(
        #     params_model, params_bvp, params_ivp
        # )

        # ------------------------------------------------------------------ Post-processing
        # Pass 2 (non-autonomous, unscaled): state y is already in physical units — no Y_ref multiply.
        n_pts = 3000
        out  = None
        out3 = None

        if sol2 is not None:
            ts_all = jnp.linspace(float(sol2.t0), float(sol2.ts[0]), n_pts)
            Y_all_phys = jax.vmap(sol2.evaluate)(ts_all)  # (n_pts, 9), already physical

            # # [scaled version — only valid when state was normalised by Y_ref before solving]
            # Y_all_phys = jax.vmap(sol2.evaluate)(ts_all) * Y_ref[1:][None, :]  # (n_pts, 9) physical

            # Diagnose which state variables go out of table bounds in the dense solution
            p_all  = Y_all_phys[:, 6]   # index 6 = p in [a1, a2, r1, r2, u1, u2, p, h1, h2]
            h1_all = Y_all_phys[:, 7]
            h2_all = Y_all_phys[:, 8]
            print(f"  Pass-2 dense p  range: {float(p_all.min()):.3e} … {float(p_all.max()):.3e}  Pa  (table [{params_model.fluid1.p_min:.2e}, {params_model.fluid1.p_max:.2e}])")
            print(f"  Pass-2 dense h1 range: {float(h1_all.min()):.3e} … {float(h1_all.max()):.3e}  J/kg  (table [{params_model.fluid1.h_min:.2e}, {params_model.fluid1.h_max:.2e}])")
            print(f"  Pass-2 dense h2 range: {float(h2_all.min()):.3e} … {float(h2_all.max()):.3e}  J/kg  (table [{params_model.fluid2.h_min:.2e}, {params_model.fluid2.h_max:.2e}])")

            out_list_pts = [
                nozzle_two_fluid_core(float(ts_all[j]), Y_all_phys[j], params_model)
                for j in range(n_pts)
            ]
            out = {k: jnp.stack([d[k] for d in out_list_pts])
                   for k in out_list_pts[0] if jnp.asarray(out_list_pts[0][k]).shape == ()}
            mdot_error = (out["m_dot"].max() - out["m_dot"].min()) / out["m_dot"][0]
            print(f"mdot error (pass2) = {float(mdot_error):.3e}")

        # ------------------------------------------------------------------ Sol3 post-processing
        # Pass 3 (non-autonomous, unscaled): state y is already in physical units.
        if sol3 is not None:
            ts_all_3     = jnp.linspace(float(sol3.t0), float(sol3.ts[0]), n_pts)
            Y_all_3_phys = jax.vmap(sol3.evaluate)(ts_all_3)  # (n_pts, 9), already physical

            # # [scaled version — only valid when state was normalised by Y_ref before solving]
            # Y_all_3_phys = jax.vmap(sol3.evaluate)(ts_all_3) * Y_ref[1:][None, :]  # (n_pts, 9)

            out_list_pts_3 = [
                nozzle_two_fluid_core(float(ts_all_3[j]), Y_all_3_phys[j], params_model)
                for j in range(n_pts)
            ]
            out3 = {k: jnp.stack([d[k] for d in out_list_pts_3])
                    for k in out_list_pts_3[0] if jnp.asarray(out_list_pts_3[0][k]).shape == ()}

        # ------------------------------------------------------------------ Sol1 post-processing
        n_sub   = 3000
        ts_sub  = jnp.linspace(sol1.t0, sol1.ts[0], n_sub)
        Y_sub_9 = jax.vmap(sol1.evaluate)(ts_sub)                               # (n_sub, 9) physical (no scaling)
        Y_sub   = jnp.concatenate([ts_sub[:, None], Y_sub_9], axis=1)          # (n_sub, 10) prepend x

        out1 = _batch_autonomous(Y_sub, params_model)

        # ------------------------------------------------------------------ Save solution
        os.makedirs("tfm_solutions", exist_ok=True)
        save_data = {"pass1": out1, "pass2": out, "pass3": out3}
        fname = (
            f"solution_elliot_subsonic"
            f"_{float(params_model.p0_in)/1e5:.1f}bar"
            f"_mr{float(row.mixture_ratio):.4g}"
            f"_nb{float(params_model.bubble_number):.2e}"
            f"_nd{float(params_model.droplet_number):.2e}"
            f".pkl"
        )
        with open(os.path.join("tfm_solutions", fname), "wb") as f:
            pkl.dump(save_data, f)

        print(f"Case with mixture ratio {row.mixture_ratio} finished and saved")

        results_list.append({
            "row": row,
            "out1": out1,
            "out2": out,
            "out3": out3,
        })

        # Free JAX compilation caches accumulated during this iteration
        jax.clear_caches()

    # ------------------------------------------------------------------ Combined plot (all cases)
    fig = plt.figure(figsize=(5, 10))
    gs  = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 3, 3, 1])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    axs = [ax0, ax1, ax2, ax3, ax4]

    for idx, res in enumerate(results_list):
        color = colors[idx % len(colors)]
        out1 = res["out1"]
        out  = res["out2"]
        out3 = res["out3"]
        label = f"case {res['row'].case} (mr={res['row'].mixture_ratio:.4g})"

        # Pass 1 is drawn with a lighter alpha since it covers the subsonic approach
        axs[0].plot(out1["x"], out1["v_mix"], color=color, alpha=0.5)
        axs[1].plot(out1["x"], out1["v1"],    color=color, alpha=0.5)
        axs[1].plot(out1["x"], out1["v2"],    color=color, alpha=0.5, linestyle="--")
        axs[2].plot(out1["x"], out1["p"],     color=color, alpha=0.5)
        axs[3].plot(out1["x"], out1["D"],     color=color, alpha=0.5)

        if out is not None:
            axs[0].plot(out["x"],  out["v_mix"], color=color)
            axs[1].plot(out["x"],  out["v1"],    color=color, label=f"{label} f1")
            axs[1].plot(out["x"],  out["v2"],    color=color, linestyle="--", label=f"{label} f2")
            axs[2].plot(out["x"],  out["p"],     color=color)
            axs[3].plot(out["x"],  out["D"],     color=color)

        if out3 is not None:
            axs[0].plot(out3["x"], out3["v_mix"], color=color)
            axs[1].plot(out3["x"], out3["v1"],    color=color)
            axs[1].plot(out3["x"], out3["v2"],    color=color, linestyle="--")
            axs[2].plot(out3["x"], out3["p"],     color=color)
            axs[3].plot(out3["x"], out3["D"],     color=color)

        # Add a proxy artist for the legend keyed by case
        axs[0].plot([], [], color=color, label=label)

    axs[0].set_ylabel("Mixture velocity (m/s)")
    axs[0].legend(fontsize=7)

    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].legend(fontsize=7)

    axs[2].set_ylabel("Pressure [Pa]")

    axs[3].set_ylabel("Determinant")
    axs[3].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)

    # Geometry: use the last case's last available pass for the nozzle outline
    _last = results_list[-1]
    _geo = _last["out3"] if _last["out3"] is not None else (_last["out2"] if _last["out2"] is not None else _last["out1"])
    xg = _geo["x"]
    rg = _geo["diameter"] / 2.0
    x_outlet = float(params_model.length)
    for ax in axs:
        ax.axvline(x_outlet, color="gray", linestyle=":", linewidth=0.9)
    axs[4].fill_between(xg, -rg, +rg, color="lightgrey")
    x_closed = jnp.concatenate([xg, xg[::-1], xg[:1]])
    y_closed = jnp.concatenate([rg, -rg[::-1], rg[:1]])
    axs[4].plot(x_closed, y_closed, "k", linewidth=1.2)
    axs[4].set_xlabel("Axial coordinate x (m)")
    fig.tight_layout(pad=1)

    # det_min sweep plot
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



# TODO: Test another Nd and Nb with same settings and see what happend. Need bigger LUT, more down in pressure
# TODO: Test second jump trhough the singularity and see what happens
