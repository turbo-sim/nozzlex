# TODO: the function with the scaling seems it is not working any more. now the one without it is working nicely check the scaling

# The difference seems to be mainly in the geom function. investigate. ok, the problem was the scaling factor S that was giving crazy results
# Now that the geom has a divergent neet to find the max mach, it is not the end any more !! ok, change to x coordinat when loking for the max mach


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
)

from nozzlex.two_fluid.nozzle_model_core import (
    nozzle_two_fluid_one_component_autonomous,
    nozzle_two_fluid_one_component_core,
    # get_speed_of_sound_mixture,
    # compute_inlet_state_hne_one_component,
)

from nozzlex.homogeneous_nonequilibrium.nozzle_model_core import (
    get_nozzle_smd,
    get_nozzle_smd_divergent,
    get_nozzle_smd_divergent_smooth,
)

# -----------------------------------------------------------------------------
# Main API — uses the lax.cond version for implicit solver compatibility
# -----------------------------------------------------------------------------
def transonic_nozzle_one_component_lax(
    params_model: NozzleParams,
    params_bvp: BVPSettings,
    params_ivp: IVPSettings,
):
    """
    Pipe / nozzle simulation using the lax.cond formulation,
    compatible with implicit ODE solvers (e.g. Kvaerno5).
    """

    fluid_vapor = params_model.fluid1
    fluid_liquid = params_model.fluid2

    print(params_model.T01_in)

    state_vapor_in = fluid_vapor.get_state_saturation(params_model.p0_in)
    state_liquid_in = fluid_liquid.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T02_in)
    # hin_l = state_liquid_in["enthalpy"]
    # # hin_v = state_vapor_in["enthalpy"]

    # # Start liquid slightly subcooled (metastable)
    # state_liquid_in = fluid_liquid.get_state(jxp.HmassP_INPUTS, hin_l, params_model.p0_in)

    print(state_liquid_in["T"])

    p_in, rho_l_in, rho_v_in, h_l_in, h_v_in = (
        state_liquid_in["pressure"],
        state_liquid_in["density"],
        state_vapor_in["density"],
        state_liquid_in["enthalpy"],
        state_vapor_in["enthalpy"],
    )

    jax.debug.print("p_in:{} | rho_l_in:{} | h_l_in:{}", p_in, rho_l_in, h_l_in)

    alpha_v_in = params_model.alphav_in
    alpha_l_in = params_model.alphal_in

    uv_in = params_model.u1_in
    ul_in = params_model.u2_in

    y_inlet = jnp.array([1e-9, alpha_v_in, alpha_l_in, rho_v_in, rho_l_in, uv_in, ul_in, p_in, h_v_in, h_l_in])

    # --- Fixed row/column scales for A_mat (equilibration from inlet A_mat) ---
    params_model = replace_param(params_model, "row_scales", jnp.ones(9))
    params_model = replace_param(params_model, "col_scales", jnp.ones(9))
    params_model = replace_param(params_model, "det_ref", jnp.float64(1.0))
    out_inlet = nozzle_two_fluid_one_component_autonomous(0.0, y_inlet, params_model)
    A_mat_inlet = out_inlet["A_mat"]
    col_scales_in = jnp.maximum(jnp.max(jnp.abs(A_mat_inlet), axis=0), 1e-30)
    A_col_scaled  = A_mat_inlet / col_scales_in[None, :]
    row_scales_in = jnp.maximum(jnp.max(jnp.abs(A_col_scaled), axis=1), 1e-30)
    params_model = replace_param(params_model, "row_scales", row_scales_in)
    params_model = replace_param(params_model, "col_scales", col_scales_in)
    out_inlet_scaled = nozzle_two_fluid_one_component_autonomous(0.0, y_inlet, params_model)
    det_ref_val = jnp.maximum(jnp.abs(out_inlet_scaled["D"]), 1e-10)
    params_model = replace_param(params_model, "det_ref", det_ref_val)

    # --- State-variable scaling ---
    # Reference scales so that Y_hat = Y / Y_ref is O(1) for all components.
    # This dramatically improves the Jacobian conditioning for the implicit solver.
    #   Y = [x,      αv,  αl,  ρv,      ρl,      uv,     ul,     p,    hv,      hl    ]
    Y_ref = jnp.array([
        params_model.length,   # x ~ L
        1.0,                   # αv already O(1)
        1.0,                   # αl already O(1)
        rho_v_in,              # ρv ~ initial vapor density
        rho_l_in,              # ρl ~ initial liquid density
        uv_in,                 # u  ~ initial velocity
        ul_in,                 # u  ~ initial velocity
        p_in,                  # p  ~ initial pressure
        h_v_in,                # hv ~ initial vapor enthalpy
        h_l_in,                # hl ~ initial liquid enthalpy
    ])

    # --- ODE wrappers with scaling ---
    def ode_full_lax(t, y_hat, args):
        """Evaluate full ODE output from scaled state."""
        y_phys = y_hat * Y_ref
        return nozzle_two_fluid_one_component_autonomous(t, y_phys, args)

    def eval_ode_rhs_scaled(t, y_hat, args):
        """RHS in scaled coordinates: dY_hat/dτ = f(Y_hat * Y_ref) / Y_ref."""
        out = ode_full_lax(t, y_hat, args)
        return out["rhs_autonomous"] / Y_ref

    y_inlet_hat = y_inlet / Y_ref

    dx_max = params_model.length/10000000

    dt_max = dx_max / out_inlet_scaled["D"]

    t0, t1 = 1e-9, 1e9
    solver = jxp.make_diffrax_solver(params_ivp.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_ivp.adjoint_name)
    term = dfx.ODETerm(eval_ode_rhs_scaled)
    ctrl = dfx.PIDController(rtol=params_ivp.rtol, atol=params_ivp.atol, dtmax=dt_max)

    def eval_end_of_domain_event_scaled(t, y, args, **kwargs):
        y_phys = y * Y_ref
        x = y_phys[0]
        L = args.length
        p_min  = jnp.maximum(args.fluid1.p_min, args.fluid2.p_min)
        h1_min, h1_max = args.fluid1.h_min, args.fluid1.h_max
        h2_min, h2_max = args.fluid2.h_min, args.fluid2.h_max
        
        out = nozzle_two_fluid_one_component_autonomous(0.0, y_phys, args)
        Ma_mix = jnp.min(out.get("Ma_mix"))
        M = 1.01 - Ma_mix
        # jax.debug.print("x={x}, L-x={leng},M={D}", x=x, leng=L - x, D=out.get("Ma_mix"))
        conditions = jnp.stack([
            x / L, 
            1 - x / L, 
            M,
            out["p"] - p_min - 100,
            out["h1"] - h1_min - 10,
            h1_max - out["h1"] + 10,
            out["h2"] - h2_min - 10,
            h2_max - out["h2"] + 10,
            ])
        
        return jnp.min(conditions)

    event = dfx.Event(
        cond_fn=eval_end_of_domain_event_scaled,
        root_finder=optx.Bisection(rtol=1e-2, atol=1e-2),
    )

    saveat = dfx.SaveAt(t1=True, dense=True, fn=ode_full_lax)
    sol1 = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=1e-6,
        y0=y_inlet_hat,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=saveat,
        event=event,
        max_steps=nsteps,
        throw=False,
    )

    return sol1, Y_ref, params_model


# -----------------------------------------------------------------------------
# Non-autonomous form: x is the independent variable, dy/dx = A^{-1}·b
# -----------------------------------------------------------------------------
def transonic_nozzle_one_component_non_autonomous(
    params_model: NozzleParams,
    params_bvp: BVPSettings,
    params_ivp: IVPSettings,
):
    """
    Non-autonomous TFM solver: integrates dy/dx = A^{-1}·b with x ∈ [0, L].

    Advantages over the autonomous form:
      - Steps are in meters → dtmax is physically meaningful and easy to set.
      - Post-processing is trivial: x is the ODE's independent variable directly.
      - Avoids the large pseudo-time step problem in the single-phase region.
    Limitation: cannot cross the sonic point (det(A) = 0 is a singularity in dy/dx).
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
    alpha_l_in = params_model.alphal_in
    uv_in = params_model.u1_in
    ul_in = params_model.u2_in

    # State: [αv, αl, ρv, ρl, uv, ul, p, hv, hl]  — 9 vars, x excluded (it is the ODE's t)
    y_inlet = jnp.array([alpha_v_in, alpha_l_in, rho_v_in, rho_l_in,
                          uv_in, ul_in, p_in, h_v_in, h_l_in])

    # --- Row/col scaling from inlet A_mat (same as autonomous solver) ---
    Y_inlet_full = jnp.concatenate([jnp.array([0.0]), y_inlet])
    params_model = replace_param(params_model, "row_scales", jnp.ones(9))
    params_model = replace_param(params_model, "col_scales", jnp.ones(9))
    params_model = replace_param(params_model, "det_ref",    jnp.float64(1.0))
    out_inlet     = nozzle_two_fluid_one_component_autonomous(0.0, Y_inlet_full, params_model)
    A_mat_inlet   = out_inlet["A_mat"]
    col_scales_in = jnp.maximum(jnp.max(jnp.abs(A_mat_inlet), axis=0), 1e-30)
    A_col_scaled  = A_mat_inlet / col_scales_in[None, :]
    row_scales_in = jnp.maximum(jnp.max(jnp.abs(A_col_scaled), axis=1), 1e-30)
    params_model  = replace_param(params_model, "row_scales", row_scales_in)
    params_model  = replace_param(params_model, "col_scales", col_scales_in)

    def ode_full(t, y, args):
        """Reconstruct full ODE output dict from physical state and position x."""
        Y_full = jnp.concatenate([jnp.array([t]), y])
        return nozzle_two_fluid_one_component_autonomous(0.0, Y_full, args)

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
        1.0,                   # αv already O(1)
        1.0,                   # αl already O(1)
        rho_v_in,              # ρv ~ initial vapor density
        rho_l_in,              # ρl ~ initial liquid density
        uv_in,                 # u  ~ initial velocity
        ul_in,                 # u  ~ initial velocity
        p_in,                  # p  ~ initial pressure
        h_v_in,                # hv ~ initial vapor enthalpy
        h_l_in,                # hl ~ initial liquid enthalpy
    ])

    return sol, Y_ref, params_model


# -----------------------------------------------------------------------------
# Two-pass transonic solver: Pass 1 autonomous (→ Ma_low), Pass 2 non-autonomous (→ outlet)
# -----------------------------------------------------------------------------
def transonic_nozzle_one_component_two_pass(
    params_model: NozzleParams,
    params_bvp: BVPSettings,
    params_ivp: IVPSettings,
):
    """
    Pass 1 (autonomous, scaled): inlet → Ma_low, stops just before throat singularity.
    Linearise at throat to jump to Ma_high.
    Pass 2 (non-autonomous, unscaled): Ma_high → outlet in physical x-space.
    No RHS negation needed: t = x is the independent variable, so dy/dx = N/D is
    well-defined in both subsonic (D>0) and supersonic (D<0) regimes.
    """

    # ---------------------------------------------------------------- fluid state at inlet
    state_vapor_in  = params_model.fluid1.get_state_saturation(params_model.p0_in)
    state_liquid_in = params_model.fluid2.get_state(
        jxp.PT_INPUTS, params_model.p0_in, params_model.T02_in
    )

    p_in    = state_liquid_in["pressure"]
    rho_l_in = state_liquid_in["density"]
    rho_v_in = state_vapor_in["density"]
    h_l_in   = state_liquid_in["enthalpy"]
    h_v_in   = state_vapor_in["enthalpy"]

    alpha_v_in = params_model.alphav_in
    alpha_l_in = params_model.alphal_in
    uv_in = params_model.u1_in
    ul_in = params_model.u2_in

    y_inlet = jnp.array([
        1e-9, alpha_v_in, alpha_l_in, rho_v_in, rho_l_in,
        uv_in, ul_in, p_in, h_v_in, h_l_in,
    ])

    # ---------------------------------------------------------------- row/col scaling
    params_model = replace_param(params_model, "row_scales", jnp.ones(9))
    params_model = replace_param(params_model, "col_scales", jnp.ones(9))
    params_model = replace_param(params_model, "det_ref",    jnp.float64(1.0))
    out_inlet     = nozzle_two_fluid_one_component_autonomous(0.0, y_inlet, params_model)
    A_mat_inlet   = out_inlet["A_mat"]
    col_scales_in = jnp.maximum(jnp.max(jnp.abs(A_mat_inlet), axis=0), 1e-30)
    A_col_scaled  = A_mat_inlet / col_scales_in[None, :]
    row_scales_in = jnp.maximum(jnp.max(jnp.abs(A_col_scaled), axis=1), 1e-30)
    params_model  = replace_param(params_model, "row_scales", row_scales_in)
    params_model  = replace_param(params_model, "col_scales", col_scales_in)
    out_inlet_sc  = nozzle_two_fluid_one_component_autonomous(0.0, y_inlet, params_model)
    det_ref_val   = jnp.maximum(jnp.abs(out_inlet_sc["D"]), 1e-10)
    params_model  = replace_param(params_model, "det_ref", det_ref_val)

    # ---------------------------------------------------------------- state-variable scaling
    Y_ref = jnp.array([
        params_model.length,
        1.0, 1.0,
        rho_v_in, rho_l_in,
        uv_in,    ul_in,
        p_in,
        h_v_in,   h_l_in,
    ])

    def ode_rhs_scaled(t, y_hat, args):
        y_phys = y_hat * Y_ref
        out    = nozzle_two_fluid_one_component_autonomous(0.0, y_phys, args)
        return out["rhs_autonomous"] / Y_ref

    y_inlet_hat = y_inlet / Y_ref

    dx_max = params_model.length / 10_000_000
    dt_max = dx_max / out_inlet_sc["D"]

    solver  = jxp.make_diffrax_solver(params_ivp.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_ivp.adjoint_name)

    # ---------------------------------------------------------------- Pass 1 event: stop at Ma_low
    def mach_low_event(t, y_hat, args, **_kwargs):
        y_phys = y_hat * Y_ref
        Ma     = nozzle_two_fluid_one_component_autonomous(0.0, y_phys, args)["Ma_mix"]
        return Ma - args.Ma_low

    ctrl1 = dfx.PIDController(rtol=params_ivp.rtol, atol=params_ivp.atol, dtmax=dt_max)
    sol1  = dfx.diffeqsolve(
        dfx.ODETerm(ode_rhs_scaled),
        solver,
        t0=1e-9, t1=1e9, dt0=1e-6,
        y0=y_inlet_hat,
        args=params_model,
        stepsize_controller=ctrl1,
        adjoint=adjoint,
        saveat=dfx.SaveAt(t1=True, dense=True),
        event=dfx.Event(
            cond_fn=mach_low_event,
            root_finder=optx.Newton(rtol=1e-7, atol=1e-7),
        ),
        max_steps=nsteps,
        throw=False,
    )

    print(f"\n--- Pass 1 (autonomous → Ma_low={params_model.Ma_low}) ---")
    print(f"  result:    {sol1.result}")
    print(f"  τ_end    = {float(sol1.ts[0]):.6e}")
    print(f"  steps    = {sol1.stats['num_steps']}")

    # ---------------------------------------------------------------- extract throat state
    y1_final_phys = sol1.evaluate(sol1.ts[0]) * Y_ref   # 10-component physical
    x_crit        = float(y1_final_phys[0])
    y_crit_9      = y1_final_phys[1:]                    # 9-component (no x)

    # ---------------------------------------------------------------- outlet check
    if x_crit / float(params_model.length) >= 0.99:
        print("\nPass 1 reached the outlet — skipping Pass 2.")
        return sol1, None, Y_ref, params_model

    # ---------------------------------------------------------------- non-autonomous RHS helper
    def rhs_nonaut(t, y9, args):
        """dy/dx = N/D in physical x-space (valid for both subsonic and supersonic)."""
        y10     = jnp.concatenate([jnp.atleast_1d(t), y9])
        out     = nozzle_two_fluid_one_component_autonomous(0.0, y10, args)
        rhs_aut = out["rhs_autonomous"]
        return rhs_aut[1:] / rhs_aut[0]

    # ---------------------------------------------------------------- linearise at throat
    f_star   = rhs_nonaut(x_crit, y_crit_9, params_model)
    det_crit = nozzle_two_fluid_one_component_autonomous(
        0.0, jnp.concatenate([jnp.atleast_1d(x_crit), y_crit_9]), params_model
    )["D"]

    def _mach_high_residual(t_val, __args):
        y_lin = y_crit_9 + f_star * (t_val - x_crit)
        y10   = jnp.concatenate([jnp.atleast_1d(t_val), y_lin])
        return nozzle_two_fluid_one_component_autonomous(0.0, y10, params_model)["Ma_mix"] - params_model.Ma_high

    sol_t0   = optx.root_find(
        _mach_high_residual,
        optx.Newton(rtol=1e-6, atol=1e-6),
        x_crit * 1.001,
    )
    t0_div   = sol_t0.value
    y0_pass2 = y_crit_9 + f_star * (t0_div - x_crit)

    print(f"\n  x_crit   = {float(x_crit):.6f} m")
    print(f"  det_crit = {float(det_crit):.3e}")
    print(f"  t0_div   = {float(t0_div):.6f} m  (Ma_lin → {params_model.Ma_high})")

    # ---------------------------------------------------------------- Pass 2: non-autonomous supersonic → outlet
    def event_pass2(t, y9, args, **_kwargs):
        x    = t
        y10  = jnp.concatenate([jnp.atleast_1d(x), y9])
        out  = nozzle_two_fluid_one_component_autonomous(0.0, y10, args)
        p_min   = jnp.maximum(args.fluid1.p_min, args.fluid2.p_min)
        h1_min, h1_max = args.fluid1.h_min, args.fluid1.h_max
        h2_min, h2_max = args.fluid2.h_min, args.fluid2.h_max
        return jnp.min(jnp.stack([
            1.0 - x / args.length,          # outlet
            out["p"] - p_min - 100,
            out["h1"] - h1_min - 10,
            h1_max - out["h1"] + 10,
            out["h2"] - h2_min - 10,
            h2_max - out["h2"] + 10,
        ]))

    ctrl2 = dfx.PIDController(
        rtol=params_ivp.rtol, atol=params_ivp.atol,
        dtmax=params_model.length / 10_000,
    )
    sol2 = dfx.diffeqsolve(
        dfx.ODETerm(rhs_nonaut),
        solver,
        t0=t0_div, t1=params_model.length * 2.0,
        dt0=1e-9,
        y0=y0_pass2,
        args=params_model,
        stepsize_controller=ctrl2,
        adjoint=adjoint,
        saveat=dfx.SaveAt(t1=True, dense=True),
        event=dfx.Event(
            cond_fn=event_pass2,
            root_finder=optx.Newton(rtol=1e-3, atol=1e-3),
        ),
        max_steps=nsteps,
        throw=False,
    )

    x2_end = float(sol2.ts[0])
    print(f"\n--- Pass 2 (non-autonomous, Ma_high → outlet) ---")
    print(f"  result:    {sol2.result}")
    print(f"  x_start  = {float(t0_div):.6f} m")
    print(f"  x_end    = {x2_end:.6f} m  (L = {float(params_model.length):.6f} m)")
    print(f"  steps    = {sol2.stats['num_steps']}")

    # diagnose which event fired
    y2_fin = sol2.evaluate(sol2.ts[0])
    y10_fin = jnp.concatenate([jnp.atleast_1d(x2_end), y2_fin])
    out_fin = nozzle_two_fluid_one_component_autonomous(0.0, y10_fin, params_model)
    _p_min  = float(jnp.maximum(params_model.fluid1.p_min, params_model.fluid2.p_min))
    _conds  = {
        "x → L  (outlet ✓)":  float(1.0 - x2_end / params_model.length),
        "p → p_min":           float(out_fin["p"] - _p_min - 100),
        "h1 → h1_min":         float(out_fin["h1"] - params_model.fluid1.h_min - 10),
        "h1 → h1_max":         float(params_model.fluid1.h_max - out_fin["h1"] + 10),
        "h2 → h2_min":         float(out_fin["h2"] - params_model.fluid2.h_min - 10),
        "h2 → h2_max":         float(params_model.fluid2.h_max - out_fin["h2"] + 10),
    }
    print(f"\n  Pass-2 event conditions at final state:")
    for name, val in _conds.items():
        marker = " ← TRIGGERED" if abs(val) < 1e-1 else ""
        print(f"    {name:<28s} = {val:+.4e}{marker}")

    return sol1, sol2, Y_ref, params_model



# -----------------------------------------------------------------------------
# SMD nozzle example — lax.cond + Kvaerno5
# -----------------------------------------------------------------------------

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


def ode_full(x, y, args):
    """Reconstruct full ODE output dict from physical state and position x."""
    return nozzle_two_fluid_one_component_core(x, y, args)

if __name__ == "__main__":

    log_file = "smd_tfm_single_solution_nb13_nd12.txt"
    tee_out = Tee(sys.stdout, log_file)
    tee_err = Tee(sys.stderr, log_file)
    sys.stdout = tee_out
    sys.stderr = tee_err

    nsteps = 100_000
    case = "b"

    table_dir = "fluid_tables"
    fluid_name = "water"
    metastable_phase = ["liquid", "vapor"]
    backend = "HEOS"
    hl_table = [300e3, 1800e3]  # J/kg
    pl_table = [1e5, 13e6]      # Pa
    # hv_table = [2200e3, 3200e3]
    # pv_table = [1e5, 13e6]      # Pa
    N_point_liquid_table = 120
    # N_point_vapor_table = 120
    # hl_table = [200e3, 1800e3]  # J/kg
    # pl_table = [5e4, 13e6]      # Pa
    hv_table = [2200e3, 3400e3]
    pv_table = [5e4, 13e6]      # Pa
    # N_point_liquid_table = 150
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

    # u1_in = 8.070846572518349
    # u1_in = 8.070846498012543
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
        bubble_number=jnp.array(1e12),
        droplet_number=1e10,
        geometry=get_nozzle_smd_divergent,
        length=0.6,  # 0.4631,
        alphav_in=1e-6,
        alphal_in=1.0-1e-6,
        u1_in=u1_in,
        u2_in=u1_in*1.000001,
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
        rtol=1e-6,
        atol=1e-6,
    )
    # Mach_max:0.17353044358840114
    # Solve the problem
    print("\n" + "-" * 60)
    print(f"Evaluating transonic solution (two-pass + {params_ivp.solver_name})")
    print("-" * 60)

    t_start = time.perf_counter()
    print(params_model)
    # sol1, sol2, Y_ref, params_model = transonic_nozzle_one_component_two_pass(
    #     params_model, params_bvp, params_ivp
    # )

    # sol1, Y_ref, params_model = transonic_nozzle_one_component_lax(
    #     params_model, params_bvp, params_ivp
    # )
    # dt_ms = (time.perf_counter() - t_start) * 1e3
    # print(f"\nTotal wall time: {dt_ms:.1f} ms")

    sol, Y_ref, params_model = transonic_nozzle_one_component_non_autonomous(params_model, params_bvp, params_ivp)

    # Post-processing: x is the independent variable directly
    n_plot = 1000
    x_grid = jnp.linspace(0.0, float(sol.ts[0]), n_plot)
    out1_list = [ode_full(x, sol.evaluate(x), params_model) for x in x_grid]
    out1 = {
        k: jnp.stack([d[k] for d in out1_list])
        for k in out1_list[0] if jnp.asarray(out1_list[0][k]).shape == ()
    }

    # ---------------------------------------------------------------- Pass 1 post-processing
    # n_fine = 5_000
    # tau0   = float(max(sol1.t0, 1e-30))
    # tau1   = float(sol1.ts[0])
    # ts1    = jnp.linspace(tau0, tau1, n_fine)

    # out1_hat = [sol1.evaluate(t) for t in ts1]
    # out1_phys = [y_hat * Y_ref for y_hat in out1_hat]
    # out1_list = [
    #     nozzle_two_fluid_one_component_autonomous(0.0, y, params_model)
    #     for y in out1_phys
    # ]
    # out1 = {
    #     k: jnp.stack([d[k] for d in out1_list])
    #     for k in out1_list[0] if jnp.asarray(out1_list[0][k]).shape == ()
    # }
    mdot_err1 = (out1["m_dot"].max() - out1["m_dot"].min()) / out1["m_dot"][0]
    print(f"Pass-1 mdot error = {float(mdot_err1):.3e}")

    # ---------------------------------------------------------------- Pass 2 post-processing
    out2 = None
    # if sol2 is not None:
    #     ts2        = jnp.linspace(float(sol2.t0), float(sol2.ts[0]), n_fine)
    #     Y_pass2    = jax.vmap(sol2.evaluate)(ts2)      # (n_fine, 9) physical (unscaled)
    #     out2_list  = [
    #         nozzle_two_fluid_one_component_autonomous(
    #             0.0,
    #             jnp.concatenate([jnp.atleast_1d(float(ts2[j])), Y_pass2[j]]),
    #             params_model,
    #         )
    #         for j in range(n_fine)
    #     ]
    #     out2 = {
    #         k: jnp.stack([d[k] for d in out2_list])
    #         for k in out2_list[0] if jnp.asarray(out2_list[0][k]).shape == ()
    #     }
    #     mdot_err2 = (out2["m_dot"].max() - out2["m_dot"].min()) / out2["m_dot"][0]
    #     print(f"Pass-2 mdot error = {float(mdot_err2):.3e}")

    # out_list = [out1]
    # input_array = jnp.array([15.0])
    # colors = plt.cm.magma(jnp.linspace(0.2, 0.8, 1))

    # ---------------------------------------------------------------- combined plot
    c2 = "tab:orange"   # Pass 1
    c1 = "tab:blue"     # Pass 2

    # _geo = out2 if out2 is not None else out1
    # xg   = _geo["x"]
    # rg   = _geo["diameter"] / 2.0

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
    axs[0].plot(out1["x"], out1["p"] * 1e-5, color=c1, label="Pass 1")
    if out2 is not None:
        axs[0].plot(out2["x"], out2["p"] * 1e-5, color=c2, label="Pass 2")
    axs[0].legend(fontsize=7)

    # --- row 2: velocity ---
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].plot(out1["x"], out1["v1"], color=c1, label="vapor")
    axs[1].plot(out1["x"], out1["v2"], color=c1, linestyle="--", label="liquid")
    if out2 is not None:
        axs[1].plot(out2["x"], out2["v1"], color=c2)
        axs[1].plot(out2["x"], out2["v2"], color=c2, linestyle="--")
    axs[1].legend(fontsize=7)

    # --- row 3: determinant ---
    axs[2].set_ylabel("Determinant")
    axs[2].plot(out1["x"], out1["D"], color=c1)
    if out2 is not None:
        axs[2].plot(out2["x"], out2["D"], color=c2)
    axs[2].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)

    # --- row 4: temperature ---
    axs[3].set_ylabel("Temperature (K)")
    axs[3].plot(out1["x"], out1["T1"], color=c1, label="vap")
    axs[3].plot(out1["x"], out1["T2"], color=c1, linestyle="--", label="liq")
    if out2 is not None:
        axs[3].plot(out2["x"], out2["T1"], color=c2)
        axs[3].plot(out2["x"], out2["T2"], color=c2, linestyle="--")
    axs[3].legend(fontsize=7)

    # --- row 5: volume fraction ---
    axs[4].set_ylabel("Vol. Fraction")
    axs[4].plot(out1["x"], out1["alpha1"], color=c1, label="α_v")
    axs[4].plot(out1["x"], out1["alpha2"], color=c1, linestyle="--", label="α_l")
    if out2 is not None:
        axs[4].plot(out2["x"], out2["alpha1"], color=c2)
        axs[4].plot(out2["x"], out2["alpha2"], color=c2, linestyle="--")
    axs[4].set_xlabel("Axial coordinate x (m)")
    axs[4].legend(fontsize=7)
    fig.tight_layout(pad=1)

    # --- nozzle geometry ---
    # fig2 = plt.figure(figsize=(8, 2))
    # plt.fill_between(xg, -rg, +rg, color="lightgrey")
    # x_closed = jnp.concatenate([xg, xg[::-1], xg[:1]])
    # y_closed = jnp.concatenate([rg, -rg[::-1], rg[:1]])
    # plt.plot(x_closed, y_closed, "k", linewidth=1.2)
    # plt.xlabel("x (m)")
    # fig2.tight_layout()

    # --- p-h diagram ---
    fluid_ph = jxp.Fluid(fluid_name, backend)
    fig_ph, ax_ph = fluid_ph.plot_phase_diagram(
        x_prop="h", y_prop="p", x_scale="linear", y_scale="log",
        plot_critical_point=True, plot_quality_isolines=True,
    )
    ax_ph.plot(out1["h1"], out1["p"], "-o", color=c1, markersize=3, label="Pass 1 vapor")
    ax_ph.plot(out1["h2"], out1["p"], "-s", color=c1, markersize=3, label="Pass 1 liquid")
    if out2 is not None:
        ax_ph.plot(out2["h1"], out2["p"], "-o", color=c2, markersize=3, label="Pass 2 vapor")
        ax_ph.plot(out2["h2"], out2["p"], "-s", color=c2, markersize=3, label="Pass 2 liquid")
    ax_ph.set_xlabel("Enthalpy (J/kg)")
    ax_ph.set_ylabel("Pressure (Pa)")
    ax_ph.legend(fontsize=7)
    fig_ph.tight_layout()

    plt.show()

    #     params_model = replace_param(params_model, "u2_in", p0_in)
    #     sol, Y_ref = transonic_nozzle_one_component_lax(params_model, params_bvp, params_ivp)
    #     dt_ms = (time.perf_counter() - t0) * 1e3
    #     out = sol.ys

    #     # Post-process: sample on a uniform *x* grid (not uniform pseudo-time)
    #     # 1. Fine τ grid to discover the x(τ) mapping (vectorized)
    #     n_fine = 5000
    #     ts_fine = jnp.linspace(sol.t0, sol.ts[0], n_fine)
    #     x_of_tau = jax.vmap(sol.evaluate)(ts_fine)[:, 0] * Y_ref[0]

    #     # 2. Build uniform x grid and interpolate back to τ
    #     n_plot = 3000
    #     x_uniform = jnp.linspace(float(x_of_tau[0]), float(x_of_tau[-1]), n_plot)
    #     ts = jnp.interp(x_uniform, x_of_tau, ts_fine)

    #     # 3. Evaluate dense output + ODE at uniform-x points
    #     out2_hat = [sol.evaluate(t) for t in ts]
    #     out2 = [y_hat * Y_ref for y_hat in out2_hat]
    #     sol2 = [
    #         nozzle_homogeneous_non_equilibrium_one_component_autonomous_lax_eigth(0.0, y, params_model)
    #         for y in out2
    #     ]
    #     keys = sol2[0].keys()
    #     out = {
    #         k: jnp.stack([d[k] for d in sol2])
    #         for k in keys if jnp.isscalar(sol2[0][k]) or sol2[0][k].shape == ()
    #     }

    #     mdot_error = (out["m_dot"].max() - out["m_dot"].min()) / out["m_dot"][0]
    #     h01_error = (out["h1"].max() - out["h1"].min()) / out["h1"][0]
    #     h02_error = (out["h2"].max() - out["h2"].min()) / out["h2"][0]
    #     s_error = (out["s1"].max() - out["s1"].min()) / out["s1"][0]

    #     print(
    #         f"Solution {i} | Solver status {sol.result} | "
    #         f"steps {int(sol.stats['num_steps']):3d} | "
    #         f"mdot error {mdot_error:0.2e} | h01 error {h01_error:0.2e} | "
    #         f"s_error {s_error:0.2e} | time {dt_ms:7.2f} ms"
    #     )

    #     solution_list.append(sol)
    #     out_list.append(out)

    # # Save results to CSV (same format as results_120a.csv)
    # for i, out in enumerate(out_list):
    #     df = pd.DataFrame({
    #         "distance": out["x"],
    #         "velocity": out["v"],
    #         "density": out["rho_mix"],
    #         "pressure": out["p"],
    #         "void_fraction": out["alpha1"],
    #         "mach_number": out["Ma_mix"],
    #         "mass_flow": out["m_dot"],
    #         "area": out["area"],
    #         "area_slope": out["dAdx"],
    #         "perimeter": out["perimeter"],
    #         "diameter": out["diameter"],
    #         "stress_wall": out["tau_w"],
    #         "determinant_D": out["D"],
    #         "enthalpy_vapor": out["h1"],
    #         "enthalpy_liquid": out["h2"],
    #         "temperature_vapor": out["T1"],
    #         "temperature_liquid": out["T2"],
    #         "entropy_vapor": out["s1"],
    #         "entropy_liquid": out["s2"],
    #         "density_vapor": out["rho1"],
    #         "density_liquid": out["rho2"],
    #         "alpha_vapor": out["alpha1"],
    #         "alpha_liquid": out["alpha2"],
    #         "Ma_vapor": out["Ma1"],
    #         "Ma_liquid": out["Ma2"],
    #         "speed_of_sound_mix": out["a_mix"],
    #         "Gruneisen_vapor": out["Gv"],
    #         "Gruneisen_liquid": out["Gl"],
    #     })
    #     df.to_csv(os.path.join("post_processing",f"results_hnem_120{case}_div_nb1e12_single.csv"), index=False)
    #     print(f"Saved results_hnem_120{case}_single_solution.csv")

    # # Create the figure
    # fig = plt.figure(figsize=(5, 10))
    # gs = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 3, 3, 3])
    # xg = out_list[0]["x"]
    # rg = out_list[0]["diameter"] / 2.0
    # ax0 = fig.add_subplot(gs[0])
    # ax1 = fig.add_subplot(gs[1], sharex=ax0)
    # ax2 = fig.add_subplot(gs[2], sharex=ax0)
    # ax3 = fig.add_subplot(gs[3], sharex=ax0)
    # ax4 = fig.add_subplot(gs[4], sharex=ax0)
    # axs = [ax0, ax1, ax2, ax3, ax4]

    # # --- row 1: pressure (bar) ---
    # axs[0].set_ylabel("Pressure (bar)")
    # for color, val, out in zip(colors, input_array, out_list):
    #     x = out["x"]
    #     axs[0].plot(x, out["p"] * 1e-5, linestyle="--", color=color)
    #     axs[0].plot(
    #         x,
    #         out["p"] * 1e-5,
    #         linestyle="-",
    #         color=color,
    #         marker="o",
    #         markersize="3",
    #         label=rf"$p_{{0,\mathrm{{in}}}} = {val:0.3f}$",
    #     )

    # # --- row 2: speed of sound ---
    # axs[1].set_ylabel("Mach mix")
    # for color, val, out in zip(colors, input_array, out_list):
    #     axs[1].plot(
    #         out["x"],
    #         out["Ma_mix"],
    #         color=color,
    #         marker="o",
    #         markersize="3",
    #     )

    # # --- row 3: mass source term ---
    # axs[2].set_ylabel("Det")
    # for color, val, out in zip(colors, input_array, out_list):
    #     axs[2].plot(out["x"], out["D"], color=color, markersize=3, marker="o")

    # # --- row 4: Gruneisen ---
    # axs[3].set_ylabel("Gruneisen")
    # for color, val, out in zip(colors, input_array, out_list):
    #     axs[3].plot(out["x"], out["Gv"], color="red")
    #     axs[3].plot(out["x"], out["Gl"], color="blue")

    # # --- row 5: volume fraction ---
    # axs[4].set_ylabel("Vol. Fraction")
    # for color, val, out in zip(colors, input_array, out_list):
    #     axs[4].plot(out["x"], out["alpha1"], markersize=3, marker="o")
    #     axs[4].plot(out["x"], out["alpha2"], markersize=3, marker="o")

    # # --- nozzle geometry ---
    # fig2 = plt.figure(figsize=(5, 10))
    # plt.fill_between(xg, -rg, +rg, color="lightgrey")
    # x_closed = jnp.concatenate([xg, xg[::-1], xg[:1]])
    # y_closed = jnp.concatenate([rg, -rg[::-1], rg[:1]])
    # plt.plot(x_closed, y_closed, "k", linewidth=1.2)

    # # --- p-h diagram ---
    # fluid_ph = jxp.Fluid(fluid_name, backend)
    # fig_ph, ax_ph = fluid_ph.plot_phase_diagram(
    #     x_prop="h", y_prop="p", x_scale="linear", y_scale="log",
    #     plot_critical_point=True, plot_quality_isolines=True,
    # )
    # for color, val, out in zip(colors, input_array, out_list):
    #     ax_ph.plot(out["h1"], out["p"], "-o", color=color, markersize=3, label="vapor (h1)")
    #     ax_ph.plot(out["h2"], out["p"], "-s", color=color, markersize=3, label="liquid (h2)")
    # ax_ph.set_xlabel("Enthalpy (kJ/kg)")
    # ax_ph.set_ylabel("Pressure (Pa)")
    # ax_ph.legend(fontsize=7)
    # fig_ph.tight_layout()

    # plt.show()

    # # Restore stdout/stderr and close log
    # sys.stdout = tee_out.stream
    # sys.stderr = tee_err.stream
    # tee_out.close()
    # tee_err.close()
    # print(f"Log saved to {log_file}")
