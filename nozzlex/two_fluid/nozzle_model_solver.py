from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import jaxprop as jxp
import diffrax as dfx

from typing import Any, Callable

jax.config.update("jax_enable_x64", True)

from .nozzle_model_core import (
    get_speed_of_sound_mixture,
    interfacial_area_dispersed_flows,
    nozzle_two_fluid_transport_autonomous,  
    nozzle_two_fluid_transport_core,
    nozzle_two_fluid_hyperbolic_autonomous,
    nozzle_two_fluid_hyperbolic_core,
)


# shorthand factory for float64 arrays
def f64(value):
    return eqx.field(
        default_factory=lambda: jnp.array(value, dtype=jnp.float64), static=False
    )


class NozzleParams(eqx.Module):
    fluid1: Any = eqx.field(static=False)
    fluid2: Any = eqx.field(static=False)
    geometry: Callable = eqx.field(static=True)
    p0_in: jnp.ndarray = f64(1.0e5)  # Pa
    T01_in: jnp.ndarray = f64(300)
    T02_in: jnp.ndarray = f64(300)
    length: jnp.ndarray = f64(1.00)  # m
    roughness: jnp.ndarray = f64(1e-6)  # m
    Ma_in: jnp.ndarray = f64(0.005)
    Ma_low: jnp.ndarray = f64(0.05)
    Ma_high: jnp.ndarray = f64(0.99)
    u1_in: jnp.ndarray = f64(1.0)
    u2_in: jnp.ndarray = f64(1.0)
    heat_transfer: jnp.ndarray = f64(1.0)
    wall_friction: jnp.ndarray = f64(1.0)
    mixture_ratio: jnp.ndarray = f64(50)
    bubble_number: jnp.ndarray = f64(1e12)
    droplet_number: jnp.ndarray = f64(1e8)
    row_scales: Any = None  # Fixed row scales for A_mat (9,), computed from inlet
    col_scales: Any = None  # Fixed column scales for A_mat (9,), computed from inlet
    det_ref: jnp.ndarray = f64(1.0)  # |det(A_scaled)| at inlet, for RHS normalization
    alpha_b: jnp.ndarray = f64(0.3)
    alpha_d: jnp.ndarray = f64(0.7)
    alphal_in: jnp.ndarray = f64(1.0)
    alphav_in: jnp.ndarray = f64(1e-6)
    we_in_bubble: jnp.ndarray = f64(12.0)
    we_in_droplet: jnp.ndarray = f64(12.0)

class BVPSettings(eqx.Module):
    num_points: int = eqx.field(static=True)
    rtol: jnp.ndarray
    atol: jnp.ndarray
    max_steps: int = eqx.field(static=True)
    jac_mode: str = eqx.field(static=True)  # "bwd" or "fwd"
    verbose: bool = eqx.field(static=True)
    method: str = eqx.field(static=True, default="GaussNewton")
    warmup_method: str = eqx.field(static=True, default="LevenbergMarquardt")
    warmup_steps: int = eqx.field(static=True, default=0)
    solve_mode: str = eqx.field(static=True, default="mach_crit")  # or "mach_crit"


class IVPSettings(eqx.Module):
    """Settings for marching initial value problem solvers."""

    solver_name: str = eqx.field(static=True, default="Dopri5")
    adjoint_name: str = eqx.field(static=True, default="DirectAdjoint")
    number_of_points: int = eqx.field(static=True, default=50)
    rtol: float = 1e-6
    atol: float = 1e-6


class ResidualParams(eqx.Module):
    x: jnp.ndarray
    Dx: jnp.ndarray
    model: NozzleParams


def replace_param(obj, field, value):
    """Return a copy of obj with a single field replaced."""
    return eqx.tree_at(lambda o: getattr(o, field), obj, replace=jnp.asarray(value), is_leaf=lambda x: x is None)
    # return eqx.tree_at(lambda o: getattr(o, field), obj, replace=jnp.asarray(value))


SOLVER_MAPPING = {
    "Newton": optx.Newton,
    "GaussNewton": optx.GaussNewton,
    "Dogleg": optx.Dogleg,
    "LevenbergMarquardt": optx.LevenbergMarquardt,
    "Bisection": optx.Bisection,
}

def transonic_nozzle_hyperbolic_non_autonomous(
    params_model: NozzleParams,
    params_bvp: BVPSettings,
    params_ivp: IVPSettings,
):
    """
    Non-autonomous hyperbolic solver: integrates dy/dx = A^{-1}·b with x ∈ [0, L].
    Mirrors transonic_nozzle_one_component_non_autonomous in
    diffrax_nozzle_tfm_sinlge_solution.py, but for the 6-variable primitive state.
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
    uv_in = params_model.u1_in
    ul_in = params_model.u2_in

    # State: [p, uv, ul, alphav, hv, hl] — 6 vars, x excluded (it is the ODE's t)
    y_inlet = jnp.array([p_in, uv_in, ul_in, alpha_v_in, h_v_in, h_l_in])


    # --- Row/col scaling from inlet A_mat (same recipe as the TFM solver) ---
    Y_inlet_full  = jnp.concatenate([jnp.array([0.0]), y_inlet])
    params_model  = replace_param(params_model, "row_scales", jnp.ones(6))
    params_model  = replace_param(params_model, "col_scales", jnp.ones(6))
    params_model  = replace_param(params_model, "det_ref",    jnp.float64(1.0))
    out_inlet     = nozzle_two_fluid_hyperbolic_autonomous(0.0, Y_inlet_full, params_model)
    A_mat_inlet   = out_inlet["A_mat"]
    col_scales_in = jnp.maximum(jnp.max(jnp.abs(A_mat_inlet), axis=0), 1e-30)
    A_col_scaled  = A_mat_inlet / col_scales_in[None, :]
    row_scales_in = jnp.maximum(jnp.max(jnp.abs(A_col_scaled), axis=1), 1e-30)
    params_model  = replace_param(params_model, "row_scales", row_scales_in)
    params_model  = replace_param(params_model, "col_scales", col_scales_in)
    out_inlet_scaled = nozzle_two_fluid_hyperbolic_autonomous(0.0, Y_inlet_full, params_model)
    # NOTE: det_ref only rescales `rhs_autonomous`; this solver marches on
    # `out["rhs"]` (= N/det_D), which is independent of det_ref. So this is a
    # no-op here -- it is kept only so the params match the autonomous solver.
    det_ref_val = jnp.sign(out_inlet_scaled["D"]) * jnp.maximum(
        jnp.abs(out_inlet_scaled["D"]), 1e-10
    )
    params_model = replace_param(params_model, "det_ref", det_ref_val)


    def ode_full(t, y, args):
        """Reconstruct full ODE output dict from physical state and position x."""
        Y_full = jnp.concatenate([jnp.array([t]), y])
        return nozzle_two_fluid_hyperbolic_autonomous(0.0, Y_full, args)

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
        root_finder=optx.Newton(rtol=1e-8, atol=1e-8),
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
        max_steps=10_000,
        throw=False,
    )

    Y_ref = jnp.array([
        params_model.length,   # x ~ L
        p_in,                  # p  ~ initial pressure
        uv_in,                 # u  ~ initial velocity
        ul_in,                 # u  ~ initial velocity
        1.0,                   # alphav already O(1)
        h_v_in,                # hv ~ initial vapor enthalpy
        h_l_in,                # hl ~ initial liquid enthalpy
    ])

    return sol, Y_ref, params_model


# --------------------------------------------------------------------
# ----- Critical Inlet Transport Eq ai
# --------------------------------------------------------------------



@eqx.filter_jit
def nozzle_det_min_two_fluid_transport(params_model, params_solver):
    """
    Integrate the two-fluid autonomous ODE and return the minimum determinant D
    along the trajectory.

    Returns:
      - min(D) along the dense trajectory  if det at end >= 0  (subsonic run)
      - -1.0                               if det at end <  0  (crossed throat)

    This makes D_min(u1_in) a continuous function with a sign change at the
    critical inlet velocity, suitable for bisection.
    """
    fluid1 = params_model.fluid1
    fluid2 = params_model.fluid2

    # Ensure A-mat scaling fields are initialised (required by core)
    params_model = replace_param(params_model, "row_scales", jnp.ones(9))
    params_model = replace_param(params_model, "col_scales", jnp.ones(9))

    state1_in = fluid1.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T01_in)
    state2_in = fluid2.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T02_in)

    p_in   = state1_in["p"]
    rho1_in = state1_in["rho"]
    rho2_in = state2_in["rho"]
    h1_in  = state1_in["h"]
    h2_in  = state2_in["h"]

    alpha1_in = params_model.alphal_in
    alpha2_in = 1.0 - alpha1_in
    u1_in = params_model.u1_in
    u2_in = u1_in * (rho1_in / rho2_in) * (alpha1_in / alpha2_in) * (1.0 / params_model.mixture_ratio)

    We_crit_b, We_crit_d, sigma_WN = params_model.we_in_bubble, params_model.we_in_droplet, 0.0728
    u_rel_abs   = jnp.maximum(jnp.abs(u1_in - u2_in), 1e-6)
    d_bubb_we   = (We_crit_b * sigma_WN) / (rho1_in * u_rel_abs**2)
    d_drop_we   = (We_crit_d * sigma_WN) / (rho2_in * u_rel_abs**2)
    ai_b, ai_d  = interfacial_area_dispersed_flows(alpha2_in, d_bubb_we, d_drop_we)
    nb_in = ai_b / (jnp.pi * d_bubb_we**2)
    nd_in = ai_d / (jnp.pi * d_drop_we**2)
    A_inlet, _, _, _ = params_model.geometry(1e-9)
    NB_in = A_inlet * u2_in * nb_in
    ND_in = A_inlet * u1_in * nd_in
    # ND_in = 1e3  # TEMP A/B test: match single_solution seed

    y_variable = jnp.array([alpha1_in, alpha2_in, rho1_in, rho2_in, u1_in, u2_in, p_in, h1_in, h2_in, NB_in, ND_in])
    x_in = jnp.array([1e-9])
    y0 = jnp.concatenate([x_in, y_variable])

    # --- A-matrix scaling (11×11) ---
    params_model = replace_param(params_model, "row_scales", jnp.ones(11))
    params_model = replace_param(params_model, "col_scales", jnp.ones(11))
    x_tmp   = jnp.array([1e-9])
    Y_tmp   = jnp.concatenate([x_tmp, y_variable])
    out_raw = nozzle_two_fluid_transport_autonomous(0.0, Y_tmp, params_model)
    col_scales = jnp.maximum(jnp.max(jnp.abs(out_raw["A_mat"]), axis=0), 1e-30)
    row_scales = jnp.maximum(
        jnp.max(jnp.abs(out_raw["A_mat"] / col_scales[None, :]), axis=1), 1e-30
    )
    params_model = replace_param(params_model, "col_scales", col_scales)
    params_model = replace_param(params_model, "row_scales", row_scales)
    out_scaled = nozzle_two_fluid_transport_autonomous(0.0, Y_tmp, params_model)
    params_model = replace_param(
        params_model, "det_ref", jnp.maximum(jnp.abs(out_scaled["D"]), 1e-10)
    )

    # --- State-variable scaling so the solver sees O(1) variables ---
    Y_ref = jnp.array([
        params_model.length, 1.0, 1.0,
        rho1_in, rho2_in,
        u1_in, u2_in,
        p_in, h1_in, h2_in, NB_in, ND_in,
    ])
    y0_hat = y0 / Y_ref

    def ode_full(t, y, args):
        y_phys = y * Y_ref
        return nozzle_two_fluid_transport_autonomous(0.0, y_phys, args)

    def ode_rhs(t, y, args):
        out = ode_full(t, y, args)
        return out["rhs_autonomous"] / Y_ref

    def event_fn(t, y, args, **kwargs):
        y_phys = y * Y_ref
        x = y_phys[0]
        L = args.length
        out = nozzle_two_fluid_transport_autonomous(0.0, y_phys, args)
        Ma = out["Ma_mix"]
        D = out["D"]
        p = y_phys[7]
        h1 = y_phys[8]
        h2 = y_phys[9]

        h1_min = args.fluid1.h_min
        h1_max = args.fluid1.h_max

        h2_min = args.fluid2.h_min
        h2_max = args.fluid2.h_max
        p_min = jnp.maximum(args.fluid1.p_min, args.fluid2.p_min)

        # All conditions positive at the inlet; the event fires when any one
        # crosses zero. D + 10 catches the sonic fold-back (det crosses zero
        # long before Ma_mix reaches 1.1, so a Mach stop alone never fires).
        conditions = jnp.stack([
            x / L,                  # 0: x >= 0
            1 - x / L,              # 1: x <= L (subsonic run reaches outlet)
            D + 10.0,               # 2: det folded past zero (choked run)
            1.1 - Ma,               # 3: max-Mach guard
            p - p_min - 10,         # 4: p above table min
            h1 - h1_min - 10,
            h1_max - h1 + 10,
            h2 - h2_min - 10,
            h2_max - h2 + 10,

        ])
        return jnp.min(conditions)

    t0, t1 = 1e-9, 1e9
    solver  = jxp.make_diffrax_solver(params_solver.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_solver.adjoint_name)
    term = dfx.ODETerm(ode_rhs)
    ctrl = dfx.PIDController(
        rtol=params_solver.rtol,
        atol=params_solver.atol,
        dtmax=1e-3,
    )
    event = dfx.Event(
        cond_fn=event_fn,
        root_finder=optx.Bisection(rtol=1e-4, atol=1e-4),
    )

    saveat = dfx.SaveAt(t1=True, dense=True, fn=ode_full)
    sol = dfx.diffeqsolve(
        term, solver,
        t0=t0, t1=t1, dt0=1e-9,
        y0=y0_hat, args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=saveat,
        event=event,
        max_steps=4_000,
        throw=False,
    )
    jax.debug.print("det in ys:{d}", d=sol.ys["D"])
    det_end = jnp.squeeze(sol.ys["D"])
    mach_end = jnp.squeeze(sol.ys["Ma_mix"])
    x_end = jnp.squeeze(sol.ys["x"])

    jax.debug.print("Det_end:{D} | Mach_end:{M} | x_end:{xe}", xe=x_end, D=det_end, M=mach_end)
    jax.debug.print("Result solution:{res} \n", res=sol.result)
    return det_end

def compute_critical_inlet_det_two_fluid_transport(u1_lower, u1_upper, params_model, params_solver):
    """
    Finds the critical inlet velocity u1_in using the minimum determinant along
    the trajectory.  D_min > 0 (subsonic) → D_min = -1 sentinel (supersonic):
    continuous sign change, suitable for bisection.

    Returns the last subsonic u1_in seen during bisection (D_min > 0), so the
    returned solution is guaranteed to be subsonic — same strategy as
    compute_critical_inlet_one_component_lax_eight in the HNEM solver.
    """

    # Mutable container to track the last subsonic u1_in via side effect
    last_subsonic_u1 = [None]

    def store_subsonic(u1_in, det_min):
        """Called at each bisection step — stores u1_in if still subsonic."""
        jax.debug.print("In store subsonic. u_in:{uin}, det_min:{detmin}", uin=u1_in, detmin=det_min)
        if float(det_min) > 0.0:
            last_subsonic_u1[0] = float(u1_in)
            jax.debug.print("True if, the det is positive")

        jax.debug.print("Last subsonic stored is:{ls}", ls=last_subsonic_u1[0])
        jax.debug.print("\n")
        jax.debug.print("\n")

    def residual(u1_in, params_model):
        pm = replace_param(params_model, "u1_in", u1_in)
        det = nozzle_det_min_two_fluid_transport(pm, params_solver)
        jax.debug.print("u1_in={u} | det_min={d}", u=u1_in, d=det)
        jax.debug.callback(store_subsonic, u1_in, det)
        return det

    solver = optx.Bisection(rtol=1e-8, atol=1e-7) # optx.Bisection(rtol=1e-8, atol=1e-7) 
    sol_root = optx.root_find(
        residual,
        solver,
        u1_lower,
        args=params_model,
        throw=False,
        options={"lower": u1_lower, "upper": u1_upper},
        max_steps=40,
    )
    # Block until all async XLA callbacks (jax.debug.callback) have fired,
    # so that last_subsonic_u1 is fully populated before we read it.
    jax.effects_barrier()

    if last_subsonic_u1[0] is not None:
        u1_crit = last_subsonic_u1[0]
        jax.debug.print("Using last subsonic u1_in: {u}", u=u1_crit)
    else:
        u1_crit = sol_root.value
        jax.debug.print("Warning: no subsonic solution found, using bisection result: {u}", u=u1_crit)

    return jnp.array(u1_crit)


# --------------------------------------------------------------------
# ----- Compute subsonic solution
# --------------------------------------------------------------------

def nozzle_subsonic_solution_transport(
    params_model: NozzleParams,
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

    alpha1_in = params_model.alphal_in
    alpha2_in = 1.0 - alpha1_in

    u1_in = params_model.u1_in        
    u2_in = u1_in * (rho1_in / rho2_in) * (alpha1_in / alpha2_in) * (1.0 / params_model.mixture_ratio)

    We_crit_b, We_crit_d, sigma_WN = params_model.we_in_bubble, params_model.we_in_droplet, 0.0728
    u_rel_abs   = jnp.maximum(jnp.abs(u1_in - u2_in), 1e-6)
    d_bubb_we   = (We_crit_b * sigma_WN) / (rho1_in * u_rel_abs**2)
    d_drop_we   = (We_crit_d * sigma_WN) / (rho2_in * u_rel_abs**2)
    ai_b, ai_d  = interfacial_area_dispersed_flows(alpha2_in, d_bubb_we, d_drop_we)
    nb_in = ai_b / (jnp.pi * d_bubb_we**2)
    nd_in = ai_d / (jnp.pi * d_drop_we**2)
    A_inlet, _, _, _ = params_model.geometry(1e-9)
    NB_in = A_inlet * u2_in * nb_in
    ND_in = A_inlet * u1_in * nd_in

    y_variable = jnp.array([alpha1_in, alpha2_in, rho1_in, rho2_in, u1_in, u2_in, p_in, h1_in, h2_in, NB_in, ND_in])

    # --- A-matrix scaling (11×11) ---
    params_model = replace_param(params_model, "row_scales", jnp.ones(11))
    params_model = replace_param(params_model, "col_scales", jnp.ones(11))
    x_tmp   = jnp.array([1e-9])
    Y_tmp   = jnp.concatenate([x_tmp, y_variable])
    out_raw = nozzle_two_fluid_transport_autonomous(0.0, Y_tmp, params_model)
    col_scales = jnp.maximum(jnp.max(jnp.abs(out_raw["A_mat"]), axis=0), 1e-30)
    row_scales = jnp.maximum(
        jnp.max(jnp.abs(out_raw["A_mat"] / col_scales[None, :]), axis=1), 1e-30
    )
    params_model = replace_param(params_model, "col_scales", col_scales)
    params_model = replace_param(params_model, "row_scales", row_scales)
    out_scaled = nozzle_two_fluid_transport_autonomous(0.0, Y_tmp, params_model)
    params_model = replace_param(
        params_model, "det_ref", jnp.maximum(jnp.abs(out_scaled["D"]), 1e-10)
    )

    def rhs_nonaut(t, y, args):
        """Non-autonomous RHS dy/dx from the autonomous form."""
        out     = nozzle_two_fluid_transport_core(t, y, args)
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
        NB_in,
        ND_in
    ])

    solver  = jxp.make_diffrax_solver(params_ivp.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_ivp.adjoint_name)
    ctrl = dfx.PIDController(
        rtol=params_ivp.rtol,
        atol=params_ivp.atol,
        dtmax=1e-3,
    )

    # ------------------------------------------------------------------ 
    # Non-autonomous form: integrate inlet → outlet
    # ------------------------------------------------------------------
    
    def event_subsonic(t, y, args, **_kwargs):
        y_phys = y
        x = t
        out    = nozzle_two_fluid_transport_core(t, y_phys, args)
        p_min  = jnp.maximum(args.fluid1.p_min, args.fluid2.p_min)
        h1_min, h1_max = args.fluid1.h_min, args.fluid1.h_max
        h2_min, h2_max = args.fluid2.h_min, args.fluid2.h_max
        return jnp.min(jnp.stack([
            1 - x / args.length,
            out["p"] - p_min - 100,
            out["h1"] - h1_min - 10,
            h1_max - out["h1"] + 10,
            out["h2"] - h2_min - 10,
            h2_max - out["h2"] + 10,
        ]))

    sol = dfx.diffeqsolve(
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
            cond_fn=event_subsonic,
            root_finder=optx.Newton(rtol=1e-6, atol=1e-6),
        ),
        max_steps=10_000,
        throw=False,
    )

    return sol, Y_ref, params_model

def compute_inlet_from_outlet_pressure_transport(u1_lower, u1_upper, p_out, params_model, params_solver):
    """
    Finds the critical inlet velocity u1_in using the minimum determinant along
    the trajectory.  D_min > 0 (subsonic) → D_min = -1 sentinel (supersonic):
    continuous sign change, suitable for bisection.

    Returns the last subsonic u1_in seen during bisection (D_min > 0), so the
    returned solution is guaranteed to be subsonic — same strategy as
    compute_critical_inlet_one_component_lax_eight in the HNEM solver.
    """

    def residual(u1_in, params_model):
        pm = replace_param(params_model, "u1_in", u1_in)
        sol, _, _ = nozzle_subsonic_solution_transport(pm, params_solver)
        p_out_model = nozzle_two_fluid_transport_core(sol.ts[-1], sol.ys[-1], params_model)["p"]
        res = 1 - (p_out_model/p_out)
        jax.debug.print("u1_in:{u} | p/p_exp:{ddp}", u=u1_in, ddp=res)
        return res

    solver = optx.Bisection(rtol=1e-8, atol=1e-7)
    sol_root = optx.root_find(
        residual,
        solver,
        u1_lower,
        args=params_model,
        throw=False,
        options={"lower": u1_lower, "upper": u1_upper},
        max_steps=100,
    )
    # Block until all async XLA callbacks (jax.debug.callback) have fired,
    # so that last_subsonic_u1 is fully populated before we read it.
    jax.effects_barrier()
    
    u1_subsonic = sol_root.value
    jax.debug.print("Subsonic velocity:{u1}", u1=u1_subsonic)

    return u1_subsonic


def compute_inlet_from_outlet_pressure_hyperbolic(u1_lower, u1_upper, p_out, params_model, params_solver):
    """
    Shoot the inlet velocity u1_in of the 6-variable hyperbolic march so that the
    solution exits at the prescribed static pressure `p_out`.  Bisection on
    res(u1_in) = 1 - p_exit(u1_in) / p_out over [u1_lower, u1_upper].

    CAVEAT: the march terminates early if it reaches Ma_mix = 0.97 or a table
    bound, and `p_exit` is then the pressure AT THAT POINT, not at x = length.
    On a case that chokes inside the bracket the residual is therefore
    discontinuous and the bracket may not contain a sign change -- watch the
    `x_end` value in the per-iteration debug print.
    """

    def residual(u1_in, params_model):
        pm = replace_param(params_model, "u1_in", u1_in)
        pm = replace_param(pm, "u2_in", u1_in * 0.999999)
        # The march is transonic_nozzle_hyperbolic_non_autonomous;
        # nozzle_two_fluid_hyperbolic_autonomous is only its ODE right-hand side
        # (t, y, args).  params_bvp is never read by the march, hence None.
        sol, _, _ = transonic_nozzle_hyperbolic_non_autonomous(pm, None, params_solver)
        # That march uses SaveAt(..., fn=ode_full), so sol.ys is the output dict
        # rather than a state vector -- read the pressure straight out of it.
        p_out_model = sol.ys["p"][-1]
        res = 1 - (p_out_model/p_out)
        # x_end < length means the march was cut short (Ma_mix = 0.97 or a table
        # bound), so p_out_model is NOT the outlet pressure -- see the note below.
        jax.debug.print(
            "u1_in:{u} | x_end:{x} | p/p_exp:{ddp}", u=u1_in, x=sol.ts[-1], ddp=res
        )
        return res

    solver = optx.Bisection(rtol=1e-10, atol=1e-10)
    sol_root = optx.root_find(
        residual,
        solver,
        u1_lower,
        args=params_model,
        throw=False,
        options={"lower": u1_lower, "upper": u1_upper},
        max_steps=10,
    )
    # Block until all async XLA callbacks (jax.debug.callback) have fired,
    # so that last_subsonic_u1 is fully populated before we read it.
    jax.effects_barrier()
    
    u1_subsonic = sol_root.value
    jax.debug.print("Subsonic velocity:{u1}", u1=u1_subsonic)

    return u1_subsonic
