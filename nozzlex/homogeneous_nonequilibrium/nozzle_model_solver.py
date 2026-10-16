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
    nozzle_homogeneous_non_equilibrium_autonomous,
    nozzle_homogeneous_non_equilibrium_core,
    nozzle_homogeneous_non_equilibrium_one_component_autonomous,
    nozzle_homogeneous_non_equilibrium_one_component_autonomous_lax_eigth,
    nozzle_homogeneous_non_equilibrium_one_component_core,
    compute_inlet_state_hne,
    compute_inlet_state_hne_one_component,
)

# shorthand factory for float64 arrays
def f64(value):
    return eqx.field(
        default_factory=lambda: jnp.array(value, dtype=jnp.float64), static=False
    )

class NozzleParams(eqx.Module):
    geometry: Callable = eqx.field(static=True)
    p0_in: jnp.ndarray = f64(1.0e5)  # Pa
    T01_in: jnp.ndarray = f64(300)
    T02_in: jnp.ndarray = f64(300)
    length: jnp.ndarray = f64(5.00)  # m
    roughness: jnp.ndarray = f64(1e-6)  # m
    Ma_in: jnp.ndarray = f64(0.005)
    Ma_low: jnp.ndarray = f64(0.05)
    Ma_high: jnp.ndarray = f64(0.99)
    heat_transfer: jnp.ndarray = f64(1.0)
    wall_friction: jnp.ndarray = f64(1.0)
    alpha1_in: jnp.ndarray = f64(1e-12)
    alpha2_in: jnp.ndarray = f64(1.0)
    mixture_ratio: jnp.ndarray = f64(50)
    bubble_number: jnp.ndarray = f64(1e12)
    droplet_number: jnp.ndarray = f64(1e8)
    alpha_b: jnp.ndarray = f64(0.3)
    alpha_d: jnp.ndarray = f64(0.7)
    fluid1: Any = eqx.field(static=False, default=None)
    fluid2: Any = eqx.field(static=False, default=None)


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
    return eqx.tree_at(lambda o: getattr(o, field), obj, replace=jnp.asarray(value))


SOLVER_MAPPING = {
    "Newton": optx.Newton,
    "GaussNewton": optx.GaussNewton,
    "Dogleg": optx.Dogleg,
    "LevenbergMarquardt": optx.LevenbergMarquardt,
    "Bisection": optx.Bisection,
}

# -----------------------------------------------------------------------------------------
# One component
# -----------------------------------------------------------------------------------------


# @eqx.filter_jit
def nozzle_homogeneuous_non_equilibrium(params_model, params_solver):
    """
    1D variable-area nozzle with friction and optional heat transfer (Reynolds analogy).
    State vector: y = [x, alpha1, alpha2, rho1, rho2, u, p, h1, h2].
    """
    # --- inlet state ---
    # state1_in, state2_in = compute_static_state(
    #     params_model.p0_in,
    #     params_model.T01_in,
    #     params_model.T02_in,
    #     params_model.Ma_in,
    #     params_model.mixture_ratio,
    #     params_model.fluid1,
    #     params_model.fluid2
    # )

    fluid1 = params_model.fluid1
    fluid2 = params_model.fluid2

    state1_in = fluid1.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T01_in)
    state2_in = fluid2.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T02_in)

    p_in, rho1_in, rho2_in, h1_in, h2_in = (
        state1_in["p"],
        state1_in["rho"],
        state2_in["rho"],
        state1_in["h"],
        state2_in["h"],
    )
    q_in = (1) / (1 + params_model.mixture_ratio)
    G1, a1 = state1_in["G"], state1_in["a"]
    alpha2 = 1 / (1 + ((1 - q_in) / q_in) * (rho1_in / rho2_in))
    alpha1 = 1 - alpha2

    # h = st["h"]
    G2, a2 = state2_in["G"], state2_in["a"]

    a_in_mix = get_speed_of_sound_mixture(
        G1, alpha1, a1, rho1_in, G2, alpha2, a2, rho2_in
    )

    u_in = params_model.Ma_in * a_in_mix
    q_in = (1) / (1 + params_model.mixture_ratio)
    alpha2_in = 1 / (1 + ((1 - q_in) / q_in) * (rho2_in / rho1_in))
    alpha1_in = 1 - alpha2_in

    x_in = 1e-9  # start slightly after inlet
    y0 = jnp.array(
        [x_in, alpha1_in, alpha2_in, rho1_in, rho2_in, u_in, p_in, h1_in, h2_in]
    )

    # --- solver setup ---
    t0, t1 = 1e-9, 1e9
    solver = jxp.make_diffrax_solver(params_solver.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_solver.adjoint_name)
    term = dfx.ODETerm(eval_ode_rhs)
    ctrl = dfx.PIDController(rtol=params_solver.rtol, atol=params_solver.atol)

    event = dfx.Event(
        cond_fn=eval_end_of_domain_event,
        root_finder=optx.Newton(rtol=1e-6, atol=1e-6),
    )

    # --- first solve (find domain end) ---
    saveat = dfx.SaveAt(t1=True, dense=True, fn=eval_ode_full)
    # saveat = dfx.SaveAt(dense = True, fn=eval_ode_full)
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t1,
        dt0=1e-9,
        y0=y0,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=saveat,
        event=event,
        max_steps=200_000,
        # throw=False,
    )

    # Relative error diagnostics
    ts = jnp.linspace(sol.t0, sol.ts[0], 1000)

    # Vectorize the first step: sol.evaluate(t) for all t in ts
    evaluate_all = jax.vmap(sol.evaluate)
    out2 = evaluate_all(ts)

    # Vectorize the second step: apply the model to each (t, out)
    model_all = jax.vmap(lambda t, y: eval_ode_full(t, y, params_model))
    sol = model_all(ts, out2)

    return sol


# @eqx.filter_jit
def nozzle_homogeneuous_non_equilibrium_mach_out_one_component(params_model, params_solver):
    """
    1D variable-area nozzle with friction and optional heat transfer (Reynolds analogy).
    State vector: y = [x, alpha1, alpha2, rho1, rho2, u, p, h1, h2].
    """
    
    y_variable = compute_inlet_state_hne_one_component(params_model.p0_in,
                                                    params_model.T01_in,
                                                    params_model.T02_in,
                                                    params_model.Ma_in,
                                                    params_model.alpha1_in,
                                                    params_model.alpha2_in,
                                                    params_model.fluid1,
                                                    params_model.fluid2,
                                                    )

    x_in = jnp.array([1e-9])  # start slightly after inlet
    y0 = jnp.concatenate([x_in, y_variable])

    # --- solver setup ---
    t0, t1 = 1e-9, 1e9
    solver = jxp.make_diffrax_solver(params_solver.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_solver.adjoint_name)
    term = dfx.ODETerm(eval_ode_rhs_1c)
    ctrl = dfx.PIDController(rtol=params_solver.rtol, atol=params_solver.atol)

    event = dfx.Event(
        cond_fn=eval_end_of_domain_event,
        root_finder=optx.Newton(rtol=1e-6, atol=1e-6),
    )

    # --- first solve (find domain end) ---
    # --- first solve (find domain end) ---
    saveat = dfx.SaveAt(t1=True, fn=eval_ode_full_1c)
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=1e-9,
        y0=y0,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=saveat,
        event=event,
        max_steps=200_000,
        # throw=False,
    )

    lb = jnp.array(t0)
    ub = jnp.array(sol.ts[0])

    saveat = dfx.SaveAt(dense=True, t1=True, fn=eval_ode_full_1c)
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=sol.ts[0],
        dt0=1e-9,
        y0=y0,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=saveat,
        event=event,
        max_steps=200_000,
        # throw=False,
    )

    x_end = sol.ys["x"]

    # Optimization problem to find the max(Mach)
        # Define the conditional branches
    def true_fun(_):
        # Case: det_end < 0, return the value as is
        # return jnp.squeeze(det_end)
        return jnp.asarray(1.1)

    def false_fun(_):
        # Case: det_end >= 0, we need to find the peak (choking point)
        lb = t0
        ub = sol.ts[0]
        x0 = 0.5 * (lb + ub)
        
        # We want to minimize "D" over the trajectory saved in 'sol'
        optimizer = optx.BFGS(rtol=1e-3, atol=1e-3)
        # TODO: the solver crashes in finding the zero. but it also crasheh if the res are too low for the ode solver. it also creashes if the res of the minimizar are too high because it does not find a zero
        # Run optimization
        res = optx.minimise(
            lambda t, *args: -nozzle_homogeneous_non_equilibrium_one_component_autonomous(
                0.0, sol.evaluate(t), params_model
            )["Ma_mix"],
            optimizer,
            y0=x0,  # A guess inside the bracket
            throw=False
        )

        # Evaluate the physics at the optimized time
        yy_max = sol.evaluate(res.value)
        return nozzle_homogeneous_non_equilibrium_one_component_autonomous(0.0, yy_max, params_model)["Ma_mix"]

    # Execute the branch and return
    f_max = jax.lax.cond(jnp.squeeze(x_end < params_model.length*0.99999), true_fun, false_fun, operand=None)

    return f_max




# -----------------------------------------------------------------------------
# Helper ODE evaluation functions
# -----------------------------------------------------------------------------
def eval_ode_full_1c(t, y, args):
    return nozzle_homogeneous_non_equilibrium_one_component_autonomous(t, y, args)


def eval_ode_rhs_1c(t, y, args):
    return nozzle_homogeneous_non_equilibrium_one_component_autonomous(t, y, args)["rhs_autonomous"]


def eval_ode_mach_1c(t, y, args):
    return nozzle_homogeneous_non_equilibrium_one_component_autonomous(t, y, args)["Ma_mix"]



@eqx.filter_jit
def nozzle_homogeneuous_non_equilibrium_mach_out_one_component_lax_eight(params_model, params_solver):
    """
    1D variable-area nozzle with friction and optional heat transfer (Reynolds analogy).
    State vector: y = [x, alphav, rhov, rhol, u, p, hv, hl] (8-variable formulation).
    Uses state-variable normalization (Y_ref) so the solver sees O(1) variables.
    Returns (f_max, sol, Y_ref) where sol is the dense diffrax solution for postprocessing.
    """

    y_variable = compute_inlet_state_hne_one_component(params_model.p0_in,
                                                    params_model.T01_in,
                                                    params_model.T02_in,
                                                    params_model.Ma_in,
                                                    params_model.alpha1_in,
                                                    params_model.alpha2_in,
                                                    params_model.fluid1,
                                                    params_model.fluid2,
                                                    )

    y_variable = jnp.concatenate([y_variable[:1], y_variable[2:]])
    x_in = jnp.array([1e-9])  # start slightly after inlet
    y0 = jnp.concatenate([x_in, y_variable])

    # --- State-variable scaling ---
    Y_ref = jnp.array([
        params_model.length,  # x ~ L
        1.0,                  # αv already O(1)
        y0[2],                # ρv ~ initial vapor density
        y0[3],                # ρl ~ initial liquid density
        y0[4],                # u  ~ initial velocity
        y0[5],                # p  ~ initial pressure
        y0[6],                # hv ~ initial vapor enthalpy
        y0[7],                # hl ~ initial liquid enthalpy
    ])

    y0_hat = y0 / Y_ref

    # --- ODE wrappers with scaling ---
    def ode_full(t, y, args):
        """Evaluate full ODE output from scaled state."""
        y_phys = y * Y_ref
        return nozzle_homogeneous_non_equilibrium_one_component_autonomous_lax_eigth(t, y_phys, args)

    def ode_rhs(t, y, args):
        """RHS in scaled coordinates: dY_hat/dτ = f(Y_hat * Y_ref) / Y_ref."""
        out = ode_full(t, y, args)
        return out["rhs_autonomous"] / Y_ref

    def event_fn(t, y, args, **kwargs):
        y_phys = y * Y_ref
        x = y_phys[0]
        L = args.length
        out = nozzle_homogeneous_non_equilibrium_one_component_autonomous_lax_eigth(0.0, y_phys, args)
        Ma_mix = jnp.min(out.get("Ma_mix"))
        pmin = jnp.min(out.get("p"))
        pmin_table = jnp.stack([args.fluid1.p_min, args.fluid2.p_min])
        pmin_table = jnp.min(pmin_table)
        M = 1.1 - Ma_mix
        conditions = jnp.stack([x / L, 1 - x / L, M, pmin - pmin_table - 1])
        return jnp.min(conditions)

    # --- solver setup ---
    t0, t1 = 1e-9, 1e9
    solver = jxp.make_diffrax_solver(params_solver.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_solver.adjoint_name)
    term = dfx.ODETerm(ode_rhs)
    ctrl = dfx.PIDController(
        rtol=params_solver.rtol,
        atol=params_solver.atol,
        dtmax=5e-2,      # hard cap on pseudo-time step size
    )
    event = dfx.Event(
        cond_fn=event_fn,
        root_finder=optx.Newton(rtol=1e-3, atol=1e-3),
    )

    # --- first solve (find domain end) ---
    saveat = dfx.SaveAt(t1=True,dense=True, fn=ode_full)
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=1e-9,
        y0=y0_hat,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=saveat,
        event=event,
        max_steps=100_000,
        # throw=False,
    )

    # saveat = dfx.SaveAt(dense=True, t1=True, fn=ode_full)
    # sol = dfx.diffeqsolve(
    #     term,
    #     solver,
    #     t0=t0,
    #     t1=sol.ts[0],
    #     dt0=1e-9,
    #     y0=y0_hat,
    #     args=params_model,
    #     stepsize_controller=ctrl,
    #     adjoint=adjoint,
    #     saveat=saveat,
    #     event=event,
    #     max_steps=500_000,
    #     # throw=False,
    # )

    # x_end = sol.ys["x"]
    det_end = sol.ys["D"]

    # Check if the solver stopped due to minimum pressure
    yy_end = sol.evaluate(sol.ts[0]) * Y_ref
    p_end = nozzle_homogeneous_non_equilibrium_one_component_autonomous_lax_eigth(
        0.0, yy_end, params_model
    )["p"]
    pmin_table = jnp.minimum(params_model.fluid1.p_min, params_model.fluid2.p_min)
    hit_pmin = jnp.squeeze(p_end <= pmin_table + 15)
    # jax.debug.print("Pmin_solver:{pms}",pms=p_end)
    # jax.debug.print("Pmin_table:{pmt}",pmt=pmin_table)
    # jax.debug.print("P<pmin:{p}",p=hit_pmin)

    # Optimization problem to find the max(Mach)
    def true_fun(_):
        return jnp.asarray(1.1)

    def false_fun(_):
        # Uniform-τ grid: the solver places more steps near the throat,
        # so uniform τ gives denser coverage there where the Mach peak is sharp.
        n_coarse = 200
        ts_coarse = jnp.linspace(t0, sol.ts[0], n_coarse)

        def mach_at_tau(t):
            yy = sol.evaluate(t) * Y_ref
            return nozzle_homogeneous_non_equilibrium_one_component_autonomous_lax_eigth(
                0.0, yy, params_model
            )["Ma_mix"]

        Ma_coarse = jax.vmap(mach_at_tau)(ts_coarse)

        # BFGS in τ-space from every coarse seed
        def neg_mach_of_tau(tau, _args=None):
            yy = sol.evaluate(tau) * Y_ref
            return -nozzle_homogeneous_non_equilibrium_one_component_autonomous_lax_eigth(
                0.0, yy, params_model
            )["Ma_mix"]

        optimizer = optx.BFGS(rtol=1e-4, atol=1e-4)

        def refine_seed(tau_seed):
            res = optx.minimise(neg_mach_of_tau, optimizer, y0=tau_seed, throw=False)
            return -neg_mach_of_tau(res.value)

        Ma_refined = jax.vmap(refine_seed)(ts_coarse)

        Ma_max = jnp.max(jnp.maximum(Ma_coarse, Ma_refined))
        # jax.debug.print("Ma_coarse_max:{mc} | Ma_refined_max:{mr}", mc=jnp.max(Ma_coarse), mr=jnp.max(Ma_refined))
        return Ma_max

    # Execute the branch and return
    f_max = jax.lax.cond(jnp.squeeze(det_end < 0) | hit_pmin, true_fun, false_fun, operand=None)

    return f_max, sol, Y_ref

    


# -----------------------------------------------------------------------------
# Helper ODE evaluation functions
# -----------------------------------------------------------------------------
def eval_ode_full_1c_8(t, y, args):
    return nozzle_homogeneous_non_equilibrium_one_component_autonomous_lax_eigth(t, y, args)


def eval_ode_rhs_1c_8(t, y, args):
    return nozzle_homogeneous_non_equilibrium_one_component_autonomous_lax_eigth(t, y, args)["rhs_autonomous"]


def eval_ode_mach_1c_8(t, y, args):
    return nozzle_homogeneous_non_equilibrium_one_component_autonomous_lax_eigth(t, y, args)["Ma_mix"]


# -----------------------------------------------------------------------------
# Determinant-based critical inlet solver
# -----------------------------------------------------------------------------
# @eqx.filter_jit
def nozzle_det_at_outlet_one_component_lax_eight(params_model, params_solver):
    """
    Integrate the 8-variable HNEM ODE to the outlet and return the
    determinant D at the end.  D > 0 means subsonic, D < 0 means supersonic.
    The zero of D(Ma_in) is the critical inlet condition.
    """
    y_variable = compute_inlet_state_hne_one_component(
        params_model.p0_in,
        params_model.T01_in,
        params_model.T02_in,
        params_model.Ma_in,
        params_model.alpha1_in,
        params_model.alpha2_in,
        params_model.fluid1,
        params_model.fluid2,
    )
    y_variable = jnp.concatenate([y_variable[:1], y_variable[2:]])
    x_in = jnp.array([1e-9])
    y0 = jnp.concatenate([x_in, y_variable])

    def ode_full(t, y, args):
        return nozzle_homogeneous_non_equilibrium_one_component_autonomous_lax_eigth(t, y, args)

    def ode_rhs(t, y, args):
        return ode_full(t, y, args)["rhs_autonomous"]

    def event_fn(t, y, args, **kwargs):
        x = y[0]
        L = args.length
        out = nozzle_homogeneous_non_equilibrium_one_component_autonomous_lax_eigth(0.0, y, args)
        Ma_mix = jnp.min(out.get("Ma_mix"))
        pmin = jnp.min(out.get("p"))
        pmin_table = jnp.stack([args.fluid1.p_min, args.fluid2.p_min])
        pmin_table = jnp.min(pmin_table)
        M = 1.1 - Ma_mix
        conditions = jnp.stack([x / L, 1 - x / L, M, pmin - pmin_table])
        return jnp.min(conditions)

    t0, t1 = 1e-9, 1e9
    solver = jxp.make_diffrax_solver(params_solver.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_solver.adjoint_name)
    term = dfx.ODETerm(ode_rhs)
    ctrl = dfx.PIDController(
        rtol=params_solver.rtol,
        atol=params_solver.atol,
        dtmax=5e-2,
    )
    event = dfx.Event(
        cond_fn=event_fn,
        root_finder=optx.Newton(rtol=1e-6, atol=1e-6),
    )

    saveat = dfx.SaveAt(t1=True, fn=ode_full)
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=1e-9,
        y0=y0,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=saveat,
        event=event,
        max_steps=200_000,
    )

    return jnp.squeeze(sol.ys["D"])


def nozzle_det_min_one_component_lax_eight(params_model, params_solver):
    """
    Integrate the HNEM ODE and return the minimum determinant D along the trajectory.

    Uses the same scaled formulation as the mach_out function for numerical
    conditioning.  The return value is:
      - min(D) along the dense trajectory  if D at the end is >= 0
      - -1.0                               if D at the end is <  0
             (i.e. the solver crossed the sonic throat)

    This makes D_min(Ma_in) a continuous function with a sign change at the
    critical inlet condition, suitable for bisection.
    """
    y_variable = compute_inlet_state_hne_one_component(
        params_model.p0_in,
        params_model.T01_in,
        params_model.T02_in,
        params_model.Ma_in,
        params_model.alpha1_in,
        params_model.alpha2_in,
        params_model.fluid1,
        params_model.fluid2,
    )
    y_variable = jnp.concatenate([y_variable[:1], y_variable[2:]])
    x_in = jnp.array([1e-9])
    y0 = jnp.concatenate([x_in, y_variable])

    Y_ref = jnp.array([
        params_model.length,
        1.0,
        y0[2],
        y0[3],
        y0[4],
        y0[5],
        y0[6],
        y0[7],
    ])
    y0_hat = y0 / Y_ref

    def ode_full(t, y, args):
        y_phys = y * Y_ref
        return nozzle_homogeneous_non_equilibrium_one_component_autonomous_lax_eigth(t, y_phys, args)

    def ode_rhs(t, y, args):
        out = ode_full(t, y, args)
        return out["rhs_autonomous"] / Y_ref

    def event_fn(t, y, args, **kwargs):
        y_phys = y * Y_ref
        x = y_phys[0]
        L = args.length
        out = nozzle_homogeneous_non_equilibrium_one_component_autonomous_lax_eigth(0.0, y_phys, args)
        Ma_mix = jnp.min(out.get("Ma_mix"))
        pmin = jnp.min(out.get("p"))
        pmin_table = jnp.stack([args.fluid1.p_min, args.fluid2.p_min])
        pmin_table = jnp.min(pmin_table)
        M = 1.1 - Ma_mix
        conditions = jnp.stack([x / L, 1 - x / L, M, pmin - pmin_table - 1])
        return jnp.min(conditions)

    t0, t1 = 1e-9, 1e9
    solver = jxp.make_diffrax_solver(params_solver.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_solver.adjoint_name)
    term = dfx.ODETerm(ode_rhs)
    ctrl = dfx.PIDController(
        rtol=params_solver.rtol,
        atol=params_solver.atol,
        dtmax=5e-2,
    )
    event = dfx.Event(
        cond_fn=event_fn,
        root_finder=optx.Newton(rtol=1e-3, atol=1e-3),
    )

    saveat = dfx.SaveAt(t1=True, dense=True, fn=ode_full)
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=1e-9,
        y0=y0_hat,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=saveat,
        event=event,
        max_steps=100_000,
    )

    det_end = jnp.squeeze(sol.ys["D"])

    # Scan the dense solution on a uniform-τ grid to find the minimum D.
    # Near the throat, dx/dτ → 0 so τ stretches out → uniform τ gives dense
    # coverage exactly where D is smallest.
    n_scan = 500
    ts_scan = jnp.linspace(t0, sol.ts[0], n_scan)

    def det_at_tau(t):
        y_phys = sol.evaluate(t) * Y_ref
        return nozzle_homogeneous_non_equilibrium_one_component_autonomous_lax_eigth(
            0.0, y_phys, params_model
        )["D"]

    D_scan = jax.vmap(det_at_tau)(ts_scan)
    D_min = jnp.min(D_scan)

    # If the solver crossed the throat (det_end < 0) return the sentinel -1.
    result = jnp.where(det_end < 0, jnp.asarray(-1.0), D_min)
    jax.debug.print("Ma_in={M} | det_end={de} | det_min={dm}", M=params_model.Ma_in, de=det_end, dm=D_min)
    return result


def compute_critical_inlet_det_one_component_lax_eight(Ma_lower, Ma_upper, params_model, params_solver):
    """
    Finds the critical inlet Mach number using the minimum determinant along the
    trajectory.  D_min > 0 (subsonic) → D_min < 0 or sentinel -1 (supersonic):
    continuous sign change, no bifurcation jump.
    """

    def critical_det_residual(Mach_in, params_model):
        pm_modified = replace_param(params_model, "Ma_in", Mach_in)
        det = nozzle_det_min_one_component_lax_eight(pm_modified, params_solver)
        return det

    solver = optx.Bisection(rtol=1e-3, atol=1e-3)
    x0_initial = Ma_lower

    sol_root = optx.root_find(
        critical_det_residual,
        solver,
        x0_initial,
        args=params_model,
        throw=True,
        options={"lower": Ma_lower, "upper": Ma_upper},
        max_steps=500,
    )
    Ma_crit = sol_root.value

    return Ma_crit


# -----------------------------------------------------------------------------
# Mach-based critical inlet computation
# -----------------------------------------------------------------------------
# @eqx.filter_jit  # Don't JIT the bisection — inner mach_out function is already JIT-compiled
def compute_critical_inlet_one_component_lax_eight(Ma_lower, Ma_upper, params_model, params_solver):
    """
    Finds the inlet Mach number that makes the flow reach Mach 1 using jaxprop.
    Fully JAX-traceable (compatible with jit/tracers).
    Returns the last subsonic Ma_in (Ma_max < 1) encountered during bisection.
    """

    jax.debug.print("Entering in the critical mach function")

    # Mutable container to track the last subsonic Ma_in via side effect
    last_subsonic_Ma = [None]

    def store_subsonic(Ma_in, Ma_max):
        """Called at each bisection step — stores Ma_in if subsonic."""
        if float(Ma_max) < 1.0:
            last_subsonic_Ma[0] = float(Ma_in)

    def critical_det_residual(Mach_in, params_model):
        # update model with candidate Mach
        pm_modified = replace_param(params_model, "Ma_in", Mach_in)
        Ma_max, _, _ = nozzle_homogeneuous_non_equilibrium_mach_out_one_component_lax_eight(
            pm_modified, params_solver
        )
        jax.debug.print("Ma_in={M} | Ma_max={Ma_max}", Ma_max=Ma_max, M=Mach_in)
        jax.debug.callback(store_subsonic, Mach_in, Ma_max)
        return 1.0 - Ma_max

    # Tight tolerances needed because the residual 1-Ma_max jumps sharply
    # near the critical Ma_in (the system is extremely sensitive)
    solver = optx.Bisection(rtol=1e-3, atol=2e-3)
    x0_initial = Ma_lower

    sol_root = optx.root_find(
        critical_det_residual,
        solver,
        x0_initial,
        args=params_model,
        throw=False,
        options={"lower": Ma_lower, "upper": Ma_upper},
        max_steps=200,
        # throw=False,
    )

    # Use the last subsonic Ma_in if available, otherwise fall back to bisection result
    if last_subsonic_Ma[0] is not None:
        Ma_crit = last_subsonic_Ma[0]
        jax.debug.print("Using last subsonic Ma_in: {M}", M=Ma_crit)
    else:
        Ma_crit = sol_root.value
        jax.debug.print("Warning: no subsonic solution found, using bisection result: {M}", M=Ma_crit)
    # Ma_crit = sol_root.value # - solver.atol * 0.5

    return Ma_crit


def compute_critical_inlet_one_component(Ma_lower, Ma_upper, params_model, params_solver):
    """
    Finds the inlet Mach number that makes the flow reach Mach 1 using jaxprop.
    Fully JAX-traceable (compatible with jit/tracers).
    """
    
    def critical_det_residual(Mach_in, params_model):
        # update model with candidate Mach
        pm_modified = replace_param(params_model, "Ma_in", Mach_in)
        Ma_max = nozzle_homogeneuous_non_equilibrium_mach_out_one_component(
            pm_modified, params_solver
        )
        # max_mach = jnp.max(sol["Ma_mix"])
        # x_final = sol["x"][-1]
        # jax.debug.print("x={x} | M_in={M_in} | M_max={M}",x=x_final, M_in=Mach_in, M=max_mach,)
        jax.debug.print("Ma_in={M} | Ma_max={Ma_max}",Ma_max=Ma_max, M=Mach_in)
        return 1.0 - Ma_max**2

    # Use JAX-safe Bisection
    solver = optx.Bisection(rtol=1e-3, atol=1e-3)  # flip=True)
    x0_initial = Ma_lower
    # x0_initial = u_lower

    # JAX-friendly root find (do not convert to float inside trace)
    sol_root = optx.root_find(
        critical_det_residual,
        solver,
        x0_initial,
        args=params_model,
        throw=True,
        options={"lower": Ma_lower, "upper": Ma_upper},
    )
    Ma_crit = sol_root.value

    return Ma_crit


def eval_end_of_domain_event(t, y, args, **kwargs):
    x = y[0]
    L = args.length

    # Determinant condition
    out = nozzle_homogeneous_non_equilibrium_autonomous(0.0, y, args)
    Ma_mix = jnp.min(out.get("Ma_mix"))

    pmin = jnp.min(out.get("p"))
    pmin_table = jnp.stack([args.fluid1.p_min, args.fluid2.p_min])
    pmin_table = jnp.min(pmin_table)


    M = 1.1 - Ma_mix
    # jax.debug.print("x={x}, L-x={leng},M={D}",x=x, leng=L-x, D = out.get("Ma_mix"))
    # M = 5
    # jax.debug.print("Not considering mach as event")
    # Stack all event conditions
    # L = 0.15
    conditions = jnp.stack([x / L, 1 - x / L, M,]) # pmin - pmin_table])  # D])
    output = jnp.min(conditions)
    # jax.debug.print("out={output}",output=output)
    return output

def eval_end_of_domain_event_8(t, y, args, **kwargs):
    x = y[0]
    L = args.length

    # Determinant condition
    out = nozzle_homogeneous_non_equilibrium_one_component_autonomous_lax_eigth(0.0, y, args)
    Ma_mix = jnp.min(out.get("Ma_mix"))

    pmin = jnp.min(out.get("p"))
    pmin_table = jnp.stack([args.fluid1.p_min, args.fluid2.p_min])
    pmin_table = jnp.min(pmin_table)


    M = 1.1 - Ma_mix
    # jax.debug.print("x={x}, L-x={leng},M={D}",x=x, leng=L-x, D = out.get("Ma_mix"))
    # M = 5
    # jax.debug.print("Not considering mach as event")
    # Stack all event conditions
    # L = 0.15
    conditions = jnp.stack([x / L, 1 - x / L, M,pmin - pmin_table]) # pmin - pmin_table])  # D])
    output = jnp.min(conditions)
    # jax.debug.print("out={output}",output=output)
    return output


def eval_end_of_domain_event_classic(t, y, args, **kwargs):

    L = args.length

    x = t

    # Determinant condition
    out = nozzle_homogeneous_non_equilibrium_core(x, y, args)
    Ma_mix = jnp.min(out.get("Ma_mix"))

    M = 15.0 - Ma_mix
    # jax.debug.print("x={x}, L-x={leng},M={D}",x=x, leng=L-x, D = out.get("Ma_mix"))
    # M = 5
    # jax.debug.print("Not considering mach as event")
    # Stack all event conditions
    # L = 0.15
    conditions = jnp.stack([x / L, 1 - x / L, M])  # D])
    output = jnp.min(conditions)
    # jax.debug.print("out={output}",output=output)
    return output


    
# @eqx.filter_jit
def nozzle_homogeneuous_non_equilibrium(params_model, params_solver):
    """
    1D variable-area nozzle with friction and optional heat transfer (Reynolds analogy).
    State vector: y = [x, alpha1, alpha2, rho1, rho2, u, p, h1, h2].
    """
    # --- inlet state ---
    # state1_in, state2_in = compute_static_state(
    #     params_model.p0_in,
    #     params_model.T01_in,
    #     params_model.T02_in,
    #     params_model.Ma_in,
    #     params_model.mixture_ratio,
    #     params_model.fluid1,
    #     params_model.fluid2
    # )

    fluid1 = params_model.fluid1
    fluid2 = params_model.fluid2

    state1_in = fluid1.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T01_in)
    state2_in = fluid2.get_state(jxp.PT_INPUTS, params_model.p0_in, params_model.T02_in)

    p_in, rho1_in, rho2_in, h1_in, h2_in = (
        state1_in["p"],
        state1_in["rho"],
        state2_in["rho"],
        state1_in["h"],
        state2_in["h"],
    )
    q_in = (1) / (1 + params_model.mixture_ratio)
    G1, a1 = state1_in["G"], state1_in["a"]
    alpha2 = 1 / (1 + ((1 - q_in) / q_in) * (rho1_in / rho2_in))
    alpha1 = 1 - alpha2

    # h = st["h"]
    G2, a2 = state2_in["G"], state2_in["a"]

    a_in_mix = get_speed_of_sound_mixture(
        G1, alpha1, a1, rho1_in, G2, alpha2, a2, rho2_in
    )

    u_in = params_model.Ma_in * a_in_mix
    q_in = (1) / (1 + params_model.mixture_ratio)
    alpha2_in = 1 / (1 + ((1 - q_in) / q_in) * (rho2_in / rho1_in))
    alpha1_in = 1 - alpha2_in

    x_in = 1e-9  # start slightly after inlet
    y0 = jnp.array(
        [x_in, alpha1_in, alpha2_in, rho1_in, rho2_in, u_in, p_in, h1_in, h2_in]
    )

    # --- solver setup ---
    t0, t1 = 1e-9, 1e9
    solver = jxp.make_diffrax_solver(params_solver.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_solver.adjoint_name)
    term = dfx.ODETerm(eval_ode_rhs)
    ctrl = dfx.PIDController(rtol=params_solver.rtol, atol=params_solver.atol)

    event = dfx.Event(
        cond_fn=eval_end_of_domain_event,
        root_finder=optx.Newton(rtol=1e-6, atol=1e-6),
    )

    # --- first solve (find domain end) ---
    saveat = dfx.SaveAt(t1=True, dense=True, fn=eval_ode_full)
    # saveat = dfx.SaveAt(dense = True, fn=eval_ode_full)
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t1,
        dt0=1e-9,
        y0=y0,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=saveat,
        event=event,
        max_steps=200_000,
        # throw=False,
    )

    # Relative error diagnostics
    ts = jnp.linspace(sol.t0, sol.ts[0], 1000)

    # Vectorize the first step: sol.evaluate(t) for all t in ts
    evaluate_all = jax.vmap(sol.evaluate)
    out2 = evaluate_all(ts)

    # Vectorize the second step: apply the model to each (t, out)
    model_all = jax.vmap(lambda t, y: eval_ode_full(t, y, params_model))
    sol = model_all(ts, out2)

    return sol


# @eqx.filter_jit
def nozzle_homogeneuous_non_equilibrium_mach_out(params_model, params_solver):
    """
    1D variable-area nozzle with friction and optional heat transfer (Reynolds analogy).
    State vector: y = [x, alpha1, alpha2, rho1, rho2, u, p, h1, h2].
    """
    y_variable = compute_inlet_state_hne(params_model.p0_in,
                                                    params_model.T01_in,
                                                    params_model.T02_in,
                                                    params_model.Ma_in,
                                                    params_model.mixture_ratio,
                                                    params_model.fluid1,
                                                    params_model.fluid2,
                                                    params_model.mixture_ratio,
                                                    )


    x_in = jnp.array([1e-9])  # start slightly after inlet
    y0 = jnp.concatenate([x_in, y_variable])

    # --- solver setup ---
    t0, t1 = 1e-9, 1e9
    solver = jxp.make_diffrax_solver(params_solver.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_solver.adjoint_name)
    term = dfx.ODETerm(eval_ode_rhs)
    ctrl = dfx.PIDController(rtol=params_solver.rtol, atol=params_solver.atol)

    event = dfx.Event(
        cond_fn=eval_end_of_domain_event,
        root_finder=optx.Newton(rtol=1e-6, atol=1e-6),
    )

    # --- first solve (find domain end) ---
    # --- first solve (find domain end) ---
    saveat = dfx.SaveAt(t1=True, fn=eval_ode_full)
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=1e-9,
        y0=y0,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=saveat,
        event=event,
        max_steps=200_000,
        throw=False,
    )

    saveat = dfx.SaveAt(dense=True, t1=True, fn=eval_ode_full)
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=sol.ts[0],
        dt0=1e-9,
        y0=y0,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=saveat,
        event=event,
        max_steps=500_000,
        throw=False,
    )

    det_end = sol.ys["D"]

    def true_fun(_):
        return jnp.asarray(1.1)

    def false_fun(_):
        # Build x(τ) mapping on a fine grid
        n_fine = 1_000
        ts_fine = jnp.linspace(t0, sol.ts[0], n_fine)
        x_of_tau = jax.vmap(lambda t: sol.evaluate(t)[0])(ts_fine)

        # Grid search: evaluate Ma at all τ-grid points
        def mach_at_tau(t):
            yy = sol.evaluate(t)
            return nozzle_homogeneous_non_equilibrium_autonomous(
                0.0, yy, params_model
            )["Ma_mix"]

        Ma_grid = jax.vmap(mach_at_tau)(ts_fine)
        i_max = jnp.argmax(Ma_grid)
        Ma_grid_max = Ma_grid[i_max]
        x_grid_max = x_of_tau[i_max]

        # Refine with BFGS in x-space
        def neg_mach_of_x(x_query, _args=None):
            tau_query = jnp.interp(x_query, x_of_tau, ts_fine)
            yy = sol.evaluate(tau_query)
            return -nozzle_homogeneous_non_equilibrium_autonomous(
                0.0, yy, params_model
            )["Ma_mix"]

        optimizer = optx.BFGS(rtol=1e-3, atol=1e-3)
        res = optx.minimise(neg_mach_of_x, optimizer, y0=x_grid_max, throw=False)
        Ma_opt = -neg_mach_of_x(res.value)

        # Also check outlet Mach
        yy_end = sol.evaluate(sol.ts[0])
        Ma_end = nozzle_homogeneous_non_equilibrium_autonomous(
            0.0, yy_end, params_model
        )["Ma_mix"]

        Ma_max = jnp.maximum(jnp.maximum(Ma_opt, Ma_end), Ma_grid_max)
        # jax.debug.print("Ma_grid_max:{mg} | Ma_opt:{mp} | Ma_end:{me}", mg=Ma_grid_max, mp=Ma_opt, me=Ma_end)
        return Ma_max

    f_max = jax.lax.cond(jnp.squeeze(det_end < 0), true_fun, false_fun, operand=None)

    return f_max, sol


# -----------------------------------------------------------------------------
# Helper ODE evaluation functions
# -----------------------------------------------------------------------------
def eval_ode_full(t, y, args):
    return nozzle_homogeneous_non_equilibrium_autonomous(t, y, args)


def eval_ode_rhs(t, y, args):
    return nozzle_homogeneous_non_equilibrium_autonomous(t, y, args)["rhs_autonomous"]


def eval_ode_mach(t, y, args):
    return nozzle_homogeneous_non_equilibrium_autonomous(t, y, args)["Ma_mix"]


# -----------------------------------------------------------------------------
# Critical inlet computation
# -----------------------------------------------------------------------------
def compute_critical_inlet(Ma_lower, Ma_upper, params_model, params_solver):
    """
    Finds the inlet Mach number that makes the flow reach Mach 1 using jaxprop.
    Fully JAX-traceable (compatible with jit/tracers).
    """

    def critical_mach_residual(Mach_in, params_model):
        # update model with candidate Mach
        pm_modified = replace_param(params_model, "Ma_in", Mach_in)
        max_mach, _ = nozzle_homogeneuous_non_equilibrium_mach_out(
            pm_modified, params_solver
        )
        jax.debug.print("Ma_in:{Mi} |Ma_max:{Mm}", Mi= Mach_in, Mm=max_mach)
        return 1.0 - max_mach

    def critical_det_residual_det(Mach_in, params_model):
        # update model with candidate Mach
        pm_modified = replace_param(params_model, "Ma_in", Mach_in)
        f_min, _ = nozzle_homogeneuous_non_equilibrium_mach_out(
            pm_modified, params_solver
        )
        jax.debug.print("Ma_in={M} | det={det}",det=f_min, M=Mach_in)
        return f_min

    # Use JAX-safe Bisection
    solver = optx.Bisection(rtol=1e-3, atol=1e-3)  # flip=True)
    x0_initial = Ma_lower
    # x0_initial = u_lower

    # JAX-friendly root find (do not convert to float inside trace)
    sol_root = optx.root_find(
        critical_mach_residual,
        solver,
        x0_initial,
        args=params_model,
        throw=False,
        max_steps=300,
        options={"lower": Ma_lower, "upper": Ma_upper},
    )
    Ma_crit = sol_root.value
    

    return Ma_crit


def eval_end_of_domain_event(t, y, args, **kwargs):
    x = y[0]
    L = args.length

    # Determinant condition
    out = nozzle_homogeneous_non_equilibrium_autonomous(0.0, y, args)
    Ma_mix = jnp.min(out.get("Ma_mix"))

    M = 1.1 - Ma_mix
    # jax.debug.print("x={x}, L-x={leng},M={D}",x=x, leng=L-x, D = out.get("Ma_mix"))
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

    x = t

    # Determinant condition
    out = nozzle_homogeneous_non_equilibrium_core(x, y, args)
    Ma_mix = jnp.min(out.get("Ma_mix"))

    M = 15.0 - Ma_mix
    # jax.debug.print("x={x}, L-x={leng},M={D}",x=x, leng=L-x, D = out.get("Ma_mix"))
    # M = 5
    # jax.debug.print("Not considering mach as event")
    # Stack all event conditions
    # L = 0.15
    conditions = jnp.stack([x / L, 1 - x / L, M])  # D])
    output = jnp.min(conditions)
    # jax.debug.print("out={output}",output=output)
    return output
