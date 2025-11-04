# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
# This script implements a collocation-based solver for 1D quasi-1D nozzle flow
# using JAX, Optimistix, and Equinox. It supports two problem formulations:
#   1. "mach_in"    – solve with a prescribed inlet Mach number.
#   2. "mach_crit"  – solve with a prescribed maximum (critical) Mach number.
#
# Main components:
#
# 1. Data containers
#    - NozzleParams     : Stores all geometric, inlet, and target flow parameters.
#    - SolverSettings   : Stores numerical solver configuration (e.g., tolerances,
#                         solver type, collocation points, solve mode).
#    - ResidualParams   : Groups model, fluid, and discretization data for residuals.
#
# 2. Main solver
#    - solve_nozzle_model_collocation() :
#        Given an initial guess and problem parameters, builds the appropriate
#        residual function (based on solve_mode), sets up the nonlinear solver,
#        and returns the converged flowfield and solver statistics.
#
# 3. Residual functions
#    - build_residual_vector_mach_inlet()   : Residual formulation enforcing a
#                                             target inlet Mach number.
#    - build_residual_vector_mach_critical(): Residual formulation enforcing a
#                                             target maximum Mach number within
#                                             the domain, found via Newton search.
#
# 4. Flowfield utilities
#    - find_maximum_mach()    : Finds the location and value of the maximum Mach
#                               number in the domain from Chebyshev-Lobatto data.
#    - compute_static_state() : Computes static flow state from stagnation state
#                               and Mach number.
#    - split_z()              : Unpacks concatenated solution vector into velocity,
#                               log-density, and log-pressure arrays.
#    - evaluate_ode_rhs()     : Evaluates the nozzle right-hand-side model at all
#                               collocation points.
#    - initialize_flowfield() : Generates an initial guess for the solver using
#                               a parabolic Mach profile.
#
# 5. Chebyshev-Lobatto utilities
#    - chebyshev_lobatto_basis()                : Returns collocation nodes and
#                                                 differentiation matrix.
#    - chebyshev_lobatto_interpolate()          : Interpolates nodal data at given
#                                                 points (value only).
#    - chebyshev_lobatto_interpolate_and_derivative():
#                                                 Interpolates nodal data and
#                                                 returns derivative.
#
# 6. Helper
#    - replace_param() : Creates a copy of an Equinox module with one field replaced.
#
# Workflow:
#    (a) Define NozzleParams and SolverSettings.
#    (b) Generate an initial guess with initialize_flowfield().
#    (c) Call solve_nozzle_model_collocation() with chosen solve_mode.
#    (d) Post-process output data (e.g., interpolate for plotting).
# -----------------------------------------------------------------------------

from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import jaxprop as jxp
import diffrax as dfx

from typing import Any, Callable

jax.config.update("jax_enable_x64", True)

from .nozzle_model_core import nozzle_single_phase_core, nozzle_single_phase_autonomous_ph


# shorthand factory for float64 arrays
def f64(value):
    return eqx.field(
        default_factory=lambda: jnp.array(value, dtype=jnp.float64),
        static=False
    )

class NozzleParams(eqx.Module):
    fluid: Any = eqx.field(static=False)
    geometry: Callable = eqx.field(static=True)
    p0_in: jnp.ndarray = f64(1.0e5)       # Pa
    d0_in: jnp.ndarray = f64(1.20)        # kg/m³
    h0_in: jnp.ndarray = f64(300e3)
    D_in: jnp.ndarray = f64(0.050)        # m
    length: jnp.ndarray = f64(5.00)       # m
    roughness: jnp.ndarray = f64(1e-6)    # m
    T_wall: jnp.ndarray = f64(300.0)      # K
    Ma_in: jnp.ndarray = f64(0.1)
    Ma_low: jnp.ndarray = f64(0.95)
    Ma_high: jnp.ndarray = f64(1.05)
    Ma_target: jnp.ndarray = f64(1.0)
    heat_transfer: jnp.ndarray = f64(0.0)
    wall_friction: jnp.ndarray = f64(0.0)
    p_termination: jnp.ndarray = f64(0.0)


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
}


# ---------- Main function call ----------
@eqx.filter_jit
def solve_nozzle_model_collocation(
    initial_guess,
    params_model,
    params_solver,
):
    """
    Solves the collocation system using a warmup solver (e.g., Levenberg-Marquardt)
    followed by a main solver (e.g., Gauss-Newton).
    Returns the evaluated flowfield and the final solver result.
    """

    # Compute the Chebyshev basis (only once per call)
    x, D = chebyshev_lobatto_basis(params_solver.num_points, 0.0, params_model.length)

    # Build the function to compute the residual vector
    residual_args = ResidualParams(x=x, Dx=D, model=params_model)

    # Select correct residual function
    if params_solver.solve_mode == "mach_in":
        residual_fn = get_residual_mach_inlet
    elif params_solver.solve_mode == "mach_crit":
        residual_fn = get_residual_mach_critical
    else:
        raise ValueError(f"Unknown solve_mode: {params_solver.solve_mode}")

    # Create the warmup and main solvers
    vars = {"step", "loss", "accepted", "step_size"}
    vset = frozenset(vars) if params_solver.verbose else frozenset()
    solver_warmup = SOLVER_MAPPING[params_solver.warmup_method](
        rtol=params_solver.rtol,
        atol=params_solver.atol,
        verbose=vset,
    )
    solver_main = SOLVER_MAPPING[params_solver.method](
        rtol=params_solver.rtol,
        atol=params_solver.atol,
        # verbose=vset,
    )

    # Use a robust solver for a fet iterations to warmup
    solution_warmup = optx.root_find(
        fn=residual_fn,
        args=residual_args,
        y0=initial_guess,
        solver=solver_warmup,
        options={"jac": params_solver.jac_mode},
        max_steps=params_solver.warmup_steps,     
        throw=False,
    )

    # Solve the problem using the main solver
    solution = optx.root_find(
        fn=residual_fn,
        args=residual_args,
        y0=solution_warmup.value,
        solver=solver_main,
        options={"jac": params_solver.jac_mode},
        max_steps=params_solver.max_steps,
    )

    # Evaluate the flowfield at the converged solution
    out_data = evaluate_ode_rhs(x, solution.value, params_model)

    return out_data, solution


# ---------- Create function handle for the residual vector ----------
def get_residual_mach_inlet(z, params: ResidualParams):

    # We solve for ln(d) and ln(p) instead of d and p directly.
    #   - This enforces strict positivity of density and pressure without explicit constraints.
    #   - The PDE in ln(d) or ln(p) form contains a 1/d or 1/p factor in the derivative term:
    #       d/dx[ln(d)] = (1/d) * d(d)/dx
    #       d/dx[ln(p)] = (1/p) * d(p)/dx
    #     so when forming the residuals, the nonlinear terms N_all[:,1] and N_all[:,2] must be divided
    #     by the current d and p values, respectively.
    #   - This scaling also normalizes the residual magnitude for variables with very different units
    #     and prevents density/pressure from dominating the Jacobian purely due to scale.

    # Unpack solution vector
    u, ln_d, ln_p = split_z(z, params.x.shape[0])
    d = jnp.exp(ln_d)
    p = jnp.exp(ln_p)

    # Compute right hand side of the ODE
    out = evaluate_ode_rhs(params.x, z, params.model)
    N_all = out["N"]
    D_tau = out["D"]

    # Evaluate residuals at collocation points
    # Multiply-through PDE residuals (no division by D)
    R_u = (params.Dx @ u) - N_all[:, 0] / D_tau
    R_d = (params.Dx @ ln_d) - N_all[:, 1] / D_tau / d
    R_p = (params.Dx @ ln_p) - N_all[:, 2] / D_tau / p

    # Evaluate residual at boundary condition
    R_u = R_u.at[0].set(params.model.Ma_in - out["Ma"][0])
    R_d = R_d.at[0].set(jnp.log(params.model.d0_in / out["d0"][0]))
    R_p = R_p.at[0].set(jnp.log(params.model.p0_in / out["p0"][0]))

    return jnp.concatenate([R_u, R_d, R_p])


# ---------- Create function handle for the residual vector ----------
def get_residual_mach_critical(z, params: ResidualParams):

    # We solve for ln(d) and ln(p) instead of d and p directly.
    #   - This enforces strict positivity of density and pressure without explicit constraints.
    #   - The PDE in ln(d) or ln(p) form contains a 1/d or 1/p factor in the derivative term:
    #       d/dx[ln(d)] = (1/d) * d(d)/dx
    #       d/dx[ln(p)] = (1/p) * d(p)/dx
    #     so when forming the residuals, the nonlinear terms N_all[:,1] and N_all[:,2] must be divided
    #     by the current d and p values, respectively.
    #   - This scaling also normalizes the residual magnitude for variables with very different units
    #     and prevents density/pressure from dominating the Jacobian purely due to scale.

    # Unpack solution vector
    u, ln_d, ln_p = split_z(z, params.x.shape[0])
    d = jnp.exp(ln_d)
    p = jnp.exp(ln_p)

    # Compute right hand side of the ODE
    out = evaluate_ode_rhs(params.x, z, params.model)
    N_all = out["N"]
    D_tau = out["D"]

    # Evaluate residuals at collocation points
    R_u = (params.Dx @ u) - N_all[:, 0] / D_tau
    R_d = (params.Dx @ ln_d) - N_all[:, 1] / D_tau / d
    R_p = (params.Dx @ ln_p) - N_all[:, 2] / D_tau / p

    # Evaluate residual at the boundary conditions
    x_star, Ma_max = find_maximum_mach(params.x, out["Ma"])
    R_u = R_u.at[0].set(params.model.Ma_target - Ma_max)
    R_d = R_d.at[0].set(jnp.log(params.model.d0_in / out["d0"][0]))
    R_p = R_p.at[0].set(jnp.log(params.model.p0_in / out["p0"][0]))

    return jnp.concatenate([R_u, R_d, R_p])



# ---------- Calculate maximum Mach number within the domain ----------


def find_maximum_mach(x_nodes, y_nodes, newton_steps=50, rtol=1e-10, atol=1e-10):
    """
    Locate the single interior maximum of the Chebyshev-Lobatto interpolant.

    Uses a smooth soft-argmax to pick an initial guess (avoids non-diff argmax),
    then runs a fixed number of Newton iterations on p'(x) = 0.

    Parameters
    ----------
    x_nodes : (N+1,) array
        Chebyshev-Lobatto nodes in the domain.
    y_nodes : (N+1,) array
        Function values at the nodes (e.g., Mach number).
    newton_steps : int, optional
        Maximum Newton iterations. Default 50.
    rtol, atol : float, optional
        Relative and absolute tolerances for Newton.

    Returns
    -------
    x_star : float
        Location of the maximum in [x1, x2].
    p_star : float
        Value of the interpolant at x_star.
    """
    x1, x2 = x_nodes[0], x_nodes[-1]

    # Smooth initial guess: soft-argmax over node values
    alpha = 50.0  # higher → sharper, closer to discrete argmax
    y_shift = y_nodes - jnp.max(y_nodes)
    w = jax.nn.softmax(alpha * y_shift)
    x0 = jnp.sum(w * x_nodes)

    # Residual for p'(x) = 0
    def resid(x, _):
        _, dp = chebyshev_lobatto_interpolate_and_derivative(x_nodes, y_nodes, x)
        return dp

    # Newton solve (ignore success flag, fixed iteration count)
    solver = optx.Newton(rtol=rtol, atol=atol, cauchy_termination=False)
    sol = optx.root_find(resid, solver, y0=x0, args=None, max_steps=newton_steps, throw=False)
    x_star = sol.value

    # Clip to domain and guard against NaN fallback
    x_star = jnp.clip(jnp.nan_to_num(x_star, nan=x0), x1, x2)

    # Value at maximum
    p_star, _ = chebyshev_lobatto_interpolate_and_derivative(x_nodes, y_nodes, x_star)
    return x_star, p_star


# ---------- helpers: pack/unpack and per-node wrapper ----------
def split_z(z, num_points):
    u = z[0:num_points]
    ln_d = z[num_points : 2 * num_points]
    ln_p = z[2 * num_points : 3 * num_points]
    return u, ln_d, ln_p


def evaluate_ode_rhs(x, z, args):
    """Vectorized full-model eval at all nodes from z=[u, ln(rho), ln(p)]."""
    u, ln_d, ln_p = split_z(z, x.shape[0])

    def per_node(ui, ln_di, ln_pi, xi):
        di = jnp.exp(ln_di)
        pi = jnp.exp(ln_pi)
        yi = jnp.array([ui, di, pi])
        return nozzle_single_phase_core(xi, yi, args)

    return jax.vmap(per_node)(u, ln_d, ln_p, x)


# ---------- Generate flow field for initial guess ----------
def initialize_flowfield(num_points, params, Ma_min=0.1, Ma_max=0.2):
    """
    Generate an initial guess for the flowfield using a concave Mach number profile.

    The Mach profile is defined as a symmetric parabola with its maximum (Ma_max)
    at the domain midpoint and its minimum (Ma_min) at both inlet and outlet.
    The corresponding velocity, density, and pressure fields are computed
    from the specified inlet stagnation state.

    Parameters
    ----------
    num_points : int
        Number of interior collocation points. The Chebyshev-Lobatto grid will
        contain num_points + 1 points in total.
    params : dict
        Must contain:
            p0_in : float
                Inlet stagnation pressure.
            d0_in : float
                Inlet stagnation density.
    fluid : object
        Fluid property object
    Ma_min : float, optional
        Minimum Mach number at the inlet and outlet. Default is 0.1.
    Ma_max : float, optional
        Maximum Mach number at the domain midpoint. Default is 0.5.

    Returns
    -------
    z0 : ndarray, shape (3*(num_points+1),)
        Initial guess vector at collocation points, concatenated as:
        [velocity, ln_density, ln_pressure].
    """
    # Inlet stagnation state
    fluid = params.fluid
    state0_in = fluid.get_state(jxp.DmassP_INPUTS, params.d0_in, params.p0_in)
    a_in = state0_in["a"]  # use inlet speed of sound for initial guess everywhere
    h_in = state0_in["h"]
    s_in = state0_in["s"]

    # Create coordinate array from 0 to 1 (Chebyshev–Lobatto nodes not needed for init)
    x_uniform = jnp.linspace(0.0, 1.0, num_points + 1)

    # Parabolic Mach profile: peak at x=0.5, symmetric, concave
    # Parabola passing through (0, M_min), (0.5, M_max), (1, M_min)
    mach_profile = Ma_min + (Ma_max - Ma_min) * (1.0 - 4.0 * (x_uniform - 0.5) ** 2)

    # Velocity from Mach (constant a_in for initial guess)
    flowfield_v = mach_profile * a_in

    # Static density/pressure from h0 = h + v^2/2, s = s_in
    h_static = h_in - 0.5 * flowfield_v**2
    state_static = fluid.get_state(jxp.HmassSmass_INPUTS, h_static, s_in)
    d_static = jnp.maximum(state_static["rho"], 1e-12)
    p_static = jnp.maximum(state_static["p"], 1e-12)

    # Log variables
    flowfield_ln_d = jnp.log(d_static)
    flowfield_ln_p = jnp.log(p_static)

    # Concatenate into initial guess vector
    return jnp.concatenate([flowfield_v, flowfield_ln_d, flowfield_ln_p])


# ---------- Define Chebyshev-Lobatto nodes and differentiation matrix ----------
def chebyshev_lobatto_basis(N: int, x1: float, x2: float):
    """
    Return:
      x : (N+1,) physical nodes in [x1, x2]
      D : (N+1, N+1) differentiation s.t. (u_x)(x_i) ≈ sum_j D[i,j] * u(x_j)

    Built from Trefethen's formula. D acts on nodal values to produce nodal derivatives.
    """

    # Standard Trefethen ordering
    k = jnp.arange(N + 1)
    x_hat = jnp.cos(jnp.pi * k / N)
    x = 0.5 * (x_hat + 1.0) * (x2 - x1) + x1

    c = jnp.where((k == 0) | (k == N), 2.0, 1.0) * ((-1.0) ** k)
    X = jnp.tile(x_hat, (N + 1, 1))
    dX = X - X.T + jnp.eye(N + 1)
    C = jnp.outer(c, 1.0 / c)
    D_hat = C / dX
    D_hat = D_hat - jnp.diag(jnp.sum(D_hat, axis=1))

    # Scale derivative from [-1,1] to [0,L]
    D = -(2.0 / (x2 - x1)) * D_hat

    # Reorder so x[0] = 0, x[-1] = L
    idx = jnp.arange(N + 1)[::-1]
    x = x[idx]
    D = D[idx][:, idx]

    return x, D


def chebyshev_lobatto_interpolate(x_nodes, y_nodes, x_eval):
    """
    Evaluate the Chebyshev-Lobatto barycentric interpolant at one or more points.

    This function is a thin wrapper around `chebyshev_lobatto_interpolate_and_derivative`
    that discards the derivative and returns only the interpolated value.

    Parameters
    ----------
    x_nodes : array_like, shape (N+1,)
        The Chebyshev-Lobatto nodes in the physical domain [x_min, x_max].
    y_nodes : array_like, shape (N+1,)
        Function values at the Chebyshev-Lobatto nodes.
    x_eval : float or array_like
        Point(s) in the domain where the interpolant should be evaluated.

    Returns
    -------
    p : float or ndarray
        Interpolated value(s) p(x_eval).
    """
    p, _ = chebyshev_lobatto_interpolate_and_derivative(x_nodes, y_nodes, x_eval)
    return p


def chebyshev_lobatto_interpolate_and_derivative(x_nodes, y_nodes, x_eval):
    """
    Evaluate the Chebyshev-Lobatto barycentric interpolant and its derivative.

    Parameters
    ----------
    x_nodes : array_like, shape (N+1,)
        The Chebyshev-Lobatto nodes in the physical domain [x_min, x_max].
    y_nodes : array_like, shape (N+1,)
        Function values at the Chebyshev-Lobatto nodes.
    x_eval : float or array_like
        Point(s) in the domain where the interpolant and its derivative
        should be evaluated.

    Returns
    -------
    p : float or ndarray
        Interpolated value(s) p(x_eval).
    dp : float or ndarray
        First derivative p'(x_eval) with respect to x.

    Notes
    -----
    - Uses the barycentric interpolation formula, which is numerically stable
      even for high-degree polynomials.
    - Correctly handles the case where x_eval coincides with one of the nodes,
      returning the exact nodal value and the exact derivative at that node.
    - Works for scalar or vector x_eval.
    """
    n = x_nodes.size - 1
    k = jnp.arange(n + 1)
    w = jnp.where((k == 0) | (k == n), 0.5, 1.0) * ((-1.0) ** k)

    def _scalar_interp_and_deriv(x):
        diff = x - x_nodes
        is_node = diff == 0.0

        def at_node():
            # Build terms excluding idx manually
            idx = jnp.argmax(is_node).astype(int)
            p = y_nodes[idx]

            # diff and weights excluding idx
            diff_ex = x_nodes[idx] - x_nodes
            ydiff_ex = y_nodes[idx] - y_nodes

            # Set excluded self-term to 0 safely
            diff_ex = diff_ex.at[idx].set(1.0)  # avoid division by 0
            ydiff_ex = ydiff_ex.at[idx].set(0.0)
            w_ex = w.at[idx].set(0.0)

            dp = jnp.sum((w_ex / w[idx]) * (ydiff_ex) / (diff_ex))
            return p, dp

        def generic():
            r = w / diff
            S = jnp.sum(r)
            N = jnp.sum(r * y_nodes)
            p = N / S
            rp = -w / (diff * diff)
            S1 = jnp.sum(rp)
            N1 = jnp.sum(rp * y_nodes)
            dp = (N1 - p * S1) / S
            return p, dp

        return jax.lax.cond(jnp.any(is_node), at_node, generic)

    if jnp.ndim(x_eval) == 0:
        return _scalar_interp_and_deriv(x_eval)
    else:
        return jax.vmap(_scalar_interp_and_deriv)(x_eval)


def compute_static_state(p0, h0, Ma, fluid):
    # --- Reference stagnation state ---
    st0 = fluid.get_state(jxp.HmassP_INPUTS, h0, p0)
    s0 = st0["s"]

    # --- Define residual system: f(p, h) = [f1, f2] ---
    def residual(x, _):
        p, h = x
        st = fluid.get_state(jxp.HmassP_INPUTS, h, p)
        a, s = st["a"], st["s"]
        f1 = 1.0 - (h + 0.5 * (a * Ma) ** 2) / h0   # energy balance
        f2 = s/s0 - 1.0                             # isentropic condition
        return jnp.array([f1, f2])

    # --- Initial guess ---
    # start slightly below stagnation pressure, same enthalpy
    x0 = jnp.array([0.95*p0, 0.95*h0])

    # --- Solve ---
    solver = optx.GaussNewton(rtol=1e-3, atol=1e-3)
    sol = optx.root_find(residual, solver, y0=x0)

    # --- Evaluate final state ---
    p, h = sol.value
    state = fluid.get_state(jxp.HmassP_INPUTS, h, p)
    return state


# # ---------- Compute static state from stagnation and Mach number ----------
# def compute_static_state(p0, h0, Ma, fluid):
#     st0 = fluid.get_state(jxp.HmassP_INPUTS, h0, p0)
#     s0 = st0["s"]

#     # Scalar residual for Bisection
#     def residual(p, _):
#         st = fluid.get_state(jxp.PSmass_INPUTS, p, s0)
#         a, h = st["a"], st["h"]
#         return h0 - h - 0.5 * (a * Ma)**2

#     solver = optx.Newton(rtol=1e-3, atol=1e-3)
#     lower, upper = 0.2 * p0, p0
#     # sol = optx.root_find(residual, solver, y0=p0, options={"lower": lower, "upper": upper})
#     sol = optx.root_find(residual, solver, y0=0.95* p0)
#     state = fluid.get_state(jxp.PSmass_INPUTS, sol.value, s0)
#     return state

# def compute_static_state(p0, h0, Ma, fluid):
#     st0 = fluid.get_state(jxp.HmassP_INPUTS, h0, p0)
#     s0 = st0["s"]

#     # Residual: total - static enthalpy = ½ (a * Ma)^2
#     def residual(h, _):
#         st = fluid.get_state(jxp.HmassSmass_INPUTS, h, s0)
#         a = st["a"]
#         return h0 - h - 0.5 * (a * Ma)**2

#     # Solve for static enthalpy
#     solver = optx.Bisection(rtol=1e-3, atol=1e-3)
#     # You can estimate lower/upper bounds around h0
#     lower, upper = 0.4 * h0, h0
#     sol = optx.root_find(residual, solver, y0=h0, options={"lower": lower, "upper": upper})

#     # Get final state
#     state = fluid.get_state(jxp.HmassSmass_INPUTS, sol.value, s0)
#     return state

@eqx.filter_jit
def nozzle_single_phase_max_mach(
    params_model,
    params_solver,
):
    """
    1D variable-area nozzle with friction and optional heat transfer (Reynolds analogy).
    State vector: y = [v, rho, p].
    """
    # Compute inlet conditions iteratively
    state_in = compute_static_state(
        params_model.p0_in,
        params_model.h0_in,
        params_model.Ma_in,
        params_model.fluid,
    )
    p_in, rho_in, a_in, h_in = state_in["p"], state_in["rho"], state_in["a"], state_in["h"]
    v_in = params_model.Ma_in * a_in
    x_in = 1e-9  # Start slightly after the nozzle inlet
    y0 = jnp.array([x_in, p_in, v_in, h_in])

    # Create and configure the solver
    t0 = 0.0  # Start at tau=0 (arbitrary)
    t1 = 1e9  # Large value that will not be reached
    solver = jxp.make_diffrax_solver(params_solver.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_solver.adjoint_name)
    term = dfx.ODETerm(eval_ode_rhs)
    ctrl = dfx.PIDController(rtol=params_solver.rtol, atol=params_solver.atol)
    event = dfx.Event(
        cond_fn=eval_end_of_domain_event,
        root_finder=optx.Bisection(rtol=1e-10, atol=1e-10),
    )

    saveat = dfx.SaveAt(t1=True, dense=True, fn=eval_ode_full)
    # Solve the ODE system without saving solution
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        saveat=saveat,
        dt0=1e-12,
        y0=y0,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        event=event,
        max_steps=20_000,
    )

    # Optimization problem to find the max(Mach)
    x0 = sol.ts[0]/2
   
    solver = optx.BFGS(rtol=1e-3, atol=1e-3)
   
    # Run optimization
    res = optx.minimise(
                        lambda t, *args: -nozzle_single_phase_autonomous_ph(0.0, sol.evaluate(t), params_model)["Ma"],
                        solver,
                        y0=x0,
                        max_steps=1000,
                    )
 
 
    t_max = res.value
    yy_max = sol.evaluate(t_max)
    Ma_max = nozzle_single_phase_autonomous_ph(0.0, yy_max, params_model)["Ma"]
    # jax.debug.print("Max mach {M}", M=Ma_max)

    return Ma_max


def eval_end_of_domain_event(t, y, args, **kwargs):
    L = args.length

    # Unpack variables
    x, p, v, h = y

    # jax.debug.print("x = {x:.3e}, p = {p:.6e}", x=x, p=p)

    fluid = args.fluid

    # Compute determinant D
    state = fluid.get_state(jxp.HmassP_INPUTS, h, p)
    a = state["a"]
    d = state["rho"]
    G = state["gruneisen"]

    dddp = (1 + G) / a**2
    dddh = - (d * G) / a**2

    A_mat = jnp.array([
        [v * dddp, d, v * dddh],
        [1.0, d * v, 0.0],
        [v, 0.0, -d * v],
    ])

    D = jnp.linalg.det(A_mat)

    # Two event conditions:
    # 1. Domain end: L - x
    # 2. Determinant-negative condition: 0.2 - x (active only if D < 0)

    cond_1 = L - x
    # cond_2 = jnp.where(D < 0, x - 25e-3, 1e9)
    cond_2 = 1.2 - v/a
    # cond_2 = x

    # Trigger when either becomes zero
    val = jnp.minimum(cond_1, cond_2)

    return val 


# -----------------------------------------------------------------------------
# Helper ODE evaluation functions
# -----------------------------------------------------------------------------
def eval_ode_full(t, y, args):
    return nozzle_single_phase_autonomous_ph(t, y, args)


def eval_ode_rhs(t, y, args):
    return nozzle_single_phase_autonomous_ph(t, y, args)["rhs_autonomous"]


# -----------------------------------------------------------------------------
# Critical inlet computation
# -----------------------------------------------------------------------------
def compute_critical_inlet(Ma_lower, Ma_upper, params_model, params_solver):
    """
    Finds the inlet Mach number that makes the flow reach Mach 1 using jaxprop.
    Fully JAX-traceable (compatible with jit/tracers).
    """

    # Residual using only jax operations and jxp for property evaluation
    def critical_mach_residual(Mach_in, params_model):

        pm = replace_param(params_model, "Ma_in", Mach_in)
        max_mach = nozzle_single_phase_max_mach(pm, params_solver)
        # jax.debug.print("1-mach = {}", 1-max_mach)
        # min_det = jnp.min(sol.ys["D"])
        # jax.debug.print("min(D) = {}", min_det)
        # return min_det
        return 0.99 - max_mach

    # Use JAX-safe Bisection
    solver = optx.Bisection(rtol=1e-3, atol=1e-3)
    x0_initial = Ma_lower

    # JAX-friendly root find (do not convert to float inside trace)
    sol_root = optx.root_find(
        critical_mach_residual,
        solver,
        x0_initial,
        args=params_model,
        throw=False,
        options={"lower": Ma_lower, "upper": Ma_upper},
    )
    Ma_in_crit = sol_root.value  

    return Ma_in_crit