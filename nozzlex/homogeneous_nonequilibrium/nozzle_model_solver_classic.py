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

from .nozzle_model_core_classic import get_speed_of_sound_mixture, nozzle_homogeneous_non_equilibrium_classic


# shorthand factory for float64 arrays
def f64(value):
    return eqx.field(
        default_factory=lambda: jnp.array(value, dtype=jnp.float64),
        static=False
    )

class NozzleParams(eqx.Module):
    fluid1: Any = eqx.field(static=False)
    fluid2: Any = eqx.field(static=False)
    geometry: Callable = eqx.field(static=True)
    p0_in: jnp.ndarray = f64(1.0e5)       # Pa
    T01_in:jnp.ndarray = f64(300)
    T02_in:jnp.ndarray = f64(300)
    length: jnp.ndarray = f64(5.00)       # m
    roughness: jnp.ndarray = f64(1e-6)    # m
    Ma_in: jnp.ndarray = f64(0.1)
    Ma_low: jnp.ndarray = f64(0.05)
    Ma_high: jnp.ndarray = f64(0.99)
    heat_transfer: jnp.ndarray = f64(0.0)
    wall_friction: jnp.ndarray = f64(0.0)
    mixture_ratio:jnp.ndarray  = f64(50)


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
    "Bisection":optx.Bisection
}


# ---------- Compute static state from stagnation and Mach number ----------
# def compute_static_state(p0, d0, Ma, fluid):
#     st0 = fluid.get_state(jxp.DmassP_INPUTS, d0, p0)
#     s0, h0 = st0["s"], st0["h"]

#     # Scalar residual for Bisection
#     def residual(p, _):
#         st = fluid.get_state(jxp.PSmass_INPUTS, p, s0)
#         a, h = st["a"], st["h"]
#         return h0 - h - 0.5 * (a * Ma)**2

#     solver = optx.Bisection(rtol=1e-3, atol=1e-3)
#     lower, upper = 0.4 * p0, p0
#     sol = optx.root_find(residual, solver, y0=0.99 * p0, options={"lower": lower, "upper": upper})
#     state = fluid.get_state(jxp.PSmass_INPUTS, sol.value, s0)
#     return state

# Specific for 

# def compute_static_state(p0, T01, T02, Ma, R, fluid1, fluid2):
#     jax.debug.print("p={p}, T1={T01}, T2={T02}, Ma={Ma}",p=p0, T01=T01, T02=T02, Ma=Ma)
#     st01 = fluid1.get_state(jxp.PT_INPUTS, p0, T01)
#     s01, h01, G1, a1, rho1 = st01["s"], st01["h"], st01["G"], st01["a"], st01["d"] 

#     st02 = fluid2.get_state(jxp.PT_INPUTS, p0, T02)
#     s02, G2, a2, rho2 = st02["s"], st02["G"], st02["a"], st02["d"]

#     q_in = 1/ (1 + R)
#     # alpha2 = 1 / (1 + ((1 - q_in) / q_in) * (rho1 / rho2))
#     # alpha1 = 1 - alpha2

#     # a_mix = get_speed_of_sound_mixture(G1, alpha1, a1, rho1, G2, alpha2, a2, rho2)

#     # Scalar residual for Bisection
#     def residual1(p, _):
#         st = fluid1.get_state(jxp.PSmass_INPUTS, p, s01)
#         h = st["h"]
#         G1, a1, rho1 = st["G"], st["a"], st["d"]


#         st = fluid2.get_state(jxp.PSmass_INPUTS, p, s02)
#         # h = st["h"]
#         G2, a2, rho2 = st["G"], st["a"], st["d"]

#         alpha2 = 1 / (1 + ((1 - q_in) / q_in) * (rho1 / rho2))
#         alpha1 = 1 - alpha2

#         jax.debug.print("alpha1={alpha1}, alpha2={alpha2}",alpha1=alpha1, alpha2=alpha2)

#         a_mix = get_speed_of_sound_mixture(G1, alpha1, a1, rho1, G2, alpha2, a2, rho2)
#         jax.debug.print("a={a}", a=a_mix)
#         return h01 - h - 0.5 * (Ma*a_mix)**2

#     solver = optx.Bisection(rtol=1e-9, atol=1e-9)
#     lower1, upper1 = 0.1 * p0, p0
#     sol1 = optx.root_find(residual1, solver, y0=0.99 * p0, options={"lower": lower1, "upper": upper1})
#     state1 = fluid1.get_state(jxp.PSmass_INPUTS, sol1.value, s01)
#     state2 = fluid2.get_state(jxp.PSmass_INPUTS, sol1.value, s02)
#     return state1, state2

@eqx.filter_jit
def  nozzle_homogeneuous_non_equilibrium(params_model, params_solver):
    """
    1D variable-area nozzle with friction and optional heat transfer (Reynolds analogy).
    State vector: y = [alpha1, alpha2, rho1, rho2, u, p, h1, h2].
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
        state2_in["h"]
    )

    q_in = (1)/(1 + params_model.mixture_ratio)

    G1, a1 = state1_in["G"], state1_in["a"]
    alpha2_in = 1 / (1 + ((1 - q_in) / q_in) * (rho1_in / rho2_in))
    alpha1_in = 1 - alpha2_in

    # h = st["h"]
    G2, a2 = state2_in["G"], state2_in["a"]

    a_in_mix = get_speed_of_sound_mixture(G1, alpha1_in, a1, rho1_in, G2, alpha2_in, a2, rho2_in)

    u_in = (params_model.Ma_in * a_in_mix)

    # q_in = (1)/(1 + params_model.mixture_ratio)
    # alpha2_in = 1 / (1 + ((1 - q_in) / q_in) * (rho1_in / rho2_in))
    # alpha1_in = 1 - alpha2_in

    x_in = 1e-9  # start slightly after inlet
    y0 = jnp.array([alpha1_in, alpha2_in, rho1_in, rho2_in, u_in, p_in, h1_in, h2_in])

    # --- solver setup ---
    t0, t1 = 1e-9, params_model.length
    # t1, t0 = 0.0, 1.0 # not working!
    solver = jxp.make_diffrax_solver(params_solver.solver_name)
    adjoint = jxp.make_diffrax_adjoint(params_solver.adjoint_name)
    term = dfx.ODETerm(eval_ode_rhs)
    ctrl = dfx.PIDController(rtol=params_solver.rtol, atol=params_solver.atol)

    # --- event: stop at nozzle exit ---
    # def eval_end_of_domain_event(t, y, args, **kwargs):
    #     x = y[0]
    #     L = args.length
    #     # jax.debug.print("{y}", y = y)
    #     out = nozzle_homogeneous_non_equilibrium_autonomous(0.0, y, args)
    #     jax.debug.print("x={x}, L-x={leng},D={D}",x=x, leng=L-x, D = out.get("D"))
    #     deter = jnp.min(out.get("D"))
    #     return jnp.minimum(jnp.minimum(x, L - x), deter)

    # def eval_end_of_domain_event(t, y, args, **kwargs):
    #     x = y[0]
    #     L = args.length

    #     # Determinant condition
    #     out = nozzle_homogeneous_non_equilibrium_autonomous(0.0, y, args)
    #     Ma_mix = jnp.min(out.get("Ma_mix"))
        
    #     M = 5 - Ma_mix
    #     # jax.debug.print("x={x}, L-x={leng},D={D}",x=x, leng=L-x, D = out.get("D"))

    #     # Stack all event conditions
    #     conditions = jnp.stack([x/L, 1 - x/L, M]) # D])
    #     output = jnp.min(conditions)
    #     # jax.debug.print("out={output}",output=output)
    #     return output

    
    
    # TODO: 
    # def stop_at_zero_det(t, y):
    #      = nozzle_homogeneous_non_equilibrium_core(y)
    #     det_M = out["determinant"]
    #     # print(f"t={t:.5f}, det={det_M:.5e}")
    #     return det_M

    event = dfx.Event(
        cond_fn=eval_end_of_domain_event,
        root_finder=optx.Newton(rtol=1e-12, atol=1e-12),
    )



    # --- first solve (find domain end) ---
    saveat = dfx.SaveAt(t1=True, fn=eval_ode_full)
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t1,
        dt0=1e-8,
        y0=y0,
        args=params_model,
        stepsize_controller=ctrl,
        adjoint=adjoint,
        saveat=saveat,
        event=event,
        max_steps=2_000_000,
    )

    # jax.debug.print("print{a}", a=sol.ys)

    # # --- second solve (save fields) ---
    # ts = jnp.linspace(t0+1e-12, sol.ts[-1], params_solver.number_of_points)
    # saveat = dfx.SaveAt(ts=ts, t1=True, fn=eval_ode_full)

    # # ts = jnp.linspace(t0, sol.ts[-1], params_solver.number_of_points)
    # # saveat = dfx.SaveAt( t1=True, fn=eval_ode_full)
    # sol_dense = dfx.diffeqsolve(
    #     term,
    #     solver,
    #     t0=t0,
    #     t1=sol.ts[-1],
    #     dt0=1e-8,
    #     y0=y0,
    #     args=params_model,
    #     saveat=saveat,
    #     stepsize_controller=ctrl,
    #     adjoint=adjoint,
    #     max_steps=200_000,
    # )

    # return sol_dense

    return sol



# -----------------------------------------------------------------------------
# Helper ODE evaluation functions
# -----------------------------------------------------------------------------
def eval_ode_full(t, y, args):
    return nozzle_homogeneous_non_equilibrium_classic(t, y, args)


def eval_ode_rhs(t, y, args):
    return nozzle_homogeneous_non_equilibrium_classic(t, y, args)["rhs"]


# -----------------------------------------------------------------------------
# Critical inlet computation
# -----------------------------------------------------------------------------
def compute_critical_inlet(Ma_lower, Ma_upper, params_model, params_solver):
    """
    Finds the inlet Mach number that makes the flow reach Mach 1 using jaxprop.
    Fully JAX-traceable (compatible with jit/tracers).
    """

    # Residual using only jax operations and jxp for property evaluation
    # def critical_mach_residual(u_in, params_model):
    #     # update model with candidate Mach
    #     pm = replace_param(params_model, "u_in", u_in)
    #     jax.debug.print("Inlet tentative velocity is:{u}", u=u_in)
    #     sol = nozzle_homogeneuous_non_equilibrium(pm, params_solver)
    #     # min_det = jnp.min(sol.ys["D"])
    #     # return min_det
    #     mach_max = jnp.max(sol.ys["Ma_mix"])
    #     jax.debug.print("{u}, {M}", u=u_in,M=mach_max)
    #     return 1 - mach_max
    def critical_mach_residual(Mach_in, params_model):
        # update model with candidate Mach
        pm = replace_param(params_model, "Ma_in", Mach_in)
        sol = nozzle_homogeneuous_non_equilibrium(pm, params_solver)
        max_mach = jnp.max(sol.ys["Ma_mix"])
        x_final = sol.ys["x"][-1]
        # jax.debug.print("x={x},M_in={M_in}, M_max={M}",x=x_final, M_in=Mach_in, M=max_mach,)
        return 1.0 - max_mach
    
    # Use JAX-safe Bisection
    solver = optx.Bisection(rtol=1e-12, atol=1e-12,) # flip=True)
    x0_initial = 0.5 * (Ma_lower + Ma_upper)
    # x0_initial = u_lower

    # JAX-friendly root find (do not convert to float inside trace)
    sol_root = optx.root_find(
        critical_mach_residual,
        solver,
        x0_initial,
        args=params_model,
        throw=True,
        options={"lower": Ma_lower, "upper": Ma_upper},
    )
    u_in_crit = sol_root.value  

    return u_in_crit

def compute_critical_inlet_test(Ma_lower, Ma_upper, params_model, params_solver):
    """
    Finds the inlet Mach number that makes the flow reach Mach 1 using jaxprop.
    Fully JAX-traceable (compatible with jit/tracers).
    """

    def critical_section_position(Mach_in, params_model, params_solver):
        # update model with candidate Mach
        pm = replace_param(params_model, "Ma_in", Mach_in)
        sol = nozzle_homogeneuous_non_equilibrium(pm, params_solver)
        max_mach = jnp.max(sol.ys["Ma_mix"])
        x_final = sol.ys["x"][-1]
        # jax.debug.print("x={x},M_in={M_in}, M_max={M}",x=x_final, M_in=Mach_in, M=max_mach,)
        return 1.0 - max_mach

    pif_iterations = 0

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
    alpha2 = 1 / (1 + ((1 - q_in) / q_in) * (rho1_in / rho2_in))
    alpha1 = 1 - alpha2

    G2, a2 = state2_in["G"], state2_in["a"]

    a_in_mix = get_speed_of_sound_mixture(G1, alpha1, a1, rho1_in, G2, alpha2, a2, rho2_in)

    u_in_crit = params_model.Ma_in * a_in_mix
    q_in = (1)/(1 + params_model.mixture_ratio)
    alpha2_in = 1 / (1 + ((1 - q_in) / q_in) * (rho2_in / rho1_in))
    alpha1_in = 1 - alpha2_in

    Ma_impossible = Ma_upper
    Ma_possible = Ma_lower
    Ma_guess = (Ma_lower + Ma_upper)/2
    u_guess = (Ma_lower * a_in_mix + Ma_upper * a_in_mix) / 2


    tol = 1e-3
    error = Ma_lower - Ma_upper

    while error > tol:
        pif_iterations += 1  
        # raw_solution = scipy.integrate.solve_ivp(
        #     lambda t, y: odefun(y)[0],
        #     [0, 1],
        #     y0 = [0.00, alpha1_in, alpha2_in, rho1_in, rho2_in, u_guess_1, u_guess_2, pressure_in_1, h1_in, h2_in],
        #     # t_eval=np.linspace(0, convergent_length, number_of_points) if number_of_points else None,
        #     method="RK45",
        #     rtol=1e-6,
        #     atol=1e-6,
        #     events=[ stop_at_zero_det] #, stop_at_length]
        # )
        # solution = postprocess_ode_autonomous(raw_solution.t, raw_solution.y, odefun)

        t0, t1 = 1e-9, params_model.length
        solver = jxp.make_diffrax_solver(params_solver.solver_name)
        adjoint = jxp.make_diffrax_adjoint(params_solver.adjoint_name)
        term = dfx.ODETerm(eval_ode_rhs)
        ctrl = dfx.PIDController(rtol=params_solver.rtol, atol=params_solver.atol) #,dtmin=1e-40) # , dtmin=1e-9) #,dtmax=1e-4)

        # --- event: stop at nozzle exit ---
        event = dfx.Event(
            cond_fn=eval_end_of_domain_event,
            root_finder=optx.Newton(rtol=1e-0, atol=1e-0),
        )

        y_inlet = [alpha1_in, alpha2_in, rho1_in, rho2_in, u_guess, p_in, h1_in, h2_in]

        # --- first solve (find domain end) ---
        saveat = dfx.SaveAt(t1 = True, dense = True, fn=eval_ode_full)
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
            max_steps=200_000,
            # throw=False,
        )
        jax.debug.print("last_x={t1}", t1=sol1.ts)

        if sol1.ts[0] < params_model.length:
            Ma_impossible = Ma_guess
        else:
            Ma_possible = Ma_guess

        u_guess = (u_impossible + u_possible) / 2
        u_guess_2 = (u_guess_1 * rho1_in * A_1_in) / (mixture_ratio * rho2_in * A_2_in)
        error = abs(u_impossible_1-u_possible_1)/u_possible_1
        print(u_impossible_1)
        print(u_possible_1)
        print(pif_iterations)
















    
    # # Use JAX-safe Bisection
    # solver = optx.Bisection(rtol=1e-12, atol=1e-12,) # flip=True)
    # x0_initial = 0.5 * (Ma_lower + Ma_upper)
    # # x0_initial = u_lower

    # JAX-friendly root find (do not convert to float inside trace)
    # sol_root = optx.root_find(
    #     critical_mach_residual,
    #     solver,
    #     x0_initial,
    #     args=params_model,
    #     throw=True,
    #     options={"lower": Ma_lower, "upper": Ma_upper},
    # )
    # u_in_crit = sol_root.value  

    return u_in_crit

def eval_end_of_domain_event(t, y, args, **kwargs):
        x = t
        L = args.length

        # Determinant condition
        out = nozzle_homogeneous_non_equilibrium_classic(t, y, args)
        Ma_mix = out.get("Ma_mix")
        det_check = out.get("D")
        # det_last
        
        M = -Ma_mix + 0.9999999 # 999999
        # M = det_check*1e-24 + 1e7
        # jax.debug.print("x={x} | L-x={leng} | det={D}",x=x, leng=L-x, D = M)
        # M = 5
        # jax.debug.print("Not considering mach as event")
        # Stack all event conditions
        conditions = jnp.stack([1 - x/L, M]) # D])
        output = jnp.min(conditions)
        # jax.debug.print("out={output}",output=output)
        return jnp.minimum(M, 1-x/L)
