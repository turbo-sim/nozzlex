"""Real-gas 1D marching solver to recover A(x) from imposed p(x).

Assumptions:
- steady, quasi-1D flow
- adiabatic, frictionless (no wall heat or shear)
- real-gas thermodynamics via `jaxprop`
"""

from __future__ import annotations

import math
from typing import Sequence, Dict, Any
import numpy as np
import jaxprop as jxp
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def march_from_p_ode(x_range: Sequence[float],
                     p_profile: Sequence[float],
                     inlet_state: Dict[str, float],
                     fluid_name: str,
                     backend: str = 'HEOS') -> Dict[str, Any]:

    # 1. Setup Interpolants
    p_func = interp1d(x_range, p_profile, kind='cubic')
    dpdx_func = interp1d(x_range, np.gradient(p_profile, x_range), kind='cubic')

    fluid = jxp.Fluid(fluid_name, backend=backend)

    # 2. Define the System of ODEs
    def ode_system(x, state_vec):
        # state_vec = [u, h, A]
        u, h, A = state_vec

        p = p_func(x)
        dpdx = dpdx_func(x)

        # Local static state
        state = fluid.get_state(jxp.HmassP_INPUTS, h, p)

        d = state.rho
        a = state.a
        G = state.gruneisen # Grüneisen parameter

        # Real-gas derivatives
        dddp = (1 + G) / a**2
        dddh = - (d * G) / a**2

        A_mat = np.array([
            [A * d,  u * A * dddh,  d * u], # Simplified Row 1
            [d * u,      0.0,       0.0  ], # Row 2
            [0,             d,      0.0  ]  # Row 3
        ])

        b_vec = np.array([
            -u * A * dddp * dpdx,
            -dpdx,
            dpdx
        ])

        # Solve for derivatives: [dudx, dhdx, dAdx]
        try:
            derivatives = np.linalg.solve(A_mat, b_vec)
        except np.linalg.LinAlgError:
            return [np.nan, np.nan, np.nan]

        return derivatives

    # 3. Initial Conditions
    # We need all three: u, h, A
    y0 = [inlet_state['u_in'], inlet_state['h_in'], inlet_state['A_in']]
    x_span = (x_range[0], x_range[-1])

    # 4. Integrate
    sol = solve_ivp(ode_system, x_span, y0, t_eval=x_range, method='RK45')

    return {
        'x': sol.t,
        'u': sol.y[0],
        'h': sol.y[1],
        'A': sol.y[2],
        'success': sol.success
    }


def march_from_p_isentropic(x: Sequence[float], p: Sequence[float],
                            h_in: float, s_in: float, A_in: float, u_in: float,
                            fluid: str = 'Air', backend: str = 'HEOS') -> Dict[str, Any]:
    """Closed-form isentropic solution: recover A(x) from p(x) without marching.

    Valid only for frictionless, adiabatic flow, where the process is
    isentropic and A(x) follows algebraically from p(x), point-by-point,
    with no ODE integration:

        s(x) = s_in                      (isentropic)
        h(x), rho(x), a(x) = EOS(p(x), s_in)
        u(x) = sqrt(2*(h0 - h(x)))        (energy / Bernoulli)
        A(x) = mdot / (rho(x) * u(x))     (mass)

    This is the frictionless limit of `march_from_p_ode`, solved directly
    instead of by marching — useful as a fast cross-check when there is no
    wall friction.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    N = x.size
    if p.size != N:
        raise ValueError('x and p must have same length')

    fluid_obj = jxp.Fluid(fluid, backend=backend)
    state_in = fluid_obj.get_state(jxp.HmassSmass_INPUTS, h_in, s_in)

    rho_in = state_in.rho
    mdot = rho_in * A_in * u_in
    h0 = h_in + 0.5 * u_in ** 2

    A = np.full(N, np.nan)
    u = np.full(N, np.nan)
    T = np.full(N, np.nan)
    rho = np.full(N, np.nan)
    h = np.full(N, np.nan)
    a = np.full(N, np.nan)
    Q = np.full(N, np.nan)
    M = np.full(N, np.nan)

    converged = True
    for i in range(N):
        st = fluid_obj.get_state(jxp.PSmass_INPUTS, p[i], s_in)
        h[i] = st.h
        rho[i] = st.rho
        T[i] = st.T
        a[i] = st.a
        # NOTE: st.Q is unreliable once the state leaves the two-phase dome
        # (jaxprop/CoolProp report Q=0 rather than Q=1 for superheated vapor,
        # since the underlying Q=-1 "single phase" sentinel gets clipped).
        # Derive quality from enthalpy bounds instead, which is robust on
        # both sides of the dome and clips correctly to [0, 1].
        st_g = fluid_obj.get_state(jxp.PQ_INPUTS, p[i], 1.0)
        st_l = fluid_obj.get_state(jxp.PQ_INPUTS, p[i], 0.0)
        Q[i] = np.clip((h[i] - st_l.h) / (st_g.h - st_l.h), 0.0, 1.0)
        dh = h0 - h[i]
        if dh <= 0:
            converged = False
            break
        u[i] = np.sqrt(2.0 * dh)
        M[i] = u[i] / a[i]
        A[i] = mdot / (rho[i] * u[i])

    return {
        'x': x, 'p': p, 'A': A, 'u': u, 'T': T, 'rho': rho,
        'h': h, 'a': a, 'Q': Q, 'M': M, 'mdot': mdot,
        's_in': s_in, 'converged': converged, 'fluid': fluid_obj,
    }


def march_from_p(x: Sequence[float], p: Sequence[float],
                 h_in: float, s_in: float, A_in: float, u_in: float,
                 fluid: str = 'Air', backend: str = 'HEOS') -> Dict[str, Any]:
    """Discrete station-to-station marching: recover A(x) from p(x) by solving
    the momentum integral at each step with a root-finding iteration.

    Unlike `march_from_p_ode` (continuous ODE, differential Jacobian form)
    and `march_from_p_isentropic` (closed-form algebraic, frictionless only),
    this steps from station i to i+1 by solving the discrete momentum balance

        mdot * (u_{i+1} - u_i) + A_avg * (p_{i+1} - p_i) = 0

    for u_{i+1} directly (a 1D root-finding problem, solved here with a
    secant-like iteration), where A_avg = 0.5*(A_i + A_{i+1}) and A_{i+1}
    follows from mass conservation once u_{i+1} and the local density (from
    the energy equation, h_{i+1} = h0 - 0.5*u_{i+1}**2) are known. No
    entropy/isentropic assumption is required — this is a general real-gas
    marching scheme, equivalent in principle to `march_from_p_ode` but
    solved as a sequence of scalar root-finds instead of an ODE integration.
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    N = x.size
    if p.size != N:
        raise ValueError('x and p must have same length')

    fluid_obj = jxp.Fluid(fluid, backend=backend)
    state_in = fluid_obj.get_state(jxp.HmassSmass_INPUTS, h_in, s_in)

    A = np.zeros(N, dtype=float)
    u = np.zeros(N, dtype=float)
    T = np.zeros(N, dtype=float)
    rho = np.zeros(N, dtype=float)
    h = np.zeros(N, dtype=float)
    Q = np.zeros(N, dtype=float)
    a = np.zeros(N, dtype=float)

    # initialize at inlet
    A[0] = A_in
    u[0] = u_in
    rho[0] = state_in.rho
    h[0] = h_in
    T[0] = state_in.T
    Q[0] = state_in.Q
    a[0] = state_in.a

    h0 = h[0] + 0.5 * u[0] ** 2  # total enthalpy
    mdot = rho[0] * A[0] * u[0]  # mass flow rate

    converged = True
    for i in range(N - 1):
        dp = p[i + 1] - p[i]

        # residual function for u_{i+1}
        def residual(u_next: float) -> float:
            # energy -> h_static at i+1
            h_static_next = h0 - 0.5 * u_next * u_next
            state_next = fluid_obj.get_state(jxp.HmassP_INPUTS, h_static_next, p[i + 1])
            rho_next = state_next.rho
            A_next = mdot / (rho_next * u_next)
            A_avg = 0.5 * (A[i] + A_next)
            # momentum integral discrete: mdot*(u_next - u_i) + A_avg*(p_{i+1}-p_i) = 0
            return mdot * (u_next - u[i]) + A_avg * dp

        # initial guesses
        u_guess = max(1e-3, u[i] - (A[i] * (p[i + 1] - p[i]) / mdot))
        u_low = max(1e-6, u_guess * 0.5)
        u_high = u_guess * 1.5 + 10.0

        # use simple secant-like iteration
        u_next = u_guess
        try:
            for k in range(60):
                r = residual(u_next)
                if abs(r) < 1e-8:
                    break
                # finite-difference derivative
                dud = max(1e-6, 1e-3 * (abs(u_next) + 1.0))
                r2 = residual(u_next + dud)
                drdu = (r2 - r) / dud
                if drdu == 0 or math.isnan(drdu):
                    u_next = (u_low + u_high) / 2.0
                else:
                    u_next = u_next - r / drdu
                # clamp
                if u_next <= 0:
                    u_next = 1e-6
            else:
                raise RuntimeError('velocity solver did not converge')
        except Exception:
            converged = False
            # set remaining values to nan and break
            A[i + 1:] = np.nan
            u[i + 1:] = np.nan
            T[i + 1:] = np.nan
            rho[i + 1:] = np.nan
            h[i + 1:] = np.nan
            break

        # accept u_next and fill fields
        u[i + 1] = float(u_next)
        h_static_next = h0 - 0.5 * u_next ** 2
        state_final = fluid_obj.get_state(jxp.HmassP_INPUTS, h_static_next, p[i + 1])
        T_next = state_final.T
        rho_next = state_final.rho
        A_next = mdot / (rho_next * u_next)
        T[i + 1] = float(T_next)
        rho[i + 1] = float(rho_next)
        A[i + 1] = float(A_next)
        h[i + 1] = float(h_static_next)
        a[i + 1] = float(state_final.a)
        Q[i + 1] = float(state_final.Q)

    return {
        'x': x,
        'p': p,
        'A': A,
        'u': u,
        'T': T,
        'rho': rho,
        'h': h,
        'a': a,
        'Q': Q,
        'mdot': mdot,
        'converged': converged,
        'fluid': fluid_obj,
    }
