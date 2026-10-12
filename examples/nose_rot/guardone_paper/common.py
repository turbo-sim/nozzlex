"""
Shared building blocks for reproducing Gajoni & Guardone (JFM 2023)
figures: the planar radial equilibrium equations (2.8), (2.9), (2.10),
and the numerical machinery (root-finding + RK45 integration) used
across figure2/4/6a/7a/8a.

All functions work at a fixed global entropy s0 and total enthalpy h0
(the stagnation state), consistent with the paper's assumption of a
uniform upstream flow.
"""

import numpy as np
import jaxprop as jxp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def get_fluid(name):
    return jxp.Fluid(name=name, backend="HEOS")


def mach_of_rho(fluid, rho, s0, h0):
    """M(rho) at fixed entropy s0, Eq. (2.10)."""
    state = fluid.get_state(jxp.DmassSmass_INPUTS, rho, s0)
    dh = h0 - state.h
    if dh <= 0:
        return 0.0
    return np.sqrt(2.0 * dh) / state.a


def gamma_of_rho(fluid, rho, s0, dP_frac=0.002):
    """Gamma = 1 + rho*c*(dc/dP)_s via central finite difference at fixed s0."""
    state = fluid.get_state(jxp.DmassSmass_INPUTS, rho, s0)
    P0 = state.p
    dP = dP_frac * P0
    sp = fluid.get_state(jxp.PSmass_INPUTS, P0 + dP, s0)
    sm = fluid.get_state(jxp.PSmass_INPUTS, P0 - dP, s0)
    dcdP = (sp.a - sm.a) / (2 * dP)
    return 1.0 + state.rho * state.a * dcdP


def find_rho_for_mach(fluid, M_target, s0, h0, rho_stag):
    """
    Root-find the density at fixed s0 that gives Mach number M_target.

    brentq needs a bracket [rho_lo, rho_hi] where the residual changes
    sign. rho_hi (near stagnation) always gives M~0, so residual<0 there.
    rho_lo is searched downward (further expansion -> higher M) until
    the residual becomes >=0, guaranteeing a sign change to bracket.
    """
    def residual(rho):
        return mach_of_rho(fluid, rho, s0, h0) - M_target

    rho_hi = rho_stag * (1 - 1e-6)
    rho_lo = rho_stag * 0.3
    while residual(rho_lo) < 0 and rho_lo > rho_stag * 1e-5:
        rho_lo *= 0.7
    return brentq(residual, rho_lo, rho_hi, xtol=1e-8, rtol=1e-10)


def integrate_radial_equilibrium(fluid, s0, h0, rho_i, r_tilde_end, n_eval=200,
                                  method="RK45"):
    """
    Integrate Eq. (2.9): drho/dr~ = rho*M(rho)^2/r~, from r~=1 (rho=rho_i)
    to r~=r_tilde_end. Returns (r_tilde, rho, M) arrays.
    """
    def rhs(r_tilde, rho):
        M = mach_of_rho(fluid, rho[0], s0, h0)
        return [rho[0] * M**2 / r_tilde]

    r_eval = np.linspace(1.0, r_tilde_end, n_eval)
    sol = solve_ivp(rhs, [1.0, r_tilde_end], [rho_i], method=method,
                     t_eval=r_eval, rtol=1e-9, atol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)

    M = np.array([mach_of_rho(fluid, rho, s0, h0) for rho in sol.y[0]])
    return sol.t, sol.y[0], M


def ideal_gas_gamma(fluid, T_K, P_low=100.0):
    """gamma_ideal = lim_{P->0} cp(Tc,P)/cv(Tc,P), Eq. (3.1)."""
    state = fluid.get_state(jxp.PT_INPUTS, P_low, T_K)
    return state.cp / state.cv


def ideal_gas_mach_profile(M_i, gamma, r_tilde):
    """Analytic ideal-gas solution, Eq. (2.8)."""
    term = 0.5 * (gamma - 1.0) * M_i**2
    return M_i / np.sqrt((1.0 + term) * r_tilde**2 - term)


def label_curves_staggered(ax, curves, r_labels):
    """
    Direct on-curve labels (paper style) for NESTED, non-crossing curves,
    ordered by value at r~=2 and staggered vertically. Use
    label_curves_crossing() instead if the curves cross each other.

    curves: list of (label, r_tilde, M) tuples.
    """
    curves_sorted = sorted(curves, key=lambda c: np.interp(2.0, c[1], c[2]))
    for i, (label, r_tilde, M) in enumerate(curves_sorted):
        r_lab = r_labels[i]
        M_lab = np.interp(r_lab, r_tilde, M)
        dy = -20 if i == 0 else 22
        va = "top" if i == 0 else "bottom"
        ax.annotate(rf"${label}$", xy=(r_lab, M_lab),
                    xytext=(0, dy), textcoords="offset points",
                    fontsize=10, ha="center", va=va,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="black"))


def label_curves_crossing(ax, curves, r_range):
    """
    Direct on-curve labels for curves that CROSS each other: for each
    curve, find the r~ position (within r_range) where it is farthest
    (in M) from every other curve, and label it there, offset away from
    the local crowd.

    curves: list of (label, r_tilde, M) tuples.
    """
    r_common = np.linspace(r_range[0], r_range[1], 200)
    M_interp = {label: np.interp(r_common, r_tilde, M)
                for label, r_tilde, M in curves}

    for label, r_tilde, M in curves:
        M_self = M_interp[label]
        others = np.array([M_interp[l2] for l2, _, _ in curves if l2 != label])
        min_dist = np.min(np.abs(others - M_self), axis=0)
        j = np.argmax(min_dist)
        r_lab, M_lab = r_common[j], M_self[j]

        above = M_lab > np.mean(others[:, j])
        dy = 18 if above else -18
        va = "bottom" if above else "top"
        ax.annotate(rf"${label}$", xy=(r_lab, M_lab),
                    xytext=(0, dy), textcoords="offset points",
                    fontsize=10, ha="center", va=va,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="black"))
