import jax
import jax.numpy as jnp
import jaxprop as jxp
from jax.numpy.linalg import det
import equinox as eqx
import optimistix as optx
import nurbspy.jax as nrb
from jax import lax
from typing import NamedTuple

from CoolProp import AbstractState
import CoolProp.CoolProp as CP

def nozzle_homogeneous_non_equilibrium_core(x, y, args):
    """Wrapper that adapts from (x, y) to autonomous form."""
    Y = jnp.concatenate([jnp.atleast_1d(x), y])
    return nozzle_homogeneous_non_equilibrium_autonomous(0.0, Y, args)

# @eqx.filter_jit
def nozzle_homogeneous_non_equilibrium_autonomous(tau, Y, args):
    """
    Autonomous right-hand side for a homogeneous non-equilibrium two-component
    nozzle model (fluid1=liquid, fluid2=gas, e.g. water + nitrogen) with no
    inter-phase mass transfer.

    Uses Cramer's rule on the linear system  A·(dY/dx) = b  rewritten as an
    autonomous system to handle sonic-point singularities:

        dx/dτ   = det(A)
        dY_i/dτ = det(A with column i replaced by b)

    State vector: Y = [x, alpha1, alpha2, rho1, rho2, u, p, h1, h2]
        x       – axial coordinate [m]
        alpha1  – volume fraction of fluid 1 (liquid)
        alpha2  – volume fraction of fluid 2 (gas)
        rho1    – density of fluid 1 [kg/m³]
        rho2    – density of fluid 2 [kg/m³]
        u       – shared mixture velocity [m/s]
        p       – pressure [Pa]
        h1      – specific enthalpy of fluid 1 [J/kg]
        h2      – specific enthalpy of fluid 2 [J/kg]
    """
    x, alpha1, alpha2, rho1, rho2, u, p, h1, h2 = Y

    # v = jnp.where(jnp.abs(v) < 1e-6, jnp.sign(v)*1e-6, v)
    # Debug print using jax.debug.print
    # jax.debug.print(
    # "x={:.2f}, alpha1={:.4e}, alpha2={:.4e}, rho1={:.4e}, rho2={:.4e}, u={:.4e}, p={:.4e}, h1={:.4e}, h2={:.4e}",
    # x, alpha1, alpha2, rho1, rho2, u, p, h1, h2
    # )   

    # --- Geometry / model parameters ---
    fluid1 = args.fluid1 # liquid
    fluid2 = args.fluid2 # vapor
    L = args.length
    eps_wall = args.roughness
    wall_friction = args.wall_friction
    heat_transfer = args.heat_transfer
    Nb = args.bubble_number
    Nd = args.droplet_number
    alpha_b = args.alpha_b
    alpha_d = args.alpha_d

    # --- Thermodynamic state fluid 1 ---
    alpha1 = jnp.clip(alpha1, 0, 1)
    alpha2 = jnp.clip(alpha2, 0, 1)
    state1 = fluid1.get_state(jxp.HmassP_INPUTS, h1, p)
    T1 = state1["T"]
    s1 = state1["s"]
    a1 = state1["a"]
    cp1 = state1["cp"]
    mu1 = state1["mu"]
    G1 = state1["gruneisen"]

    # --- Thermodynamic state fluid 2 ---
    state2 = fluid2.get_state(jxp.HmassP_INPUTS, h2, p)
    T2 = state2["T"]
    s2 = state2["s"]
    a2 = state2["a"]
    cp2 = state2["cp"]
    mu2 = state2["mu"]
    G2 = state2["gruneisen"]

    rho_mix = rho1 * alpha1 + rho2 * alpha2
    mu_mix = mu1 * alpha1 + mu2 * alpha2

    S = (rho1**2 * rho2**2 * u**3) ** (1/8)

    # --- Geometry ---
    # A, dAdx, perimeter, radius = args.geometry(x, L) # When using symmetric geometry
    A, dAdx, perimeter, radius = args.geometry(x) # When using linear convergent divergent geometry
    diameter = 2 * radius

    # --- Lookart-Martinelli friction factor description ---
    # Re1 = u * rho1 * diameter * jnp.sqrt(alpha1) / mu1 # jnp.maximum(mu_mix, 1e-12)
    # f_D1 = get_friction_factor_haaland(Re1, eps_wall, diameter)
    # tau_w1 = get_wall_viscous_stress(f_D1, rho1, u)
    # tau_w1 = tau_w1 * wall_friction

    # phi_LM_squared = richardson(alpha2)

    # tau_w_mix = tau_w1 * phi_LM_squared

    # # --- Wall heat transfer and friction ---
    Re_mix = u * rho_mix * diameter / mu_mix # jnp.maximum(mu_mix, 1e-12)
    f_D = get_friction_factor_haaland(Re_mix, eps_wall, diameter)
    tau_w_mix = get_wall_viscous_stress(f_D, rho_mix, u)


    Ai, d_1 = interfacial_area(alpha2, alpha_b=alpha_b, alpha_d=alpha_d, Nb=Nb, Nd=Nd)
    # ht_1 = 6 # water
    Nu_1 = 12
    k_1 = state1["k"]
    ht_1 = (k_1 * Nu_1) / d_1 # water

    # nitrogen is consider the continuous phase
    ht_2 = ranz_marshall_htc_two_components(alpha2, alpha1, state2, state1, fluid2, Nb=Nb, Nd=Nd, alpha_b=alpha_b, alpha_d=alpha_d)
    ht_12 = (1 / ((1/ht_1) + (1/ht_2)))
    # the flow is considered to be a dispersed flow with liquid droplets being the dispersed phase
    Q12 = ht_12 * Ai * (T1 - T2)


    a_mix = get_speed_of_sound_mixture(G1, alpha1, a1, rho1, G2, alpha2, a2, rho2)

    # Mask with booleans (convert to 0.0 if disabled)
    # tau_w_mix = wall_friction * tau_w_mix
    # f_D = wall_friction * f_D

    # --- Build A matrix and b vector ---
    # A_mat = jnp.array([[d, v, 0.0], [d * v, 0.0, 1.0], [0.0, -(a**2), 1.0]])
    A_mat = jnp.asarray(
                [
                    [-1.0,               -1.0,               0.0,               0.0,               0.0,                    0.0,                    0.0,                    0.0],
                    [rho1 * u,          0.0,               alpha1 * u,        0.0,               alpha1 * rho1,          0.0,                    0.0,                    0.0],
                    [0.0,               rho2 * u,          0.0,               alpha2 * u,        alpha2 * rho2,          0.0,                    0.0,                    0.0],
                    [0.0,               0.0,               0.0,               0.0,               rho_mix * u,            1.0,                    0.0,                    0.0],
                    [0.0,               0.0,               0.0,               0.0,               alpha1 * rho1 * u**2,   0.0,                    alpha1 * rho1 * u,      0.0],
                    [0.0,               0.0,               0.0,               0.0,               alpha2 * rho2 * u**2,   0.0,                    0.0,                    alpha2 * rho2 * u],
                    [0.0,               0.0,              -1.0,               0.0,               0.0,                    (1 + G1) / a1**2,       -(rho1 * G1) / a1**2,    0.0],
                    [0.0,               0.0,               0.0,              -1.0,               0.0,                    (1 + G2) / a2**2,       0.0,                    -(rho2 * G2) / a2**2],
                ]
            )
    
    # AS1 = AbstractState("HEOS", "Water")
    # AS2 = AbstractState("HEOS", "Nitrogen")
    # AS1.update(CP.HmassP_INPUTS, h1, p)
    # AS2.update(CP.HmassP_INPUTS, h2, p)

    
    # drho_dP_1 = AS1.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
    # drho_dh_1 = AS1.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)

    # drho_dP_2 = AS2.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
    # drho_dh_2 = AS2.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)

    # deltap1 = drho_dP_1 - (1 + G1) / a1**2
    # deltap2 = drho_dP_2 - ((1 + G2) / a2**2)
    # deltah1 = drho_dh_1 - (-(rho1 * G1) / a1**2)
    # deltah2 = drho_dh_2 - (-(rho2 * G2) / a2**2)

    # jax.debug.print("dp1={dp1} | dp2={dp2} | dh1={dh1} | dh2={dh2}", dp1=deltap1, dp2=deltap2, dh1=deltah1, dh2=deltah2)




    b_vec = jnp.asarray(
                [
                    0.0,
                    -(dAdx/A) * alpha1 * rho1 *  u,
                    -(dAdx/A) * alpha2 * rho2 *  u,
                    -(tau_w_mix * perimeter) / A,
                    -Q12,
                    Q12,
                    0.0,
                    0.0
                ]
            )
    
    # eigenvalues = compute_eigenvalues(A_mat)

    A_mat = A_mat / S
    b_vec = b_vec / S

    # --- Determinants ---
    # D = det(A_mat)
    M1 = A_mat.at[:, 0].set(b_vec)
    M2 = A_mat.at[:, 1].set(b_vec)
    M3 = A_mat.at[:, 2].set(b_vec)
    M4 = A_mat.at[:, 3].set(b_vec)
    M5 = A_mat.at[:, 4].set(b_vec)
    M6 = A_mat.at[:, 5].set(b_vec)
    M7 = A_mat.at[:, 6].set(b_vec)
    M8 = A_mat.at[:, 7].set(b_vec)

    # det_sympy = 1.0*alpha1*alpha2*rho1*rho2*u**3*(-G1*a2**2*alpha1*rho1*rho2*u**2 + G1*a2**2*alpha1*rho2*rho_mix*u**2 - G2*a1**2*alpha2*rho1*rho2*u**2 + G2*a1**2*alpha2*rho1*rho_mix*u**2 - a1**2*a2**2*alpha1*rho1*rho2 - a1**2*a2**2*alpha2*rho1*rho2 + a1**2*alpha2*rho1*rho_mix*u**2 + a2**2*alpha1*rho2*rho_mix*u**2)/(a1**2*a2**2)
    # det_sympy_actual=1.0*alpha1*alpha2*rho1*rho2*u**3*(G1*a2**2*alpha1*rho1*rho2*u**2 - G1*a2**2*alpha1*rho2*rho_mix*u**2 + G2*a1**2*alpha2*rho1*rho2*u**2 - G2*a1**2*alpha2*rho1*rho_mix*u**2 + a1**2*a2**2*alpha1*rho1*rho2 + a1**2*a2**2*alpha2*rho1*rho2 - a1**2*alpha2*rho1*rho_mix*u**2 - a2**2*alpha1*rho2*rho_mix*u**2)/(a1**2*a2**2)
    # det_sympy_actua2=1.0*alpha1*alpha2*rho1*rho2*u**3*(G1*a2**2*alpha1*rho1*rho2*u**2 - G1*a2**2*alpha1*rho2*rho_mix*u**2 + G2*a1**2*alpha2*rho1*rho2*u**2 - G2*a1**2*alpha2*rho1*rho_mix*u**2 + a1**2*a2**2*alpha1*rho1*rho2 + a1**2*a2**2*alpha2*rho1*rho2 - a1**2*alpha2*rho1*rho_mix*u**2 - a2**2*alpha1*rho2*rho_mix*u**2)/(a1**2*a2**2)
    # jax.debug.print("delta sympy={dd}", dd=det_sympy_actual-det_sympy_actua2)

    # det_N1 = 1.0*alpha1*alpha2*rho1*rho2*u**2*(-A*Ai*G1*G2*T1*alpha1*ht_21*rho1*u**2 + A*Ai*G1*G2*T1*alpha1*ht_21*rho_mix*u**2 - A*Ai*G1*G2*T1*alpha2*ht_21*rho2*u**2 + A*Ai*G1*G2*T1*alpha2*ht_21*rho_mix*u**2 + A*Ai*G1*G2*T2*alpha1*ht_21*rho1*u**2 - A*Ai*G1*G2*T2*alpha1*ht_21*rho_mix*u**2 + A*Ai*G1*G2*T2*alpha2*ht_21*rho2*u**2 - A*Ai*G1*G2*T2*alpha2*ht_21*rho_mix*u**2 - A*Ai*G1*T1*a2**2*alpha2*ht_21*rho2 + A*Ai*G1*T1*alpha2*ht_21*rho_mix*u**2 + A*Ai*G1*T2*a2**2*alpha2*ht_21*rho2 - A*Ai*G1*T2*alpha2*ht_21*rho_mix*u**2 - A*Ai*G2*T1*a1**2*alpha1*ht_21*rho1 + A*Ai*G2*T1*alpha1*ht_21*rho_mix*u**2 + A*Ai*G2*T2*a1**2*alpha1*ht_21*rho1 - A*Ai*G2*T2*alpha1*ht_21*rho_mix*u**2 - G1*G2*alpha1*alpha2*perimeter*rho1*tau_w_mix*u**3 + G1*G2*alpha1*alpha2*perimeter*rho2*tau_w_mix*u**3 + G1*a2**2*alpha1*alpha2*dAdx*rho1*rho2*u**3 - G1*a2**2*alpha1*alpha2*dAdx*rho2*rho_mix*u**3 + G1*a2**2*alpha1*alpha2*perimeter*rho2*tau_w_mix*u - G1*alpha1*alpha2*perimeter*rho1*tau_w_mix*u**3 - G2*a1**2*alpha1*alpha2*dAdx*rho1*rho2*u**3 + G2*a1**2*alpha1*alpha2*dAdx*rho1*rho_mix*u**3 - G2*a1**2*alpha1*alpha2*perimeter*rho1*tau_w_mix*u + G2*alpha1*alpha2*perimeter*rho2*tau_w_mix*u**3 + a1**2*alpha1*alpha2*dAdx*rho1*rho_mix*u**3 - a1**2*alpha1*alpha2*perimeter*rho1*tau_w_mix*u - a2**2*alpha1*alpha2*dAdx*rho2*rho_mix*u**3 + a2**2*alpha1*alpha2*perimeter*rho2*tau_w_mix*u)/(A*a1**2*a2**2)
    # det_N2 = 1.0*alpha1*alpha2*rho1*rho2*u**2*(A*Ai*G1*G2*T1*alpha1*ht_21*rho1*u**2 - A*Ai*G1*G2*T1*alpha1*ht_21*rho_mix*u**2 + A*Ai*G1*G2*T1*alpha2*ht_21*rho2*u**2 - A*Ai*G1*G2*T1*alpha2*ht_21*rho_mix*u**2 - A*Ai*G1*G2*T2*alpha1*ht_21*rho1*u**2 + A*Ai*G1*G2*T2*alpha1*ht_21*rho_mix*u**2 - A*Ai*G1*G2*T2*alpha2*ht_21*rho2*u**2 + A*Ai*G1*G2*T2*alpha2*ht_21*rho_mix*u**2 + A*Ai*G1*T1*a2**2*alpha2*ht_21*rho2 - A*Ai*G1*T1*alpha2*ht_21*rho_mix*u**2 - A*Ai*G1*T2*a2**2*alpha2*ht_21*rho2 + A*Ai*G1*T2*alpha2*ht_21*rho_mix*u**2 + A*Ai*G2*T1*a1**2*alpha1*ht_21*rho1 - A*Ai*G2*T1*alpha1*ht_21*rho_mix*u**2 - A*Ai*G2*T2*a1**2*alpha1*ht_21*rho1 + A*Ai*G2*T2*alpha1*ht_21*rho_mix*u**2 + G1*G2*alpha1*alpha2*perimeter*rho1*tau_w_mix*u**3 - G1*G2*alpha1*alpha2*perimeter*rho2*tau_w_mix*u**3 - G1*a2**2*alpha1*alpha2*dAdx*rho1*rho2*u**3 + G1*a2**2*alpha1*alpha2*dAdx*rho2*rho_mix*u**3 - G1*a2**2*alpha1*alpha2*perimeter*rho2*tau_w_mix*u + G1*alpha1*alpha2*perimeter*rho1*tau_w_mix*u**3 + G2*a1**2*alpha1*alpha2*dAdx*rho1*rho2*u**3 - G2*a1**2*alpha1*alpha2*dAdx*rho1*rho_mix*u**3 + G2*a1**2*alpha1*alpha2*perimeter*rho1*tau_w_mix*u - G2*alpha1*alpha2*perimeter*rho2*tau_w_mix*u**3 - a1**2*alpha1*alpha2*dAdx*rho1*rho_mix*u**3 + a1**2*alpha1*alpha2*perimeter*rho1*tau_w_mix*u + a2**2*alpha1*alpha2*dAdx*rho2*rho_mix*u**3 - a2**2*alpha1*alpha2*perimeter*rho2*tau_w_mix*u)/(A*a1**2*a2**2)
    # det_N3 = 1.0*alpha2*rho1**2*rho2*u**2*(A*Ai*G1*G2*T1*alpha1*ht_21*rho1*u**2 - A*Ai*G1*G2*T1*alpha1*ht_21*rho_mix*u**2 + A*Ai*G1*G2*T1*alpha2*ht_21*rho2*u**2 - A*Ai*G1*G2*T1*alpha2*ht_21*rho_mix*u**2 - A*Ai*G1*G2*T2*alpha1*ht_21*rho1*u**2 + A*Ai*G1*G2*T2*alpha1*ht_21*rho_mix*u**2 - A*Ai*G1*G2*T2*alpha2*ht_21*rho2*u**2 + A*Ai*G1*G2*T2*alpha2*ht_21*rho_mix*u**2 + A*Ai*G1*T1*a2**2*alpha1*ht_21*rho2 + A*Ai*G1*T1*a2**2*alpha2*ht_21*rho2 - A*Ai*G1*T1*alpha2*ht_21*rho_mix*u**2 - A*Ai*G1*T2*a2**2*alpha1*ht_21*rho2 - A*Ai*G1*T2*a2**2*alpha2*ht_21*rho2 + A*Ai*G1*T2*alpha2*ht_21*rho_mix*u**2 - A*Ai*G2*T1*alpha1*ht_21*rho_mix*u**2 + A*Ai*G2*T2*alpha1*ht_21*rho_mix*u**2 + G1*G2*alpha1*alpha2*perimeter*rho1*tau_w_mix*u**3 - G1*G2*alpha1*alpha2*perimeter*rho2*tau_w_mix*u**3 - G1*a2**2*alpha1**2*dAdx*rho1*rho2*u**3 + G1*a2**2*alpha1**2*dAdx*rho2*rho_mix*u**3 - G1*a2**2*alpha1**2*perimeter*rho2*tau_w_mix*u - G1*a2**2*alpha1*alpha2*dAdx*rho1*rho2*u**3 + G1*a2**2*alpha1*alpha2*dAdx*rho2*rho_mix*u**3 - G1*a2**2*alpha1*alpha2*perimeter*rho2*tau_w_mix*u + G1*alpha1*alpha2*perimeter*rho1*tau_w_mix*u**3 - G2*alpha1*alpha2*perimeter*rho2*tau_w_mix*u**3 + a2**2*alpha1**2*dAdx*rho2*rho_mix*u**3 - a2**2*alpha1**2*perimeter*rho2*tau_w_mix*u + a2**2*alpha1*alpha2*dAdx*rho2*rho_mix*u**3 - a2**2*alpha1*alpha2*perimeter*rho2*tau_w_mix*u)/(A*a1**2*a2**2)
    # det_N4 = 1.0*alpha1*rho1*rho2**2*u**2*(-A*Ai*G1*G2*T1*alpha1*ht_21*rho1*u**2 + A*Ai*G1*G2*T1*alpha1*ht_21*rho_mix*u**2 - A*Ai*G1*G2*T1*alpha2*ht_21*rho2*u**2 + A*Ai*G1*G2*T1*alpha2*ht_21*rho_mix*u**2 + A*Ai*G1*G2*T2*alpha1*ht_21*rho1*u**2 - A*Ai*G1*G2*T2*alpha1*ht_21*rho_mix*u**2 + A*Ai*G1*G2*T2*alpha2*ht_21*rho2*u**2 - A*Ai*G1*G2*T2*alpha2*ht_21*rho_mix*u**2 + A*Ai*G1*T1*alpha2*ht_21*rho_mix*u**2 - A*Ai*G1*T2*alpha2*ht_21*rho_mix*u**2 - A*Ai*G2*T1*a1**2*alpha1*ht_21*rho1 - A*Ai*G2*T1*a1**2*alpha2*ht_21*rho1 + A*Ai*G2*T1*alpha1*ht_21*rho_mix*u**2 + A*Ai*G2*T2*a1**2*alpha1*ht_21*rho1 + A*Ai*G2*T2*a1**2*alpha2*ht_21*rho1 - A*Ai*G2*T2*alpha1*ht_21*rho_mix*u**2 - G1*G2*alpha1*alpha2*perimeter*rho1*tau_w_mix*u**3 + G1*G2*alpha1*alpha2*perimeter*rho2*tau_w_mix*u**3 - G1*alpha1*alpha2*perimeter*rho1*tau_w_mix*u**3 - G2*a1**2*alpha1*alpha2*dAdx*rho1*rho2*u**3 + G2*a1**2*alpha1*alpha2*dAdx*rho1*rho_mix*u**3 - G2*a1**2*alpha1*alpha2*perimeter*rho1*tau_w_mix*u - G2*a1**2*alpha2**2*dAdx*rho1*rho2*u**3 + G2*a1**2*alpha2**2*dAdx*rho1*rho_mix*u**3 - G2*a1**2*alpha2**2*perimeter*rho1*tau_w_mix*u + G2*alpha1*alpha2*perimeter*rho2*tau_w_mix*u**3 + a1**2*alpha1*alpha2*dAdx*rho1*rho_mix*u**3 - a1**2*alpha1*alpha2*perimeter*rho1*tau_w_mix*u + a1**2*alpha2**2*dAdx*rho1*rho_mix*u**3 - a1**2*alpha2**2*perimeter*rho1*tau_w_mix*u)/(A*a1**2*a2**2)
    # det_N5 = 1.0*alpha1*alpha2*rho1*rho2*u**3*(-A*Ai*G1*T1*a2**2*ht_21*rho2 + A*Ai*G1*T2*a2**2*ht_21*rho2 + A*Ai*G2*T1*a1**2*ht_21*rho1 - A*Ai*G2*T2*a1**2*ht_21*rho1 + G1*a2**2*alpha1*perimeter*rho2*tau_w_mix*u + G2*a1**2*alpha2*perimeter*rho1*tau_w_mix*u - a1**2*a2**2*alpha1*dAdx*rho1*rho2*u - a1**2*a2**2*alpha2*dAdx*rho1*rho2*u + a1**2*alpha2*perimeter*rho1*tau_w_mix*u + a2**2*alpha1*perimeter*rho2*tau_w_mix*u)/(A*a1**2*a2**2)
    # det_N6 = 1.0*alpha1*alpha2*rho1*rho2*u**3*(A*Ai*G1*T1*a2**2*ht_21*rho2*rho_mix*u - A*Ai*G1*T2*a2**2*ht_21*rho2*rho_mix*u - A*Ai*G2*T1*a1**2*ht_21*rho1*rho_mix*u + A*Ai*G2*T2*a1**2*ht_21*rho1*rho_mix*u - G1*a2**2*alpha1*perimeter*rho1*rho2*tau_w_mix*u**2 - G2*a1**2*alpha2*perimeter*rho1*rho2*tau_w_mix*u**2 + a1**2*a2**2*alpha1*dAdx*rho1*rho2*rho_mix*u**2 - a1**2*a2**2*alpha1*perimeter*rho1*rho2*tau_w_mix + a1**2*a2**2*alpha2*dAdx*rho1*rho2*rho_mix*u**2 - a1**2*a2**2*alpha2*perimeter*rho1*rho2*tau_w_mix)/(A*a1**2*a2**2)
    # det_N7 = 1.0*alpha2*rho2*u**2*(A*Ai*G1*T1*a2**2*alpha1*ht_21*rho2*rho_mix*u**2 - A*Ai*G1*T2*a2**2*alpha1*ht_21*rho2*rho_mix*u**2 - A*Ai*G2*T1*a1**2*alpha1*ht_21*rho1**2*u**2 - A*Ai*G2*T1*a1**2*alpha2*ht_21*rho1*rho2*u**2 + A*Ai*G2*T1*a1**2*alpha2*ht_21*rho1*rho_mix*u**2 + A*Ai*G2*T2*a1**2*alpha1*ht_21*rho1**2*u**2 + A*Ai*G2*T2*a1**2*alpha2*ht_21*rho1*rho2*u**2 - A*Ai*G2*T2*a1**2*alpha2*ht_21*rho1*rho_mix*u**2 - A*Ai*T1*a1**2*a2**2*alpha1*ht_21*rho1*rho2 - A*Ai*T1*a1**2*a2**2*alpha2*ht_21*rho1*rho2 + A*Ai*T1*a1**2*alpha2*ht_21*rho1*rho_mix*u**2 + A*Ai*T1*a2**2*alpha1*ht_21*rho2*rho_mix*u**2 + A*Ai*T2*a1**2*a2**2*alpha1*ht_21*rho1*rho2 + A*Ai*T2*a1**2*a2**2*alpha2*ht_21*rho1*rho2 - A*Ai*T2*a1**2*alpha2*ht_21*rho1*rho_mix*u**2 - A*Ai*T2*a2**2*alpha1*ht_21*rho2*rho_mix*u**2 - G1*a2**2*alpha1**2*perimeter*rho1*rho2*tau_w_mix*u**3 - G2*a1**2*alpha1*alpha2*perimeter*rho1**2*tau_w_mix*u**3 + a1**2*a2**2*alpha1**2*dAdx*rho1**2*rho2*u**3 + a1**2*a2**2*alpha1*alpha2*dAdx*rho1**2*rho2*u**3 - a1**2*alpha1*alpha2*perimeter*rho1**2*tau_w_mix*u**3 - a2**2*alpha1**2*perimeter*rho1*rho2*tau_w_mix*u**3)/(A*a1**2*a2**2)
    # det_N8 = 1.0*alpha1*rho1*u**2*(A*Ai*G1*T1*a2**2*alpha1*ht_21*rho1*rho2*u**2 - A*Ai*G1*T1*a2**2*alpha1*ht_21*rho2*rho_mix*u**2 + A*Ai*G1*T1*a2**2*alpha2*ht_21*rho2**2*u**2 - A*Ai*G1*T2*a2**2*alpha1*ht_21*rho1*rho2*u**2 + A*Ai*G1*T2*a2**2*alpha1*ht_21*rho2*rho_mix*u**2 - A*Ai*G1*T2*a2**2*alpha2*ht_21*rho2**2*u**2 - A*Ai*G2*T1*a1**2*alpha2*ht_21*rho1*rho_mix*u**2 + A*Ai*G2*T2*a1**2*alpha2*ht_21*rho1*rho_mix*u**2 + A*Ai*T1*a1**2*a2**2*alpha1*ht_21*rho1*rho2 + A*Ai*T1*a1**2*a2**2*alpha2*ht_21*rho1*rho2 - A*Ai*T1*a1**2*alpha2*ht_21*rho1*rho_mix*u**2 - A*Ai*T1*a2**2*alpha1*ht_21*rho2*rho_mix*u**2 - A*Ai*T2*a1**2*a2**2*alpha1*ht_21*rho1*rho2 - A*Ai*T2*a1**2*a2**2*alpha2*ht_21*rho1*rho2 + A*Ai*T2*a1**2*alpha2*ht_21*rho1*rho_mix*u**2 + A*Ai*T2*a2**2*alpha1*ht_21*rho2*rho_mix*u**2 - G1*a2**2*alpha1*alpha2*perimeter*rho2**2*tau_w_mix*u**3 - G2*a1**2*alpha2**2*perimeter*rho1*rho2*tau_w_mix*u**3 + a1**2*a2**2*alpha1*alpha2*dAdx*rho1*rho2**2*u**3 + a1**2*a2**2*alpha2**2*dAdx*rho1*rho2**2*u**3 - a1**2*alpha2**2*perimeter*rho1*rho2*tau_w_mix*u**3 - a2**2*alpha1*alpha2*perimeter*rho2**2*tau_w_mix*u**3)/(A*a1**2*a2**2)

    # # Compute determinants
    det_D = det(A_mat)
    det_N1 = det(M1)
    det_N2 = det(M2)
    det_N3 = det(M3)
    det_N4 = det(M4)
    det_N5 = det(M5)
    det_N6 = det(M6)
    det_N7 = det(M7)
    det_N8 = det(M8)
    # det_D = det_sympy_actual


    # jax.debug.print("x={x}, det={det_D}, Ma={Ma_mix}, delta_det={dd}", x=x, det_D=det_D, Ma_mix=u/a_mix, dd=det_D-det_sympy_actual)

    # jax.debug.print("x={x} | Ma={Ma} | det={D} | d={d}", x=x, Ma=u/a_mix,D=det_D,d=d_2,)
    # jax.debug.print(
    #     "x={x}, Ma={Ma}, Ma_st={Ma_st}, det={D}", x=x, Ma=u/a_mix, Ma_st=u/a_mix_stadke,D=det_D
    # )

    # jax.debug.print("x={x} | determinant={D} |  Ma={Ma_mix}", Ma_mix=u/a_mix, D=det_D, x=x)

    N = [det_N1, det_N2, det_N3, det_N4, det_N5, det_N6, det_N7, det_N8]
    N = jnp.array(N)
    # --- Autonomous system: dx/dτ = D, dy/dτ = N_i ---
    rhs = jnp.array([det_N1 / det_D, det_N2 / det_D, det_N3 / det_D, det_N4 / det_D, det_N5 / det_D, det_N6 / det_D, det_N7 / det_D, det_N8 / det_D])
    rhs_autonomous = jnp.array([det_D, det_N1, det_N2, det_N3, det_N4, det_N5, det_N6, det_N7, det_N8])
    # jax.debug.print("x={x} | determinant={D} |  Ma={Ma_mix}", Ma_mix=u/a_mix, D=det_D, x=x)

    # Export data
    out = {
        "x": x,
        "v": u,
        "alpha1":alpha1,
        "alpha2":alpha2,
        "rho1": rho1,
        "rho2":rho2,
        "p": p,
        "h1":h1,
        "h2":h2,
        "T1":T1,
        "T2":T2,
        "s1":s1,
        "s2":s2,
        "rhs": rhs,
        "rhs_autonomous": rhs_autonomous,
        "area": A,
        "dAdx": dAdx,
        "diameter": diameter,
        "perimeter": perimeter,
        # "h01": h01,
        # "p01": p01,
        # "T01": T01,
        # "d01": rho01,
        "Ma1": u / a1,
        # "h02": h02,
        # "p02": p02,
        # "T02": T02,
        # "d02": rho02,
        "rho_mix":rho_mix,
        "a_mix":a_mix,
        "Ma2": u / a2,
        "Ma_mix":u / a_mix,
        "Re": Re_mix,
        "f_D": f_D,
        "tau_w": tau_w_mix,
        "A_mat": A_mat,
        "b_vec": b_vec,
        "D": det_D,
        "N": N,
        "m_dot": u * rho_mix * A,
        # "eigenvalues":eigenvalues,
    }

    return {**out, **state1.to_dict(include_aliases=True),  **state2.to_dict(include_aliases=True)}
    # return {**out, **state}

def nozzle_homogeneous_non_equilibrium_one_component_core(x, y, args):
    """Wrapper that adapts from (x, y) to autonomous form."""
    Y = jnp.concatenate([jnp.atleast_1d(x), y])
    return nozzle_homogeneous_non_equilibrium_one_component_autonomous(0.0, Y, args)

def nozzle_homogeneous_non_equilibrium_one_component_core_lax_eight(x, y, args):
    """Wrapper that adapts from (x, y) to autonomous form."""
    Y = jnp.concatenate([jnp.atleast_1d(x), y])
    return nozzle_homogeneous_non_equilibrium_one_component_autonomous_lax_eigth(0.0, Y, args)


# # @eqx.filter_jit
# def nozzle_homogeneous_non_equilibrium_one_component_autonomous_old(tau, Y, args):
#     """
#     Autonomous formulation of the nozzle equations:
#         dx/dt   = det(A)
#         dy_i/dt = det(A with column i replaced by b)

#     State vector: Y = [x, v, rho, p]
#     """
#     # Matrix written for flashing, where v is the dispersed phase while l is the continuous phase
#     x, alphav, alphal, rhov, rhol, u, p, hv, hl = Y

#     # v = jnp.where(jnp.abs(v) < 1e-6, jnp.sign(v)*1e-6, v)
#     # Debug print using jax.debug.print
#     jax.debug.print(
#     "x={:.2f}, alpha1={:.4e}, alpha2={:.4e}, rho1={:.4e}, rho2={:.4e}, u={:.4e}, p={:.4e}, h1={:.4e}, h2={:.4e}",
#     x, alphav, alphal, rhov, rhol, u, p, hv, hl
#     )   

#     # --- Geometry / model parameters ---
#     fluidV = args.fluid1
#     fluidL = args.fluid2
#     L = args.length
#     eps_wall = args.roughness
#     wall_friction = args.wall_friction
#     heat_transfer = args.heat_transfer

#     # --- Thermodynamic state fluid ---
#     ALPHA_MIN = 1e-6

#     # --- Step 1: Clamp volume fractions and enforce constraint ---
#     alphav = jnp.clip(alphav, ALPHA_MIN, 1.0 - ALPHA_MIN)
#     alphal = jnp.clip(alphal, ALPHA_MIN, 1.0 - ALPHA_MIN)

#     p = jnp.clip(p, 1e2, 1e9)   # enforce valid pressure range
    
#     statev = fluidV.get_state(jxp.HmassP_INPUTS, hv, p)
#     Tv = statev["T"]
#     sv = statev["s"]
#     av = statev["a"]
#     cpv = statev["cp"]
#     muv = statev["mu"]
#     Gv = statev["gruneisen"]

#     # # Stagnation state
#     # h01 = state1["h"] + 0.5 * u**2
#     # state01 = fluid1.get_state(jxp.HmassSmass_INPUTS, h01, state1["s"])
#     # p01 = state01["p"]
#     # T01 = state01["T"]
#     # rho01 = state01["d"]

#     # --- Thermodynamic state fluid 2 ---
#     statel = fluidL.get_state(jxp.HmassP_INPUTS, hl, p)
#     Tl = statel["T"]
#     sl = statel["s"]
#     al = statel["a"]
#     cpl = statel["cp"]
#     mul = statel["mu"]
#     Gl = statel["gruneisen"]

#     # # Stagnation state
#     # h02 = state2["h"] + 0.5 * u**2
#     # state02 = fluid2.get_state(jxp.HmassSmass_INPUTS, h02, state1["s"])
#     # p02 = state02["p"]
#     # T02 = state02["T"]
#     # rho02 = state02["d"]

#     rho_mix = rhov * alphav + rhol * alphal
#     mu_mix = muv * alphav + mul * alphal

#     S = (rhov**2 * rhol**2 * u**3) ** (1/9)

#     # --- Geometry ---
#     # A, dAdx, perimeter, radius = args.geometry(x, L) # When using symmetric geometry
#     A, dAdx, perimeter, radius = args.geometry(x) # When using linear convergent divergent geometry
#     diameter = 2 * radius

#     # --- Wall heat transfer and friction ---
#     Re_mix = u * rho_mix * diameter / mu_mix # jnp.maximum(mu_mix, 1e-12)
#     f_D = get_friction_factor_haaland(Re_mix, eps_wall, diameter)
#     tau_w_mix = get_wall_viscous_stress(f_D, rho_mix, u)
#     Ai, d_v = interfacial_area(alphal)
#     # ht_1 = 6 # water
#     Nu_v = 12 # the two resistence model is used to model the heat transfer coefficient between the phases. The htc in the dispersed phase side (vapor) its calculated using a constant htc and a sensitivity analysis is needed to finde the appropriate value
#     k_v = statev["k"]
#     ht_v = (k_v * Nu_v) / d_v # nitrogen
#     ht_l = ranz_marshall_htc(alphav, alphal, statel, statev,)
#     ht_vl = (1 / ((1/ht_v) + (1/ht_l)))
#     # the flow is considered to be a dispersed flow with liquid droplets being the dispersed phase
#     Qvl = ht_vl * Ai * (Tv - Tl)
    
#     liquid_sat = fluidL.get_state_saturation(p)
#     vapor_sat = fluidV.get_state_saturation(p)
    
    
#     h_sat_l = liquid_sat["enthalpy"]
#     h_sat_v = vapor_sat["enthalpy"]
#     Tsat = liquid_sat["temperature"]

#     # mass transfer from liquid to vapor
#     # check that the Temperature is smaller of the saturation temperature and that it is smaller than the critical temperature
#     m_dot_lv = calculate_mass_transfer_lv(Tl, Tv, Tsat, ht_l, ht_v, Ai, h_sat_l, h_sat_v, hl, hv)


#     a_mix = get_speed_of_sound_mixture(Gv, alphav, av, rhov, Gl, alphal, al, rhol)

#     # Mask with booleans (convert to 0.0 if disabled)
#     # tau_w_mix = wall_friction * tau_w_mix
#     # f_D = wall_friction * f_D

#     kappa = jnp.where(alphav < 1e-4, 1e-10, 1.0)

#     # --- Build A matrix and b vector ---
#     # A_mat = jnp.array([[d, v, 0.0], [d * v, 0.0, 1.0], [0.0, -(a**2), 1.0]])
#     # A_mat = jnp.asarray(
#     #             [
#     #                 [-1.0,               -1.0,               0.0,               0.0,               0.0,                    0.0,                    0.0,                    0.0],
#     #                 kappa * [rhov * u,          0.0,               alphav * u,        0.0,               alphav * rhov,          0.0,                    0.0,                    0.0],
#     #                 [0.0,               rhol * u,          0.0,               alphal * u,        alphal * rhol,          0.0,                    0.0,                    0.0],
#     #                 [0.0,               0.0,               0.0,               0.0,               rho_mix * u,            1.0,                    0.0,                    0.0],
#     #                 kappa * [0.0,               0.0,               0.0,               0.0,               alphav * rhov * u**2,   0.0,                    alphav * rhov * u,      0.0],
#     #                 [0.0,               0.0,               0.0,               0.0,               alphal * rhol * u**2,   0.0,                    0.0,                    alphal * rhol * u],
#     #                 [0.0,               0.0,              -1.0,               0.0,               0.0,                    (1 + Gv) / av**2,       -(rhov * Gv) / av**2,    0.0],
#     #                 [0.0,               0.0,               0.0,              -1.0,               0.0,                    (1 + Gl) / al**2,       0.0,                    -(rhol * Gl) / al**2],
#     #             ]
#     #         )

    
#     A_mat = jnp.stack([
#         jnp.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),

#         kappa * jnp.array([rhov * u, 0.0, alphav * u, 0.0, alphav * rhov, 0.0, 0.0, 0.0]),

#         jnp.array([0.0, rhol * u, 0.0, alphal * u, alphal * rhol, 0.0, 0.0, 0.0]),

#         jnp.array([0.0, 0.0, 0.0, 0.0, rho_mix * u, 1.0, 0.0, 0.0]),

#         kappa * jnp.array([0.0, 0.0, 0.0, 0.0, alphav * rhov * u**2, 0.0, alphav * rhov * u, 0.0]),

#         jnp.array([0.0, 0.0, 0.0, 0.0, alphal * rhol * u**2, 0.0, 0.0, alphal * rhol * u]),

#         jnp.array([0.0, 0.0, -1.0, 0.0, 0.0, (1.0 + Gv) / av**2, -(rhov * Gv) / av**2, 0.0]),

#         jnp.array([0.0, 0.0, 0.0, -1.0, 0.0, (1.0 + Gl) / al**2, 0.0, -(rhol * Gl) / al**2]),
#     ], axis=0)


#     b_vec = jnp.asarray(
#                 [
#                     0.0,
#                     kappa * (m_dot_lv - (dAdx/A) * alphav * rhov *  u),
#                     - m_dot_lv - (dAdx/A) * alphal * rhol *  u,
#                     -(tau_w_mix * perimeter) / A,
#                     kappa * (m_dot_lv * (h_sat_v) - Qvl),
#                     kappa * (Qvl - m_dot_lv * (h_sat_v)),
#                     0.0,
#                     0.0
#                 ]
#             )

#     A_mat = A_mat / S
#     b_vec = b_vec / S

#     # --- Determinants ---
#     # D = det(A_mat)
#     A_mat_copy = 1
#     M1 = A_mat.at[:, 0].set(b_vec)
#     M2 = A_mat.at[:, 1].set(b_vec)
#     M3 = A_mat.at[:, 2].set(b_vec)
#     M4 = A_mat.at[:, 3].set(b_vec)
#     M5 = A_mat.at[:, 4].set(b_vec)
#     M6 = A_mat.at[:, 5].set(b_vec)
#     M7 = A_mat.at[:, 6].set(b_vec)
#     M8 = A_mat.at[:, 7].set(b_vec)

#     # # Compute determinants
#     det_D = det(A_mat)
#     det_N1 = det(M1)
#     det_N2 = det(M2)
#     det_N3 = det(M3)
#     det_N4 = det(M4)
#     det_N5 = det(M5)
#     det_N6 = det(M6)
#     det_N7 = det(M7)
#     det_N8 = det(M8)
#     # det_D = det_sympy_actual

#     # jax.debug.print("x={x} | determinant={D} |  Ma={Ma_mix}", Ma_mix=u/a_mix, D=det_D, x=x)

#     N = [det_N1, det_N2, det_N3, det_N4, det_N5, det_N6, det_N7, det_N8]
#     N = jnp.array(N)
#     # --- Autonomous system: dx/dτ = D, dy/dτ = N_i ---
#     rhs = jnp.array([det_N1 / det_D, det_N2 / det_D, det_N3 / det_D, det_N4 / det_D, det_N5 / det_D, det_N6 / det_D, det_N7 / det_D, det_N8 / det_D])
#     rhs_autonomous = jnp.array([det_D, det_N1, det_N2, det_N3, det_N4, det_N5, det_N6, det_N7, det_N8])
#     jax.debug.print("x={x} | determinant={D} |  a={Ma_mix}", Ma_mix=a_mix, D=det_D, x=x)
#     jax.debug.print("Qvl:{qv}", qv=Qvl)
#     # Export data
#     out = {
#         "x": x,
#         "v": u,
#         "alpha1":alphav,
#         "alpha2":alphal,
#         "rho1": rhov,
#         "rho2":rhol,
#         "p": p,
#         "h1":hv,
#         "h2":hl,
#         "T1":Tv,
#         "T2":Tl,
#         "s1":sv,
#         "s2":sl,
#         "rhs": rhs,
#         "rhs_autonomous": rhs_autonomous,
#         "area": A,
#         "dAdx": dAdx,
#         "diameter": diameter,
#         "perimeter": perimeter,
#         # "h01": h01,
#         # "p01": p01,
#         # "T01": T01,
#         # "d01": rho01,
#         "Ma1": u / av,
#         # "h02": h02,
#         # "p02": p02,
#         # "T02": T02,
#         # "d02": rho02,
#         "rho_mix":rho_mix,
#         "a_mix":a_mix,
#         "Ma2": u / al,
#         "Ma_mix":u / a_mix,
#         "Re": Re_mix,
#         "f_D": f_D,
#         "tau_w": tau_w_mix,
#         "A_mat": A_mat,
#         "b_vec": b_vec,
#         "D": det_D,
#         "N": N,
#         "m_dot": u * rho_mix * A,
#     }

#     return {**out, **statev.to_dict(include_aliases=True),  **statel.to_dict(include_aliases=True)}
#     # return {**out, **state}

def nozzle_homogeneous_non_equilibrium_one_component_autonomous(tau, Y, args):
    """
    Autonomous formulation WITHOUT ramping or matrix modification.
    Ghost phase strategy: clamp alpha >= ALPHA_MIN, use actual (not saturation)
    properties for the ghost phase, and set heat/mass transfer to zero when
    the flow is effectively single-phase.
    State vector: Y = [x, alphav, alphal, rhov, rhol, u, p, hv, hl]
    """

    x, alphav, alphal, rhov, rhol, u, p, hv, hl = Y
    # jax.debug.print(
    #     "RAW Y: x={:.4f} αv={:.3e} αl={:.3e} ρv={:.2f} ρl={:.2f} u={:.4f} p={:.1f} hv={:.1f} hl={:.1f}",
    #     x, alphav, alphal, rhov, rhol, u, p, hv, hl
    # )

    # --- Geometry / model parameters ---
    fluidV = args.fluid1
    fluidL = args.fluid2
    L = args.length
    eps_wall = args.roughness
    wall_friction = args.wall_friction
    heat_transfer = args.heat_transfer

    # --- Clamp volume fractions ---
    ALPHA_MIN = 1e-6
    # alphav = jnp.clip(alphav, ALPHA_MIN, 1.0 - ALPHA_MIN)
    # alphal = jnp.clip(alphal, ALPHA_MIN, 1.0 - ALPHA_MIN)
    p = jnp.clip(p, 1.0, 1e9)

    # --- Saturation properties (needed for mass transfer) ---
    liquid_sat = fluidL.get_state_saturation(p)
    vapor_sat = fluidV.get_state_saturation(p)
    h_sat_l = liquid_sat["enthalpy"]
    h_sat_v = vapor_sat["enthalpy"]
    Tsat = liquid_sat["temperature"]

    # --- Thermodynamic states ---
    statel = fluidL.get_state(jxp.HmassP_INPUTS, hl, p)
    Tl = statel["T"]

    # --- Two-phase flag based on enthalpy and temperature ---
    is_two_phase = (hl >= h_sat_l) & (Tl >= Tsat)
    # Vapor: real properties from get_state when two-phase, saturation when
    # single-phase (ghost phase has non-physical hv).
    # Clamp ghost-phase enthalpy to saturation so get_state always receives
    # valid inputs (avoids NaN in unused branch during Jacobian evaluation).
    hv_safe = jnp.clip(hv, fluidV.h_min, fluidV.h_max)
    statev_actual = fluidV.get_state(jxp.HmassP_INPUTS, hv_safe, p)
    Tv = jnp.where(is_two_phase, statev_actual["T"], vapor_sat["temperature"])
    sv = jnp.where(is_two_phase, statev_actual["s"], vapor_sat["entropy"])
    av = jnp.where(is_two_phase, statev_actual["a"], vapor_sat["speed_of_sound"])
    muv = jnp.where(is_two_phase, statev_actual["mu"], vapor_sat["viscosity"])
    Gv = jnp.where(is_two_phase, statev_actual["gruneisen"], vapor_sat["gruneisen"])
    sl = statel["s"]
    al = statel["a"]
    mul = statel["mu"]
    Gl = statel["gruneisen"]

    # --- Mixture properties ---
    rho_mix = rhov * alphav + rhol * alphal
    mu_mix = muv * alphav + mul * alphal
    S = (rhov**2 * rhol**2 * u**3) ** (1 / 9)

    # --- Geometry ---
    A, dAdx, perimeter, radius = args.geometry(x)
    diameter = 2 * radius

    # --- Wall friction ---
    Re_mix = u * rho_mix * diameter / mu_mix
    f_D = get_friction_factor_haaland(Re_mix, eps_wall, diameter)
    tau_w_mix = get_wall_viscous_stress(f_D, rho_mix, u)
    tau_w_mix = tau_w_mix * wall_friction

    # --- Interfacial heat transfer ---
    # Use Tsat for the vapor-side interface temperature (interface is at saturation)
    Ai, d_v = interfacial_area(alphav)
    Nu_v = 12
    Nu_l = 18
    k_v = statev_actual["conductivity"]
    k_l = statel["k"]
    ht_v = (k_v * Nu_v) / d_v
    # ht_l = (k_l * Nu_l) / (diameter * alphal)
    ht_l = ranz_marshall_htc(alphav, alphal, statel, statev_actual, fluidL)
    # ht_vl = 1.0 / (1.0 / ht_v + 1.0 / ht_l)
    # Qvl = ht_vl * Ai * (Tsat - Tl)
    Qvi = ht_v * Ai * (Tsat - Tv)
    Qli = ht_l * Ai * (Tsat - Tl)

    # --- Mass transfer ---

    # --- Zero out heat/mass transfer in single-phase region ---
    # Qvl = jnp.where(is_two_phase, Qvl, 0.0)
    Qvi = jnp.where(is_two_phase, Qvi, 0.0)
    Qli = jnp.where(is_two_phase, Qli, 0.0)
    m_dot_lv = calculate_mass_transfer_lv(
        Tl, Tv, Tsat, ht_l, ht_v, Ai, h_sat_l, h_sat_v, hl, hv, Qvi, Qli,
    )
    m_dot_lv = jnp.where(is_two_phase, m_dot_lv, 0.0)

    # --- Speed of sound (always use mixture formula) ---
    a_mix = get_speed_of_sound_mixture(
        Gv, alphav, av, rhov, Gl, alphal, al, rhol
    )

    # --- Energy source terms (non-conservative total enthalpy form) ---

    # --- Build A matrix and b vector (no ramping, full two-phase rows always) ---
    # Variables: [dαv, dαl, dρv, dρl, du, dp, dhv, dhl]
    b4 = m_dot_lv * (h_sat_v - hv) + Qvi
    b5 = Qli - m_dot_lv * (h_sat_l - hl)
    A_mat = jnp.stack(
        [
            jnp.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([rhov * u, 0.0, alphav * u, 0.0, alphav * rhov, 0.0, 0.0, 0.0]),
            jnp.array([0.0, rhol * u, 0.0, alphal * u, alphal * rhol, 0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0, 0.0, rho_mix * u, 1.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0, 0.0, alphav * rhov * u**2, 0.0, alphav * rhov * u, 0.0]),
            jnp.array([0.0, 0.0, 0.0, 0.0, alphal * rhol * u**2, 0.0, 0.0, alphal * rhol * u]),
            jnp.array([0.0, 0.0, -1.0, 0.0, 0.0, (1.0 + Gv) / av**2, -(rhov * Gv) / av**2, 0.0]),
            jnp.array([0.0, 0.0, 0.0, -1.0, 0.0, (1.0 + Gl) / al**2, 0.0, -(rhol * Gl) / al**2]),
        ],
        axis=0,
    )
    b_vec = jnp.asarray(
        [
            0.0,
            m_dot_lv - (dAdx / A) * alphav * rhov * u,
            -m_dot_lv - (dAdx / A) * alphal * rhol * u,
            -(tau_w_mix * perimeter) / A,
            b4,
            b5,
            0.0,
            0.0,
        ]
    )
    A_mat = A_mat / S
    b_vec = b_vec / S

    # --- Determinants via Cramer's rule ---
    M1 = A_mat.at[:, 0].set(b_vec)
    M2 = A_mat.at[:, 1].set(b_vec)
    M3 = A_mat.at[:, 2].set(b_vec)
    M4 = A_mat.at[:, 3].set(b_vec)
    M5 = A_mat.at[:, 4].set(b_vec)
    M6 = A_mat.at[:, 5].set(b_vec)
    M7 = A_mat.at[:, 6].set(b_vec)
    M8 = A_mat.at[:, 7].set(b_vec)

    det_D = det(A_mat)
    det_N1 = det(M1)
    det_N2 = det(M2)
    det_N3 = det(M3)
    det_N4 = det(M4)
    det_N5 = det(M5)
    det_N6 = det(M6)
    det_N7 = det(M7)
    det_N8 = det(M8)
    N = jnp.array([det_N1, det_N2, det_N3, det_N4, det_N5, det_N6, det_N7, det_N8])

    # --- Ghost phase: force vapor to follow saturation curve when single-phase ---
    # dhv/dτ = dh_sat_v/dp * dp/dτ,  dρv/dτ = dρ_sat_v/dp * dp/dτ
    dh_sat_v_dp = jax.grad(lambda p_: fluidV.get_state_saturation(p_)["enthalpy"])(p)
    drho_sat_v_dp = jax.grad(lambda p_: fluidV.get_state_saturation(p_)["density"])(p)
    det_N3_sat = drho_sat_v_dp * det_N6 # dρv/dτ following saturation
    det_N7_sat = dh_sat_v_dp * det_N6   # dhv/dτ following saturation
    det_N3 = jnp.where(is_two_phase, det_N3, det_N3_sat)
    det_N7 = jnp.where(is_two_phase, det_N7, det_N7_sat)
    rhs = jnp.array([
        det_N1 / det_D, det_N2 / det_D, det_N3 / det_D, det_N4 / det_D,
        det_N5 / det_D, det_N6 / det_D, det_N7 / det_D, det_N8 / det_D
    ])
    rhs_autonomous = jnp.array([
        det_D, det_N1, det_N2, det_N3, det_N4, det_N5, det_N6, det_N7, det_N8
    ])
    # jax.debug.print(
    #     "x={x:.4f} | amix={am:.1f} | av={av:.1f} | al={al:.1f} | Gv={Gv:.3f} | Gl={Gl:.3f} | Ma={Ma:.4f} | det={D:.3e} | αv={alv:.3e} | ρv={rv:.2f} | ρl={rl:.2f} | 2ph={tp} | b4={b4:.3e} | b5={b5:.3e} | hv={hv:.1f} | hl={hl:.1f} | mdot={md:.3e}",
    #     x=x, am=a_mix, av=av, al=al, Gv=Gv, Gl=Gl, Ma=u/a_mix,
    #     D=det_D, alv=alphav, rv=rhov, rl=rhol, tp=is_two_phase,
    #     b4=b4, b5=b5, hv=hv, hl=hl, md=m_dot_lv
    # )

    # --- Export ---
    out = {
        "x": x,
        "v": u,
        "alpha1": alphav,
        "alpha2": alphal,
        "rho1": rhov,
        "rho2": rhol,
        "p": p,
        "h1": hv,
        "h2": hl,
        "T1": Tv,
        "T2": Tl,
        "s1": sv,
        "s2": sl,
        "rhs": rhs,
        "rhs_autonomous": rhs_autonomous,
        "area": A,
        "dAdx": dAdx,
        "diameter": diameter,
        "perimeter": perimeter,
        "Ma1": u / av,
        "rho_mix": rho_mix,
        "a_mix": a_mix,
        "Ma2": u / al,
        "Ma_mix": u / a_mix,
        "Re": Re_mix,
        "f_D": f_D,
        "tau_w": tau_w_mix,
        "A_mat": A_mat,
        "b_vec": b_vec,
        "D": det_D,
        "N": N,
        "m_dot": u * rho_mix * A,
        "m_dot_lv": m_dot_lv,
        "Qvi": Qvi,
        "Qli": Qli,
        "Gv":Gv,
        "Gl":Gl,
        }
    return {**out, **statev_actual.to_dict(include_aliases=True), **statel.to_dict(include_aliases=True)}



def nozzle_homogeneous_non_equilibrium_one_component_autonomous_lax(tau, Y, args, use_lax_cond=True):
    """
    Autonomous formulation WITHOUT ramping or matrix modification.

    Ghost phase strategy: clamp alpha >= ALPHA_MIN, use actual (not saturation)
    properties for the ghost phase, and set heat/mass transfer to zero when
    the flow is effectively single-phase.

    State vector: Y = [x, alphav, alphal, rhov, rhol, u, p, hv, hl]

    Parameters
    ----------
    use_lax_cond : bool
        If True (default), use jax.lax.cond for two-phase switching (implicit
        solver compatible, but higher memory with vmap).
        If False, use jnp.where (vmap friendly, but NaN gradients with
        implicit solvers).
    """
    x, alphav, alphal, rhov, rhol, u, p, hv, hl = Y

    jax.debug.print(
        "RAW Y: x={:.4f} αv={:.3e} αl={:.3e} ρv={:.2f} ρl={:.2f} u={:.4f} p={:.1f} hv={:.1f} hl={:.1f}",
        x, alphav, alphal, rhov, rhol, u, p, hv, hl
    )

    # --- Geometry / model parameters ---
    fluidV = args.fluid1
    fluidL = args.fluid2
    L = args.length
    eps_wall = args.roughness
    wall_friction = args.wall_friction
    heat_transfer = args.heat_transfer

    # --- Clamp volume fractions ---
    ALPHA_MIN = 1e-6
    alphav_safe = jnp.clip(alphav, ALPHA_MIN, 1.0 - ALPHA_MIN)
    alphal_safe = jnp.clip(alphal, ALPHA_MIN, 1.0 - ALPHA_MIN)
        # Force pressures and enthalpies into the table's strictly valid ranges
    p_safe = jnp.clip(p, fluidV.p_min, fluidV.p_max)
    hv_safe = jnp.clip(hv, fluidV.h_min, fluidV.h_max)
    hl_safe = jnp.clip(hl, fluidL.h_min, fluidL.h_max)


    p = jnp.clip(p, 1.0, 1e9)

    # --- Saturation properties (needed for mass transfer) ---
    liquid_sat = fluidL.get_state_saturation(p_safe)
    vapor_sat = fluidV.get_state_saturation(p_safe)

    h_sat_l = liquid_sat["enthalpy"]
    h_sat_v = vapor_sat["enthalpy"]
    Tsat = liquid_sat["temperature"]

    # --- Thermodynamic states ---
    statel = fluidL.get_state(jxp.HmassP_INPUTS, hl_safe, p_safe)
    Tl = statel["T"]

    # --- Two-phase flag based on enthalpy and temperature ---
    is_two_phase = (hl >= h_sat_l) & (Tl >= Tsat)

    # Vapor: real properties from get_state when two-phase, saturation when
    # single-phase (ghost phase has non-physical hv).
    # Use lax.cond to only trace the taken branch (implicit solver compatible).
    # --- OLD (jnp.where — evaluates both branches, NaN in Jacobian) ---
    # hv_safe = jnp.clip(hv, fluidV.h_min, fluidV.h_max)
    # statev_actual = fluidV.get_state(jxp.HmassP_INPUTS, hv_safe, p)
    # Tv = jnp.where(is_two_phase, statev_actual["T"], vapor_sat["temperature"])
    # sv = jnp.where(is_two_phase, statev_actual["s"], vapor_sat["entropy"])
    # av = jnp.where(is_two_phase, statev_actual["a"], vapor_sat["speed_of_sound"])
    # muv = jnp.where(is_two_phase, statev_actual["mu"], vapor_sat["viscosity"])
    # Gv = jnp.where(is_two_phase, statev_actual["gruneisen"], vapor_sat["gruneisen"])

    def _vapor_from_actual(_args):
        _hv, _p = _args
        st = fluidV.get_state(jxp.HmassP_INPUTS, _hv, _p)
        return (st["T"], st["s"], st["a"], st["mu"], st["gruneisen"], st["conductivity"])

    def _vapor_from_sat(_args):
        _hv, _p = _args  # unused, but signature must match
        return (vapor_sat["temperature"], vapor_sat["entropy"],
                vapor_sat["speed_of_sound"], vapor_sat["viscosity"],
                vapor_sat["gruneisen"], vapor_sat["conductivity"])

    Tv, sv, av, muv, Gv, k_v = jax.lax.cond(
        is_two_phase, _vapor_from_actual, _vapor_from_sat, (hv_safe, p_safe)
    )

    sl = statel["s"]
    al = statel["a"]
    mul = statel["mu"]
    Gl = statel["gruneisen"]

    # --- Mixture properties ---
    rho_mix = rhov * alphav_safe + rhol * alphal_safe
    mu_mix = muv * alphav_safe + mul * alphal_safe

    S = (rhov**2 * rhol**2 * u**3) ** (1 / 9)

    # --- Geometry ---
    A, dAdx, perimeter, radius = args.geometry(x)
    diameter = 2 * radius

    # --- Wall friction ---
    Re_mix = u * rho_mix * diameter / mu_mix
    f_D = get_friction_factor_haaland(Re_mix, eps_wall, diameter)
    tau_w_mix = get_wall_viscous_stress(f_D, rho_mix, u)
    tau_w_mix = tau_w_mix * wall_friction

    # --- Interfacial heat transfer ---
    # Use Tsat for the vapor-side interface temperature (interface is at saturation)
    Ai, d_v = interfacial_area(alphav_safe)

    Nu_v = 12
    k_l = statel["k"]
    ht_v = (k_v * Nu_v) / d_v
    # ht_l = (k_l * Nu_l) / (diameter * alphal)
    ht_l = ranz_marshall_htc(alphav_safe, alphal_safe, statel, None, fluidL)
    # ht_vl = 1.0 / (1.0 / ht_v + 1.0 / ht_l)
    # Qvl = ht_vl * Ai * (Tsat - Tl)

    # --- Heat/mass transfer via lax.cond (only evaluate in two-phase) ---
    # --- OLD (jnp.where — evaluates both branches) ---
    # Qvi = ht_v * Ai * (Tsat - Tv)
    # Qli = ht_l * Ai * (Tsat - Tl)
    # Qvi = jnp.where(is_two_phase, Qvi, 0.0)
    # Qli = jnp.where(is_two_phase, Qli, 0.0)
    # m_dot_lv = calculate_mass_transfer_lv(
    #     Tl, Tv, Tsat, ht_l, ht_v, Ai, h_sat_l, h_sat_v, hl, hv, Qvi, Qli,
    # )
    # m_dot_lv = jnp.where(is_two_phase, m_dot_lv, 0.0)

    def _source_terms_two_phase(_args):
        _Tl, _Tv, _Tsat, _ht_l, _ht_v, _Ai, _h_sat_l, _h_sat_v, _hl, _hv = _args
        _Qvi = _ht_v * _Ai * (_Tsat - _Tv)
        _Qli = _ht_l * _Ai * (_Tsat - _Tl)
        _m_dot_lv = calculate_mass_transfer_lv(
            _Tl, _Tv, _Tsat, _ht_l, _ht_v, _Ai, _h_sat_l, _h_sat_v, _hl, _hv, _Qvi, _Qli,
        )
        return _Qvi, _Qli, _m_dot_lv

    def _source_terms_single_phase(_args):
        return (0.0, 0.0, 0.0)

    Qvi, Qli, m_dot_lv = jax.lax.cond(
        is_two_phase,
        _source_terms_two_phase,
        _source_terms_single_phase,
        (Tl, Tv, Tsat, ht_l, ht_v, Ai, h_sat_l, h_sat_v, hl, hv),
    )

    # --- Speed of sound (always use mixture formula) ---
    a_mix = get_speed_of_sound_mixture(
        Gv, alphav_safe, av, rhov, Gl, alphal_safe, al, rhol
    )

    # --- Energy source terms (non-conservative total enthalpy form) ---


    # --- Build A matrix and b vector (no ramping, full two-phase rows always) ---
    # Variables: [dαv, dαl, dρv, dρl, du, dp, dhv, dhl]

    b4 = m_dot_lv * (h_sat_v - hv) + Qvi
    b5 = Qli - m_dot_lv * (h_sat_l - hl)

    A_mat = jnp.stack(
        [
            jnp.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([rhov * u, 0.0, alphav * u, 0.0, alphav * rhov, 0.0, 0.0, 0.0]),
            jnp.array([0.0, rhol * u, 0.0, alphal * u, alphal * rhol, 0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0, 0.0, rho_mix * u, 1.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0, 0.0, alphav * rhov * u**2, 0.0, alphav * rhov * u, 0.0]),
            jnp.array([0.0, 0.0, 0.0, 0.0, alphal * rhol * u**2, 0.0, 0.0, alphal * rhol * u]),
            jnp.array([0.0, 0.0, -1.0, 0.0, 0.0, (1.0 + Gv) / av**2, -(rhov * Gv) / av**2, 0.0]),
            jnp.array([0.0, 0.0, 0.0, -1.0, 0.0, (1.0 + Gl) / al**2, 0.0, -(rhol * Gl) / al**2]),
        ],
        axis=0,
    )

    b_vec = jnp.asarray(
        [
            0.0,
            m_dot_lv - (dAdx / A) * alphav * rhov * u,
            -m_dot_lv - (dAdx / A) * alphal * rhol * u,
            -(tau_w_mix * perimeter) / A,
            b4,
            b5,
            0.0,
            0.0,
        ]
    )

    A_mat = A_mat / S
    b_vec = b_vec / S

    # --- Determinants via Cramer's rule ---
    M1 = A_mat.at[:, 0].set(b_vec)
    M2 = A_mat.at[:, 1].set(b_vec)
    M3 = A_mat.at[:, 2].set(b_vec)
    M4 = A_mat.at[:, 3].set(b_vec)
    M5 = A_mat.at[:, 4].set(b_vec)
    M6 = A_mat.at[:, 5].set(b_vec)
    M7 = A_mat.at[:, 6].set(b_vec)
    M8 = A_mat.at[:, 7].set(b_vec)

    det_D  = det(A_mat)
    det_N1 = det(M1)
    det_N2 = det(M2)
    det_N3 = det(M3)
    det_N4 = det(M4)
    det_N5 = det(M5)
    det_N6 = det(M6)
    det_N7 = det(M7)
    det_N8 = det(M8)

    N = jnp.array([det_N1, det_N2, det_N3, det_N4, det_N5, det_N6, det_N7, det_N8])

    # --- Ghost phase: force vapor to follow saturation curve when single-phase ---
    # dhv/dτ = dh_sat_v/dp * dp/dτ,  dρv/dτ = dρ_sat_v/dp * dp/dτ
    # Use finite differences instead of jax.grad to avoid second-derivative issues
    # with implicit solvers (Jacobian of jax.grad needs d²/dp² of saturation table).
    # _dp = 1.0  # Pa perturbation
    # _sat_plus = fluidV.get_state_saturation(p + _dp)
    # _sat_minus = fluidV.get_state_saturation(p - _dp)
    # dh_sat_v_dp = (_sat_plus["enthalpy"] - _sat_minus["enthalpy"]) / (2.0 * _dp)
    # drho_sat_v_dp = (_sat_plus["density"] - _sat_minus["density"]) / (2.0 * _dp)

    # det_N3_sat = drho_sat_v_dp * det_N6  # dρv/dτ following saturation
    # det_N7_sat = dh_sat_v_dp * det_N6    # dhv/dτ following saturation

    # det_N3 = jnp.where(is_two_phase, det_N3, det_N3_sat)
    # det_N7 = jnp.where(is_two_phase, det_N7, det_N7_sat)

    rhs = jnp.array([
        det_N1 / det_D, det_N2 / det_D, det_N3 / det_D, det_N4 / det_D,
        det_N5 / det_D, det_N6 / det_D, det_N7 / det_D, det_N8 / det_D
    ])
    rhs_autonomous = jnp.array([
        det_D, det_N1, det_N2, det_N3, det_N4, det_N5, det_N6, det_N7, det_N8
    ])

    # jax.debug.print(
    #     "x={x:.4f} | amix={am:.1f} | av={av:.1f} | al={al:.1f} | Gv={Gv:.3f} | Gl={Gl:.3f} | Ma={Ma:.4f} | det={D:.3e} | αv={alv:.3e} | ρv={rv:.2f} | ρl={rl:.2f} | 2ph={tp} | b4={b4:.3e} | b5={b5:.3e} | hv={hv:.1f} | hl={hl:.1f} | mdot={md:.3e}",
    #     x=x, am=a_mix, av=av, al=al, Gv=Gv, Gl=Gl, Ma=u/a_mix,
    #     D=det_D, alv=alphav, rv=rhov, rl=rhol, tp=is_two_phase,
    #     b4=b4, b5=b5, hv=hv, hl=hl, md=m_dot_lv
    # )

    # --- Export ---
    out = {
        "x": x,
        "v": u,
        "alpha1": alphav,
        "alpha2": alphal,
        "rho1": rhov,
        "rho2": rhol,
        "p": p,
        "h1": hv,
        "h2": hl,
        "T1": Tv,
        "T2": Tl,
        "s1": sv,
        "s2": sl,
        "rhs": rhs,
        "rhs_autonomous": rhs_autonomous,
        "area": A,
        "dAdx": dAdx,
        "diameter": diameter,
        "perimeter": perimeter,
        "Ma1": u / av,
        "rho_mix": rho_mix,
        "a_mix": a_mix,
        "Ma2": u / al,
        "Ma_mix": u / a_mix,
        "Re": Re_mix,
        "f_D": f_D,
        "tau_w": tau_w_mix,
        "A_mat": A_mat,
        "b_vec": b_vec,
        "D": det_D,
        "N": N,
        "m_dot": u * rho_mix * A,
        "m_dot_lv": m_dot_lv,
        "Qvi": Qvi,
        "Qli": Qli,
        "Gv":Gv,
        "Gl":Gl,
    }

    return {**out, **statel.to_dict(include_aliases=True)}

def nozzle_homogeneous_non_equilibrium_one_component_autonomous_lax(tau, Y, args, use_lax_cond=True):
    """
    Autonomous formulation WITHOUT ramping or matrix modification.

    Ghost phase strategy: clamp alpha >= ALPHA_MIN, use actual (not saturation)
    properties for the ghost phase, and set heat/mass transfer to zero when
    the flow is effectively single-phase.

    State vector: Y = [x, alphav, alphal, rhov, rhol, u, p, hv, hl]

    Parameters
    ----------
    use_lax_cond : bool
        If True (default), use jax.lax.cond for two-phase switching (implicit
        solver compatible, but higher memory with vmap).
        If False, use jnp.where (vmap friendly, but NaN gradients with
        implicit solvers).
    """
    x, alphav, alphal, rhov, rhol, u, p, hv, hl = Y

    jax.debug.print(
        "RAW Y: x={:.4f} αv={:.3e} αl={:.3e} ρv={:.2f} ρl={:.2f} u={:.4f} p={:.1f} hv={:.1f} hl={:.1f}",
        x, alphav, alphal, rhov, rhol, u, p, hv, hl
    )

    # --- Geometry / model parameters ---
    fluidV = args.fluid1
    fluidL = args.fluid2
    L = args.length
    eps_wall = args.roughness
    wall_friction = args.wall_friction
    heat_transfer = args.heat_transfer

    # --- Clamp volume fractions ---
    ALPHA_MIN = 1e-6
    alphav_safe = jnp.clip(alphav, ALPHA_MIN, 1.0 - ALPHA_MIN)
    alphal_safe = jnp.clip(alphal, ALPHA_MIN, 1.0 - ALPHA_MIN)
        # Force pressures and enthalpies into the table's strictly valid ranges
    # p_safe = jnp.clip(p, fluidV.p_min, fluidV.p_max)
    # hv_safe = jnp.clip(hv, fluidV.h_min, fluidV.h_max)
    # hl_safe = jnp.clip(hl, fluidL.h_min, fluidL.h_max)
    p_safe = p
    hv_safe = hv
    hl_safe = hl


    p = jnp.clip(p, 1.0, 1e9)

    # --- Saturation properties (needed for mass transfer) ---
    liquid_sat = fluidL.get_state_saturation(p_safe)
    vapor_sat = fluidV.get_state_saturation(p_safe)

    h_sat_l = liquid_sat["enthalpy"]
    h_sat_v = vapor_sat["enthalpy"]
    Tsat = liquid_sat["temperature"]

    # --- Thermodynamic states ---
    statel = fluidL.get_state(jxp.HmassP_INPUTS, hl_safe, p_safe)
    Tl = statel["T"]

    # --- Two-phase flag based on enthalpy and temperature ---
    is_two_phase = (hl >= h_sat_l) & (Tl >= Tsat)

    # Vapor: real properties from get_state when two-phase, saturation when
    # single-phase (ghost phase has non-physical hv).
    # Use lax.cond to only trace the taken branch (implicit solver compatible).
    # --- OLD (jnp.where — evaluates both branches, NaN in Jacobian) ---
    # hv_safe = jnp.clip(hv, fluidV.h_min, fluidV.h_max)
    # statev_actual = fluidV.get_state(jxp.HmassP_INPUTS, hv_safe, p)
    # Tv = jnp.where(is_two_phase, statev_actual["T"], vapor_sat["temperature"])
    # sv = jnp.where(is_two_phase, statev_actual["s"], vapor_sat["entropy"])
    # av = jnp.where(is_two_phase, statev_actual["a"], vapor_sat["speed_of_sound"])
    # muv = jnp.where(is_two_phase, statev_actual["mu"], vapor_sat["viscosity"])
    # Gv = jnp.where(is_two_phase, statev_actual["gruneisen"], vapor_sat["gruneisen"])

    def _vapor_from_actual(_args):
        _hv, _p = _args
        st = fluidV.get_state(jxp.HmassP_INPUTS, _hv, _p)
        return (st["T"], st["s"], st["a"], st["mu"], st["gruneisen"], st["conductivity"])

    def _vapor_from_sat(_args):
        _hv, _p = _args  # unused, but signature must match
        return (vapor_sat["temperature"], vapor_sat["entropy"],
                vapor_sat["speed_of_sound"], vapor_sat["viscosity"],
                vapor_sat["gruneisen"], vapor_sat["conductivity"])

    Tv, sv, av, muv, Gv, k_v = jax.lax.cond(
        is_two_phase, _vapor_from_actual, _vapor_from_sat, (hv_safe, p_safe)
    )

    sl = statel["s"]
    al = statel["a"]
    mul = statel["mu"]
    Gl = statel["gruneisen"]

    # --- Mixture properties ---
    rho_mix = rhov * alphav_safe + rhol * alphal_safe
    mu_mix = muv * alphav_safe + mul * alphal_safe

    S = (rhov**2 * rhol**2 * u**3) ** (1 / 8)

    # --- Geometry ---
    A, dAdx, perimeter, radius = args.geometry(x)
    diameter = 2 * radius

    # --- Wall friction ---
    Re_mix = u * rho_mix * diameter / mu_mix
    f_D = get_friction_factor_haaland(Re_mix, eps_wall, diameter)
    tau_w_mix = get_wall_viscous_stress(f_D, rho_mix, u)
    tau_w_mix = tau_w_mix * wall_friction

    # --- Interfacial heat transfer ---
    # Use Tsat for the vapor-side interface temperature (interface is at saturation)
    Ai, d_v = interfacial_area(alphav_safe)

    Nu_v = 12
    k_l = statel["k"]
    ht_v = (k_v * Nu_v) / d_v
    # ht_l = (k_l * Nu_l) / (diameter * alphal)
    ht_l = ranz_marshall_htc(alphav_safe, alphal_safe, statel, None, fluidL)
    # ht_vl = 1.0 / (1.0 / ht_v + 1.0 / ht_l)
    # Qvl = ht_vl * Ai * (Tsat - Tl)

    # --- Heat/mass transfer via lax.cond (only evaluate in two-phase) ---
    # --- OLD (jnp.where — evaluates both branches) ---
    # Qvi = ht_v * Ai * (Tsat - Tv)
    # Qli = ht_l * Ai * (Tsat - Tl)
    # Qvi = jnp.where(is_two_phase, Qvi, 0.0)
    # Qli = jnp.where(is_two_phase, Qli, 0.0)
    # m_dot_lv = calculate_mass_transfer_lv(
    #     Tl, Tv, Tsat, ht_l, ht_v, Ai, h_sat_l, h_sat_v, hl, hv, Qvi, Qli,
    # )
    # m_dot_lv = jnp.where(is_two_phase, m_dot_lv, 0.0)

    def _source_terms_two_phase(_args):
        _Tl, _Tv, _Tsat, _ht_l, _ht_v, _Ai, _h_sat_l, _h_sat_v, _hl, _hv = _args
        _Qvi = _ht_v * _Ai * (_Tsat - _Tv)
        _Qli = _ht_l * _Ai * (_Tsat - _Tl)
        _m_dot_lv = calculate_mass_transfer_lv(
            _Tl, _Tv, _Tsat, _ht_l, _ht_v, _Ai, _h_sat_l, _h_sat_v, _hl, _hv, _Qvi, _Qli,
        )
        return _Qvi, _Qli, _m_dot_lv

    def _source_terms_single_phase(_args):
        return (0.0, 0.0, 0.0)

    Qvi, Qli, m_dot_lv = jax.lax.cond(
        is_two_phase,
        _source_terms_two_phase,
        _source_terms_single_phase,
        (Tl, Tv, Tsat, ht_l, ht_v, Ai, h_sat_l, h_sat_v, hl, hv),
    )

    # --- Speed of sound (always use mixture formula) ---
    a_mix = get_speed_of_sound_mixture(
        Gv, alphav_safe, av, rhov, Gl, alphal_safe, al, rhol
    )

    # --- Energy source terms (non-conservative total enthalpy form) ---


    # --- Build A matrix and b vector (no ramping, full two-phase rows always) ---
    # Variables: [dαv, dαl, dρv, dρl, du, dp, dhv, dhl]

    b4 = m_dot_lv * (h_sat_v - hv) + Qvi
    b5 = Qli - m_dot_lv * (h_sat_l - hl)

    A_mat = jnp.stack(
        [
            jnp.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            jnp.array([rhov * u, 0.0, alphav * u, 0.0, alphav * rhov, 0.0, 0.0, 0.0]),
            jnp.array([0.0, rhol * u, 0.0, alphal * u, alphal * rhol, 0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0, 0.0, rho_mix * u, 1.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0, 0.0, alphav * rhov * u**2, 0.0, alphav * rhov * u, 0.0]),
            jnp.array([0.0, 0.0, 0.0, 0.0, alphal * rhol * u**2, 0.0, 0.0, alphal * rhol * u]),
            jnp.array([0.0, 0.0, -1.0, 0.0, 0.0, (1.0 + Gv) / av**2, -(rhov * Gv) / av**2, 0.0]),
            jnp.array([0.0, 0.0, 0.0, -1.0, 0.0, (1.0 + Gl) / al**2, 0.0, -(rhol * Gl) / al**2]),
        ],
        axis=0,
    )

    b_vec = jnp.asarray(
        [
            0.0,
            m_dot_lv - (dAdx / A) * alphav * rhov * u,
            -m_dot_lv - (dAdx / A) * alphal * rhol * u,
            -(tau_w_mix * perimeter) / A,
            b4,
            b5,
            0.0,
            0.0,
        ]
    )

    A_mat = A_mat / S
    b_vec = b_vec / S

    # --- Determinants via Cramer's rule ---
    M1 = A_mat.at[:, 0].set(b_vec)
    M2 = A_mat.at[:, 1].set(b_vec)
    M3 = A_mat.at[:, 2].set(b_vec)
    M4 = A_mat.at[:, 3].set(b_vec)
    M5 = A_mat.at[:, 4].set(b_vec)
    M6 = A_mat.at[:, 5].set(b_vec)
    M7 = A_mat.at[:, 6].set(b_vec)
    M8 = A_mat.at[:, 7].set(b_vec)

    det_D  = det(A_mat)
    det_N1 = det(M1)
    det_N2 = det(M2)
    det_N3 = det(M3)
    det_N4 = det(M4)
    det_N5 = det(M5)
    det_N6 = det(M6)
    det_N7 = det(M7)
    det_N8 = det(M8)

    N = jnp.array([det_N1, det_N2, det_N3, det_N4, det_N5, det_N6, det_N7, det_N8])

    # --- Ghost phase: force vapor to follow saturation curve when single-phase ---
    # dhv/dτ = dh_sat_v/dp * dp/dτ,  dρv/dτ = dρ_sat_v/dp * dp/dτ
    # Use finite differences instead of jax.grad to avoid second-derivative issues
    # with implicit solvers (Jacobian of jax.grad needs d²/dp² of saturation table).
    # _dp = 1.0  # Pa perturbation
    # _sat_plus = fluidV.get_state_saturation(p + _dp)
    # _sat_minus = fluidV.get_state_saturation(p - _dp)
    # dh_sat_v_dp = (_sat_plus["enthalpy"] - _sat_minus["enthalpy"]) / (2.0 * _dp)
    # drho_sat_v_dp = (_sat_plus["density"] - _sat_minus["density"]) / (2.0 * _dp)

    # det_N3_sat = drho_sat_v_dp * det_N6  # dρv/dτ following saturation
    # det_N7_sat = dh_sat_v_dp * det_N6    # dhv/dτ following saturation

    # det_N3 = jnp.where(is_two_phase, det_N3, det_N3_sat)
    # det_N7 = jnp.where(is_two_phase, det_N7, det_N7_sat)

    rhs = jnp.array([
        det_N1 / det_D, det_N2 / det_D, det_N3 / det_D, det_N4 / det_D,
        det_N5 / det_D, det_N6 / det_D, det_N7 / det_D, det_N8 / det_D
    ])
    rhs_autonomous = jnp.array([
        det_D, det_N1, det_N2, det_N3, det_N4, det_N5, det_N6, det_N7, det_N8
    ])

    # jax.debug.print(
    #     "x={x:.4f} | amix={am:.1f} | av={av:.1f} | al={al:.1f} | Gv={Gv:.3f} | Gl={Gl:.3f} | Ma={Ma:.4f} | det={D:.3e} | αv={alv:.3e} | ρv={rv:.2f} | ρl={rl:.2f} | 2ph={tp} | b4={b4:.3e} | b5={b5:.3e} | hv={hv:.1f} | hl={hl:.1f} | mdot={md:.3e}",
    #     x=x, am=a_mix, av=av, al=al, Gv=Gv, Gl=Gl, Ma=u/a_mix,
    #     D=det_D, alv=alphav, rv=rhov, rl=rhol, tp=is_two_phase,
    #     b4=b4, b5=b5, hv=hv, hl=hl, md=m_dot_lv
    # )

    # --- Export ---
    out = {
        "x": x,
        "v": u,
        "alpha1": alphav,
        "alpha2": alphal,
        "rho1": rhov,
        "rho2": rhol,
        "p": p,
        "h1": hv,
        "h2": hl,
        "T1": Tv,
        "T2": Tl,
        "s1": sv,
        "s2": sl,
        "rhs": rhs,
        "rhs_autonomous": rhs_autonomous,
        "area": A,
        "dAdx": dAdx,
        "diameter": diameter,
        "perimeter": perimeter,
        "Ma1": u / av,
        "rho_mix": rho_mix,
        "a_mix": a_mix,
        "Ma2": u / al,
        "Ma_mix": u / a_mix,
        "Re": Re_mix,
        "f_D": f_D,
        "tau_w": tau_w_mix,
        "A_mat": A_mat,
        "b_vec": b_vec,
        "D": det_D,
        "N": N,
        "m_dot": u * rho_mix * A,
        "m_dot_lv": m_dot_lv,
        "Qvi": Qvi,
        "Qli": Qli,
        "Gv":Gv,
        "Gl":Gl,
    }

    return {**out, **statel.to_dict(include_aliases=True)}

@eqx.filter_jit
def nozzle_homogeneous_non_equilibrium_one_component_autonomous_lax_eigth(tau, Y, args, use_lax_cond=True):
    """
    Autonomous formulation WITHOUT ramping or matrix modification.

    Ghost phase strategy: clamp alpha >= ALPHA_MIN, use actual (not saturation)
    properties for the ghost phase, and set heat/mass transfer to zero when
    the flow is effectively single-phase.

    State vector: Y = [x, alphav, rhov, rhol, u, p, hv, hl]

    Parameters
    ----------
    use_lax_cond : bool
        If True (default), use jax.lax.cond for two-phase switching (implicit
        solver compatible, but higher memory with vmap).
        If False, use jnp.where (vmap friendly, but NaN gradients with
        implicit solvers).
    """
    x, alphav, rhov, rhol, u, p, hv, hl = Y

    # jax.debug.print(
    #     "RAW Y: x={:.4f} αv={:.3e} ρv={:.2f} ρl={:.2f} u={:.4f} p={:.1f} hv={:.1f} hl={:.1f}",
    #     x, alphav, rhov, rhol, u, p, hv, hl
    # )

    # --- Geometry / model parameters ---
    fluidV = args.fluid1
    fluidL = args.fluid2
    L = args.length
    eps_wall = args.roughness
    wall_friction = args.wall_friction
    heat_transfer = args.heat_transfer
    Nb = args.bubble_number
    Nd = args.droplet_number
    alpha_b = args.alpha_b
    alpha_d = args.alpha_d

    # --- Clamp volume fractions ---
    ALPHA_MIN = 1e-6
    alphal = 1.0 - alphav
    alphav_safe = jnp.clip(alphav, ALPHA_MIN, 1.0 - ALPHA_MIN)
    alphal_safe = jnp.clip(alphal, ALPHA_MIN, 1.0 - ALPHA_MIN)
        # Force pressures and enthalpies into the table's strictly valid ranges
    # p_safe = jnp.clip(p, fluidV.p_min, fluidV.p_max)
    # hv_safe = jnp.clip(hv, fluidV.h_min, fluidV.h_max)
    # hl_safe = jnp.clip(hl, fluidL.h_min, fluidL.h_max)

    p_safe = p
    hv_safe = hv
    hl_safe = hl


    p = jnp.clip(p, 1.0, 1e9)

    # --- Saturation properties (needed for mass transfer) ---
    liquid_sat = fluidL.get_state_saturation(p_safe)
    vapor_sat = fluidV.get_state_saturation(p_safe)

    h_sat_l = liquid_sat["enthalpy"]
    h_sat_v = vapor_sat["enthalpy"]
    Tsat = liquid_sat["temperature"]

    # --- Thermodynamic states ---
    statel = fluidL.get_state(jxp.HmassP_INPUTS, hl_safe, p_safe)
    Tl = statel["T"]

    # --- Two-phase flag based on enthalpy and temperature ---
    is_two_phase = (hl >= h_sat_l) & (Tl >= Tsat)

    def _vapor_from_actual(_args):
        _hv, _p = _args
        st = fluidV.get_state(jxp.HmassP_INPUTS, _hv, _p)
        return (st["T"], st["s"], st["a"], st["mu"], st["gruneisen"], st["conductivity"])

    def _vapor_from_sat(_args):
        _hv, _p = _args  # unused, but signature must match
        return (vapor_sat["temperature"], vapor_sat["entropy"],
                vapor_sat["speed_of_sound"], vapor_sat["viscosity"],
                vapor_sat["gruneisen"], vapor_sat["conductivity"])

    Tv, sv, av, muv, Gv, k_v = jax.lax.cond(
        is_two_phase, _vapor_from_actual, _vapor_from_sat, (hv_safe, p_safe)
    )

    sl = statel["s"]
    al = statel["a"]
    mul = statel["mu"]
    Gl = statel["gruneisen"]

    # --- Mixture properties ---
    rho_mix = rhov * alphav_safe + rhol * alphal_safe
    mu_mix = muv * alphav_safe + mul * alphal_safe

    S = (rhov * rhol * u**3) ** (1 / 8)

    # --- Geometry ---
    A, dAdx, perimeter, radius = args.geometry(x)
    diameter = 2 * radius

    # --- Wall friction ---
    Re_mix = u * rho_mix * diameter / mu_mix
    f_D = get_friction_factor_haaland(Re_mix, eps_wall, diameter)
    tau_w_mix = get_wall_viscous_stress(f_D, rho_mix, u)
    tau_w_mix = tau_w_mix * wall_friction

    # --- Interfacial heat transfer ---
    # Use Tsat for the vapor-side interface temperature (interface is at saturation)

    # Critical nucleus radius from classical nucleation theory:
    #   r* = 2σ·T_sat / (h_lv·ρ_sat_v·ΔT),  ΔT = T_l - T_sat (liquid superheat)
    # Bubbles smaller than r* collapse → use 2·r* as the minimum bubble diameter.
    sigma_i = surface_tension(fluidL.fluid_name, Tl)
    h_lv = h_sat_v - h_sat_l
    rho_sat_v = vapor_sat["density"]
    DeltaT = jnp.maximum(Tl - Tsat, 1e-10)
    r_crit = 2.0 * sigma_i * Tsat / (h_lv * rho_sat_v * DeltaT)
    d_min_crit = 2.0 * r_crit
    d_min_crit = 1e-5
    Ai, d_v = interfacial_area(alphav_safe, alpha_b=alpha_b, alpha_d=alpha_d, Nb=Nb, Nd=Nd, d_min_b=d_min_crit)

    Nu_v = 12
    k_l = statel["k"]
    ht_v = (k_v * Nu_v) / d_v
    # ht_l = (k_l * Nu_l) / (diameter * alphal)
    ht_l = ranz_marshall_htc(alphav_safe, alphal_safe, statel, None, fluidL, Nb=Nb, Nd=Nd, alpha_b=alpha_b, alpha_d=alpha_d)
    # ht_vl = 1.0 / (1.0 / ht_v + 1.0 / ht_l)
    # Qvl = ht_vl * Ai * (Tsat - Tl)

    # --- Heat/mass transfer via lax.cond (only evaluate in two-phase) ---
    # --- OLD (jnp.where — evaluates both branches) ---
    # Qvi = ht_v * Ai * (Tsat - Tv)
    # Qli = ht_l * Ai * (Tsat - Tl)
    # Qvi = jnp.where(is_two_phase, Qvi, 0.0)
    # Qli = jnp.where(is_two_phase, Qli, 0.0)
    # m_dot_lv = calculate_mass_transfer_lv(
    #     Tl, Tv, Tsat, ht_l, ht_v, Ai, h_sat_l, h_sat_v, hl, hv, Qvi, Qli,
    # )
    # m_dot_lv = jnp.where(is_two_phase, m_dot_lv, 0.0)

    def _source_terms_two_phase(_args):
        _Tl, _Tv, _Tsat, _ht_l, _ht_v, _Ai, _h_sat_l, _h_sat_v, _hl, _hv = _args
        _Qvi = _ht_v * _Ai * (_Tsat - _Tv)
        _Qli = _ht_l * _Ai * (_Tsat - _Tl)
        _m_dot_lv = calculate_mass_transfer_lv(
            _Tl, _Tv, _Tsat, _ht_l, _ht_v, _Ai, _h_sat_l, _h_sat_v, _hl, _hv, _Qvi, _Qli,
        )
        return _Qvi, _Qli, _m_dot_lv

    def _source_terms_single_phase(_args):
        return (0.0, 0.0, 0.0)

    Qvi, Qli, m_dot_lv = jax.lax.cond(
        is_two_phase,
        _source_terms_two_phase,
        _source_terms_single_phase,
        (Tl, Tv, Tsat, ht_l, ht_v, Ai, h_sat_l, h_sat_v, hl, hv),
    )

    # jax.debug.print(
    #     "x={x:.4f} | is_2ph={tp} | Tl={Tl:.4f} Tsat={Ts:.4f} hl={hl:.2f} hsl={hsl:.2f} | "
    #     "Qvi={Qvi:.3e} Qli={Qli:.3e} mdot={m:.3e} | Ai={Ai:.3e} d={d:.3e} | alphav={av:.3e}",
    #     x=x, tp=is_two_phase, Tl=Tl, Ts=Tsat, hl=hl, hsl=h_sat_l,
    #     Qvi=Qvi, Qli=Qli, m=m_dot_lv, Ai=Ai, d=d_v, av=alphav,
    # )  # DEBUG: remove after investigation

    # --- Speed of sound (always use mixture formula) ---
    a_mix = get_speed_of_sound_mixture(
        Gv, alphav_safe, av, rhov, Gl, alphal_safe, al, rhol
    )

    # --- Energy source terms (non-conservative total enthalpy form) ---


    # --- Build A matrix and b vector (no ramping, full two-phase rows always) ---
    # Variables: [dαv, dαl, dρv, dρl, du, dp, dhv, dhl]

    b4 = m_dot_lv * (h_sat_v - hv) + Qvi
    b5 = Qli - m_dot_lv * (h_sat_l - hl)

    A_mat = jnp.stack(
        [
            jnp.array([rhov * u, alphav * u, 0.0, alphav * rhov, 0.0, 0.0, 0.0]),
            jnp.array([- rhol * u, 0.0, alphal * u, alphal * rhol, 0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0, rho_mix * u, 1.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0, alphav * rhov * u**2, 0.0, alphav * rhov * u, 0.0]),
            jnp.array([0.0, 0.0, 0.0, alphal * rhol * u**2, 0.0, 0.0, alphal * rhol * u]),
            jnp.array([0.0, -1.0, 0.0, 0.0, (1.0 + Gv) / av**2, -(rhov * Gv) / av**2, 0.0]),
            jnp.array([0.0, 0.0, -1.0, 0.0, (1.0 + Gl) / al**2, 0.0, -(rhol * Gl) / al**2]),
        ],
        axis=0,
    )

    b_vec = jnp.asarray(
        [
            m_dot_lv - (dAdx / A) * alphav * rhov * u,
            -m_dot_lv - (dAdx / A) * alphal * rhol * u,
            -(tau_w_mix * perimeter) / A,
            b4,
            b5,
            0.0,
            0.0,
        ]
    )

    # Rows: [MassV, MassL, MixMomentum, V-Energy, L-Energy, V-EOS, L-EOS]
    row_scales = jnp.array([
            jnp.abs(rhov) * alphav,                  
            jnp.abs(rhol) * alphal,                  
            jnp.abs(rho_mix * u + p_safe),   
            jnp.abs(rhov * u * hv_safe) * alphav,        
            jnp.abs(rhol * u * hl_safe) * alphal,        
            1.0,                                      
            1.0                                       
        ])
    
    # Scale both the matrix and the RHS vector
    A_mat_scaled = A_mat / row_scales[:, None]
    b_vec_scaled = b_vec / row_scales

    A_mat = A_mat / S
    b_vec = b_vec / S

    # --- Determinants via Cramer's rule ---
    M1 = A_mat.at[:, 0].set(b_vec)
    M2 = A_mat.at[:, 1].set(b_vec)
    M3 = A_mat.at[:, 2].set(b_vec)
    M4 = A_mat.at[:, 3].set(b_vec)
    M5 = A_mat.at[:, 4].set(b_vec)
    M6 = A_mat.at[:, 5].set(b_vec)
    M7 = A_mat.at[:, 6].set(b_vec)
    # M1 = A_mat_scaled.at[:, 0].set(b_vec_scaled)
    # M2 = A_mat_scaled.at[:, 1].set(b_vec_scaled)
    # M3 = A_mat_scaled.at[:, 2].set(b_vec_scaled)
    # M4 = A_mat_scaled.at[:, 3].set(b_vec_scaled)
    # M5 = A_mat_scaled.at[:, 4].set(b_vec_scaled)
    # M6 = A_mat_scaled.at[:, 5].set(b_vec_scaled)
    # M7 = A_mat_scaled.at[:, 6].set(b_vec_scaled)

    det_D  = det(A_mat)
    # det_D = det(A_mat_scaled)
    det_N1 = det(M1)
    det_N2 = det(M2)
    det_N3 = det(M3)
    det_N4 = det(M4)
    det_N5 = det(M5)
    det_N6 = det(M6)
    det_N7 = det(M7)

    N = jnp.array([det_N1, det_N2, det_N3, det_N4, det_N5, det_N6, det_N7,])

    # --- Ghost phase: force vapor to follow saturation curve when single-phase ---
    # dhv/dτ = dh_sat_v/dp * dp/dτ,  dρv/dτ = dρ_sat_v/dp * dp/dτ
    # Use finite differences instead of jax.grad to avoid second-derivative issues
    # with implicit solvers (Jacobian of jax.grad needs d²/dp² of saturation table).
    # Use lax.cond so that only the active branch is differentiated (implicit-safe).
    _dp = 1.0  # Pa perturbation
    _sat_plus = fluidV.get_state_saturation(p + _dp)
    _sat_minus = fluidV.get_state_saturation(p - _dp)
    dh_sat_v_dp = (_sat_plus["enthalpy"] - _sat_minus["enthalpy"]) / (2.0 * _dp)
    drho_sat_v_dp = (_sat_plus["density"] - _sat_minus["density"]) / (2.0 * _dp)

    det_N3_sat = drho_sat_v_dp * det_N6  # dρv/dτ following saturation
    det_N7_sat = dh_sat_v_dp * det_N6    # dhv/dτ following saturation

    def _ghost_two_phase(_args):
        _det_N3, _det_N7, _det_N3_sat, _det_N7_sat = _args
        return _det_N3, _det_N7

    def _ghost_single_phase(_args):
        _det_N3, _det_N7, _det_N3_sat, _det_N7_sat = _args
        return _det_N3_sat, _det_N7_sat

    det_N3, det_N7 = jax.lax.cond(
        is_two_phase,
        _ghost_two_phase,
        _ghost_single_phase,
        (det_N3, det_N7, det_N3_sat, det_N7_sat),
    )

    rhs = jnp.array([
        det_N1 / det_D, det_N2 / det_D, det_N3 / det_D, det_N4 / det_D,
        det_N5 / det_D, det_N6 / det_D, det_N7 / det_D,
    ])
    rhs_autonomous = jnp.array([
        det_D, det_N1, det_N2, det_N3, det_N4, det_N5, det_N6, det_N7,
    ])


    # jax.debug.print(
    #     "x={x:.4f} | amix={am:.1f} | av={av:.1f} | al={al:.1f} | Gv={Gv:.3f} | Gl={Gl:.3f} | Ma={Ma:.4f} | det={D:.3e} | αv={alv:.3e} | ρv={rv:.2f} | ρl={rl:.2f} | 2ph={tp} | b4={b4:.3e} | b5={b5:.3e} | hv={hv:.1f} | hl={hl:.1f} | mdot={md:.3e}",
    #     x=x, am=a_mix, av=av, al=al, Gv=Gv, Gl=Gl, Ma=u/a_mix,
    #     D=det_D, alv=alphav, rv=rhov, rl=rhol, tp=is_two_phase,
    #     b4=b4, b5=b5, hv=hv, hl=hl, md=m_dot_lv
    # )

    # --- Export ---
    out = {
        "x": x,
        "v": u,
        "alpha1": alphav,
        "alpha2": alphal,
        "rho1": rhov,
        "rho2": rhol,
        "p": p,
        "h1": hv,
        "h2": hl,
        "T1": Tv,
        "T2": Tl,
        "s1": sv,
        "s2": sl,
        "rhs": rhs,
        "rhs_autonomous": rhs_autonomous,
        "area": A,
        "dAdx": dAdx,
        "diameter": diameter,
        "perimeter": perimeter,
        "Ma1": u / av,
        "rho_mix": rho_mix,
        "a_mix": a_mix,
        "Ma2": u / al,
        "Ma_mix": u / a_mix,
        "Re": Re_mix,
        "f_D": f_D,
        "tau_w": tau_w_mix,
        "A_mat": A_mat,
        "b_vec": b_vec,
        "D": det_D,
        "N": N,
        "m_dot": u * rho_mix * A,
        "m_dot_lv": m_dot_lv,
        "Qvi": Qvi,
        "Qli": Qli,
        "Gv":Gv,
        "Gl":Gl,
    }

    return {**out, **statel.to_dict(include_aliases=True)}





# -----------------------------------------------------------------------------
# Functions to calculate heat transfer and friction
# -----------------------------------------------------------------------------


def get_friction_factor_haaland(Reynolds, roughness, diameter):
    """
    Computes the Darcy-Weisbach friction factor using the Haaland equation.

    The Haaland equation provides an explicit formulation for the friction factor
    that is simpler to use than the Colebrook equation, with an acceptable level
    of accuracy for most engineering applications.
    This function implements the Haaland equation as it is presented in many fluid
    mechanics textbooks, such as "Fluid Mechanics Fundamentals and Applications"
    by Cengel and Cimbala (equation 12-93).

    Parameters
    ----------
    reynolds : float
        The Reynolds number, dimensionless, indicating the flow regime.
    roughness : float
        The absolute roughness of the pipe's internal surface (m).
    diameter : float
        The inner diameter of the pipe (m).

    Returns
    -------
    f : float
        The computed friction factor, dimensionless.
    """
    Re_safe = jnp.maximum(Reynolds, 1.0)
    term = 6.9 / Re_safe + (roughness / diameter / 3.7) ** 1.11
    f = (-1.8 * jnp.log10(term)) ** -2
    return f


def get_wall_viscous_stress(darcy_friction_factor, density, velocity):
    """Wall shear stress from Darcy-Weisbach friction factor.

    Parameters
    ----------
    darcy_friction_factor : float
        Darcy-Weisbach friction factor (dimensionless).
    density : float
        Fluid density (kg/m^3).
    velocity : float
        Fluid velocity (m/s).

    Returns
    -------
    float
        Wall shear stress (Pa).
    """
    return 0.125 * darcy_friction_factor * density * velocity**2

def surface_tension(fluid_name, T):
    """
    Surface tension [N/m] from Mulero et al. (JPCRD 2012) correlations.

    Supported fluids: water, nitrogen, CO2.
    Uses the standard form: sigma = sum_i( sigma_i * (1 - T/Tc)^n_i )

    Parameters
    ----------
    fluid_name : str
        CoolProp-style fluid name (case-insensitive).
    T : scalar
        Temperature [K].
    """
    name = fluid_name.lower()
    if name == "water":
        # Water: Tc = 647.096 K
        # σ = 0.2358 (1 - T/Tc)^1.256 [1 - 0.625 (1 - T/Tc)]
        theta = jnp.clip(1.0 - T / 647.096, 0.0, 1.0)
        sigma = 0.2358 * theta**1.256 * (1.0 - 0.625 * theta)
    elif name == "nitrogen":
        # Nitrogen: Tc = 126.192 K
        # σ = 0.02898 (1 - T/Tc)^1.2457
        theta = jnp.clip(1.0 - T / 126.192, 0.0, 1.0)
        sigma = 0.02898 * theta**1.2457
    elif name in ("co2", "carbondioxide", "carbon dioxide"):
        # CO2: Tc = 304.128 K
        # σ = 0.07863 (1 - T/Tc)^1.2845
        theta = jnp.clip(1.0 - T / 304.128, 0.0, 1.0)
        sigma = 0.07863 * theta**1.2845
    else:
        raise ValueError(f"No surface tension correlation for fluid '{fluid_name}'. "
                         f"Supported: water, nitrogen, CO2.")
    return jnp.maximum(sigma, 1e-10)


def ranz_marshall_htc(
        alpha_vapor,
        alpha_liquid,
        state_continuous,
        state_dispersed,  # reserved — not needed for liquid-side HTC
        fluid_liquid,
        Nb,
        Nd=1e8,
        alpha_b=0.3,
        alpha_d=0.7,
        We=8.0,
):
    """
    Liquid-side HTC for vapor bubbles dispersed in liquid (Ranz-Marshall 1952).

    Caller convention (vapor-in-liquid regime):
        state_continuous = statel  (liquid — the continuous phase)
        fluid_liquid     = liquid fluid object (needs fluid_name for surface tension)

    The bubble diameter comes from interfacial_area(alpha_continuous).
    The slip velocity is estimated from the critical Weber number (bubble breakup).
    All thermophysical properties are taken from the CONTINUOUS phase (liquid).
    """
    # --- Continuous-phase (liquid) properties ---
    rho_c = state_continuous["rho"]    # kg/m³
    mu_c  = state_continuous["mu"]     # Pa·s
    k_c   = state_continuous["k"]      # W/(m·K)
    cp_c  = state_continuous["cp"]     # J/(kg·K)
    T_c   = state_continuous["T"]      # K

    # --- Bubble diameter from interfacial-area model ---
    _, d_bubble = interfacial_area(alpha_vapor, alpha_b=alpha_b, alpha_d=alpha_d, Nb=Nb, Nd=Nd)

    # --- Surface tension from fluid-specific correlation ---
    sigma = surface_tension(fluid_liquid.fluid_name, T_c)

    # --- Slip velocity from critical Weber number (bubble-breakup criterion) ---
    # We = rho_continuous * u_slip^2 * d / sigma  →  u_slip = sqrt(We*sigma / (rho_c*d))
    u_slip = jnp.sqrt(We * sigma / (rho_c * d_bubble))

    # --- Ranz-Marshall: Re and Pr use continuous-phase (liquid) properties ---
    Re = rho_c * u_slip * d_bubble / mu_c
    Pr = cp_c * mu_c / k_c

    Nu = 2.0 + 0.6 * Re**0.5 * Pr**(1.0 / 3.0)

    htc_l = k_c * Nu / d_bubble

    return htc_l

def ranz_marshall_htc_two_components(
        alpha_vapor,
        alpha_liquid,
        state_continuous,
        state_dispersed,  # reserved — not needed for liquid-side HTC
        fluid_continuous,
        Nb,
        Nd=1e8,
        alpha_b=0.3,
        alpha_d=0.7,
        We=8.0,
):
    """
    Liquid-side HTC for vapor bubbles dispersed in liquid (Ranz-Marshall 1952).

    Caller convention (vapor-in-liquid regime):
        state_continuous  = statel  (liquid — the continuous phase)
        fluid_continuous  = liquid fluid object (needs fluid_name for surface tension)

    The bubble diameter comes from interfacial_area(alpha_continuous).
    The slip velocity is estimated from the critical Weber number (bubble breakup).
    All thermophysical properties are taken from the CONTINUOUS phase (liquid).
    """
    # --- Continuous-phase (liquid) properties ---
    rho_c = state_continuous["rho"]    # kg/m³
    mu_c  = state_continuous["mu"]     # Pa·s
    k_c   = state_continuous["k"]      # W/(m·K)
    cp_c  = state_continuous["cp"]     # J/(kg·K)
    T_c   = state_continuous["T"]      # K

    # --- Bubble diameter from interfacial-area model ---
    _, d_bubble = interfacial_area(alpha_vapor, alpha_b=alpha_b, alpha_d=alpha_d, Nb=Nb, Nd=Nd)

    # --- Surface tension from fluid-specific correlation ---
    sigma = surface_tension(fluid_continuous.fluid_name, T_c)

    # --- Slip velocity from critical Weber number (bubble-breakup criterion) ---
    # We = rho_continuous * u_slip^2 * d / sigma  →  u_slip = sqrt(We*sigma / (rho_c*d))
    u_slip = jnp.sqrt(We * sigma / (rho_c * d_bubble))

    # --- Ranz-Marshall: Re and Pr use continuous-phase (liquid) properties ---
    Re = rho_c * u_slip * d_bubble / mu_c
    Pr = cp_c * mu_c / k_c

    Nu = 2.0 + 0.6 * Re**0.5 * Pr**(1.0 / 3.0)

    htc_l = k_c * Nu / d_bubble

    return htc_l

def interfacial_area_old(alpha, alpha_b=0.3, alpha_d=0.7, Nb=1e10, Nd=1e8):
    """
    Compute Ai and equivalent spherical diameter based on alpha.
    Returns (Ai, d), where d is the equivalent spherical diameter.
    """

    # --- Bubbly regime ---
    if alpha <= alpha_b:
        A_ib = (6 * alpha)**(2/3) * (jnp.pi * Nb)**(1/3)
        Ai = A_ib
        N = Nb

    # --- Droplet regime ---
    elif alpha >= alpha_d:
        A_id = (6 * (1 - alpha))**(2/3) * (jnp.pi * Nd)**(1/3)
        Ai = A_id
        N = Nd

    # --- Transition regime (linear interpolation) ---
    else:
        A_ib_b = (6 * alpha_b)**(2/3) * (jnp.pi * Nb)**(1/3)
        A_id_d = (6 * (1 - alpha_d))**(2/3) * (jnp.pi * Nd)**(1/3)
        Ai = A_ib_b + (A_id_d - A_ib_b) * (alpha - alpha_b) / (alpha_d - alpha_b)
        
        # Smooth transition in N as well (optional)
        N = Nb + (Nd - Nb) * (alpha - alpha_b) / (alpha_d - alpha_b)

    # --- Equivalent spherical diameter ---
    d = jnp.sqrt(Ai / (N * jnp.pi))
    
    return Ai, d

def interfacial_area_old(alpha, alpha_b=0.3, alpha_d=0.7, Nb=1e13, Nd=1e8):
    """
    alpha is the volume fraction of the vapor. 
    if less than 0.3 -> bubbly flow
    if more than 0.7 -> droplet flow
    in between, transition between the two regimes

    JAX-compatible version of interfacial_area.
    Uses lax.cond for smooth JIT and autodiff.
    """

    def bubbly(_):
        A_ib = (6 * alpha) ** (2/3) * (jnp.pi * Nb) ** (1/3)
        return A_ib, Nb

    def droplet(_):
        A_id = (6 * (1 - alpha)) ** (2/3) * (jnp.pi * Nd) ** (1/3)
        return A_id, Nd

    def transition(_):
        A_ib_b = (6 * alpha_b) ** (2/3) * (jnp.pi * Nb) ** (1/3)
        A_id_d = (6 * (1 - alpha_d)) ** (2/3) * (jnp.pi * Nd) ** (1/3)
        Ai = A_ib_b + (A_id_d - A_ib_b) * (alpha - alpha_b) / (alpha_d - alpha_b)
        N = Nb + (Nd - Nb) * (alpha - alpha_b) / (alpha_d - alpha_b)
        return Ai, N

    # Nested condition: alpha <= alpha_b → bubbly, alpha >= alpha_d → droplet, else → transition
    Ai, N = jax.lax.cond(
        alpha <= alpha_b,
        bubbly,
        lambda _: jax.lax.cond(alpha >= alpha_d, droplet, transition, None),
        None
    )

    N = jnp.maximum(N, 1e-10)

    d = jnp.sqrt(Ai / (N * jnp.pi))
    d = jnp.clip(d, 1e-7, 1.0)

    # Recompute Ai from the clamped diameter for consistency
    Ai = N * jnp.pi * d**2

    return Ai, d

def interfacial_area(
    alpha,
    alpha_b=0.3, # 0.3 was the original
    alpha_d=0.7,
    Nb=1e13,
    Nd=1e12,
    d_min_b=1e-6, # 1e-5
    d_min_d=1e-6, # 1e-5
):
    """
    TA-BD interfacial area density with nucleation and breakup floors.

    Five regions as a function of vapor volume fraction alpha:
      1. alpha <= alpha_b_star         : nucleation floor (constant Ai, d = d_min_b)
      2. alpha_b_star < alpha <= 0.3   : bubbly regime
      3. 0.3 < alpha < 0.7             : linear transition
      4. 0.7 <= alpha < alpha_d_star   : droplet regime
      5. alpha >= alpha_d_star         : breakup floor (constant Ai, d = d_min_d)

    Parameters
    ----------
    alpha : scalar
        Vapor volume fraction in [0, 1].
    alpha_b, alpha_d : float
        TA-BD transition bounds (0.3 and 0.7 in the paper).
    Nb, Nd : float
        Bubble and droplet number densities [1/m^3].
    d_min_b, d_min_d : float
        Minimum physical diameters for bubbles and droplets [m].

    Returns
    -------
    Ai : scalar
        Interfacial area density [1/m].
    d : scalar
        Characteristic length scale [m] (bubble or droplet diameter).
    """

    # Derived thresholds where the natural cube-root diameter hits the floor
    alpha_b_star = jnp.pi * Nb * d_min_b**3 / 6.0
    alpha_d_star = 1.0 - jnp.pi * Nd * d_min_d**3 / 6.0

    # Endpoint values at the TA-BD transition boundaries (alpha_b = 0.3, alpha_d = 0.7)
    Ai_b_at_03 = (6.0 * alpha_b) ** (2.0 / 3.0) * (jnp.pi * Nb) ** (1.0 / 3.0)
    Ai_d_at_07 = (6.0 * (1.0 - alpha_d)) ** (2.0 / 3.0) * (jnp.pi * Nd) ** (1.0 / 3.0)
    d_b_at_03 = (6.0 * alpha_b / (jnp.pi * Nb)) ** (1.0 / 3.0)
    d_d_at_07 = (6.0 * (1.0 - alpha_d) / (jnp.pi * Nd)) ** (1.0 / 3.0)

    def nucleation_floor(_):
        Ai = Nb * jnp.pi * d_min_b**2
        d = d_min_b
        return Ai, d

    def bubbly(_):
        Ai = (6.0 * alpha) ** (2.0 / 3.0) * (jnp.pi * Nb) ** (1.0 / 3.0)
        d = (6.0 * alpha / (jnp.pi * Nb)) ** (1.0 / 3.0)
        return Ai, d

    def transition(_):
        w = (alpha - alpha_b) / (alpha_d - alpha_b)
        Ai = Ai_b_at_03 + (Ai_d_at_07 - Ai_b_at_03) * w
        d = d_b_at_03 + (d_d_at_07 - d_b_at_03) * w
        return Ai, d

    def droplet(_):
        Ai = (6.0 * (1.0 - alpha)) ** (2.0 / 3.0) * (jnp.pi * Nd) ** (1.0 / 3.0)
        d = (6.0 * (1.0 - alpha) / (jnp.pi * Nd)) ** (1.0 / 3.0)
        return Ai, d

    def breakup_floor(_):
        Ai = Nd * jnp.pi * d_min_d**2
        d = d_min_d
        return Ai, d

    # Nested lax.cond cascade — walks through the five branches in order.
    # Structure: if alpha <= alpha_b_star -> nucleation_floor
    #            elif alpha <= alpha_b    -> bubbly
    #            elif alpha <  alpha_d    -> transition
    #            elif alpha <  alpha_d_star -> droplet
    #            else                     -> breakup_floor
    Ai, d = jax.lax.cond(
        alpha <= alpha_b_star,
        nucleation_floor,
        lambda _: jax.lax.cond(
            alpha <= alpha_b,
            bubbly,
            lambda _: jax.lax.cond(
                alpha < alpha_d,
                transition,
                lambda _: jax.lax.cond(
                    alpha < alpha_d_star,
                    droplet,
                    breakup_floor,
                    None,
                ),
                None,
            ),
            None,
        ),
        None,
    )

    return Ai, d


def interfacial_area_stadtke(
    alpha_g,
    alpha_b_cr=0.15,
    alpha_d_cr=0.15,
    alpha_b_max=0.35,
    alpha_d_max=0.35,
    d_bub=4e-3,
    d_drop=0.8e-3,
):
    """
    Interfacial area concentration from Städtke (2006), Sec. 8.1.2, Eqs. (8.11)–(8.16).

    Two flow regimes coexist with volumetric weighting Xb + Xd = 1.
    The bubble (α_b) and droplet (α_d) subregion void fractions form symmetric
    bell-shaped curves (Fig. 8.2):

        Pure bubbly  (α_g < α_b_cr):           α_b = α_g,    α_d = 0
        Transition   (α_b_cr ≤ α_g ≤ 1-α_d_cr): piecewise linear, peaks at α_g=0.5
        Pure droplet (α_g > 1-α_d_cr):         α_b = 0,      α_d = 1-α_g

    Interfacial areas (Eq. 8.11, C^int = 2/r = 4/d):
        a_b^int = (3/2) · C_b^int · α_b = 6 · α_b / d_bub
        a_d^int = (3/2) · C_d^int · α_d = 6 · α_d / d_drop

    Default parameters reproduce the mono-dispersed case in Fig. 8.3
    (d_bub = 4.0 mm, d_drop = 0.8 mm, α_b_cr = α_d_cr = 0.15, α_b/d_max = 0.35).

    Parameters
    ----------
    alpha_g : float
        Gas (vapor) void fraction [-].
    alpha_b_cr, alpha_d_cr : float
        Critical void fractions bounding the transition zone [-].
    alpha_b_max, alpha_d_max : float
        Peak subregion void fractions inside the transition zone [-].
    d_bub : float
        Bubble diameter [m].
    d_drop : float
        Droplet diameter [m].

    Returns
    -------
    a_b_int : float  Bubbly-regime interfacial area density [m²/m³].
    a_d_int : float  Droplet-regime interfacial area density [m²/m³].
    d_bub   : float  Bubble diameter used [m].
    d_drop  : float  Droplet diameter used [m].
    """
    midpoint = 0.5
    alpha_g_max = 1.0 - alpha_d_cr   # upper bound of transition zone

    # ------------------------------------------------------------------
    # α_b: bubble subregion void fraction (Fig. 8.2, solid curve)
    #   - rises  from α_b_cr  to α_b_max  as α_g goes from α_b_cr  to 0.5
    #   - falls  from α_b_max to 0        as α_g goes from 0.5       to 1-α_d_cr
    # ------------------------------------------------------------------
    t_b_rise = jnp.clip((alpha_g - alpha_b_cr) / (midpoint - alpha_b_cr), 0.0, 1.0)
    t_b_fall = jnp.clip((alpha_g_max - alpha_g) / (alpha_g_max - midpoint), 0.0, 1.0)
    alpha_b_trans = jnp.where(
        alpha_g <= midpoint,
        alpha_b_cr + (alpha_b_max - alpha_b_cr) * t_b_rise,   # rising branch
        alpha_b_max * t_b_fall,                                # falling branch
    )
    alpha_b = jnp.where(
        alpha_g < alpha_b_cr, alpha_g,                         # pure bubbly
        jnp.where(alpha_g > alpha_g_max, 0.0, alpha_b_trans),  # transition / pure droplet
    )

    # ------------------------------------------------------------------
    # α_d: droplet subregion void fraction (Fig. 8.2, dashed curve)
    #   - rises  from 0        to α_d_max  as α_g goes from α_b_cr  to 0.5
    #   - falls  from α_d_max  to α_d_cr   as α_g goes from 0.5     to 1-α_d_cr
    # ------------------------------------------------------------------
    t_d_rise = jnp.clip((alpha_g - alpha_b_cr) / (midpoint - alpha_b_cr), 0.0, 1.0)
    t_d_fall = jnp.clip((alpha_g_max - alpha_g) / (alpha_g_max - midpoint), 0.0, 1.0)
    alpha_d_trans = jnp.where(
        alpha_g <= midpoint,
        alpha_d_max * t_d_rise,                                            # rising branch
        alpha_d_cr + (alpha_d_max - alpha_d_cr) * t_d_fall,               # falling branch
    )
    alpha_d = jnp.where(
        alpha_g < alpha_b_cr, 0.0,                                         # pure bubbly
        jnp.where(alpha_g > alpha_g_max, 1.0 - alpha_g, alpha_d_trans),   # transition / pure droplet
    )

    # ------------------------------------------------------------------
    # Interfacial areas: Eq. (8.11), a^int = (3/2)·(2/r)·α = 6α/d
    # ------------------------------------------------------------------
    a_b_int = 6.0 * alpha_b / d_bub
    a_d_int = 6.0 * alpha_d / d_drop

    return a_b_int, a_d_int, d_bub, d_drop


def get_speed_of_sound_mixture_old(
    G1, alpha1, a1, rho1,
    G2, alpha2, a2, rho2,
    threshold=1e-20
):
    rho_m = alpha1 * rho1 + alpha2 * rho2

    inv_a1_sq = 1.0 / (a1 * a1)
    inv_a2_sq = 1.0 / (a2 * a2)

    denominator = (
        -G1 * alpha1 * inv_a1_sq
        + G1 * alpha1 * inv_a1_sq * (rho_m / rho1)
        -G2 * alpha2 * inv_a2_sq
        + G2 * alpha2 * inv_a2_sq * (rho_m / rho2)
        + alpha1 * inv_a1_sq * (rho_m / rho1)
        + alpha2 * inv_a2_sq * (rho_m / rho2)
    )

    # First: handle near-single-phase cases
    def use_a2(_):
        return a2

    def use_a1(_):
        return a1

    def use_formula(_):
        return jnp.sqrt(1.0 / denominator)

    # Nested conds give if / elif / else
    return lax.cond(
        alpha1 < threshold,
        use_a2,
        lambda _: lax.cond(
            alpha2 < threshold,
            use_a1,
            use_formula,
            operand=None,
        ),
        operand=None,
    )

def get_speed_of_sound_mixture(
    G1, alpha1, a1, rho1,
    G2, alpha2, a2, rho2,
):
    rho_m = alpha1 * rho1 + alpha2 * rho2

    term11 = - G1 / (a1 ** 2)
    term12 = G1 * rho_m / ((a1 ** 2) * rho1) 
    term13 = rho_m /  ((a1 ** 2) * rho1)
    term1 = alpha1 * (term11 + term12 + term13)

    term21 = - G2 / (a2 ** 2)
    term22 = G2 * rho_m / ((a2 ** 2) * rho2) 
    term23 = rho_m /  ((a2 ** 2) * rho2)
    term2 = alpha2 * (term21 + term22 + term23)

    a_mix = jnp.sqrt(1 / (term1 + term2))

    return a_mix

def get_speed_of_sound_mixture_stadke(alpha1, a1, rho1, alpha2, a2, rho2):
    """
    Calculate the mixture speed of sound.
    
    Parameters:
    alpha1, alpha2 : float or jax array
        Volume fractions of each component
    a1, a2 : float or jax array
        Speed of sound in each component
    rho1, rho2 : float or jax array
        Densities of each component

    
    Returns:
    c_m : float or jax array
        Speed of sound in the mixture
    """
    rho_m = alpha1 * rho1 + alpha2 * rho2
    term1 = alpha1 / a1**2 * (rho_m / rho1)
    term2 = alpha2 / a2**2 * (rho_m / rho2)
    
    denominator = term1 + term2 

    return jnp.max(jnp.sqrt(1.0 / denominator))

def compute_inlet_state_hne_one_component(p0, T01, T02, Ma, alpha1, alpha2, fluid1, fluid2,
                                          alpha1_threshold=1e-3):
    """
    Compute the inlet state for the homogeneous non-equilibrium model.

    Phase 1 is the dispersed phase and phase 2 is the continuum phase.

    If alpha1 < alpha1_threshold, uses single-phase inlet (continuum only,
    dispersed phase initialized at saturation). Otherwise, uses two-phase inlet.
    """
    def two_phase_inlet(_):
        rho1 = fluid1.get_state(jxp.PT_INPUTS, p0, T01)["density"]
        rho2 = fluid2.get_state(jxp.PT_INPUTS, p0, T02)["density"]
        R = (alpha1 * rho1) / (alpha2  * rho2)
        return compute_inlet_state_hne(p0, T01, T02, Ma, R, fluid1, fluid2, R)

    def single_phase_inlet(_):

        # Fluid 1 = dispersed (vapor), Fluid 2 = continuum (liquid)
        h0_in_2 = fluid2.get_state(jxp.PT_INPUTS, p0, T02)["h"]

        state_2_in = compute_static_state_single_phase(p0, h0_in_2, Ma, fluid2)
        state_1_in = fluid1.get_state_saturation(state_2_in["pressure"])

        p_in, rho_2_in, rho_1_in, h_2_in, h_1_in = (
            state_2_in["p"],
            state_2_in["rho"],
            state_1_in["density"],
            state_2_in["h"],
            state_1_in["enthalpy"]
        )

        a2 = state_2_in["a"]

        u_in = Ma * a2

        # alpha1_ghost = jnp.float64(1e-6)
        # alpha2_ghost = 1.0 - alpha1_ghost

        y_inlet = jnp.array([alpha1, alpha2, rho_1_in, rho_2_in, u_in, p_in, h_1_in, h_2_in])

        return y_inlet

    return jax.lax.cond(alpha1 < alpha1_threshold, single_phase_inlet, two_phase_inlet, None)


def compute_inlet_state_hne(p0, T01, T02, Ma, R, fluid1, fluid2, mixture_ratio):
    # jax.debug.print("p={p}, T1={T01}, T2={T02}, Ma={Ma}",p=p0, T01=T01, T02=T02, Ma=Ma)

    # jax.debug.print("fluid1={f1} | fluid2={f2}", f1=fluid1, f2=fluid2)
    st01 = fluid1.get_state(jxp.PT_INPUTS, p0, T01)
    # st01.block_until_ready()

    s01, h01 = st01["s"], st01["h"]

    st02 = fluid2.get_state(jxp.PT_INPUTS, p0, T02)

    # st02.block_until_ready()
    s02= st02["s"]

    q_in = 1/ (1 + R)

    # Scalar residual for Bisection
    def residual1(p, _):
        st = fluid1.get_state(jxp.PSmass_INPUTS, p, s01)
        h, G1, a1, rho1 = st["h"], st["G"], st["a"], st["d"]

        st = fluid2.get_state(jxp.PSmass_INPUTS, p, s02)
        G2, a2, rho2 = st["G"], st["a"], st["d"]

        alpha2 = 1 / (1 + ((1 - q_in) / q_in) * (rho2 / rho1))
        alpha1 = 1 - alpha2

        # jax.debug.print("alpha1={alpha1}, alpha2={alpha2}",alpha1=alpha1, alpha2=alpha2)
        a_mix = get_speed_of_sound_mixture(G1, alpha1, a1, rho1, G2, alpha2, a2, rho2)
        f = h01 - h - 0.5 * (Ma*a_mix)**2
        # jax.debug.print("f={a}", a=f)
        
        return h01 - h - 0.5 * (Ma*a_mix)**2

    solver = optx.Newton(rtol=1e-6, atol=1e-6)
    lower1, upper1 = 0.1 * p0, p0
    sol1 = optx.root_find(residual1, solver, y0=0.99 * p0, options={"lower": lower1, "upper": upper1})
    state1_in = fluid1.get_state(jxp.PSmass_INPUTS, sol1.value, s01)
    state2_in = fluid2.get_state(jxp.PSmass_INPUTS, sol1.value, s02)

    p_in, rho1_in, rho2_in, h1_in, h2_in = (
            state1_in["p"],
            state1_in["rho"],
            state2_in["rho"],
            state1_in["h"],
            state2_in["h"]
        )
    
    G1, a1 = state1_in["G"], state1_in["a"]
    q_in = (1)/(1 + mixture_ratio)
    alpha2_in = 1 / (1 + ((1 - q_in) / q_in) * (rho2_in / rho1_in))
    alpha1_in = 1 - alpha2_in

    # h = st["h"]
    G2, a2 = state2_in["G"], state2_in["a"]

    a_in_mix = get_speed_of_sound_mixture(G1, alpha1_in, a1, rho1_in, G2, alpha2_in, a2, rho2_in)

    u_in = Ma * a_in_mix

    # x_in = 1e-9

    y = jnp.array([alpha1_in, alpha2_in, rho1_in, rho2_in, u_in, p_in, h1_in, h2_in])

    return y


def compute_static_state_single_phase(p0, h0, Ma, fluid):
    # --- Reference stagnation state ---
    st0 = fluid.get_state(jxp.HmassP_INPUTS, h0, p0)
    s0 = st0["s"]
    a0 = st0["a"]

    # --- 1D isentropic residual: find h such that h + 0.5*(a(h,s0)*Ma)^2 = h0 ---
    # Reducing to 1D (entropy fixed at s0) avoids the near-singular 2D Jacobian
    # that GaussNewton produces at low Ma where the kinetic term is tiny.
    def residual(h, _):
        st = fluid.get_state(jxp.HmassSmass_INPUTS, h, s0)
        a = st["a"]
        return h0 - h - 0.5 * (a * Ma) ** 2

    # --- Bisection bracket ---
    # f(h0) = -0.5*(a*Ma)^2 < 0  (upper bound: static h can't exceed stagnation h)
    # f(h0 - 10*ke_est) > 0      (lower bound: well below the true static h)
    # This bracket is correct for both positive and negative h0.
    ke_est = 0.5 * (Ma * a0) ** 2
    upper = h0
    lower = h0 - 10.0 * ke_est - 1.0   # +1 J/kg offset avoids zero-width bracket at Ma=0

    solver = optx.Bisection(rtol=1e-6, atol=1e-6)
    sol = optx.root_find(residual, solver, y0=(lower + upper) / 2.0,
                         options={"lower": lower, "upper": upper})

    h_static = sol.value
    state = fluid.get_state(jxp.HmassSmass_INPUTS, h_static, s0)
    return state


def calculate_mass_transfer_lv(T_l, T_v, T_sat, htc_l, htc_v, A_i, h_sat_l, h_sat_v, h_l, h_v, Qvi, Qli):
    """
    Interphase mass transfer from Rane & He (2021), Eq. (4).
    Positive m_dot = evaporation (l -> v), Negative = condensation (v -> l).
    """
    # Eq. (4): m_dot = -[htc_l * Ai * (Tsat - Tl) + htc_v * Ai * (Tsat - Tv)] / (h_qs - h_ps)
    # Distribute the minus sign:
    numerator = -htc_l * A_i * (T_sat - T_l) - htc_v * A_i * (T_sat - T_v)

    numerator = - Qvi - Qli


    # Denominator depends on direction: Eq. (5) for evaporation, Eq. (6) for condensation
    # Evaporation (m_dot > 0): h_qs = h_v(T_sat) = h_sat_v,  h_ps = h_l(T_l) = h_l
    denominator = h_sat_v - h_sat_l

    m_dot = numerator / denominator

    # jax.debug.print("Tl:{Tl} | Tv:{Tv} | Tsat:{Ts} | mdot:{md}", Tl=T_l, Tv=T_v, Ts=T_sat, md=m_dot)

    return m_dot


def calculate_mass_transfer_lv_rane(T_l, T_v, T_sat, htc_l, htc_v, A_i, h_sat_l, h_sat_v, h_l, h_v):
    """
    Interphase mass transfer from Rane & He (2021), Eq. (4).
    Positive m_dot = evaporation (l -> v), Negative = condensation (v -> l).
    """
    # Eq. (4): m_dot = -[htc_l * Ai * (Tsat - Tl) + htc_v * Ai * (Tsat - Tv)] / (h_qs - h_ps)
    # Distribute the minus sign:
    numerator = -htc_l * A_i * (T_sat - T_l) - htc_v * A_i * (T_sat - T_v)

    # Denominator depends on direction: Eq. (5) for evaporation, Eq. (6) for condensation
    # Evaporation (m_dot > 0): h_qs = h_v(T_sat) = h_sat_v,  h_ps = h_l(T_l) = h_l
    denom_evap = h_sat_v - h_l

    # Condensation (m_dot < 0): h_qs = h_v(T_v) = h_v,  h_ps = h_l(T_sat) = h_sat_l
    denom_cond = h_v - h_sat_l

    # First compute with evaporation denominator to determine direction
    # (safe division)
    denom_evap_safe = jnp.where(jnp.abs(denom_evap) < 1e-6, 1e-6, denom_evap)
    m_dot_trial = numerator / denom_evap_safe
    mass_evap = m_dot_trial >= 0.0


    # Select the correct denominator based on direction
    denom_cond_safe = jnp.where(jnp.abs(denom_cond) < 1e-6, 1e-6, denom_cond)
    m_dot = jnp.where(
        m_dot_trial >= 0.0,
        numerator / denom_evap_safe,   # evaporation
        numerator / denom_cond_safe,   # condensation
    )

    # jax.debug.print("Tl:{Tl} | Tv:{Tv} | Tsat:{Ts} | mdot:{md}", Tl=T_l, Tv=T_v, Ts=T_sat, md=m_dot)

    return m_dot


def calculate_mass_transfer_lv_stadtke(
    T_l, T_v, T_sat,
    a_b_int, htc_b_g, htc_b_l,
    a_d_int, htc_d_g, htc_d_l,
    h_sat_l, h_sat_v,
):
    """
    Interphase mass transfer from Städtke (2006), Eqs. (8.18) and (8.20).

    Positive m_dot = evaporation (l → v), Negative = condensation (v → l).

    Parameters
    ----------
    T_l, T_v, T_sat : float
        Liquid temperature, vapor temperature, saturation temperature [K].
    a_b_int : float
        Interfacial area density for the bubbly regime [m²/m³].
        (vapor bubbles dispersed in liquid — applies when alpha_v < alpha_b)
    htc_b_g : float
        Vapor-side HTC in the bubbly regime (internal bubble conduction) [W/(m²·K)].
    htc_b_l : float
        Liquid-side HTC in the bubbly regime (Ranz-Marshall external) [W/(m²·K)].
    a_d_int : float
        Interfacial area density for the droplet regime [m²/m³].
        (liquid droplets dispersed in vapor — applies when alpha_v > alpha_d)
    htc_d_g : float
        Vapor-side HTC in the droplet regime (Ranz-Marshall external) [W/(m²·K)].
    htc_d_l : float
        Liquid-side HTC in the droplet regime (internal droplet conduction) [W/(m²·K)].
    h_sat_l, h_sat_v : float
        Specific enthalpies of saturated liquid and vapor [J/kg].

    Notes
    -----
    Eq. (8.20) clips the driving temperatures so that only the *metastable*
    contribution of each phase enters the source term:
        ΔT_g = min(T_v − T_sat, 0)  → subcooled vapor  (condensation driver)
        ΔT_l = max(T_l − T_sat, 0)  → superheated liquid (evaporation driver)

    Eq. (8.18):
        σ^M = (1/Δhₛ) · [(a_b·H_b,g + a_d·H_d,g)·ΔT_g
                         + (a_b·H_b,l + a_d·H_d,l)·ΔT_l]
    """
    # Eq. (8.20): metastable temperature excesses only
    delta_T_g = jnp.minimum(T_v - T_sat, 0.0)   # ≤ 0: subcooled vapor → condensation
    delta_T_l = jnp.maximum(T_l - T_sat, 0.0)   # ≥ 0: superheated liquid → evaporation

    # Latent heat denominator (safe against degenerate states)
    delta_hs = h_sat_v - h_sat_l
    delta_hs_safe = jnp.where(jnp.abs(delta_hs) < 1.0, 1.0, delta_hs)

    # Eq. (8.18): combined vapor-side and liquid-side contributions from both regimes
    numerator = (
        (a_b_int * htc_b_g + a_d_int * htc_d_g) * delta_T_g   # condensation part
      + (a_b_int * htc_b_l + a_d_int * htc_d_l) * delta_T_l   # evaporation part
    )

    m_dot = numerator / delta_hs_safe
    return m_dot


def calculate_mass_transfer_lv_lee(
    T_l, T_v, T_sat,
    alpha_l, rho_l,
    alpha_v, rho_v,
    coeff=0.1,
):
    """
    Mass transfer rate from the Lee (1980) model (ANSYS Fluent Eq. 14.609–14.610).

    Positive m_dot = evaporation (l → v), negative = condensation (v → l).

    Evaporation  (T_l > T_sat):   m_lv = coeff · α_l · ρ_l · (T_l − T_sat) / T_sat
    Condensation (T_v < T_sat):   m_vl = coeff · α_v · ρ_v · (T_sat − T_v) / T_sat

    Net:  m_dot = m_lv − m_vl

    The `coeff` parameter [1/s] acts as a relaxation frequency and must be
    calibrated against experimental data for the fluid / flow regime of interest.
    Typical values range from 0.1 to 1e5 s⁻¹ depending on how fast the
    interfacial heat/mass exchange is.

    Only the *metastable* temperature excess of each phase contributes
    (jnp.maximum clips away the non-driving sign), consistent with
    Städtke Eq. (8.20).

    Parameters
    ----------
    T_l, T_v, T_sat : float   Liquid, vapor, saturation temperatures [K].
    alpha_l, rho_l  : float   Liquid volume fraction [-] and density [kg/m³].
    alpha_v, rho_v  : float   Vapor  volume fraction [-] and density [kg/m³].
    coeff           : float   Relaxation frequency [1/s].  Default 0.1 s⁻¹.
    """
    T_sat_safe = jnp.where(T_sat < 1.0, 1.0, T_sat)   # avoid division by zero

    # Evaporation: liquid superheat drives liquid → vapor (Eq. 14.609)
    m_lv = coeff * alpha_l * rho_l * jnp.maximum(T_l - T_sat, 0.0) / T_sat_safe

    # Condensation: vapor subcooling drives vapor → liquid (Eq. 14.610)
    m_vl = coeff * alpha_v * rho_v * jnp.maximum(T_sat - T_v, 0.0) / T_sat_safe

    return m_lv - m_vl


def compute_eigenvalues(A_mat):
    """
    Given a system matrix A_mat, returns all eigenvalues
    sorted by real part.
    """
    eigs = jnp.linalg.eigvals(A_mat)
    idx  = jnp.argsort(eigs.real)
    return eigs[idx]


def richardson(alpha_l):
    phi_lm_squared = (1-alpha_l)**(-1.75)
    return phi_lm_squared


# TODO: implement thoroughly the new hnem


class CommonQtyHNEM(NamedTuple):
    """Shared abbreviations of the homogeneous model."""
    rho_mix : jnp.ndarray   # mixture density  alpha_g rho_g + alpha_l rho_l                [kg/m^3]
    rho_a2  : jnp.ndarray   # mixture bulk modulus rho a^2 (Wood)                            [Pa]
    a_sq    : jnp.ndarray   # homogeneous frozen sound speed squared, eq. (4.63)              [m^2/s^2]
    w_g     : jnp.ndarray   # gas share of the mixture compliance  alpha_g rho a^2/(rho_g a_g^2) [-]
    w_l     : jnp.ndarray   # liquid share of the mixture compliance, w_g + w_l = 1           [-]
 
 
class HomogeneousClosure(NamedTuple):
    """
    Physical closure quantities consumed by D_h_primitive_hnem.
 
    All quantities are per unit volume and are scalars (per cell).
    Sign convention: sigma_Mg > 0 = evaporation (liquid -> gas), and sigma_Ml = -sigma_Mg.
 
    Total energy is conserved exactly when the closure satisfies the interface balance
        sigma_Mg (h_ex_g - h_ex_l) + sigma_Qg + sigma_Ql = 0,
    which the two-fluid closure enforces through sigma_Mg = -(Qvi + Qli)/(h_sat_g - h_sat_l).
 
    F is the external force on the mixture (wall friction). The wall is at rest, so F does no
    work on the total energy: its dissipation -F u returns to the phases as heat, a share xi_Fg
    to the gas and 1 - xi_Fg to the liquid. Total energy is conserved for any xi_Fg, the split
    is still to be decided. A body force such as gravity does work on the fluid and must not be
    passed in F.
    """
    sigma_Mg : jnp.ndarray  # gas mass source                                    [kg/(m^3 s)]
    sigma_Qg : jnp.ndarray  # interfacial heat to gas                            [W/m^3]
    sigma_Ql : jnp.ndarray  # interfacial heat to liquid                         [W/m^3]
    F        : jnp.ndarray  # external force on the mixture (wall friction)      [N/m^3]
    xi_Fg    : jnp.ndarray  # share of the friction heat -F u given to the gas   [-]
    Q_g      : jnp.ndarray  # wall/external heat to gas                          [W/m^3]
    Q_l      : jnp.ndarray  # wall/external heat to liquid                       [W/m^3]
    h_ex_g   : jnp.ndarray  # enthalpy of mass entering the gas phase  [J/kg]  (h_sat_g for evaporation)
    h_ex_l   : jnp.ndarray  # enthalpy of mass leaving the liquid phase [J/kg] (h_sat_l for evaporation)
    dlnA_dx  : jnp.ndarray  # (1/A) dA/dx — log area derivative                  [1/m]
 
 
def common_quantities_hnem(U_h, props_l, props_g) -> CommonQtyHNEM:
    """
    Pre-compute the shared abbreviations of the homogeneous model.
 
    Parameters
    ----------
    U_h     : (5,) primitive state [p, u, alpha_g, h_g, h_l]
    props_g : (rho_g, a_g) phasic density and speed of sound from get_state
    props_l : (rho_l, a_l)
    """
    _, _, ag, _, _ = U_h
    al = 1.0 - ag
 
    rg, ag_ = props_g
    rl, al_ = props_l
 
    rho_mix = ag * rg + al * rl
 
    # Phasic compliances alpha_i/(rho_i a_i^2), their sum is 1/(rho a^2) — Wood, eq. (4.63)
    comp_g = ag / (rg * ag_**2)
    comp_l = al / (rl * al_**2)
    rho_a2 = 1.0 / (comp_g + comp_l)
    a_sq = rho_a2 / rho_mix
 
    return CommonQtyHNEM(rho_mix=rho_mix, rho_a2=rho_a2, a_sq=a_sq,
                         w_g=comp_g * rho_a2, w_l=comp_l * rho_a2)
 
 
def G_h_matrix_hnem(U_h, props_l, props_g, cq: CommonQtyHNEM) -> jnp.ndarray:
    """
    5×5 quasi-linear coefficient matrix G_h = J_hs G_s J_sh of the homogeneous model.
 
    Rows/cols: p, u, alpha_g, h_g, h_l.
        Rows 1-3 : Städtke Table B.16 (= eq. 4.64, verified exactly). The void entry
                   -alpha_g alpha_l rho a^2 (1/(rho_g a_g^2) - 1/(rho_l a_l^2)) is written as
                   alpha_g - w_g.
        Rows 4-5 : Gibbs transform of the entropy rows, Dh_i/Dt = (1/rho_i) Dp/Dt + sources,
                   each phasic enthalpy follows the pressure isentropically.
    Gamma_i cancels exactly, so G_h needs only rho_i and a_i.
    Eigenvalues u + a, u - a and u (three times, semi-simple). det G_h = u^3 (u^2 - a^2).
    """
    p, u, ag, h_g, h_l = U_h
 
    rg, _ = props_g
    rl, _ = props_l
 
    rho_mix = cq.rho_mix
    rho_a2 = cq.rho_a2
    w_g = cq.w_g
 
    # Row 1 — pressure:        Dp/Dt       + rho a^2 du/dx
    row1 = jnp.array([u,             rho_a2,      0.0, 0.0, 0.0])
    # Row 2 — momentum:        Du/Dt       + (1/rho) dp/dx
    row2 = jnp.array([1.0 / rho_mix, u,           0.0, 0.0, 0.0])
    # Row 3 — void fraction:   Dalpha_g/Dt + (alpha_g - w_g) du/dx
    row3 = jnp.array([0.0,           ag - w_g,    u,   0.0, 0.0])
    # Row 4 — gas enthalpy:    Dh_g/Dt     + (rho a^2/rho_g) du/dx
    row4 = jnp.array([0.0,           rho_a2 / rg, 0.0, u,   0.0])
    # Row 5 — liquid enthalpy: Dh_l/Dt     + (rho a^2/rho_l) du/dx
    row5 = jnp.array([0.0,           rho_a2 / rl, 0.0, 0.0, u  ])
 
    return jnp.stack([row1, row2, row3, row4, row5])
 
 
def D_h_primitive_hnem(U_h, eos_g, eos_l, cq: CommonQtyHNEM,
                       closure: HomogeneousClosure) -> jnp.ndarray:
    """
    Primitive enthalpy-frame source vector D_h (5,) of the homogeneous model, quasi-1D.
 
    With q_i the heat input to phase i and S_i/rho_i its expansion rate at constant pressure,
        q_i = sigma_Mi (h_ex_i - h_i) + sigma_Qi + Q_i + Phi_i            (= T_i sigma_i^S)
        S_i = sigma_Mi - alpha_i rho_i u dlnA/dx + Gamma_i q_i / a_i^2
    the rows are
        D_p     = rho a^2 (S_g/rho_g + S_l/rho_l)
        D_u     = F / rho
        D_alpha = w_l S_g/rho_g - w_g S_l/rho_l
        D_hi    = D_p/rho_i + q_i/(alpha_i rho_i)
    Row 3 corrects the two typos of Table B.17. The area term is the 1D divergence correction
    du/dx -> du/dx + u dlnA/dx in every row except momentum.
    T_i is not needed: the enthalpy frame is temperature-free, temperatures enter only through
    the closure (sigma_Qi).
 
    Parameters
    ----------
    U_h          : (5,) primitive state [p, u, alpha_g, h_g, h_l]
    eos_g, eos_l : (rho_i, a_i, T_i, Gamma_i) at U_h, the same tuples as the two-fluid code
                   (T_i is not used)
    cq           : CommonQtyHNEM at U_h
    closure      : HomogeneousClosure with the physical source inputs
 
    Returns
    -------
    D_h : (5,) — [D_p, D_u, D_alpha, D_{h_g}, D_{h_l}]
    """
    p, u, ag, h_g, h_l = U_h
    al = 1.0 - ag
 
    rg, ag_, _, Gg = eos_g     # T_g not needed
    rl, al_, _, Gl = eos_l     # T_l not needed
 
    rho_mix = cq.rho_mix
    rho_a2 = cq.rho_a2
    w_g = cq.w_g
    w_l = cq.w_l
 
    # ── Unpack closure ───────────────────────────────────────────────────────
    sigma_Mg = closure.sigma_Mg
    sigma_Qg = closure.sigma_Qg
    sigma_Ql = closure.sigma_Ql
    F        = closure.F
    xi_Fg    = closure.xi_Fg
    Q_g      = closure.Q_g
    Q_l      = closure.Q_l
    h_ex_g   = closure.h_ex_g
    h_ex_l   = closure.h_ex_l
    dlnA_dx  = closure.dlnA_dx
 
    sigma_Ml = -sigma_Mg
 
    # ── Friction heat: the wall is at rest, so the work -F u stays in the fluid ─
    Phi_g = -xi_Fg * F * u
    Phi_l = -(1.0 - xi_Fg) * F * u
 
    # ── Heat input to each phase, q_i = T_i sigma_i^S (Städtke eqs. 4.56-4.57) ─
    q_g = sigma_Mg * (h_ex_g - h_g) + sigma_Qg + Q_g + Phi_g
    q_l = sigma_Ml * (h_ex_l - h_l) + sigma_Ql + Q_l + Phi_l
 
    # ── Expansion sources: mass transfer, area change, heating at constant p ──
    S_g = sigma_Mg - ag * rg * u * dlnA_dx + Gg * q_g / ag_**2
    S_l = sigma_Ml - al * rl * u * dlnA_dx + Gl * q_l / al_**2
 
    # ── Row 1: pressure ──────────────────────────────────────────────────────
    D1 = rho_a2 * (S_g / rg + S_l / rl)
 
    # ── Row 2: momentum ──────────────────────────────────────────────────────
    D2 = F / rho_mix
 
    # ── Row 3: void fraction (Table B.17 row 3, corrected) ───────────────────
    D3 = w_l * S_g / rg - w_g * S_l / rl
 
    # ── Rows 4, 5: enthalpies — isentropic pressure part + own heat per unit mass
    D4 = D1 / rg + q_g / (ag * rg)
    D5 = D1 / rl + q_l / (al * rl)
 
    return jnp.array([D1, D2, D3, D4, D5])
