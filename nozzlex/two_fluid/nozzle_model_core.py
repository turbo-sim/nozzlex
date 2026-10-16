import jax
import jax.numpy as jnp
import jaxprop as jxp
from jax.numpy.linalg import det
import equinox as eqx
import optimistix as optx
import nurbspy.jax as nrb
from typing import NamedTuple



from CoolProp import AbstractState
import CoolProp.CoolProp as CP


# ------------------------------------------------------------------------------
# Transport equation for bubble and droplet number
# ------------------------------------------------------------------------------

def nozzle_two_fluid_transport_core(x, y, args):
    """Wrapper that adapts from (x, y) to autonomous form."""
    Y = jnp.concatenate([jnp.atleast_1d(x), y])
    return nozzle_two_fluid_transport_autonomous(0.0, Y, args)

@eqx.filter_jit
def nozzle_two_fluid_transport_autonomous(tau, Y, args):
    """
    Autonomous right-hand side for a two-fluid non-equilibrium nozzle model
    (fluid1=liquid/water, fluid2=gas/nitrogen) with no inter-phase mass transfer
    and separate velocities for each phase.

    Uses Cramer's rule on the linear system  A·(dY/dx) = b  rewritten as an
    autonomous system to handle sonic-point singularities:

        dx/dτ   = det(A)
        dY_i/dτ = det(A with column i replaced by b)

    State vector: Y = [x, alpha1, alpha2, rho1, rho2, u1, u2, p, h1, h2, NB, ND]
        x       – axial coordinate [m]
        alpha1  – volume fraction of fluid 1 (liquid)
        alpha2  – volume fraction of fluid 2 (gas)
        rho1    – density of fluid 1 [kg/m³]
        rho2    – density of fluid 2 [kg/m³]
        u1      – velocity of fluid 1 [m/s]
        u2      – velocity of fluid 2 [m/s]
        p       – shared pressure [Pa]
        h1      – specific enthalpy of fluid 1 [J/kg]
        h2      – specific enthalpy of fluid 2 [J/kg]
        NB      – bubble number flux A·nb·u2 [1/s] (bubbles advect with gas)
        ND      – droplet number flux A·nd·u1 [1/s] (droplets advect with liquid)
    """
    x, alpha1, alpha2, rho1, rho2, u1, u2, p, h1, h2, NB, ND = Y

    # v = jnp.where(jnp.abs(v) < 1e-6, jnp.sign(v)*1e-6, v)
    # Debug print using jax.debug.print
    # jax.debug.print(
    # "x={:.2f}, alpha1={:.4e}, alpha2={:.4e}, rho1={:.4e}, rho2={:.4e}, u1={:.4e}, u2={:.4e}, p={:.4e}, h1={:.4e}, h2={:.4e}, NB={:.4e}, ND={:.4e}",
    # x, alpha1, alpha2, rho1, rho2, u1, u2, p, h1, h2, NB, ND
    # )

    # --- Geometry / model parameters ---
    fluid1 = args.fluid1 # water
    fluid2 = args.fluid2 # nitrogen
    L = args.length
    eps_wall = args.roughness
    wall_friction = args.wall_friction
    heat_transfer = args.heat_transfer

    # --- Thermodynamic state fluid 1 ---
    alpha1 = jnp.clip(alpha1, 1e-12, 1-1e-12)
    alpha2 = jnp.clip(alpha2, 1e-12, 1-1e-12)
    
    # h1 = jnp.clip(h1, -1e5, 1e7)
    # h2 = jnp.clip(h2, -1e5, 1e7)
    # p = jnp.clip(p, 1e2, 1e9)
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

    # TODO: find correct S

    # # --- Old scalar S scaling (working but det blows up near critical point) ---
    # # For stadke model
    S = ( rho1 * rho2 * u1 * u2) ** (1/9)
    # S = ( rho1 ** 2 * rho2 ** 2 * u1 ** 5 * u2 ** 5) ** (1/9)
    
    # # For Wallis model
    # S = (rho1**2 * rho2**2 * u1 * u2) ** (1/9)
    # S = (rho1**2 * rho2**2 * u1**2 * u2**2) ** (1/9)
    
    # # different "S" changes the result in the autonomous form. check which is the correct one
    
    # --- Geometry ---
    # A, dAdx, perimeter, radius = args.geometry(x, L) # When using symmetric geometry
    A, dAdx, perimeter, radius = args.geometry(x) # When using linear convergent divergent geometry
    diameter = 2 * radius

    # --- Wall heat transfer and friction ---
    Re1 = u1 * rho1 * diameter * jnp.sqrt(alpha1) / mu1  # sqrt(alpha) gives an effective hydraulic diameter per phase; non-standard
    f_D1 = get_friction_factor_haaland(Re1, eps_wall, diameter)
    tau_w1 = get_wall_viscous_stress(f_D1, rho1, u1)
    tau_w1 = tau_w1 * wall_friction

    Re2 = u2 * rho2 * diameter * jnp.sqrt(alpha2) / mu2  # sqrt(alpha) gives an effective hydraulic diameter per phase; non-standard
    f_D2 = get_friction_factor_haaland(Re2, eps_wall, diameter)
    tau_w2 = get_wall_viscous_stress(f_D2, rho2, u2)
    tau_w2 = tau_w2 * wall_friction

    rho_mix = alpha1 * rho1 + alpha2 * rho2
    u_mix = (alpha1 * rho1 * u1 + alpha2 * rho2 * u2) / rho_mix  # mass-averaged mixture velocity


    # --------------------------------------------------------------------- #
    # Interfacial area considering blending of two regimes
    We_crit_b = 6.0
    We_crit_d = 6.0
    sigma_WN = 0.0728 # N/m
    u_rel_abs = jnp.maximum(jnp.abs(u1 - u2), 1e-6)

    d_bubb_we = (We_crit_b * sigma_WN) / (rho1 * u_rel_abs**2)
    d_drop_we = (We_crit_d * sigma_WN) / (rho2 * u_rel_abs**2)
    
    nb = NB / (A * u2)
    nd = ND / (A * u1)

    alpha_b_transition=0.20
    alpha_d_transition=0.80

    xd = smoothstep(alpha2, alpha_b_transition, alpha_d_transition)
    xd = jnp.clip(xd, 1e-6, 1-1e-6)
    xb = 1.0 - xd
    alpha_bubble  = alpha2         * xb   # gas   volume in bubbly region
    alpha_droplet = (1.0 - alpha2) * xd   # liquid volume in droplet region

    d_bubb = jnp.maximum((6 * alpha_bubble  / (jnp.pi * nb)) ** (1/3), 1e-7)
    d_drop = jnp.maximum((6 * alpha_droplet / (jnp.pi * nd)) ** (1/3), 1e-7)

    ai_b = 6.0 * alpha_bubble  / d_bubb
    ai_d = 6.0 * alpha_droplet / d_drop

    # Random collision source term
    alpha_max_RC = 0.8
    a_for_RC = jnp.minimum(alpha2, 0.99 * alpha_max_RC)  # use TRUE gas fraction, keep below packing
    a13    = a_for_RC ** (1.0/3.0)
    amax13 = alpha_max_RC ** (1.0/3.0)
    denom  = amax13 * (amax13 - a13)
    C = 3.0 # 3.0
    C_RC = 30.0 # 0.06
    Re_mix = rho_mix * u_mix * diameter / mu_mix
    f_tw = 0.316 / Re_mix**0.25 * (1.0 / (1.0 - alpha2))**0.25
    epsilon2 = f_tw * u_mix**3 / (2 *diameter)
    u_t2 = epsilon2**(1/3) * d_bubb**(1/3)
    arg     = C * amax13 * a13 / denom
    S_RC_bubble =  - A * (C_RC / xb) * u_t2 * nb**2 * d_bubb**2 / denom * (1.0 - jnp.exp(-arg))


    alpha_max_RC_d = 0.80                 # max liquid fraction before bubbly regime takes over
    a_for_RC_d = jnp.minimum(alpha1, 0.95 * alpha_max_RC_d)
    a13_d    = a_for_RC_d ** (1.0/3.0)
    amax13_d = alpha_max_RC_d ** (1.0/3.0)
    denom_d  = amax13_d * (amax13_d - a13_d)
    arg_d    = C * amax13_d * a13_d / denom_d
    C = 3.0 # 3.0
    C_RC = 0.06 # 0.06
    # Turbulent dissipation for droplet RC: continuous phase is GAS (fluid 2), so use (1/alpha2)^0.25
    f_tw_d   = 0.316 / Re_mix**0.25 * (1.0 / alpha2)**0.25
    epsilon_d = f_tw_d * u_mix**3 / (2 * diameter)
    u_td = epsilon_d**(1/3) * d_drop**(1/3)

    S_RC_droplet = -A * (C_RC / xd) * u_td * nd**2 * d_drop**2 / denom_d * (1.0 - jnp.exp(-arg_d))
    # S_RC_droplet = 0.0

    # Breakup source term
    timescale_br = 0.01
    nb_We = (6 * alpha_bubble) / (jnp.pi * d_bubb_we**3)
    diff_breakup_b = nb_We - nb
    H_BR_B = jnp.heaviside(diff_breakup_b, 0.5)
    S_breakup_bubble = A * H_BR_B * diff_breakup_b / timescale_br
    # S_breakup_bubble = 0.0
    
    # Breakup source term
    timescale_br = 0.01 # 1.0
    nd_We = (6 * alpha_droplet) / (jnp.pi * d_drop_we**3)
    diff_breakup = nd_We - nd
    H_BR = jnp.heaviside(diff_breakup, 0.5)
    S_breakup_droplet = A * H_BR * diff_breakup / timescale_br
    # S_breakup_droplet = 0.0

    # # Morphological change source term
    xi = jnp.clip((alpha2 - alpha_b_transition) / (alpha_d_transition - alpha_b_transition), 1e-6, 1.0)
    # dxd_dalpha2 = 6.0 * xi * (1.0 - xi) / (alpha_d_transition - alpha_b_transition)
    # dxb_dalpha2 = -dxd_dalpha2   # dx_b/dx_d = -1

    dxd_dalpha2 = 6.0 * xi * (1.0 - xi) / (alpha_d_transition - alpha_b_transition)
    dxb_dalpha2 = - 6.0 * xi * (1.0 - xi) / (alpha_d_transition - alpha_b_transition)

    S_morph_b = - u2 * 6.0 / (jnp.pi * d_bubb**3) * (xb - 1.0 - alpha2 * dxb_dalpha2)
    S_morph_d = - u1 * 6.0 / (jnp.pi * d_drop**3) * (xd - 1.0 - alpha1 * dxd_dalpha2)

    # # Birth diameter = local diameter (minimal choice, prevents spurious drift)
    # K_morph_b = (6.0 * alpha2) / (jnp.pi * d_bubb**3) * dxb_dalpha2 * u2   # [1/m/s] × u_g
    # K_morph_d = (6.0 * alpha1) / (jnp.pi * d_drop**3) * dxd_dalpha2 * u1   # [1/m/s] × u_l

    # K_morph_d = 6.0 / (jnp.pi * d_drop**3) * (xb + alpha1 / (alpha_d_transition - alpha_b_transition) * 6.0 * xi * (1.0 - xi)) # the minus in the overall K is because we use alpha2 for the drop not alpha1
    # K_morph_b = -6.0 / (jnp.pi * d_bubb**3) * (xd + alpha2 / (alpha_d_transition - alpha_b_transition) * 6.0 * xi * (1.0 - xi))

    # jax.debug.print(
    #     "x={x:.4f} | K_morph_d={Sbr:.4e} | H_BR={H:.1f} | diff={diff:.4e} "
    #     "| nd_We={ndWe:.4e} | nd={nd:.4e} "
    #     "| d_drop={dd:.4e} | alpha1={a1:.4f}",
    #     x=x, Sbr=S_morph_d, H=H_BR, diff=diff_breakup,
    #     ndWe=nd_We, nd=nd,
    #     dd=d_drop, a1=alpha1,
    # )

    # Dispersed phase (liquid, phase 1): constant Nusselt number
    Nu_1 = 12
    k_1 = state1["k"]
    ht_11 = (k_1 * Nu_1) / d_drop
    
    # Continuous phase (gas, phase 2): Ranz-Marshall correlation
    ht_12 = ranz_marshall_htc_tfm(alpha1, state2, fluid2, relative_velocity=u_rel_abs, dp=d_drop) #Nb=args.bubble_number, Nd=args.droplet_number)
    ht_liquid_dispersed = (1 / ((1/ht_11) + (1/ht_12)))

    # Dispersed phase (gas, phase 2): constant Nusselt number
    Nu_2 = 12
    k_2 = state2["k"]
    ht_21 = (k_2 * Nu_2) / d_bubb

    # Continuous phase (liquid, phase 1): Ranz-Marshall correlation
    ht_22 = ranz_marshall_htc_tfm(alpha2, state1, fluid1, relative_velocity=u_rel_abs, dp=d_bubb) #Nb=args.bubble_number, Nd=args.droplet_number)
    ht_vapor_dispersed = (1 / ((1/ht_21) + (1/ht_22)))

    sigma_Q1 = (ht_liquid_dispersed * ai_d + ht_vapor_dispersed * ai_b) * (T2 - T1)
    sigma_Q2 = -sigma_Q1

    u_v = jnp.abs(u1 + u2) / 2

    # a_mix = get_speed_of_sound_mixture_stadke_tfm(alpha1, a1, rho1, alpha2, a2, rho2)
    a_mix = get_speed_of_sound_mixture_two_fluid(alpha1, a1, rho1, alpha2, a2, rho2, u2/u1)
    Ma_mix = u_mix / a_mix

    # Regime-dependent drag (Schiller-Naumann, switches bubbly/droplet/transition)
    Cd_bubb = 0.44
    Cd_drop = 0.44
    F_drag_vapor_liquid = 1/8 * (Cd_drop * ai_d * rho2 + Cd_bubb * ai_b * rho1) * u_rel_abs * (u1 - u2)


    # --- Build A matrix and b vector ---
    A_mat = jnp.array([
        # Row 1
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

        # Row 2
        [rho1 * u1, 0.0, alpha1 * u1, 0.0, alpha1 * rho1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

        # Row 3
        [0.0, rho2 * u2, 0.0, alpha2 * u2, 0.0, alpha2 * rho2, 0.0, 0.0, 0.0, 0.0, 0.0],

        # Row 4
        [0.0, 0.0, 0.0, 0.0, alpha1 * rho1 * u1, 0.0, alpha1, 0.0, 0.0, 0.0, 0.0],

        # Row 5
        [0.0, 0.0, 0.0, 0.0, 0.0, alpha2 * rho2 * u2, alpha2, 0.0, 0.0, 0.0, 0.0],

        # Row 6
        [0.0, 0.0, 0.0, 0.0, alpha1 * rho1 * u1**2, 0.0, 0.0, alpha1 * rho1 * u1, 0.0, 0.0, 0.0],

        # Row 7
        [0.0, 0.0, 0.0, 0.0, 0.0, alpha2 * rho2 * u2**2, 0.0, 0.0, alpha2 * rho2 * u2, 0.0, 0.0],

        # Row 8
        [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, (1 + G1) / a1**2, -(rho1 * G1) / a1**2, 0.0, 0.0, 0.0],

        # Row 9
        [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, (1 + G2) / a2**2, 0.0, -(rho2 * G2) / a2**2, 0.0, 0.0],

        # Row 10: bubble flux NB (state index 9) with morphological coupling to dα₂/dx
        [0.0, -A*S_morph_b, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],

        # Row 11: droplet flux ND (state index 10) with morphological coupling to dα₂/dx
        [0.0, -A*S_morph_d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],

    ])



    b_vec = jnp.asarray(
            [
            # Row 1: Mass conservation balance
            0.0,

            # Row 2: Phase 1 Momentum source
            - (1.0 / A) * alpha1 * rho1 * u1 * dAdx,

            # Row 3: Phase 2 Momentum source
            - (1.0 / A) * alpha2 * rho2 * u2 * dAdx,

            # Row 4: Phase 1 Velocity/Drag source
            -F_drag_vapor_liquid - alpha1 * tau_w1 * perimeter / A,

            # Row 5: Phase 2 Velocity/Drag source
            F_drag_vapor_liquid - alpha2 * tau_w2 * perimeter / A,

            # Row 6: Phase 1 Energy source
            sigma_Q1 - F_drag_vapor_liquid * u_v,

            # Row 7: Phase 2 Energy source
            sigma_Q2 + F_drag_vapor_liquid * u_v,

            # Row 8: Equation of State / Constraint 1
            0.0,

            # Row 9: Equation of State / Constraint 2
            0.0,

            # Row 10: Bubble number sources (coalescence sink + breakup source)
            S_RC_bubble + S_breakup_bubble,

            # Row 11: Droplet number sources (coalescence sink + breakup source)
            S_RC_droplet + S_breakup_droplet,
            ]
        )


    A_mat_copy = A_mat
    b_vec_copy = b_vec

    # # --- Old scalar S scaling (working but det blows up near critical point) ---
    A_mat = A_mat / S
    b_vec = b_vec / S

    # --- Fixed constant row/column scaling (scales computed once from inlet, passed via args) ---

    # TODO: Change the scaling considering the number of droplet and bubbles
    row_scales = args.row_scales
    col_scales = args.col_scales
    A_mat = A_mat / row_scales[:, None] / col_scales[None, :]
    b_vec = b_vec / row_scales

    # --- Determinants ---
    M1 = A_mat.at[:, 0].set(b_vec)
    M2 = A_mat.at[:, 1].set(b_vec)
    M3 = A_mat.at[:, 2].set(b_vec)
    M4 = A_mat.at[:, 3].set(b_vec)
    M5 = A_mat.at[:, 4].set(b_vec)
    M6 = A_mat.at[:, 5].set(b_vec)
    M7 = A_mat.at[:, 6].set(b_vec)
    M8 = A_mat.at[:, 7].set(b_vec)
    M9 = A_mat.at[:, 8].set(b_vec)
    M10 = A_mat.at[:, 9].set(b_vec)
    M11 = A_mat.at[:, 10].set(b_vec)

    det_D = det(A_mat)
    det_N1 = det(M1)
    det_N2 = det(M2)
    det_N3 = det(M3)
    det_N4 = det(M4)
    det_N5 = det(M5)
    det_N6 = det(M6)
    det_N7 = det(M7)
    det_N8 = det(M8)
    det_N9 = det(M9)
    det_N10 = det(M10)
    det_N11 = det(M11)

    N_raw = jnp.array([det_N1, det_N2, det_N3, det_N4, det_N5, det_N6, det_N7, det_N8, det_N9, det_N10, det_N11])
    # --- Column-scale correction for autonomous RHS ---
    # With row/col scaling: det(Ni_s) = det(Ni)*c_i / (prod(r)*prod(c))
    # but det(A_s) = det(A) / (prod(r)*prod(c)), so dy_i/dx = det(Ni_s)/det(A_s) = (det(Ni)/det(A))*c_i
    # To recover the correct dy_i/dx we divide det(Ni_s) by c_i:
    #   dy_i/dτ = det(Ni_s) / c_i,   dx/dτ = det(A_s)
    N = N_raw / col_scales
    # --- Normalize autonomous RHS by |det(A)| at inlet so RHS is O(1) near inlet ---
    det_ref = args.det_ref
    # --- Autonomous system: dx/dτ = D/det_ref, dy/dτ = N_i/det_ref (corrected) ---
    rhs = N / det_D
    rhs_autonomous = jnp.concatenate([jnp.array([det_D]), N]) / det_ref

    m_dot = alpha1 * u1 * rho1 * A + alpha2 * u2 * rho2 * A

    # jax.debug.print(
    #     "x={x:.4f} | α2={a2:.3f} | xb={xb:.4e} | nd={nd:.3e} | nd_we={nd_we:.3e} | "
    #     "d_bubb={db:.3e} | ai_b={aib:.3e} | S_RC={src:.3e} | S_br_d={sbd:.3e}",
    #     x=x, a2=alpha2, xb=xb, nd=nd, nd_we=nd_We,
    #     db=d_bubb, aib=ai_b, src=S_RC_bubble, sbd=S_breakup_droplet,
    # )

    # eigenvalues = compute_eigenvalues(A_mat_copy)

    # Export data
    out = {
        "x": x,
        "v1": u1,
        "v2": u2,
        "v_mix":u_mix,
        "alpha1":alpha1,
        "alpha2":alpha2,
        "rho1": rho1,
        "rho2": rho2,
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
        "Ma1": u1 / a1,
        # "h02": h02,
        # "p02": p02,
        # "T02": T02,
        # "d02": rho02,
        "rho_mix":rho_mix,
        "a_mix":a_mix,
        "Ma_mix":Ma_mix,
        "Ma2": u2 / a2,
        "Re1": Re1,
        "Re2": Re2,
        "f_D1": f_D1,
        "f_D2": f_D2,
        "tau_w1": tau_w1,
        "tau_w2": tau_w2,
        "A_mat": A_mat_copy,
        "b_vec": b_vec_copy,
        "D": det_D,
        "N": N,
        "m_dot": m_dot,
        "drag_force":F_drag_vapor_liquid,
        "d_bubble":d_bubb,
        "d_droplet":d_drop,
        "nb":nb,
        "nd":nd,
        "S_RC_bubble":S_RC_bubble,
        "NB":NB,
        "ND":ND,
        # "eigenvalues":eigenvalues
    }

    return {**out, **state1.to_dict(include_aliases=True),  **state2.to_dict(include_aliases=True)}
    # return {**out, **state}


@eqx.filter_jit
def nozzle_two_fluid_transport_autonomous_working(tau, Y, args):
    """
    Autonomous right-hand side for a two-fluid non-equilibrium nozzle model
    (fluid1=liquid/water, fluid2=gas/nitrogen) with no inter-phase mass transfer
    and separate velocities for each phase.

    Uses Cramer's rule on the linear system  A·(dY/dx) = b  rewritten as an
    autonomous system to handle sonic-point singularities:

        dx/dτ   = det(A)
        dY_i/dτ = det(A with column i replaced by b)

    State vector: Y = [x, alpha1, alpha2, rho1, rho2, u1, u2, p, h1, h2]
        x       – axial coordinate [m]
        alpha1  – volume fraction of fluid 1 (liquid)
        alpha2  – volume fraction of fluid 2 (gas)
        rho1    – density of fluid 1 [kg/m³]
        rho2    – density of fluid 2 [kg/m³]
        u1      – velocity of fluid 1 [m/s]
        u2      – velocity of fluid 2 [m/s]
        p       – shared pressure [Pa]
        h1      – specific enthalpy of fluid 1 [J/kg]
        h2      – specific enthalpy of fluid 2 [J/kg]
    """
    x, alpha1, alpha2, rho1, rho2, u1, u2, p, h1, h2, NB, ND = Y

    # v = jnp.where(jnp.abs(v) < 1e-6, jnp.sign(v)*1e-6, v)
    # Debug print using jax.debug.print
    # jax.debug.print(
    # "x={:.2f}, alpha1={:.4e}, alpha2={:.4e}, rho1={:.4e}, rho2={:.4e}, u1={:.4e}, u2={:.4e}, p={:.4e}, h1={:.4e}, h2={:.4e}",
    # x, alpha1, alpha2, rho1, rho2, u1, u2, p, h1, h2
    # )   

    # --- Geometry / model parameters ---
    fluid1 = args.fluid1 # water
    fluid2 = args.fluid2 # nitrogen
    L = args.length
    eps_wall = args.roughness
    wall_friction = args.wall_friction
    heat_transfer = args.heat_transfer

    # --- Thermodynamic state fluid 1 ---
    alpha1 = jnp.clip(alpha1, 1e-12, 1-1e-12)
    alpha2 = jnp.clip(alpha2, 1e-12, 1-1e-12)
    
    # h1 = jnp.clip(h1, -1e5, 1e7)
    # h2 = jnp.clip(h2, -1e5, 1e7)
    # p = jnp.clip(p, 1e2, 1e9)
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

    # TODO: find correct S

    # # --- Old scalar S scaling (working but det blows up near critical point) ---
    # # For stadke model
    S = ( rho1 * rho2 * u1 * u2) ** (1/11)
    # S = ( rho1 ** 2 * rho2 ** 2 * u1 ** 5 * u2 ** 5) ** (1/9)
    
    # # For Wallis model
    # S = (rho1**2 * rho2**2 * u1 * u2) ** (1/9)
    # S = (rho1**2 * rho2**2 * u1**2 * u2**2) ** (1/9)
    
    # # different "S" changes the result in the autonomous form. check which is the correct one

    # --- Build A matrix and b vector ---

    A_mat = jnp.array([
        # Row 1
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

        # Row 2
        [rho1 * u1, 0.0, alpha1 * u1, 0.0, alpha1 * rho1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

        # Row 3
        [0.0, rho2 * u2, 0.0, alpha2 * u2, 0.0, alpha2 * rho2, 0.0, 0.0, 0.0, 0.0, 0.0],

        # Row 4
        [0.0, 0.0, 0.0, 0.0, alpha1 * rho1 * u1, 0.0, alpha1, 0.0, 0.0, 0.0, 0.0],

        # Row 5
        [0.0, 0.0, 0.0, 0.0, 0.0, alpha2 * rho2 * u2, alpha2, 0.0, 0.0, 0.0, 0.0],

        # Row 6
        [0.0, 0.0, 0.0, 0.0, alpha1 * rho1 * u1**2, 0.0, 0.0, alpha1 * rho1 * u1, 0.0, 0.0, 0.0],

        # Row 7
        [0.0, 0.0, 0.0, 0.0, 0.0, alpha2 * rho2 * u2**2, 0.0, 0.0, alpha2 * rho2 * u2, 0.0, 0.0],

        # Row 8
        [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, (1 + G1) / a1**2, -(rho1 * G1) / a1**2, 0.0, 0.0, 0.0],

        # Row 9
        [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, (1 + G2) / a2**2, 0.0, -(rho2 * G2) / a2**2, 0.0, 0.0],

        # Row 10: bubble flux
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,],

        # Row 11: droplet flux
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,],

    ])
    
    # --- Geometry ---
    # A, dAdx, perimeter, radius = args.geometry(x, L) # When using symmetric geometry
    A, dAdx, perimeter, radius = args.geometry(x) # When using linear convergent divergent geometry
    diameter = 2 * radius

    # --- Wall heat transfer and friction ---
    Re1 = u1 * rho1 * diameter * jnp.sqrt(alpha1) / mu1  # sqrt(alpha) gives an effective hydraulic diameter per phase; non-standard
    f_D1 = get_friction_factor_haaland(Re1, eps_wall, diameter)
    tau_w1 = get_wall_viscous_stress(f_D1, rho1, u1)
    tau_w1 = tau_w1 * wall_friction

    Re2 = u2 * rho2 * diameter * jnp.sqrt(alpha2) / mu2  # sqrt(alpha) gives an effective hydraulic diameter per phase; non-standard
    f_D2 = get_friction_factor_haaland(Re2, eps_wall, diameter)
    tau_w2 = get_wall_viscous_stress(f_D2, rho2, u2)
    tau_w2 = tau_w2 * wall_friction

    rho_mix = alpha1 * rho1 + alpha2 * rho2
    u_mix = (alpha1 * rho1 * u1 + alpha2 * rho2 * u2) / rho_mix  # mass-averaged mixture velocity


    # --------------------------------------------------------------------- #
    # Interfacial area considering blending of two regimes
    We_crit_b = 2.3
    We_crit_d = 6.0
    sigma_WN = 0.0728 # N/m
    u_rel_abs = jnp.maximum(jnp.abs(u1 - u2), 1e-6)

    d_bubb_we = (We_crit_b * sigma_WN) / (rho1 * u_rel_abs**2)
    d_drop_we = (We_crit_d * sigma_WN) / (rho2 * u_rel_abs**2)
    
    nb = NB / (A * u2)
    nd = ND / (A * u1)

    alpha_b_transition=0.20
    alpha_d_transition=0.80
    alpha_max=0.80

    alpha_bubble_star = alpha_b_star(alpha2, alpha_b_transition, alpha_max)
    alpha_droplet_star = alpha_d_star(alpha2, alpha_d_transition, alpha_max)
    xd = smoothstep(alpha2, alpha_b_transition, alpha_d_transition)
    xd = jnp.maximum(xd, 1e-6)
    xb = 1.0 - xd

    alpha_bubble = alpha_bubble_star * xb
    alpha_droplet = alpha_droplet_star * xd

    # d_bubb = ((6 * alpha_bubble) / (jnp.pi * nb)) ** (1/3)
    # d_bubb = jnp.maximum(d_bubb, 5e-6)
    # d_bubb = jnp.maximum(d_bubb, d_bubb_we)
    # d_bubb = (6 * alpha_bubble / (jnp.pi * nb)  +  d_bubb_we**3 * (1.0 - xb)) ** (1.0/3.0)

    # Floor activates only when xb < xb_floor_thresh (very late in the transition)
    xb_floor_thresh = 0.1
    floor_weight = 1.0 - smoothstep(xb, 0.0, xb_floor_thresh)

    d_bubb_we_capped = jnp.minimum(d_bubb_we, 1e-3)
    d_bubb = (6 * alpha_bubble / (jnp.pi * nb)  +  d_bubb_we_capped**3 * floor_weight) ** (1.0/3.0)

    d_drop = ((6 * alpha_droplet) / (jnp.pi * nd)) ** (1/3)
    d_drop = jnp.maximum(d_drop, 5e-6)


    # # d_bubb = ((6 * alpha_bubble) / (jnp.pi * nb)) ** (1/3)
    # # d_bubb = jnp.maximum(d_bubb, 5e-6)
    # # d_bubb = jnp.maximum(d_bubb, d_bubb_we)
    # # d_bubb = (6 * alpha_bubble / (jnp.pi * nb)  +  d_bubb_we**3 * (1.0 - xb)) ** (1.0/3.0)

    # # Floor activates only when xb < xb_floor_thresh (very late in the transition)
    # xb_floor_thresh = 0.1
    # floor_weight = 1.0 - smoothstep(xb, 0.0, xb_floor_thresh)

    # d_bubb_we_capped = jnp.minimum(d_bubb_we, 1e-3)
    # d_bubb = (6 * alpha_bubble / (jnp.pi * nb)  +  d_bubb_we_capped**3 * floor_weight) ** (1.0/3.0)

    # d_drop = ((6 * alpha_droplet) / (jnp.pi * nd)) ** (1/3)
    # d_drop = jnp.maximum(d_drop, 5e-6)

    ai_b_star, ai_d_star = interfacial_area_dispersed_flows(alpha2, d_bubb, d_drop)
    # ai_b, ai_d = ai_b_star * xb, ai_d_star * xd
    ai_b, ai_d = ai_b_star, ai_d_star


    # Random collision source term
    alpha_max_RC = 0.81
    a_for_RC = jnp.minimum(alpha2, 0.95 * alpha_max_RC)  # use TRUE gas fraction, keep below packing
    a13    = a_for_RC ** (1.0/3.0)
    amax13 = alpha_max_RC ** (1.0/3.0)
    denom  = amax13 * (amax13 - a13)
    C = 3.0 # 3.0
    C_RC = 3.0 # 0.06
    Re_mix = rho_mix * u_mix * diameter / mu_mix
    f_tw = 0.316 / Re_mix**0.25 * (1.0 / (1.0 - alpha2))**0.25
    epsilon2 = f_tw * u_mix**3 / (2 *diameter)
    u_t2 = epsilon2**(1/3) * d_bubb**(1/3)
    arg     = C * amax13 * a13 / denom
    S_random_collision =  - A * C_RC * u_t2 * nb**2 * d_bubb**2 / denom * (1.0 - jnp.exp(-arg))
    # S_random_collision = 0.0

    # Breakup source term
    timescale_br = 1.0
    nb_We = (6 * alpha_bubble) / (jnp.pi * d_bubb_we**3)
    diff_breakup_b = nb_We - nb
    H_BR_B = jnp.heaviside(diff_breakup_b, 0.5)
    S_breakup_bubble = A * H_BR_B * diff_breakup_b / timescale_br
    S_breakup_bubble = 0.0
    
    # Breakup source term
    timescale_br = 1.0 # 1.0
    nd_We = (6 * alpha_droplet) / (jnp.pi * d_drop_we**3)
    diff_breakup = nd_We - nd
    H_BR = jnp.heaviside(diff_breakup, 0.5)
    S_breakup_droplet = A * H_BR * diff_breakup / timescale_br
    # S_breakup_droplet = 0.0

    jax.debug.print(
        "x={x:.4f} | S_breakup_droplet={Sbr:.4e} | H_BR={H:.1f} | diff={diff:.4e} "
        "| nd_We={ndWe:.4e} | nd={nd:.4e} "
        "| d_drop={dd:.4e} | alpha1={a1:.4f}",
        x=x, Sbr=S_breakup_droplet, H=H_BR, diff=diff_breakup,
        ndWe=nd_We, nd=nd,
        dd=d_drop, a1=alpha1,
    )

    # Dispersed phase (liquid, phase 1): constant Nusselt number
    Nu_1 = 12
    k_1 = state1["k"]
    ht_11 = (k_1 * Nu_1) / d_drop
    
    # Continuous phase (gas, phase 2): Ranz-Marshall correlation
    ht_12 = ranz_marshall_htc_tfm(alpha1, state2, fluid2, relative_velocity=u_rel_abs, dp=d_drop) #Nb=args.bubble_number, Nd=args.droplet_number)
    ht_liquid_dispersed = (1 / ((1/ht_11) + (1/ht_12)))

    # Dispersed phase (gas, phase 2): constant Nusselt number
    Nu_2 = 12
    k_2 = state2["k"]
    ht_21 = (k_2 * Nu_2) / d_bubb

    # Continuous phase (liquid, phase 1): Ranz-Marshall correlation
    ht_22 = ranz_marshall_htc_tfm(alpha2, state1, fluid1, relative_velocity=u_rel_abs, dp=d_bubb) #Nb=args.bubble_number, Nd=args.droplet_number)
    ht_vapor_dispersed = (1 / ((1/ht_21) + (1/ht_22)))

    sigma_Q1 = (ht_liquid_dispersed * ai_d + ht_vapor_dispersed * ai_b) * (T2 - T1)
    sigma_Q2 = -sigma_Q1

    u_v = jnp.abs(u1 + u2) / 2

    # a_mix = get_speed_of_sound_mixture_stadke_tfm(alpha1, a1, rho1, alpha2, a2, rho2)
    a_mix = get_speed_of_sound_mixture_two_fluid(alpha1, a1, rho1, alpha2, a2, rho2, u2/u1)
    Ma_mix = u_mix / a_mix

    # Regime-dependent drag (Schiller-Naumann, switches bubbly/droplet/transition)
    Cd_bubb = 0.44
    Cd_drop = 0.44
    F_drag_vapor_liquid = 1/8 * (Cd_drop * ai_d * rho2 + Cd_bubb * ai_b * rho1) * u_rel_abs * (u1 - u2)

    b_vec = jnp.asarray(
            [
            # Row 1: Mass conservation balance
            0.0,

            # Row 2: Phase 1 Momentum source
            - (1.0 / A) * alpha1 * rho1 * u1 * dAdx,

            # Row 3: Phase 2 Momentum source
            - (1.0 / A) * alpha2 * rho2 * u2 * dAdx,

            # Row 4: Phase 1 Velocity/Drag source
            -F_drag_vapor_liquid - alpha1 * tau_w1 * perimeter / A,

            # Row 5: Phase 2 Velocity/Drag source
            F_drag_vapor_liquid - alpha2 * tau_w2 * perimeter / A,

            # Row 6: Phase 1 Energy source
            sigma_Q1 - F_drag_vapor_liquid * u_v,

            # Row 7: Phase 2 Energy source
            sigma_Q2 + F_drag_vapor_liquid * u_v,

            # Row 8: Equation of State / Constraint 1
            0.0,

            # Row 9: Equation of State / Constraint 2
            0.0,

            # Row 10: Sources of bubble number decrease
            S_random_collision + S_breakup_bubble,

            # Row 11: Sources for number of droplet increase
            S_breakup_droplet,
            ]
        )


    A_mat_copy = A_mat
    b_vec_copy = b_vec

    # # --- Old scalar S scaling (working but det blows up near critical point) ---
    A_mat = A_mat / S
    b_vec = b_vec / S

    # --- Fixed constant row/column scaling (scales computed once from inlet, passed via args) ---

    # TODO: Change the scaling considering the number of droplet and bubbles
    row_scales = args.row_scales
    col_scales = args.col_scales
    A_mat = A_mat / row_scales[:, None] / col_scales[None, :]
    b_vec = b_vec / row_scales

    # --- Determinants ---
    M1 = A_mat.at[:, 0].set(b_vec)
    M2 = A_mat.at[:, 1].set(b_vec)
    M3 = A_mat.at[:, 2].set(b_vec)
    M4 = A_mat.at[:, 3].set(b_vec)
    M5 = A_mat.at[:, 4].set(b_vec)
    M6 = A_mat.at[:, 5].set(b_vec)
    M7 = A_mat.at[:, 6].set(b_vec)
    M8 = A_mat.at[:, 7].set(b_vec)
    M9 = A_mat.at[:, 8].set(b_vec)
    M10 = A_mat.at[:, 9].set(b_vec)
    M11 = A_mat.at[:, 10].set(b_vec)

    det_D = det(A_mat)
    det_N1 = det(M1)
    det_N2 = det(M2)
    det_N3 = det(M3)
    det_N4 = det(M4)
    det_N5 = det(M5)
    det_N6 = det(M6)
    det_N7 = det(M7)
    det_N8 = det(M8)
    det_N9 = det(M9)
    det_N10 = det(M10)
    det_N11 = det(M11)

    N_raw = jnp.array([det_N1, det_N2, det_N3, det_N4, det_N5, det_N6, det_N7, det_N8, det_N9, det_N10, det_N11])
    # --- Column-scale correction for autonomous RHS ---
    # With row/col scaling: det(Ni_s) = det(Ni)*c_i / (prod(r)*prod(c))
    # but det(A_s) = det(A) / (prod(r)*prod(c)), so dy_i/dx = det(Ni_s)/det(A_s) = (det(Ni)/det(A))*c_i
    # To recover the correct dy_i/dx we divide det(Ni_s) by c_i:
    #   dy_i/dτ = det(Ni_s) / c_i,   dx/dτ = det(A_s)
    N = N_raw / col_scales
    # --- Normalize autonomous RHS by |det(A)| at inlet so RHS is O(1) near inlet ---
    det_ref = args.det_ref
    # --- Autonomous system: dx/dτ = D/det_ref, dy/dτ = N_i/det_ref (corrected) ---
    rhs = N / det_D
    rhs_autonomous = jnp.concatenate([jnp.array([det_D]), N]) / det_ref

    m_dot = alpha1 * u1 * rho1 * A + alpha2 * u2 * rho2 * A

    # jax.debug.print(
    #     "x={x:.4f} | α2={a2:.3f} | xb={xb:.4e} | nd={nd:.3e} | nd_we={nd_we:.3e} | "
    #     "d_bubb={db:.3e} | ai_b={aib:.3e} | S_RC={src:.3e} | S_br_d={sbd:.3e}",
    #     x=x, a2=alpha2, xb=xb, nd=nd, nd_we=nd_We,
    #     db=d_bubb, aib=ai_b, src=S_random_collision, sbd=S_breakup_droplet,
    # )

    # eigenvalues = compute_eigenvalues(A_mat_copy)

    # Export data
    out = {
        "x": x,
        "v1": u1,
        "v2": u2,
        "v_mix":u_mix,
        "alpha1":alpha1,
        "alpha2":alpha2,
        "rho1": rho1,
        "rho2": rho2,
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
        "Ma1": u1 / a1,
        # "h02": h02,
        # "p02": p02,
        # "T02": T02,
        # "d02": rho02,
        "rho_mix":rho_mix,
        "a_mix":a_mix,
        "Ma_mix":Ma_mix,
        "Ma2": u2 / a2,
        "Re1": Re1,
        "Re2": Re2,
        "f_D1": f_D1,
        "f_D2": f_D2,
        "tau_w1": tau_w1,
        "tau_w2": tau_w2,
        "A_mat": A_mat_copy,
        "b_vec": b_vec_copy,
        "D": det_D,
        "N": N,
        "m_dot": m_dot,
        "drag_force":F_drag_vapor_liquid,
        "d_bubble":d_bubb,
        "d_droplet":d_drop,
        "nb":nb,
        "nd":nd,
        "S_random_collision":S_random_collision,
        "NB":NB,
        "ND":ND,
        # "eigenvalues":eigenvalues
    }

    return {**out, **state1.to_dict(include_aliases=True),  **state2.to_dict(include_aliases=True)}
    # return {**out, **state}



# ------------------------------------------------------------------------------
# One-component hyperbolic two-fluid model
# ------------------------------------------------------------------------------


def nozzle_two_fluid_hyperbolic_core(x, y, args):
    """Wrapper that adapts from (x, y) to autonomous form."""
    Y = jnp.concatenate([jnp.atleast_1d(x), y])
    return nozzle_two_fluid_hyperbolic_autonomous(0.0, Y, args)

@eqx.filter_jit
def nozzle_two_fluid_hyperbolic_autonomous(tau, Y, args):
    """
    Autonomous formulation of the nozzle equations:
        dx/dt   = det(A)
        dy_i/dt = det(A with column i replaced by b)
    v is secondary phase and liquid is secondary phase, this is designed for flashing problems.
    in case of condensation, juste define the fluds the other way
    State vector: Y = [x, v, rho, p]
    """
    x, p, ug, ul, alphag, hg, hl = Y

    k = 0.5

    alphal = 1.0 - alphag
    # jax.debug.print(
    #     "RAW Y: x={:.4f} αv={:.3e} ρv={:.2f} ρl={:.2f} u={:.4f} p={:.1f} hv={:.1f} hl={:.1f}",
    #     x, alphav, rhov, rhol, u, p, hv, hl
    # )

    # --- Geometry / model parameters ---
    fluidV = args.fluid1
    fluidL = args.fluid2
    eps_wall = args.roughness
    wall_friction = args.wall_friction
    Nb = args.bubble_number
    Nd = args.droplet_number
    alpha_b = args.alpha_b
    alpha_d = args.alpha_d

    # --- Clamp volume fractions ---
    ALPHA_MIN = 1e-6
    # alphal = 1.0 - alphav
    alphav_safe = jnp.clip(alphag, ALPHA_MIN, 1.0 - ALPHA_MIN)
    alphal_safe = jnp.clip(alphal, ALPHA_MIN, 1.0 - ALPHA_MIN)

    # --- Saturation properties (needed for mass transfer) ---
    liquid_sat = fluidL.get_state_saturation(p)
    vapor_sat = fluidV.get_state_saturation(p)

    h_sat_l = liquid_sat["enthalpy"]
    h_sat_g = vapor_sat["enthalpy"]
    Tsat = liquid_sat["temperature"]

    # --- Thermodynamic states ---
    statel = fluidL.get_state(jxp.HmassP_INPUTS, hl, p)
    Tl = statel["T"]
    rhol = statel["density"]
    al = statel["a"]

    # --- Two-phase flag based on enthalpy and temperature ---
    is_two_phase = (hl >= h_sat_l) & (Tl >= Tsat)

    def _vapor_from_actual(_args):
        _hv, _p = _args
        st = fluidV.get_state(jxp.HmassP_INPUTS, _hv, _p)
        return (st["T"], st["s"], st["a"], st["mu"], st["gruneisen"], st["conductivity"], st["density"])

    def _vapor_from_sat(_args):
        _hv, _p = _args  # unused, but signature must match
        return (vapor_sat["temperature"], vapor_sat["entropy"],
                vapor_sat["speed_of_sound"], vapor_sat["viscosity"],
                vapor_sat["gruneisen"], vapor_sat["conductivity"],
                vapor_sat["density"])

    Tg, sg, ag, mug, Gg, k_g, rhog = jax.lax.cond(
        is_two_phase, _vapor_from_actual, _vapor_from_sat, (hg, p)
    )

    sl = statel["s"]
    al = statel["a"]
    mul = statel["mu"]
    Gl = statel["gruneisen"]

    props_l = rhol, al
    props_g = rhog, ag

    cq = common_quantities(Y[1:], props_l, props_g, k)


    # --- Mixture properties ---
    rho_mix = rhog * alphav_safe + rhol * alphal_safe

    # # --- Old scalar S scaling (working but det blows up near critical point) ---
    # # For stadke model
    S = ( rhog * rhol * ug * ul) ** (1/9)


    A_mat = G_h_matrix(Y[1:], props_l, props_g, cq)
    
    # --- Geometry ---
    # A, dAdx, perimeter, radius = args.geometry(x, L) # When using symmetric geometry
    A, dAdx, perimeter, radius = args.geometry(x) # When using linear convergent divergent geometry
    diameter = 2 * radius

    dAdx = dAdx / A

    cl = two_fluid_closure(Y[1:], rhog, Tg, rhol, Tl, fluidV, fluidL, Tsat, h_sat_g, h_sat_l, dAdx, Nb=Nb, Nd=Nd, alpha_b=alpha_b, alpha_d=alpha_d )


    eos_g = rhog, ag, Tg, Gg
    eos_l = rhol, al, Tl, Gl

    b_vec = D_h_primitive(Y[1:], eos_g, eos_l, cq, cl, k)

    A_mat_copy = A_mat
    b_vec_copy = b_vec

    # # --- Old scalar S scaling (working but det blows up near critical point) ---
    A_mat = A_mat / S
    b_vec = b_vec / S

    # --- Fixed constant row/column scaling (scales computed once from inlet, passed via args) ---
    row_scales = args.row_scales
    col_scales = args.col_scales

    A_mat = A_mat / row_scales[:, None] / col_scales[None, :]
    b_vec = b_vec / row_scales

    # --- Determinants ---
    M1 = A_mat.at[:, 0].set(b_vec)
    M2 = A_mat.at[:, 1].set(b_vec)
    M3 = A_mat.at[:, 2].set(b_vec)
    M4 = A_mat.at[:, 3].set(b_vec)
    M5 = A_mat.at[:, 4].set(b_vec)
    M6 = A_mat.at[:, 5].set(b_vec)

    det_D = det(A_mat)
    det_N1 = det(M1)
    det_N2 = det(M2)
    det_N3 = det(M3)
    det_N4 = det(M4)
    det_N5 = det(M5)
    det_N6 = det(M6)

    N_raw = jnp.array([det_N1, det_N2, det_N3, det_N4, det_N5, det_N6])
    # --- Column-scale correction for autonomous RHS ---
    N = N_raw / col_scales

    # ── Extra abbreviations for eigenstructure (§13.1) ──────────────────────
    rg_hat  = cq.rg_hat;  rl_hat  = cq.rl_hat;  rh2    = cq.rh2
    rho_s   = cq.rho_s;   du      = cq.du;     a0sq   = cq.a0sq

    # k*rho_mix is not stored in cq; derive from rg_hat = rg + k*rho_mix.
    k_rho_mix  = rg_hat - rhog
    rho_hat_s  = rho_s + k_rho_mix          # hat_rho_s = rho_s + k*rho_mix
    # rh         = jnp.sqrt(rh2)              # hat_rho = sqrt(hat_rho^2)

    a_tilde_sq = (rho_hat_s / rho_s) * (rhog * rhol / rh2) * a0sq
 
    delta_a_sq = alphag * alphal * rhog * rhol * rg_hat * rl_hat / rh2**2 * du**2
    a_sq       = a_tilde_sq - delta_a_sq
    a_mix          = jnp.sqrt(jnp.maximum(a_sq, 0.0))  # real mixture sound speed

    hat_ag = alphag * rhog * rl_hat / rh2         # volume-fraction weight for u_g
    hat_al = alphal * rhol * rg_hat / rh2         # volume-fraction weight for u_l
    u_mix  = hat_ag * ug + hat_al * ul    # mixture velocity

    Ma_mix = u_mix / a_mix

    # --- Normalize autonomous RHS by |det(A)| at inlet so RHS is O(1) near inlet ---
    det_ref = args.det_ref
    # --- Autonomous system: dx/dτ = D/det_ref, dy/dτ = N_i/det_ref (corrected) ---
    rhs = N / det_D
    rhs_autonomous = jnp.concatenate([jnp.array([det_D]), N]) / det_ref

    m_dot = alphag * ug * rhog * A + alphal * ul * rhol * A

    # Export data
    out = {
        "x": x,
        "v1": ug,
        "v2": ul,
        "alpha1":alphag,
        "alpha2":alphal,
        "rho1": rhog,
        "rho2": rhol,
        "p": p,
        "h1":hg,
        "h2":hl,
        "T1":Tg,
        "T2":Tl,
        "s1":sg,
        "s2":sl,
        "rhs": rhs,
        "rhs_autonomous": rhs_autonomous,
        "area": A,
        "dAdx": dAdx,
        "diameter": diameter,
        "perimeter": perimeter,
        "rho_mix":rho_mix,
        "Ma_mix": Ma_mix,
        "A_mat": A_mat_copy,
        "b_vec": b_vec_copy,
        "D": det_D,
        "N": N,
        "m_dot": m_dot,
        "a_mix": a_mix,
        "u_mix": u_mix,
        # "eigenvalues":eigenvalues
    }

    return {**out}
    # return {**out, **state}


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
    """
    name = fluid_name.lower()
    if name == "water":
        theta = jnp.clip(1.0 - T / 647.096, 0.0, 1.0)
        sigma = 0.2358 * theta**1.256 * (1.0 - 0.625 * theta)
    elif name == "nitrogen":
        theta = jnp.clip(1.0 - T / 126.192, 0.0, 1.0)
        sigma = 0.02898 * theta**1.2457
    elif name in ("co2", "carbondioxide", "carbon dioxide"):
        theta = jnp.clip(1.0 - T / 304.128, 0.0, 1.0)
        sigma = 0.07863 * theta**1.2845
    else:
        raise ValueError(f"No surface tension correlation for fluid '{fluid_name}'. "
                         f"Supported: water, nitrogen, CO2.")
    return jnp.maximum(sigma, 1e-10)

def ranz_marshall_htc_tfm(
        alpha_dispersed,
        state_continuous,
        fluid_continuous,
        relative_velocity,
        dp,
        We=8.0,
        Nb=1e8,
        Nd=1e8,
):
    """
    Continuous-phase HTC for dispersed particles (Ranz-Marshall 1952).

    Same as ranz_marshall_htc_two_components from HNEM, but uses
    the actual relative velocity instead of the Weber-estimated slip.
    All thermophysical properties are taken from the CONTINUOUS phase.

    Parameters
    ----------
    alpha_dispersed : float
        Volume fraction of the dispersed phase.
    state_continuous : FluidState
        Thermodynamic state of the continuous phase.
    fluid_continuous : Fluid
        Fluid object for the continuous phase (needs fluid_name for surface tension).
    relative_velocity : float
        Absolute relative velocity between phases |u1 - u2| [m/s].
    We : float
        Critical Weber number (default 8.0). Reserved, not used when
        relative_velocity is provided directly.
    Nb : float
        Number density for interfacial area model (default 1e10).
    """
    # --- Continuous-phase properties ---
    rho_c = state_continuous["rho"]
    mu_c  = state_continuous["mu"]
    k_c   = state_continuous["k"]
    cp_c  = state_continuous["cp"]

    # # --- Dispersed-phase diameter from interfacial-area model ---
    # _, dp = interfacial_area(alpha_dispersed, Nb=Nb, Nd=Nd)

    # --- Re and Pr use continuous-phase properties ---
    Re = rho_c * relative_velocity * dp / mu_c
    Pr = cp_c * mu_c / k_c

    Nu = 2.0 + 0.6 * Re**0.5 * Pr**(1.0 / 3.0)

    htc = k_c * Nu / dp

    return htc


def ranz_marshall_htc_two_fluid(
        state_continuous,
        relative_velocity,
        d_dispersed
):
    """
    Continuous-phase HTC for dispersed particles (Ranz-Marshall 1952).

    Same as ranz_marshall_htc_two_components from HNEM, but uses
    the actual relative velocity instead of the Weber-estimated slip.
    All thermophysical properties are taken from the CONTINUOUS phase.

    Parameters
    ----------
    alpha_dispersed : float
        Volume fraction of the dispersed phase.
    state_continuous : FluidState
        Thermodynamic state of the continuous phase.
    fluid_continuous : Fluid
        Fluid object for the continuous phase (needs fluid_name for surface tension).
    relative_velocity : float
        Absolute relative velocity between phases |u1 - u2| [m/s].
    We : float
        Critical Weber number (default 8.0). Reserved, not used when
        relative_velocity is provided directly.
    Nb : float
        Number density for interfacial area model (default 1e10).
    """
    # --- Continuous-phase properties ---
    rho_c = state_continuous["rho"]
    mu_c  = state_continuous["mu"]
    k_c   = state_continuous["k"]
    cp_c  = state_continuous["cp"]

    # --- Re and Pr use continuous-phase properties ---
    Re = rho_c * jnp.abs(relative_velocity) * d_dispersed / mu_c
    Pr = cp_c * mu_c / k_c

    Nu = 2.0 + 0.6 * Re**0.5 * Pr**(1.0 / 3.0)

    htc = k_c * Nu / d_dispersed

    return htc


def interfacial_area_old(alpha, alpha_b=0.3, alpha_d=0.7, Nb=1e8, Nd=1e8):
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
        ai = A_ib_b + (A_id_d - A_ib_b) * (alpha - alpha_b) / (alpha_d - alpha_b)
        N = Nb + (Nd - Nb) * (alpha - alpha_b) / (alpha_d - alpha_b)
        return ai, N

    # Nested condition: alpha <= alpha_b → bubbly, alpha >= alpha_d → droplet, else → transition
    ai, N = jax.lax.cond(
        alpha <= alpha_b,
        bubbly,
        lambda _: jax.lax.cond(alpha >= alpha_d, droplet, transition, None),
        None
    )

    N = jnp.maximum(N, 1e-10)

    d = jnp.sqrt(ai / (N * jnp.pi))
    d = jnp.clip(d, 1e-7, 1.0)

    # Recompute ai from the clamped diameter for consistency
    ai = N * jnp.pi * d**2

    return ai, d

def interfacial_area(
    alpha,
    alpha_b=0.3, # 0.3 was the original
    alpha_d=0.7,
    Nb=1e10,
    Nd=1e10,
    d_min_b=1e-6,
    d_min_d=1e-6,
):
    """
    TA-BD interfacial area density with nucleation and breakup floors.

    Five regions as a function of vapor volume fraction alpha:
      1. alpha <= alpha_b_star         : nucleation floor (constant ai, d = d_min_b)
      2. alpha_b_star < alpha <= 0.3   : bubbly regime
      3. 0.3 < alpha < 0.7             : linear transition
      4. 0.7 <= alpha < alpha_d_star   : droplet regime
      5. alpha >= alpha_d_star         : breakup floor (constant ai, d = d_min_d)

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
    ai : scalar
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
        ai = Nb * jnp.pi * d_min_b**2
        d = d_min_b
        return ai, d

    def bubbly(_):
        ai = (6.0 * alpha) ** (2.0 / 3.0) * (jnp.pi * Nb) ** (1.0 / 3.0)
        d = (6.0 * alpha / (jnp.pi * Nb)) ** (1.0 / 3.0)
        return ai, d

    def transition(_):
        w = (alpha - alpha_b) / (alpha_d - alpha_b)
        ai = Ai_b_at_03 + (Ai_d_at_07 - Ai_b_at_03) * w
        d = d_b_at_03 + (d_d_at_07 - d_b_at_03) * w
        return ai, d

    def droplet(_):
        ai = (6.0 * (1.0 - alpha)) ** (2.0 / 3.0) * (jnp.pi * Nd) ** (1.0 / 3.0)
        d = (6.0 * (1.0 - alpha) / (jnp.pi * Nd)) ** (1.0 / 3.0)
        return ai, d

    def breakup_floor(_):
        ai = Nd * jnp.pi * d_min_d**2
        d = d_min_d
        return ai, d

    # Nested lax.cond cascade — walks through the five branches in order.
    # Structure: if alpha <= alpha_b_star -> nucleation_floor
    #            elif alpha <= alpha_b    -> bubbly
    #            elif alpha <  alpha_d    -> transition
    #            elif alpha <  alpha_d_star -> droplet
    #            else                     -> breakup_floor
    ai, d = jax.lax.cond(
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

    return ai, d



# @jit
def get_speed_of_sound_mixture(G1, alpha1, a1, rho1, G2, alpha2, a2, rho2):
    """
    Calculate the mixture speed of sound.
    
    Parameters:
    G1, G2 : float or jax array
        Gruneisen coefficients for each component
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
    term1 = -G1 * alpha1 / a1**2
    term2 = G1 * alpha1 / a1**2 * (rho_m / rho1)
    term3 = -G2 * alpha2 / a2**2
    term4 = G2 * alpha2 / a2**2 * (rho_m / rho2)
    term5 = alpha1 / a1**2 * (rho_m / rho1)
    term6 = alpha2 / a2**2 * (rho_m / rho2)
    
    denominator = term1 + term2 + term3 + term4 + term5 + term6
    
    return jnp.sqrt(1.0 / denominator)

def get_speed_of_sound_mixture_two_fluid_old(alpha1, a1, rho1, alpha2, a2, rho2):
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

    return jnp.sqrt(1.0 / denominator)


def get_speed_of_sound_mixture_two_fluid_frozen(alpha1, a1, rho1, alpha2, a2, rho2):
    """
    Mixture speed of sound from Städtke eq. (4.104).

    Here phase 1 = gas (g), phase 2 = liquid (l).
    Note the cross-weighting: numerator uses rho of the OPPOSITE phase,
    denominator also uses rho of the OPPOSITE phase divided by own a².
    """
    numerator   = alpha1 * rho2  +  alpha2 * rho1
    denominator = (alpha1 * rho2) / a1**2  +  (alpha2 * rho1) / a2**2

    return jnp.sqrt(numerator / denominator)


def get_speed_of_sound_mixture_two_fluid(alpha1, a1, rho1, alpha2, a2, rho2, S):
    """
    Mixture speed of sound with slip correction.

    Parameters
    ----------
    alpha1, alpha2 : volume fractions of phase 1 and 2
    a1, a2         : speeds of sound of phase 1 and 2
    rho1, rho2     : densities of phase 1 and 2
    S              : slip ratio  u2 / u1

    Formula
    -------
    a_m = ( (α1 ρ1 + α2 ρ2 S) / (α1 ρ1 + α2 ρ2) )
          * sqrt( (α1 ρ2 + α2 ρ1 / S²) / (α1 ρ2 / a1² + α2 ρ1 / a2²) )
    """
    rho_mix   = alpha1 * rho1 + alpha2 * rho2
    prefactor = (alpha1 * rho1 + alpha2 * rho2 * S) / rho_mix

    numerator   = alpha1 * rho2 + alpha2 * rho1 / S**2
    denominator = alpha1 * rho2 / a1**2 + alpha2 * rho1 / a2**2

    return prefactor * jnp.sqrt(numerator / denominator)


def calculate_Cd_IshiiZuber(rho_l, rho_v, mu_v, u_rel, sauter_mean_diameter, sigma, g=9.81):
    """
    Calculates Cd using Ishii-Zuber formulation.
    Phase 1: Liquid (dispersed droplets)
    Phase 2: Vapor (continuous gas)

    Parameters
    ----------
    rho_l : float  - Density of the liquid (dispersed) phase [kg/m³]
    rho_v : float  - Density of the vapor (continuous) phase [kg/m³]
    mu_v  : float  - Dynamic viscosity of the vapor (continuous) phase [Pa·s]
    u_rel : float  - Absolute relative velocity |u1 - u2| [m/s]
    sauter_mean_diameter : float - Mean droplet diameter [m]
    sigma : float  - Surface tension [N/m]
    g     : float  - Gravitational acceleration [m/s²]
    """
    d_l = sauter_mean_diameter
    # 1. Particle Reynolds number (continuous phase)
    Re = (rho_v * u_rel * d_l) / mu_v

    # 2. Eötvös number (buoyancy vs surface tension)
    Eo = (g * jnp.abs(rho_l - rho_v) * d_l**2) / sigma

    # 3. Viscous regime (Schiller-Naumann)
    Cd_viscous = (24.0 / Re) * (1.0 + 0.1 * Re**0.75)

    # 4. Distorted regime (ellipsoidal)
    Cd_distorted = (2.0 / 3.0) * jnp.sqrt(Eo)

    # 5. Spherical cap regime (upper bound)
    Cd_cap = 8.0 / 3.0

    # Select dominant regime, capped by spherical cap
    Cd = jnp.maximum(Cd_viscous, Cd_distorted)
    Cd = jnp.minimum(Cd, Cd_cap)

    return Cd

# TODO: this is wrong. I am giving 1 to gas and 2 to liquid for the one component case. this means that the solver is giving wrong densities to the drag forces. need to double check. Droplet regime should be rho1 not rho2 This is opposite in the two component case. betgter to have two distinguished functions that calculates differently based on which is the liquid and which the vapor
def calculate_drag_force(alpha1, alpha2, rho1, rho2, mu1, mu2, u1, u2, d,
                         alpha_b=0.3, alpha_d=0.7):
    """
    Regime-dependent drag force per unit volume (force on gas due to liquid).

    Uses lax.cond to switch between bubbly, transition, and droplet regimes
    based on the gas volume fraction (alpha2).

    - Bubbly  (alpha2 <= alpha_b): gas dispersed in liquid
    - Droplet (alpha2 >= alpha_d): liquid dispersed in gas
    - Transition: linear interpolation between the two

    Returns F such that:
      - Phase 1 (liquid) momentum source: -F
      - Phase 2 (gas)    momentum source: +F

    Parameters
    ----------
    alpha1, alpha2 : volume fractions (liquid, gas)
    rho1, rho2     : densities (liquid, gas)
    mu1, mu2       : dynamic viscosities (liquid, gas)
    u1, u2         : velocities (liquid, gas)
    d              : characteristic diameter of dispersed structures
    alpha_b        : upper bound of bubbly regime
    alpha_d        : lower bound of droplet regime
    """
    u_rel = u1 - u2
    u_rel_abs = jnp.abs(u_rel)

    # Schiller-Naumann drag coefficient for each regime
    def _Cd_SN(rho_c, mu_c):
        Re = rho_c * u_rel_abs * d / jnp.maximum(mu_c, 1e-30)
        return (24.0 / jnp.maximum(Re, 1e-12)) * (1.0 + 0.15 * Re**0.687)

    prefactor = (3.0 / 4.0) / d * u_rel_abs * u_rel

    def bubbly(_):
        # Gas dispersed in liquid: alpha_disp = alpha2, rho_cont = rho1
        C_d = _Cd_SN(rho1, mu1)
        return prefactor * C_d * alpha2 * rho1

    def droplet(_):
        # Liquid dispersed in gas: alpha_disp = alpha1, rho_cont = rho2
        C_d = _Cd_SN(rho2, mu2)
        return prefactor * C_d * alpha1 * rho2

    def transition(_):
        # Endpoints frozen at regime boundaries to anchor the interpolation.
        # Weighting forces (= Cd·alpha_boundary·rho_c) is correct here because
        # the continuous phase and volume fraction both change across the transition.
        F_b = prefactor * _Cd_SN(rho1, mu1) * alpha_b * rho1           # bubbly  at alpha2 = alpha_b
        F_d = prefactor * _Cd_SN(rho2, mu2) * (1.0 - alpha_d) * rho2  # droplet at alpha2 = alpha_d
        w = (alpha2 - alpha_b) / (alpha_d - alpha_b)
        return F_b + (F_d - F_b) * w

    F = jax.lax.cond(
        alpha2 <= alpha_b,
        bubbly,
        lambda _: jax.lax.cond(alpha2 >= alpha_d, droplet, transition, None),
        None,
    )

    return F


def calculate_drag_force_one_component(
    alpha1, alpha2, rho1, rho2, mu1, mu2, u1, u2, d,
    alpha_b=0.3, alpha_d=0.7,
):
    """
    Regime-dependent drag force per unit volume for the one-component case.

    Convention: phase 1 = vapor (gas), phase 2 = liquid.
      F > 0 means force on vapor in the direction of u2 - u1.

    Parameters
    ----------
    alpha1, alpha2 : volume fractions (vapor, liquid)
    rho1, rho2     : densities (vapor, liquid)
    mu1, mu2       : dynamic viscosities (vapor, liquid)
    u1, u2         : velocities (vapor, liquid)
    d              : characteristic diameter of dispersed structures
    alpha_b        : upper bound of bubbly regime (vapor fraction)
    alpha_d        : lower bound of droplet regime (vapor fraction)
    """
    u_rel = u1 - u2
    u_rel_abs = jnp.abs(u_rel)

    def _Cd_SN(rho_c, mu_c):
        Re = rho_c * u_rel_abs * d / jnp.maximum(mu_c, 1e-30)
        return (24.0 / jnp.maximum(Re, 1e-12)) * (1.0 + 0.15 * Re**0.687)

    prefactor = (3.0 / 4.0) / d * u_rel_abs * u_rel

    def bubbly(_):
        # Vapor dispersed in liquid: alpha_disp = alpha1 (vapor), rho_cont = rho2 (liquid)
        C_d = _Cd_SN(rho2, mu2)
        return prefactor * C_d * alpha1 * rho2

    def droplet(_):
        # Liquid dispersed in vapor: alpha_disp = alpha2 (liquid), rho_cont = rho1 (vapor)
        C_d = _Cd_SN(rho1, mu1)
        return prefactor * C_d * alpha2 * rho1

    def transition(_):
        F_b = prefactor * _Cd_SN(rho2, mu2) * alpha_b * rho2           # bubbly at alpha1 = alpha_b
        F_d = prefactor * _Cd_SN(rho1, mu1) * (1.0 - alpha_d) * rho1  # droplet at alpha1 = alpha_d
        w = (alpha1 - alpha_b) / (alpha_d - alpha_b)
        return F_b + (F_d - F_b) * w

    return jax.lax.cond(
        alpha1 <= alpha_b,
        bubbly,
        lambda _: jax.lax.cond(alpha1 >= alpha_d, droplet, transition, None),
        None,
    )


def calculate_drag_force_ishii_zuber_one_component(
    alpha1, alpha2, rho1, rho2, mu1, mu2, u1, u2, d,
    sigma, g=9.81,
    alpha_b=0.3, alpha_d=0.7,
):
    """
    Ishii-Zuber drag force for the one-component case (smooth blend, no lax.cond).

    Convention: phase 1 = vapor (gas), phase 2 = liquid.
      F > 0 means force on vapor in the direction of u2 - u1.
    """
    u_rel = u1 - u2
    u_rel_abs = jnp.abs(u_rel)

    def _Cd_IZ(rho_c, mu_c, rho_d, mu_d, alpha_disp):
        one_minus_a = jnp.maximum(1.0 - alpha_disp, 1e-6)
        mu_star = (mu_d + 0.4 * mu_c) / jnp.maximum(mu_d + mu_c, 1e-30)
        mu_m    = mu_c * one_minus_a ** (-2.5 * mu_star)
        Re_m    = rho_c * u_rel_abs * d / jnp.maximum(mu_m, 1e-30)
        Eo      = g * jnp.abs(rho_c - rho_d) * d * d / jnp.maximum(sigma, 1e-30)
        Cd_visc = (24.0 / jnp.maximum(Re_m, 1e-12)) * (1.0 + 0.1 * Re_m ** 0.75)
        f       = jnp.maximum(one_minus_a ** 1.5, 1e-12)
        Cd_dist = (2.0 / 3.0) * jnp.sqrt(Eo) * (
            (1.0 + 17.67 * f ** (6.0 / 7.0)) / (18.67 * f)
        ) ** 2
        Cd_cap  = (8.0 / 3.0) * one_minus_a ** 2
        return jnp.maximum(Cd_visc, jnp.minimum(Cd_dist, Cd_cap))

    prefactor = (3.0 / 4.0) / d * u_rel_abs * u_rel

    # Bubbly: vapor (1) dispersed in liquid (2)
    Cd_bub = _Cd_IZ(rho2, mu2, rho1, mu1, alpha1)
    F_bub  = prefactor * Cd_bub * alpha1 * rho2

    # Droplet: liquid (2) dispersed in vapor (1)
    Cd_drop = _Cd_IZ(rho1, mu1, rho2, mu2, alpha2)
    F_drop  = prefactor * Cd_drop * alpha2 * rho1

    # Blend on alpha1 (vapor fraction): 0 = bubbly, 1 = droplet
    t = jnp.clip((alpha1 - alpha_b) / (alpha_d - alpha_b), 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)
    return (1.0 - w) * F_bub + w * F_drop


def calculate_drag_force_ishii_zuber(
    alpha1, alpha2, rho1, rho2, mu1, mu2, u1, u2, d,
    sigma, g=9.81,
    alpha_b=0.3, alpha_d=0.7,
):
    """
    Ishii-Zuber (1979) regime-dependent drag force per unit volume.

    Smoothly blends the bubbly and droplet endpoints via a C^1 Hermite
    smoothstep weight on alpha2, replacing the lax.cond branch cascade.
    The asymptotic regimes (alpha2 <= alpha_b and alpha2 >= alpha_d) reduce
    to the original bubbly/droplet expressions; the transition region is
    a smooth blend of the two physical formulas (continuous force AND
    continuous slope), which removes the RHS kinks at the regime boundaries
    that were causing implicit-solver step rejection.

    Convention (matches `calculate_drag_force` docstring):
      phase 1 = liquid, phase 2 = gas
      F > 0 means force on gas in the direction of u1 - u2.

    Extra parameters
    ----------------
    sigma : surface tension (N/m)
    g     : gravitational acceleration (m/s^2)
    """
    u_rel = u1 - u2
    u_rel_abs = jnp.abs(u_rel)

    def _Cd_IZ(rho_c, mu_c, rho_d, mu_d, alpha_disp):
        one_minus_a = jnp.maximum(1.0 - alpha_disp, 1e-6)

        # Ishii-Zuber mixture viscosity:
        #   mu_m = mu_c (1 - alpha_d)^(-2.5 mu*)
        #   mu*  = (mu_d + 0.4 mu_c) / (mu_d + mu_c)
        mu_star = (mu_d + 0.4 * mu_c) / jnp.maximum(mu_d + mu_c, 1e-30)
        mu_m = mu_c * one_minus_a ** (-2.5 * mu_star)

        # Mixture Reynolds and Eötvös numbers
        Re_m = rho_c * u_rel_abs * d / jnp.maximum(mu_m, 1e-30)
        Eo   = g * jnp.abs(rho_c - rho_d) * d * d / jnp.maximum(sigma, 1e-30)

        # --- Viscous (undistorted) regime: Ishii-Zuber form ---
        Cd_visc = (24.0 / jnp.maximum(Re_m, 1e-12)) * (1.0 + 0.1 * Re_m ** 0.75)

        # --- Distorted-particle regime: surface-tension controlled ---
        # f(alpha) is the swarm-hindering factor. Common bubbly form is
        # (1 - alpha_disp)^1.5; the original I-Z form is
        # (mu_c/mu_m)(1-alpha_disp)^0.5. They agree at dilute limits and
        # differ modestly at high dispersed fractions.
        f = jnp.maximum(one_minus_a ** 1.5, 1e-12)
        Cd_dist = (2.0 / 3.0) * jnp.sqrt(Eo) * (
            (1.0 + 17.67 * f ** (6.0 / 7.0)) / (18.67 * f)
        ) ** 2

        # --- Churn-turbulent / spherical-cap plateau ---
        Cd_cap = (8.0 / 3.0) * one_minus_a ** 2

        return jnp.maximum(Cd_visc, jnp.minimum(Cd_dist, Cd_cap))

    prefactor = (3.0 / 4.0) / d * u_rel_abs * u_rel

    # Bubbly endpoint: gas dispersed in liquid (continuous = 1, dispersed = 2)
    Cd_bub = _Cd_IZ(rho1, mu1, rho2, mu2, alpha2)
    F_bub  = prefactor * Cd_bub * alpha2 * rho1

    # Droplet endpoint: liquid dispersed in gas (continuous = 2, dispersed = 1)
    Cd_drop = _Cd_IZ(rho2, mu2, rho1, mu1, alpha1)
    F_drop  = prefactor * Cd_drop * alpha1 * rho2

    # Hermite smoothstep blend on alpha2:
    #   w = 0 for alpha2 <= alpha_b  -> pure bubbly
    #   w = 1 for alpha2 >= alpha_d  -> pure droplet
    #   C^1 continuous everywhere (no slope jump at the boundaries)
    t = jnp.clip((alpha2 - alpha_b) / (alpha_d - alpha_b), 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)

    return (1.0 - w) * F_bub + w * F_drop


def compute_eigenvalues(A_mat):
    """
    Given a system matrix A_mat, returns all eigenvalues
    sorted by real part.
    """
    eigs = jnp.linalg.eigvals(A_mat)
    idx  = jnp.argsort(eigs.real)
    return eigs[idx]



def calculate_mass_transfer_lv(T_l, T_v, T_sat, htc_l, htc_v, A_i, h_sat_l, h_sat_v, h_l, h_v, Qvi, Qli):
    """
    Interphase mass transfer from Rane & He (2021), Eq. (4).
    Positive m_dot = evaporation (l -> v), Negative = condensation (v -> l).
    """
    # Eq. (4): m_dot = -[htc_l * ai * (Tsat - Tl) + htc_v * ai * (Tsat - Tv)] / (h_qs - h_ps)
    # Distribute the minus sign:
    numerator = -htc_l * A_i * (T_sat - T_l) - htc_v * A_i * (T_sat - T_v)

    numerator = - Qvi - Qli


    # Denominator depends on direction: Eq. (5) for evaporation, Eq. (6) for condensation
    # Evaporation (m_dot > 0): h_qs = h_v(T_sat) = h_sat_v,  h_ps = h_l(T_l) = h_l
    denominator = h_sat_v - h_sat_l

    m_dot = numerator / denominator

    # jax.debug.print("Tl:{Tl} | Tv:{Tv} | Tsat:{Ts} | mdot:{md}", Tl=T_l, Tv=T_v, Ts=T_sat, md=m_dot)

    return m_dot


def alpha_b_star(alpha_g, alpha_trans=0.2, alpha_max=0.35):
    """Stadtke's effective bubble void fraction (hyperbolic saturation form).

    Parameters
    ----------
    alpha_g : array-like or scalar
        Gas volume fraction, in [0, 1].
    alpha_trans : float
        End of the linear region (where alpha_b* = alpha_g exactly). Default 0.15.
    alpha_max : float
        Asymptotic upper bound (regime-transition ceiling). Default 0.35.

    Returns
    -------
    alpha_b_star : same shape as alpha_g
    """
    k = alpha_max - alpha_trans
    hyperbolic = alpha_max - (k * k) / (alpha_g - alpha_trans + k)
    return jnp.where(alpha_g <= alpha_trans, alpha_g, hyperbolic)


def alpha_d_star(alpha_g, alpha_trans=0.8, alpha_max=0.35):
    """Stadtke effective droplet (liquid) fraction.

    Equal to alpha_b_star evaluated on the liquid fraction (1 - alpha_g).
    alpha_trans is on the liquid-fraction axis, so default 0.15 puts the
    linear region at alpha_g in [0.85, 1.0] — symmetric with the bubble side.
    """
    return alpha_b_star(1.0 - alpha_g, 1 - alpha_trans, alpha_max)


def smoothstep(alpha_g, alpha_start=0.2, alpha_end=0.8):
    """Smoothstep from 0 at alpha_b_transition to 1 at alpha_d_transition."""
    t = (alpha_g - alpha_start) / (alpha_end - alpha_start)
    t = jnp.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def interfacial_area_dispersed_flows(alpha_g, d_bubble, d_droplet):
    """Raw interfacial area density for each regime, from volume conservation.

    ai = 6 * alpha / d, using the TRUE phase fractions. No saturation,
    no regime blending — the caller applies the smoothstep blend to the
    returned ai_b, ai_d.
    """
    alpha_l = 1.0 - alpha_g
    ai_b = 6.0 * alpha_g / d_bubble   # bubbles in liquid
    ai_d = 6.0 * alpha_l / d_droplet  # droplets in gas
    return ai_b, ai_d


class CommonQty(NamedTuple):
    """Abbreviations that appear in G_h, J, K, X."""
    rho_mix: jnp.ndarray  # alpha_g*rho_g + alpha_l*rho_l  (mixture density)
    rho_s:   jnp.ndarray  # alpha_g*rho_l + alpha_l*rho_g  (cross-weighted)
    rh2:     jnp.ndarray  # rho_g*rho_l + k*rho_mix^2      (virtual-mass modified)
    rg_hat:  jnp.ndarray  # rho_g + k*rho_mix
    rl_hat:  jnp.ndarray  # rho_l + k*rho_mix
    a0sq:    jnp.ndarray  # frozen sound-speed combination
    du:      jnp.ndarray  # u_g - u_l  (slip)
    u1:      jnp.ndarray  # alpha_g*u_g + alpha_l*u_l  (volume-weighted velocity)
    u2:      jnp.ndarray  # mass-weighted velocity
    G_pp:    jnp.ndarray  # pressure-row (1,1) entry of G_4
    G_uu:    jnp.ndarray  # shared diagonal entry of velocity rows
    G_aa:    jnp.ndarray  # void-fraction diagonal entry



def G_h_matrix(U_h, props_l, props_g, cq: CommonQty) -> jnp.ndarray:
    """
    6×6 quasi-linear coefficient matrix G_h = Phi^{-1} G_s Phi.

    Block structure (derivation §4.4):
        G_h = [[G_4,    0_{4×2}  ],
               [G_h21,  diag(u_g, u_l)]]

    The upper-left G_4 is identical to the verified G_s upper block.
    The lower-left G_h21 is the Gibbs-relation transform of G_s row 1.
    """
    p, u_g, u_l, ag, h_g, h_l = U_h
    al = 1.0 - ag

    rg, ag_ = props_g
    rl, al_ = props_l

    rho_s = cq.rho_s;  rh2 = cq.rh2;  rg_hat = cq.rg_hat;  rl_hat = cq.rl_hat
    a0sq  = cq.a0sq;   du  = cq.du;   G_pp   = cq.G_pp;    G_uu   = cq.G_uu;  G_aa = cq.G_aa

    # ── Upper-left G_4 (4×4) — rows/cols: p, u_g, u_l, alpha_g ─────────────
    # Row 1 — pressure equation
    r1 = jnp.array([
        G_pp,
        ag  * a0sq * rl * rg / rho_s,
        al  * a0sq * rl * rg / rho_s,
        a0sq * du  * rl * rg / rho_s,
    ])
    # Row 2 — gas velocity equation
    r2 = jnp.array([
        rl_hat / rh2,
        G_uu,
        -al * du * rl * rl_hat / rh2,
        0.0,
    ])
    # Row 3 — liquid velocity equation
    r3 = jnp.array([
        rg_hat / rh2,
        ag * du * rg * rg_hat / rh2,
        G_uu,
        0.0,
    ])
    # Row 4 — void fraction equation
    r4 = jnp.array([
        al * ag * du / rho_s * a0sq / (ag_**2 * al_**2),
        al * ag * rg / rho_s * a0sq / al_**2,
        -al * ag * rl / rho_s * a0sq / ag_**2,
        G_aa,
    ])

    # ── Lower-left G_h21 (2×4) — enthalpy rows ───────────────────────────────
    # Row 5 — gas enthalpy: (G_pp-u_g)/rho_g for col p; ratio*rho_l for cols u,al,a
    rh21_g = jnp.array([
        (G_pp - u_g) / rg,
        ag * a0sq * rl / rho_s,
        al * a0sq * rl / rho_s,
        a0sq * du  * rl / rho_s,
    ])
    # Row 6 — liquid enthalpy: (G_pp-u_l)/rho_l; ratio*rho_g for cols u,al,a
    rh21_l = jnp.array([
        (G_pp - u_l) / rl,
        ag * a0sq * rg / rho_s,
        al * a0sq * rg / rho_s,
        a0sq * du  * rg / rho_s,
    ])

    z4 = jnp.zeros(4)
    z2 = jnp.zeros(2)
    o  = jnp.ones(1)

    # Assemble full 6×6 row by row
    row1 = jnp.concatenate([r1,     z2])
    row2 = jnp.concatenate([r2,     z2])
    row3 = jnp.concatenate([r3,     z2])
    row4 = jnp.concatenate([r4,     z2])
    row5 = jnp.concatenate([rh21_g, jnp.array([u_g, 0.0])])
    row6 = jnp.concatenate([rh21_l, jnp.array([0.0, u_l])])

    return jnp.stack([row1, row2, row3, row4, row5, row6])



def common_quantities(U_h, props_l, props_g, k) -> CommonQty:
    """
    Pre-compute all shared abbreviations needed by the matrices.

    Parameters
    ----------
    U_h   : (6,) primitive state  [p, u_g, u_l, alpha_g, h_g, h_l]
    eos_g : PhaseProps for gas
    eos_l : PhaseProps for liquid
    k     : virtual-mass coefficient (scalar)
    """
    _, u_g, u_l, ag, _, _ = U_h
    al = 1.0 - ag

    rg, ag_ = props_g
    rl, al_ = props_l

    rho_mix = ag * rg + al * rl
    rho_s   = ag * rl + al * rg
    rg_hat  = rg + k * rho_mix
    rl_hat  = rl + k * rho_mix
    rh2     = rg * rl + k * rho_mix**2

    # Frozen (zero-slip) sound speed combination — derivation §2.1
    a0sq = rho_s / (ag * rl / ag_**2 + al * rg / al_**2)

    du = u_g - u_l
    u1 = ag * u_g + al * u_l
    u2 = (ag * rg * u_g + al * rl * u_l) / rho_mix

    # Diagonal/coupling scalars in G_4 — derivation §2.2
    G_pp = (al * u_l * rg / rho_s * a0sq / al_**2
          + ag * u_g * rl / rho_s * a0sq / ag_**2)
    G_uu = u1 * rl * rg / rh2 + u2 * k * rho_mix**2 / rh2
    G_aa = (al * u_g * rg / rho_s * a0sq / al_**2
          + ag * u_l * rl / rho_s * a0sq / ag_**2)

    return CommonQty(rho_mix=rho_mix, rho_s=rho_s, rh2=rh2,
                     rg_hat=rg_hat, rl_hat=rl_hat, a0sq=a0sq,
                     du=du, u1=u1, u2=u2, G_pp=G_pp, G_uu=G_uu, G_aa=G_aa)



def D_h_primitive(U_h, eos_g, eos_l, cq: CommonQty, closure, k) -> jnp.ndarray:
    """
    Primitive enthalpy-frame source vector D_h (6,)

    Parameters
    ----------
    U_h     : (6,) primitive state [p, u_g, u_l, alpha_g, h_g, h_l]
    eos_g, eos_l : PhaseProps at U_h
    cq      : CommonQty at U_h  (includes rho_mix, rho_s, rh2, etc.)
    closure : TwoFluidClosure containing physical source inputs
    k       : virtual-mass coefficient

    Returns
    -------
    D_h : (6,) — [D_p, D_{u_g}, D_{u_l}, D_{alpha}, D_{h_g}, D_{h_l}]
    """
    p, u_g, u_l, ag, h_g, h_l = U_h
    al = 1.0 - ag

    rg, ag_, T_g, Gg = eos_g 
    ag_sq = ag_ ** 2
    rg_h = -(rg * Gg) / ag_**2

    rl, al_, T_l, Gl = eos_l
    al_sq = al_ ** 2
    rl_h = -(rl * Gl) / al_**2


    rho_s = cq.rho_s;  rh2 = cq.rh2;  a0sq = cq.a0sq
    du      = cq.du

    k_rho   = cq.rg_hat - rg     # = k * rho_mix

    # ── Unpack closure ───────────────────────────────────────────────────────
    sigma_Mg = closure.sigma_Mg
    sigma_Qg = closure.sigma_Qg
    sigma_Ql = closure.sigma_Ql
    Fv_g     = closure.Fv_g
    F_g      = closure.F_g
    F_l      = closure.F_l
    Q_g      = closure.Q_g
    Q_l      = closure.Q_l
    u_ex     = closure.u_ex
    u_int    = closure.u_int
    h_ex_g   = closure.h_ex_g
    h_ex_l   = closure.h_ex_l
    dlnA_dx  = closure.dlnA_dx

    sigma_Ml = -sigma_Mg
    Fv_l     = -Fv_g

    # ── Entropy sources T_i * sigma_i^S  (Städtke eq. 4.50) ─────────────────
    Tg_sSg = (sigma_Mg * ((h_ex_g - h_g) + 0.5 * (u_ex - u_g)**2)
              + Fv_g  * (u_int - u_g)
              + sigma_Qg + Q_g)
    Tl_sSl = (sigma_Ml * ((h_ex_l - h_l) + 0.5 * (u_ex - u_l)**2)
              + Fv_l  * (u_int - u_l)
              + sigma_Ql + Q_l)
    sSg = Tg_sSg / T_g   # sigma_g^S
    sSl = Tl_sSl / T_l   # sigma_l^S

    # ── Recurring mass+entropy group (Appendix A) ────────────────────────────
    Mg = sigma_Mg - (T_g * rg_h / rg) * sSg
    Ml = sigma_Ml - (T_l * rl_h / rl) * sSl

    # ── Row 1: pressure ──────────────────────────────────────────────────────
    Ds_1 = (rl * a0sq / rho_s * Mg
          + rg * a0sq / rho_s * Ml
          - rl * rg / rho_s * (ag * u_g + al * u_l) * a0sq * dlnA_dx)

    # ── Row 2: gas velocity (k → k*rho correction applied) ───────────────────
    Ds_2 = (rl / (ag * rh2) * (Fv_g + sigma_Mg * (u_ex - u_g))
          - sigma_Mg * (k_rho * rg + al * (rl**2 - rg**2)) / (rg * rh2) * du)
        #   + (F_g * (rl + ag * k_rho) + F_l * ag * k_rho) / (ag * rh2))

    # ── Row 3: liquid velocity (k → k*rho correction applied) ────────────────
    Ds_3 = (-rg / (al * rh2) * (Fv_g + sigma_Mg * (u_ex - u_l))
           - sigma_Mg * (k_rho * rl + ag * (rg**2 - rl**2)) / (rl * rh2) * du)
        #    + (F_g * al * k_rho + F_l * (rg + al * k_rho)) / (al * rh2))

    # ── Row 4: void fraction ─────────────────────────────────────────────────
    Ds_4 = (al * a0sq / (rho_s * al_sq) * Mg
          - ag * a0sq / (rho_s * ag_sq) * Ml
          + al * ag * a0sq / rho_s * (rl * u_l / ag_sq - rg * u_g / al_sq) * dlnA_dx)

    # ── Rows 5, 6: entropy ───────────────────────────────────────────────────
    Ds_5 = sSg / (ag * rg)
    Ds_6 = sSl / (al * rl)

    D1 = Ds_1
    D2 = Ds_2
    D3 = Ds_3
    D4 = Ds_4
    D5 = (1/rg) * Ds_1 + T_g * Ds_5
    D6 = (1/rl) * Ds_1 + T_l * Ds_6

    return jnp.array([D1, D2, D3, D4, D5, D6])

class TwoFluidClosure(NamedTuple):
    """
    Physical closure quantities consumed by D_h_primitive.

    All quantities are per-unit-volume and are scalars (per-cell).
    Sign convention: sigma_Mg > 0 = evaporation (liquid → gas).
    """
    sigma_Mg : jnp.ndarray  # gas mass source  [kg/(m^3 s)]
    sigma_Qg : jnp.ndarray  # interfacial heat to gas   [W/m^3]
    sigma_Ql : jnp.ndarray  # interfacial heat to gas   [W/m^3]
    Fv_g     : jnp.ndarray  # interfacial drag on gas   [N/m^3]
    F_g      : jnp.ndarray  # external body force on gas [N/m^3]
    F_l      : jnp.ndarray  # external body force on liquid [N/m^3]
    Q_g      : jnp.ndarray  # wall/external heat to gas  [W/m^3]
    Q_l      : jnp.ndarray  # wall/external heat to liquid [W/m^3]
    u_ex     : jnp.ndarray  # velocity of transferred mass [m/s]
    u_int    : jnp.ndarray  # interface velocity [m/s]
    h_ex_g   : jnp.ndarray  # enthalpy of mass entering gas phase [J/kg]  (h_sat_g for evaporation)
    h_ex_l   : jnp.ndarray  # enthalpy of mass leaving liquid phase [J/kg] (h_sat_l for evaporation)
    dlnA_dx  : jnp.ndarray  # (1/A) dA/dx — log area derivative [1/m]


def evaporation_weight(h_l, T_l, h_sat_l, T_sat, ag, gate_dT=0.5, gate_dh=2.0e3,
                       dh_gate=1.0, dT_gate=0.001):
    """Evaporation-gate weight in [0, 1] — liquid superheat OR appreciable α.

    w → 1 where the liquid is superheated (h AND T criteria), w → 0 where
    subcooled.  Single source of truth shared by the mass-transfer closure
    (σ gate) and the solver's ghost-vapor ENERGY pin (pin weight = 1 − w): the
    pin releases exactly where evaporation switches on, so the gate can never
    prevent flashing from starting.

    α-AVAILABILITY CLAUSE (`| ag > 5e-2`): w is ALSO forced to 1 once the vapor
    is physically real (α > 5%), regardless of liquid superheat.  This
    deliberately breaks the older "liquid quantities only" rule, and it is the
    fix for the shock-condensation case flagged in TODO_two_phase.md: with a
    pure superheat gate, a recompression shock re-subcools the liquid at α~0.5
    and would RE-PIN genuinely real vapor to h_sat — i.e. force equilibrium
    condensation.  Keyed on α, a real phase is never pinned.

    Safe w.r.t. the solver's NO-DEADLOCK invariant: the clause only WIDENS the
    w=1 (evaporation-on / pin-off) region, never narrows it, so nothing in the
    α-growth chain can be gated off by it.

    SIDE EFFECT to be aware of: σ_Mg is likewise ungated wherever α > 5%, so in
    a subcooled region with real vapor the closure now returns σ < 0 — the
    CONDENSATION branch is active there (intended for shocks; it was previously
    suppressed by the evaporation-only choice).
    """
    # BINARY on/off gate (2026-07-21 A/B, nozzlex parity): hard switch on the
    # same h-AND-T criterion.  Flips BOTH the σ gate and the ghost pin weight
    # (1−w) together — pre-onset: σ=0 & fully pinned; post-onset: σ on & free.
    # Watch for: (a) per-cell flicker in time at the moving front (add
    # hysteresis if seen, don't go back to the blend silently), (b) Newton
    # Jacobian sees zero derivative across the switch (pin removes the fragile
    # h_g dof, expected harmless).
    # NOTE the parentheses around (ag > ...): in Python `&`/`|` bind TIGHTER than
    # `>`, so `A & B | ag > 5e-2` parses as `((A & B) | ag) > 5e-2` → bitwise-OR
    # of a bool array with the float64 α (TypeError under JAX).
    return jnp.where((h_l > h_sat_l + dh_gate) & (T_l > T_sat + dT_gate)
                     | (ag > 5e-2),
                     1.0, 0.0)
    # SMOOTH C¹ version (WORKING baseline 2026-07-21 — full-physics steady
    # state, SSR 3.4e-6; kept for revert if the binary gate flickers):
    # _sig = lambda z: 0.5 * (1.0 + jnp.tanh(z))
    # return _sig(jnp.minimum((h_l - h_sat_l - dh_gate) / gate_dh,
    #                         (T_l - T_sat   - dT_gate) / gate_dT))



def two_fluid_closure(
    U_h,
    rg,
    T_g,
    rl,
    T_l,
    fluid_g,               # FluidBicubic — for viscosity/conductivity lookup
    fluid_l,               # FluidBicubic
    T_sat,                 # saturation temperature at current pressure [K]
    h_sat_g,               # saturation vapour enthalpy [J/kg]
    h_sat_l,               # saturation liquid enthalpy [J/kg]
    dlnA_dx=0.0,           # area derivative [1/m]
    F_g=0.0,               # gravity or other external force on gas [N/m^3]
    F_l=0.0,               # gravity or other external force on liquid [N/m^3]
    Q_g=0.0,               # wall heat to gas [W/m^3]
    Q_l=0.0,               # wall heat to liquid [W/m^3]
    Nb=1e10,               # bubble/droplet number density [1/m^3]
    Nd=1e10,
    alpha_b=0.3,           # TA-BD transition void fractions
    alpha_d=0.7,
    mass_transfer=True,
    gate_dT= 1e-3, # 0.5,           # smooth-gate temperature width [K].  NOTE 2026-07-21:
                           # widening to 2.0 K (onset spread ~4-8 cells) was
                           # PROBED and is NEGATIVE — died EARLIER (step 292 vs
                           # 386, dt collapse at t=5.9e-4 vs 1.55e-3 s).  The
                           # front-cell blow-up is not a σ(x) sharpness problem.
    gate_dh= 0.5 #2.0e3,         # smooth-gate enthalpy width [J/kg]  (~cp_l·gate_dT)
) -> TwoFluidClosure:
    """
    Assemble a TwoFluidClosure from state + saturation properties.

    Uses:
      - interfacial_area    → dispersed-structure diameter d and area ai
      - ranz_marshall_htc_two_fluid  → HTC for each phase
      - calculate_drag_force_one_component  → Fv_g
      - calculate_mass_transfer_lv  → sigma_Mg

    The saturation properties (T_sat, h_sat_g, h_sat_l) must be supplied
    externally from a saturation EOS call so that this function stays
    differentiable inside jax.jit (no secondary EOS solve needed).

    Parameters
    ----------
    U_h : (6,)  primitive state  [p, u_g, u_l, alpha_g, h_g, h_l]
    eos_g, eos_l : PhaseProps pre-computed at U_h
    fluid_g, fluid_l : FluidBicubic tables (needed for mu, k, cp properties)
    T_sat, h_sat_g, h_sat_l : saturation properties at pressure U_h[0]
    dlnA_dx : logarithmic area gradient [1/m]
    F_g, F_l : external body forces [N/m^3]
    Q_g, Q_l : wall heat sources [W/m^3]
    Nb, Nd   : number densities for TA-BD interfacial area model
    alpha_b, alpha_d : TA-BD transition void fractions
    """
    p, u_g, u_l, ag, h_g, h_l = U_h
    al = 1.0 - ag

    # rg = eos_g.rho;  T_g = eos_g.T
    # rl = eos_l.rho;  T_l = eos_l.T

    # --- Two-phase gate (SYMMETRIC, enthalpy AND temperature) ----------------
    # Mass transfer activates only when a phase is away from saturation in the
    # direction that drives a real phase change, requiring BOTH the enthalpy and
    # the temperature to agree:
    #   evaporation  if the LIQUID is superheated  (h_l > h_sat_l AND T_l > T_sat)
    #   condensation if the VAPOUR is subcooled     (h_g < h_sat_g AND T_g < T_sat)
    # and OFF otherwise.  Requiring both h and T is the conservative choice and
    # stays well-posed toward the CRITICAL region, where the saturation split (and
    # any single h- or T-only test) becomes degenerate: if one variable is
    # spuriously triggered by table extrapolation, the other guards it.  Small
    # deadbands (dh_gate, dT_gate) cover the ~mK / ~tens-of-J/kg inconsistency
    # between the metastable property tables and the saturation table (test_T0).
    #
    # Replaces the old `is_two_phase = True` override, which forced mass transfer
    # even at saturation and let a superheated VAPOUR spuriously drive evaporation.
    if mass_transfer == True:
        # SMOOTH C¹ gate (2026-07-21) — replaces the binary lax.cond gate.
        # The binary gate is a checkerboard PUMP at the flashing front: with
        # mK/J-scale deadbands, neighbouring cells straddle the threshold and
        # σ_Mg flips full-on/off (even evap/cond) cell-to-cell — measured 76%
        # 2Δx oscillation in σ_Mg itself at the blow-up.  Worse, lax.cond has
        # ZERO derivative across the branch switch, so the implicit Newton's
        # Jacobian cannot see the source activating and overshoots into the
        # flip every step.  Same physics and the same h-AND-T conservatism,
        # expressed as sigmoid weights over the LEAST-activated of the two
        # criteria (smooth AND); the two directions combine as a smooth OR.
        # Widths gate_dT/gate_dh set the transition scale; the old deadbands
        # (dh_gate, dT_gate) remain as the sigmoid centres, covering the
        # table/saturation inconsistency (test_T0).  As widths → 0 this
        # recovers the binary gate exactly.
        # OLD binary gate (kept for reference / quick revert):
        # can_evaporate = (h_l > h_sat_l + dh_gate) & (T_l > T_sat + dT_gate)
        # can_condense  = (h_g < h_sat_g - dh_gate) & (T_g < T_sat - dT_gate)
        # is_two_phase  = can_evaporate | can_condense
        # Shared helper — SAME weight the solver's ghost-vapor pin uses as 1−w.
        w_evap  = evaporation_weight(h_l, T_l, h_sat_l, T_sat, ag, 
                                     gate_dT=gate_dT, gate_dh=gate_dh)
        # EVAPORATION-ONLY (2026-07-21, nozzlex parity) — nozzlex gates with
        # is_two_phase = (hl >= h_sat_l) & (Tl >= Tsat): NO condensation branch.
        # Our condensation weight half-opens AT the saturation line, so the
        # subcooled pre-flashing ghost (h_g ≈ h_sat_g, T_g ≈ T_sat) was DRAINED
        # by σ<0 onto the α clip floor; the non-conservative clip then fights
        # the source every step — measured as the u_g/h_g 2Δx pump pinned on
        # the floor plateau cells at flashing onset (x≈0.08, α=1e-3 exactly,
        # u_g=-5.7 m/s between +199 m/s neighbours).  nozzlex's implied σ in
        # that region is EXACTLY 0; ours was −100..−470 kg/(m³s).
        # w_cond = _sig(jnp.minimum((h_sat_g - h_g - dh_gate) / gate_dh,
        #                           (T_sat   - T_g - dT_gate) / gate_dT))
        # w_mt   = 1.0 - (1.0 - w_evap) * (1.0 - w_cond)   # smooth OR
        w_mt    = w_evap
        # ── Extended EOS for transport properties (mu, k, cp) ───────────────────
        p_arr  = jnp.expand_dims(p, 0)

        # analytic EOS (FluidPerfectGas) has no bounds → PHYSICAL floors, not ±inf
        # (p<0 → rho<0 → sqrt NaN; observed 2026-07-14 Bernoulli ghost case).
        p_arr = jnp.clip(p_arr, getattr(fluid_l, "p_min", 1e2), getattr(fluid_l, "p_max", 1e9))
        p_arr = jnp.clip(p_arr, getattr(fluid_g, "p_min", 1e2), getattr(fluid_g, "p_max", 1e9))

        h_g = jnp.clip(jnp.expand_dims(h_g, 0), getattr(fluid_g, "h_min", 1e3), getattr(fluid_g, "h_max", 5e6))
        h_l = jnp.clip(jnp.expand_dims(h_l, 0), getattr(fluid_l, "h_min", 1e3), getattr(fluid_l, "h_max", 5e6))

        stg_ex = fluid_g.get_state(jxp.HmassP_INPUTS, h_g, p_arr)
        stl_ex = fluid_l.get_state(jxp.HmassP_INPUTS, h_l, p_arr)

        # ── Interfacial area and dispersed-structure diameter ────────────────────
        # d_min_b=5e-5 matches nozzlex (nozzle_two_fluid_one_component_autonomous).
        ai, d = interfacial_area(ag, alpha_b=alpha_b, alpha_d=alpha_d,
                                Nb=Nb, Nd=Nd, d_min_b=5e-5)
        # ai, d = 100, 5e-5
        # ── Drag force on gas per unit volume (sign: F > 0 accelerates gas) ─────
        mu_g = stg_ex["viscosity"][0]
        k_g = stg_ex["conductivity"][0]   
        mu_l = stl_ex["viscosity"][0]
        # Drag force per unit volume: F > 0 means force on gas in direction of u_l - u_g
        Fv_g = -calculate_drag_force_one_component(
            ag, al, rg, rl, mu_g, mu_l, u_g, u_l, d,
            alpha_b=alpha_b, alpha_d=alpha_d,
        )

        # ── Heat transfer coefficients (Ranz-Marshall, continuous phase) ─────────
        du_abs = jnp.abs(u_g - u_l)
        # Gas-phase HTC: continuous phase depends on flow regime (use liquid as
        # continuous in bubbly, gas in droplet; transition blended).
        # For simplicity use liquid-continuous HTC for gas bubbles.
        # htc_g = ranz_marshall_htc_two_fluid(
        #     {k: v[0] for k, v in stl_ex.items()},   # liquid as continuous
        #     du_abs, d,
        # )

        Nu_g = 120
        htc_g = (k_g * Nu_g) / d

        htc_l = ranz_marshall_htc_two_fluid(
            {                                         # liquid as continuous (bubbly regime)
                "rho": stl_ex["rho"][0],             # FluidState alias → density
                "mu":  stl_ex["mu"][0],              # FluidState alias → viscosity
                "k":   stl_ex["k"][0],               # FluidState alias → conductivity
                "cp":  stl_ex["cp"][0],              # FluidState alias → isobaric_heat_capacity
            },
            du_abs, d,
        )

        # # ── Interfacial heat transfer rates ──────────────────────────────────────
        # Qvi = htc_g * ai * (T_sat - T_g)    # heat from interface to gas [W/m^3]
        # Qli = htc_l * ai * (T_sat - T_l)    # heat from interface to liquid [W/m^3]

        # # ── Mass transfer (evaporation/condensation) ─────────────────────────────
        # sigma_Mg = calculate_mass_transfer_lv(
        #     T_l, T_g, T_sat, htc_l, htc_g, ai,
        #     h_sat_l, h_sat_g, h_l, h_g, Qvi, Qli,
        # )

        # ── OLD binary dispatch (kept for reference / quick revert) ──────────
        # def _source_terms_two_phase(_args):
        #     _Tl, _Tv, _Tsat, _ht_l, _ht_v, _Ai, _h_sat_l, _h_sat_v, _hl, _hv = _args
        #     _Qvi = _ht_v * _Ai * (_Tsat - _Tv)
        #     _Qli = _ht_l * _Ai * (_Tsat - _Tl)
        #     _m_dot_lv = calculate_mass_transfer_lv(
        #         _Tl, _Tv, _Tsat, _ht_l, _ht_v, _Ai, _h_sat_l, _h_sat_v, _hl, _hv, _Qvi, _Qli,
        #     )
        #     return _Qvi, _Qli, _m_dot_lv
        #
        # def _source_terms_single_phase(_args):
        #     return (0.0, 0.0, 0.0)
        #
        # Qvi, Qli, sigma_Mg = jax.lax.cond(
        #     is_two_phase,
        #     _source_terms_two_phase,
        #     _source_terms_single_phase,
        #     (T_l, T_g, T_sat, htc_l, htc_g, ai, h_sat_l, h_sat_g, h_l, h_g),
        # )

        # Always evaluate, then scale ALL THREE interfacial exchanges by the
        # SAME smooth weight — this keeps the interface energy balance
        # σ_Mg = -(Qvi+Qli)/(h_sat_g - h_sat_l) exact at every gate level
        # (scaling σ alone would create/destroy interfacial energy).
        _Qvi = htc_g * ai * (T_sat - T_g)
        _Qli = htc_l * ai * (T_sat - T_l)
        _sMg = calculate_mass_transfer_lv(
            T_l, T_g, T_sat, htc_l, htc_g, ai, h_sat_l, h_sat_g, h_l, h_g, _Qvi, _Qli,
        )
        Qvi      = w_mt * _Qvi
        Qli      = w_mt * _Qli
        sigma_Mg = w_mt * _sMg
    else:     # No mass transfer, just heat and drag

        # ── Extended EOS for transport properties (mu, k, cp) ───────────────────
        p_arr  = jnp.expand_dims(p, 0)

        # analytic EOS (FluidPerfectGas) has no bounds → PHYSICAL floors, not ±inf
        # (p<0 → rho<0 → sqrt NaN; observed 2026-07-14 Bernoulli ghost case).
        p_arr = jnp.clip(p_arr, getattr(fluid_l, "p_min", 1e2), getattr(fluid_l, "p_max", 1e9))
        p_arr = jnp.clip(p_arr, getattr(fluid_g, "p_min", 1e2), getattr(fluid_g, "p_max", 1e9))

        h_g = jnp.clip(jnp.expand_dims(h_g, 0), getattr(fluid_g, "h_min", 1e3), getattr(fluid_g, "h_max", 5e6))
        h_l = jnp.clip(jnp.expand_dims(h_l, 0), getattr(fluid_l, "h_min", 1e3), getattr(fluid_l, "h_max", 5e6))

        stg_ex = fluid_g.get_state(jxp.HmassP_INPUTS, h_g, p_arr)
        stl_ex = fluid_l.get_state(jxp.HmassP_INPUTS, h_l, p_arr)

        # ── Interfacial area and dispersed-structure diameter ────────────────────
        # d_min_b=5e-5 matches nozzlex (nozzle_two_fluid_one_component_autonomous).
        ai, d = interfacial_area(ag, alpha_b=alpha_b, alpha_d=alpha_d,
                                Nb=Nb, Nd=Nd, d_min_b=5e-5)
        # ai, d = 100, 5e-5
        # ── Drag force on gas per unit volume (sign: F > 0 accelerates gas) ─────
        mu_g = stg_ex["viscosity"][0]
        k_g = stg_ex["conductivity"][0]   
        mu_l = stl_ex["viscosity"][0]
        # Drag force per unit volume: F > 0 means force on gas in direction of u_l - u_g
        Fv_g = -calculate_drag_force_one_component(
            ag, al, rg, rl, mu_g, mu_l, u_g, u_l, d,
            alpha_b=alpha_b, alpha_d=alpha_d,
        )

        # ── Heat transfer coefficients (Ranz-Marshall, continuous phase) ─────────
        du_abs = jnp.abs(u_g - u_l)
        # Gas-phase HTC: continuous phase depends on flow regime (use liquid as
        # continuous in bubbly, gas in droplet; transition blended).
        # For simplicity use liquid-continuous HTC for gas bubbles.
        # htc_g = ranz_marshall_htc_two_fluid(
        #     {k: v[0] for k, v in stl_ex.items()},   # liquid as continuous
        #     du_abs, d,
        # )

        Nu_g = 12
        htc_g = (k_g * Nu_g) / d

        htc_l = ranz_marshall_htc_two_fluid(
            {                                         # liquid as continuous (bubbly regime)
                "rho": stl_ex["rho"][0],             # FluidState alias → density
                "mu":  stl_ex["mu"][0],              # FluidState alias → viscosity
                "k":   stl_ex["k"][0],               # FluidState alias → conductivity
                "cp":  stl_ex["cp"][0],              # FluidState alias → isobaric_heat_capacity
            },
            du_abs, d,
        )

        htc = 1 / (1/htc_g + 1/htc_l)

        # # ── Interfacial heat transfer rates ──────────────────────────────────────
        Qvi = htc * ai * (T_l - T_g)    # heat from interface to gas [W/m^3]
        Qli = - Qvi                     # heat from interface to liquid [W/m^3]

        # # ── Mass transfer (evaporation/condensation) ─────────────────────────────
        sigma_Mg = 0.0

    # ── Interface kinematics ─────────────────────────────────────────────────
    # Donor/upwind exchanged-mass velocity: the transferred mass carries the
    # velocity of the phase it LEAVES (evaporation σ_Mg>0: from liquid → u_l;
    # condensation σ_Mg<0: from gas → u_g).  This keeps the entropy slip term
    # ½(u_ex - u_i)² ≥ 0 unambiguous (RELAP5/CATHARE/TRACE convention).
    u_mix2 = (ag * rg * u_g + al * rl * u_l) / (ag * rg + al * rl)  # mass-weighted mixture
    u_ex   = jax.lax.cond(
        sigma_Mg >= 0.0,
        lambda _: u_l,       # evaporation: mass leaves the liquid phase
        lambda _: u_g,       # condensation: mass leaves the gas phase
        operand=None,
    )
    u_int  = u_mix2          # mass-weighted mixture interface velocity (viscous-work)
    # Enthalpy of transferred mass: saturation enthalpy of appearing phase

    # TODO: check the source terms implementation and compare it with nozzlex and with thee sources in Stadtke
    # TODO: check the sign of u_ex and u_v
    # TODO: h_ex is different beetween the phases. The difference is in the fact that theee phase appears at saturatio condition but on the liquid side it appears as sat liquid, onthe gas side as saturated gas
    # TODO: there should be the velocity differencee at the denominator of the mass source term?

    return TwoFluidClosure(
        sigma_Mg=sigma_Mg,
        sigma_Qg=Qvi,
        sigma_Ql=Qli,
        Fv_g=Fv_g,
        F_g=F_g,
        F_l=F_l,
        Q_g=Q_g,
        Q_l=Q_l,
        u_ex=u_ex,
        u_int=u_int,
        h_ex_g=h_sat_g,
        h_ex_l=h_sat_l,
        dlnA_dx=dlnA_dx,
    )



def get_nozzle_from_csv(x, roughness=0.0, file_path="nozzle_geometry.csv"):
    """
    Axisymmetric nozzle from a tabulated wall radius r(x), e.g. the geometry
    exported by the nozzlex inverse design (examples/inverse_design/run_design.py).

    file_path : CSV with a header row and at least the columns 'x' and 'r' [m]
                (extra columns such as D, A are ignored).

    r(x) is fitted with a C2 cubic spline, so A and the analytic dA/dx are
    smooth across the table points.  Outside the tabulated range r is held at
    its end value (dA/dx = 0); set the grid to [x_first, x_last] of the table.
    """
    from scipy.interpolate import CubicSpline

    x = jnp.asarray(x, dtype=float)
    data = jnp.genfromtxt(file_path, delimiter=",", names=True)
    x_tab = jnp.asarray(data["x"], dtype=float)
    r_tab = jnp.asarray(data["r"], dtype=float)

    spline = CubicSpline(x_tab, r_tab)
    inside = (x >= x_tab[0]) & (x <= x_tab[-1])
    x_c = jnp.clip(x, x_tab[0], x_tab[-1])
    r = spline(x_c)
    drdx = jnp.where(inside, spline(x_c, 1), 0.0)

    A    = jnp.pi * r**2
    dAdx = 2.0 * jnp.pi * r * drdx
    D_h  = 2.0 * r

    return A, dAdx, jnp.pi*r, 0.5*D_h
