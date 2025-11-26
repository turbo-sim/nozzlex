import jax
import jax.numpy as jnp
import jaxprop as jxp



def nozzle_single_phase_core(t, y, args):
    """Wrapper that adapts from (t, y) to autonomous form."""
    Y = jnp.concatenate([jnp.atleast_1d(t), y])
    return nozzle_single_phase_autonomous_ph(0.0, Y, args)


def nozzle_single_phase_autonomous(tau, Y, args):
    """
    Autonomous formulation of the nozzle equations:
        dx/dt   = det(A)
        dy_i/dt = det(A with column i replaced by b)

    State vector: Y = [x, v, rho, p]
    """
    x, v, d, p = Y

    # v = jnp.where(jnp.abs(v) < 1e-6, jnp.sign(v)*1e-6, v)
    # Debug print using jax.debug.print
    # jax.debug.print("x={:.6f}, v={:.6f}, rho={:.6f}, p={:.6f}", x, v, d, p)


    # --- Geometry / model parameters ---
    fluid = args.fluid
    L = args.length
    eps_wall = args.roughness
    T_ext = args.T_wall
    wall_friction = args.wall_friction
    heat_transfer = args.heat_transfer

    # --- Thermodynamic state ---
    # d = jnp.clip(d, 1e-10, 1e6)  # enforce valid density range
    # p = jnp.clip(p, 1e2, 1e9)   # enforce valid pressure range

    state = fluid.get_state(jxp.DmassP_INPUTS, d, p)
    T = state["T"]
    h = state["h"]
    s = state["s"]
    a = state["a"]
    cp = state["cp"]
    mu = state["mu"]
    G = state["gruneisen"]

    # Stagnation state
    h0 = state["h"] + 0.5 * v**2
    state0 = fluid.get_state(jxp.HmassSmass_INPUTS, h0, state["s"])
    p0 = state0["p"]
    T0 = state0["T"]
    d0 = state0["d"]

    # --- Geometry ---
    A, dAdx, perimeter, diameter, D_h= args.geometry(x, L) # When using symmetric geometry
    # A, dAdx, perimeter, radius = args.geometry(x) # When using linear convergent divergent geometry
    # diameter = 2 * radius

    # --- Wall heat transfer and friction ---
    Re = v * d * diameter / jnp.maximum(mu, 1e-12)
    f_D = get_friction_factor_haaland(Re, eps_wall, diameter)
    tau_w = get_wall_viscous_stress(f_D, d, v)
    htc = 10000*get_heat_transfer_coefficient(v, d, cp, f_D)
    htc = jnp.clip(htc, 0.0, 1e6)   # pick bound based on your scaling
    q_w = htc * (T_ext - T)

    # Mask with booleans (convert to 0.0 if disabled)
    tau_w = wall_friction * tau_w
    f_D = wall_friction * f_D
    q_w = heat_transfer * q_w
    htc = heat_transfer * htc

    # jax.debug.print(
    #     "raw htc={:.4e}, raw q_w={:.4e}, T_ext={:.2f}, T={:.2f}", 
    #     htc, q_w, T_ext, T
    # )

    # --- Build A matrix and b vector ---
    A_mat = jnp.array([[d, v, 0.0], [d * v, 0.0, 1.0], [0.0, -(a**2), 1.0]])

    b_vec = jnp.array(
        [
            -d * v / A * dAdx,
            -(perimeter / A) * tau_w,
            (perimeter / A) * (G / v) * (tau_w * v + q_w),
        ]
    )

    # --- Determinants ---
    D = jnp.linalg.det(A_mat)

    # Replace columns one by one to compute N_i
    N = []
    for i in range(3):
        A_mod = A_mat.at[:, i].set(b_vec)
        N.append(jnp.linalg.det(A_mod))
    N = jnp.array(N)

    # --- Autonomous system: dx/dτ = D, dy/dτ = N_i ---
    dx_dtau = D
    dv_dtau = N[0]
    dd_dtau = N[1]
    dp_dtau = N[2]
    rhs = jnp.array([dv_dtau / dx_dtau, dd_dtau / dx_dtau, dp_dtau / dx_dtau])
    rhs_autonomous = jnp.array([dx_dtau, dv_dtau, dd_dtau, dp_dtau])

    # Export data
    out = {
        "x": x,
        "v": v,
        "d": d,
        "p": p,
        "rhs": rhs,
        "rhs_autonomous": rhs_autonomous,
        "A": A,
        "dAdx": dAdx,
        "diameter": diameter,
        "perimeter": perimeter,
        "h0": h0,
        "p0": p0,
        "T0": T0,
        "d0": d0,
        "Ma": v / state["a"],
        "Re": Re,
        "f_D": f_D,
        "tau_w": tau_w,
        "q_w": q_w,
        "htc": htc,
        "m_dot": d * v * A,
        "A_mat": A_mat,
        "b_vec": b_vec,
        "D": D,
        "N": N,
    }

    return {**out, **state.to_dict(include_aliases=True)}
    # return {**out, **state}

def nozzle_single_phase_autonomous_ph(tau, Y, args):
    """
    Autonomous formulation of the nozzle equations:
        dx/dt   = det(A)
        dy_i/dt = det(A with column i replaced by b)

    State vector: Y = [x, p, v, h]
    """
    x, p, v, h = Y

    # v = jnp.where(jnp.abs(v) < 1e-6, jnp.sign(v)*1e-6, v)
    # Debug print using jax.debug.print
    # jax.debug.print("x={:.6f}, v={:.6f}, p={:.6f}, h={:.6f}", x, v, p, h)

    # --- Geometry / model parameters ---
    fluid = args.fluid
    L = args.length
    eps_wall = args.roughness
    T_ext = args.T_wall
    wall_friction = args.wall_friction
    heat_transfer = args.heat_transfer

    # --- Thermodynamic state ---
    # d = jnp.clip(d, 1e-10, 1e6)  # enforce valid density range
    # p = jnp.clip(p, 1e2, 1e9)   # enforce valid pressure range
    # jax.debug.print("h = {}", h)
    # jax.debug.print("x = {}", x)

    state = fluid.get_state(jxp.HmassP_INPUTS, h, p)
    T = state["T"]
    d = state["rho"]
    s = state["s"]
    a = state["a"]
    cp = state["cp"]
    mu = state["mu"]
    G = state["gruneisen"]

    try:
        Q = state["quality_mass"]
        two_phase = jnp.logical_and(Q > 0.0, Q < 1.0)    
    except AttributeError:
        two_phase = False

    # Stagnation state
    h0 = state["h"] + 0.5 * v**2
    # state0 = fluid.get_state(jxp.HmassSmass_INPUTS, h0, state["s"])
    # p0 = state0["p"]
    # T0 = state0["T"]
    # d0 = state0["d"]
    # h0 = state0["h"]

    # --- Geometry ---
    # A, dAdx, perimeter, diameter = args.geometry(x, L) # When using symmetric geometry
    A, dAdx, perimeter, height, D_h = args.geometry(x, L) # When using linear convergent divergent geometry
    # diameter = 2 * radius


    # --- Wall heat transfer and friction ---
    Re = v * d * D_h / jnp.maximum(mu, 1e-12)
    f_D = get_friction_factor_haaland(Re, eps_wall, D_h)
    tau_w = get_wall_viscous_stress(f_D, d, v)
    # tau_w = jax.lax.cond(
    #     two_phase,
    #     lambda _: get_wall_viscous_stress_two_phase(args, p, d, Q, f_D, args.two_phase_friction, D_h, v),
    #     lambda _: get_wall_viscous_stress(f_D, d, v),
    #     operand=None
    # )
    htc = 10000*get_heat_transfer_coefficient(v, d, cp, f_D)
    htc = jnp.clip(htc, 0.0, 1e6)   # pick bound based on your scaling
    q_w = htc * (T_ext - T)

    # Mask with booleans (convert to 0.0 if disabled)
    tau_w = wall_friction * tau_w
    f_D = wall_friction * f_D
    q_w = heat_transfer * q_w
    htc = heat_transfer * htc

    dddp = (1 + G) / a**2
    dddh = - (d * G) / a**2
    
    # --- Build A matrix and b vector ---
    # A_mat = jnp.array([[d, v, 0.0, 0.0], [d * v, 0.0, 1.0, 0.0], [v, 0.0, 0.0, 1.0], [0, 1.0, - dddp, - dddh]])

    # b_vec = jnp.array(
    #     [
    #         -d * v / A * dAdx,
    #         -(perimeter / A) * tau_w,
    #         0.00,
    #         0.00,
    #     ]
    # )

    A_mat = jnp.array([[v * dddp, d, v * dddh], [1.00, d * v, 0], [v, 0, -d * v]])

    b_vec = jnp.array(
        [
            -d * v / A * dAdx,
            -(perimeter / A) * tau_w,
            -v * tau_w * perimeter / A,
        ]
    )
    S = d **(1/3)
    A_mat = A_mat / S 
    b_vec = b_vec / S

    # --- Determinants ---
    D = jnp.linalg.det(A_mat)

    # Replace columns one by one to compute N_i
    N = []
    for i in range(3):
        A_mod = A_mat.at[:, i].set(b_vec)
        N.append(jnp.linalg.det(A_mod))
    N = jnp.array(N)

    # --- Autonomous system: dx/dτ = D, dy/dτ = N_i ---
    dx_dtau = D
    dp_dtau = N[0]
    dv_dtau = N[1]
    dh_dtau = N[2]
    rhs = jnp.array([dp_dtau / dx_dtau, dv_dtau / dx_dtau, dh_dtau / dx_dtau])
    rhs_autonomous = jnp.array([dx_dtau, dp_dtau, dv_dtau, dh_dtau])

    # Export data
    out = {
        "x": x,
        "v": v,
        "d": d,
        "h": h,
        "p": p,
        "rhs": rhs,
        "rhs_autonomous": rhs_autonomous,
        "A": A,
        "dAdx": dAdx,
        # "diameter": diameter,
        "diameter": height,
        "perimeter": perimeter,
        "h0": h0,
        # "p0": p0,
        # "T0": T0,
        # "d0": d0,
        "a": state["a"],
        "Ma": v / state["a"],
        "Q": state["Q"],
        "Re": Re,
        "f_D": f_D,
        "tau_w": tau_w,
        "q_w": q_w,
        "htc": htc,
        "m_dot": d * v * A,
        "A_mat": A_mat,
        "b_vec": b_vec,
        "D": D,
        "N": N,
    }

    return {**out, **state.to_dict(include_aliases=True)}
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
    ratio = roughness / diameter
    # jax.debug.print("f_D = {f_D}, Re = {Re}, term={term}, roughness/D_h = {ratio}", f_D=f, Re=Reynolds, term=term, ratio = ratio)

    return f

def viscosity_MacAdams(mu_L, mu_v, x):
    mu = (x/mu_v + (1-x)/mu_L)**(-1)
    return mu

def f_HEM_purdue(Re):
    if Re < 2000:
        f = 16 / Re
    elif 200 <= Re < 20000:
        f = 0.079 * (Re ** -0.25)
    else:
        f = 0.046 * (Re ** -0.2)
        
    return f


# def LM_gronnerud(rho_l, rho_v, mu_l, mu_v, x, G_l, d, g=9.81):
#     """
#     Calculate the Lockhart-Martinelli parameter (Phi_LM) based on the given parameters.
   
#     Parameters:
#     rho_l (float): Liquid density (kg/m^3)
#     rho_v (float): Vapor density (kg/m^3)
#     mu_l (float): Liquid viscosity (Pa·s or kg/m·s)
#     mu_v (float): Vapor viscosity (Pa·s or kg/m·s)
#     x (float): Void fraction (dimensionless)
#     g_l (float): Liquid mass flux (kg/m^2/s)
#     g (float): Gravitational acceleration (m/s^2)
#     d (float): Pipe diameter (m)
   
#     Returns:
#     float: Phi_LM (dimensionless)
#     """
#     # Step 1: Calculate Fr_L
#     Fr_l = G_l**2 / (g * d * rho_l**2)
   
#     # Step 2: Determine f_FR based on Fr_L
#     if Fr_l >= 1:
#         f_fr = 1
#     else:
#         f_fr = Fr_l**0.3 + 0.0055 * (math.log(1 / Fr_l))**2
   
#     # Step 3: Calculate (dp/dz)_Fr
#     dp_dz_fr = f_fr * (x + 4 * (x**1.8 - x**10 * f_fr**0.5))
   
#     # Step 4: Calculate Phi_LM^2
#     phi_lm_squared = 1 + dp_dz_fr * (((rho_l / rho_v) / (mu_l / mu_v)**0.25) - 1)
   
#     return phi_lm_squared  # Return Phi_LM^2

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

def get_wall_viscous_stress_two_phase(args, p, rho, Q, f_D, correlation, Dh, v):
    """
    Compute wall viscous stress for two-phase flow using different correlations.
    
    correlations: "Beattie", "Richardson"
    """
    fluid = args.fluid
    state_L = fluid.get_state(jxp.PQ_INPUTS, p, 0.0)
    state_V = fluid.get_state(jxp.PQ_INPUTS, p, 1.0)

    rho_L = state_L.rho
    rho_V = state_V.rho
    mu_L = state_L.mu
    mu_V = state_V.mu

    roughness = args.roughness

    # Strings are static → normal Python if/elif works
    if correlation == "Beattie":

        """
        Calculate the mixture viscosity with Beattie formula
            x: vapour quality
            y: void fraction
        """

        y = rho_L * Q / (rho_L * Q + rho_V * (1 - Q))
        mu = mu_L * (1 - y) * (1 + 2.5 * y) + mu_V * y 
        Reynolds = v * rho * Dh / jnp.maximum(mu, 1e-12)

        Re_safe = jnp.maximum(Reynolds, 1.0)
        term = 6.9 / Re_safe + (roughness / Dh / 3.7) ** 1.11
        f = (-1.8 * jnp.log10(term)) ** -2

        tau_w = 0.125 * f * rho * v**2


    elif correlation == "Richardson": 

        """
        Calculate the mixture viscosity with Richardson formula
            x: vapour quality
            y: void fraction
        """
        gamma = rho_L * Q / (rho_L * Q + rho_V * (1 - Q))
        phi_lm_squared = (1-gamma)**(-1.75)

        tau_w = f_D * Dh * v**2 * 0.3

    else:
        # Default for single phase
        tau_w = 0.125 * f_D * rho * v**2

    return tau_w

def get_heat_transfer_coefficient(
    velocity, density, heat_capacity, darcy_friction_factor
):
    """
    Estimates the heat transfer using the Reynolds analogy.

    This function is an adaptation of the Reynolds analogy which relates the heat transfer
    coefficient to the product of the Fanning friction factor, velocity, density, and heat
    capacity of the fluid. The Fanning friction factor one fourth of the Darcy friction factor.

    Parameters
    ----------
    velocity : float
        Velocity of the fluid (m/s).
    density : float
        Density of the fluid (kg/m^3).
    heat_capacity : float
        Specific heat capacity of the fluid at constant pressure (J/kg·K).
    darcy_friction_factor : float
        Darcy friction factor, dimensionless.

    Returns
    -------
    float
        Estimated heat transfer coefficient (W/m^2·K).

    """
    fanning_friction_factor = darcy_friction_factor / 4
    return 0.5 * fanning_friction_factor * velocity * density * heat_capacity

# ------------------------------------------------------------------
# Describe the geometry of the converging diverging nozzle
# ------------------------------------------------------------------
def symmetric_nozzle_geometry(x, L, A_inlet=0.30, A_throat=0.15):
    """
    Return A (m^2), dA/dx (m), perimeter (m), diameter (m) for a symmetric parabolic CD nozzle.
    x: position in m (scalar or array)
    L: total length in m (scalar)
    """

    def area_fn(x_):
        xi = x_ / L
        return A_inlet - 4.0 * (A_inlet - A_throat) * xi * (1.0 - xi)

    # make it work for both scalar and array x
    A = area_fn(x)

    # jacfwd works for vector outputs directly
    dAdx = jax.jacfwd(area_fn)(x)

    radius = jnp.sqrt(A / jnp.pi)          # m
    diameter = 2.0 * radius                # m
    perimeter = jnp.pi * diameter          # m
    D_h = 4.0 * A / perimeter

    return A, dAdx, perimeter, diameter, D_h

# Nakagawa geometry
def linear_convergent_divergent_nozzle(
    x,
    L,
    # L_convergent=(83.50e-3 - 56.15e-3),
    L_convergent=0.02735,
    height_in=0.005,
    height_throat=0.00012,
    height_out=0.00027,
    width=0.003,
):
    """
    JAX-safe linear convergent–divergent nozzle (planar geometry).
    Uses jax.lax.cond (JAX's version of 'if') so only the active branch runs.
    Returns (A, dAdx, perimeter, height).
    """
    L_divergent = L - L_convergent

    def convergent(x_):
        h = height_in + (height_throat - height_in) * x_ / L_convergent
        A = 2.0 * h * width
        dAdx = 2.0 * width * (height_throat - height_in) / L_convergent
        return A, dAdx, h

    def divergent(x_):
        h = height_throat + (height_out - height_throat) * (x_ - L_convergent) / L_divergent
        A = 2.0 * h * width
        dAdx = 2.0 * width * (height_out - height_throat) / L_divergent
        return A, dAdx, h

    A, dAdx, h = jax.lax.cond(x <= L_convergent, convergent, divergent, operand=x)
    perimeter = 2.0 * (width + 2.0 * h)
    D_h = 4.0 * A / perimeter

    return A, dAdx, perimeter, h, D_h