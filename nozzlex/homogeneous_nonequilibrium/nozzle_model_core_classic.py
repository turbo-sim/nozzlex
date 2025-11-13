import jax
import jax.numpy as jnp
import jaxprop as jxp
from jax.numpy.linalg import det
import equinox as eqx




# def nozzle_homogeneous_non_equilibrium_core(t, y, args):
#     """Wrapper that adapts from (t, y) to autonomous form."""
#     Y = jnp.concatenate([jnp.atleast_1d(t), y])
#     return nozzle_homogeneous_non_equilibrium_autonomous(0.0, Y, args)

@eqx.filter_jit
def nozzle_homogeneous_non_equilibrium_classic(x, Y, args):
    """
    Autonomous formulation of the nozzle equations:
        dx/dt   = det(A)
        dy_i/dt = det(A with column i replaced by b)

    State vector: Y = [x, v, rho, p]
    """
    alpha1, alpha2, rho1, rho2, u, p, h1, h2 = Y

    # v = jnp.where(jnp.abs(v) < 1e-6, jnp.sign(v)*1e-6, v)
    # Debug print using jax.debug.print
    # jax.debug.print(
    # "x={:.2f}, alpha1={:.4e}, alpha2={:.4e}, rho1={:.4e}, rho2={:.4e}, u={:.4e}, p={:.4e}, h1={:.4e}, h2={:.4e}",
    # x, alpha1, alpha2, rho1, rho2, u, p, h1, h2
    # )   

    # --- Geometry / model parameters ---
    fluid1 = args.fluid1
    fluid2 = args.fluid2
    L = args.length
    eps_wall = args.roughness
    wall_friction = args.wall_friction
    heat_transfer = args.heat_transfer

    # --- Thermodynamic state fluid 1 ---
    alpha1 = jnp.clip(alpha1, 0, 1)
    alpha2 = jnp.clip(alpha2, 0, 1)
    h1 = jnp.clip(h1, -1e5, 1e7)  # enforce valid density range
    h2 = jnp.clip(h2, -1e5, 1e7)  # enforce valid density range
    p = jnp.clip(p, 1e2, 1e9)   # enforce valid pressure range
    state1 = fluid1.get_state(jxp.HmassP_INPUTS, h1, p)
    T1 = state1["T"]
    s1 = state1["s"]
    a1 = state1["a"]
    cp1 = state1["cp"]
    mu1 = state1["mu"]
    G1 = state1["gruneisen"]

    # # Stagnation state
    # h01 = state1["h"] + 0.5 * u**2
    # state01 = fluid1.get_state(jxp.HmassSmass_INPUTS, h01, state1["s"])
    # p01 = state01["p"]
    # T01 = state01["T"]
    # rho01 = state01["d"]

    # --- Thermodynamic state fluid 2 ---
    state2 = fluid2.get_state(jxp.HmassP_INPUTS, h2, p)
    T2 = state2["T"]
    s2 = state2["s"]
    a2 = state2["a"]
    cp2 = state2["cp"]
    mu2 = state2["mu"]
    G2 = state2["gruneisen"]

    # # Stagnation state
    # h02 = state2["h"] + 0.5 * u**2
    # state02 = fluid2.get_state(jxp.HmassSmass_INPUTS, h02, state1["s"])
    # p02 = state02["p"]
    # T02 = state02["T"]
    # rho02 = state02["d"]

    rho_mix = rho1 * alpha1 + rho2 * alpha2
    mu_mix = mu1 * alpha1 + mu2 * alpha2

    S = (rho1**2 * rho2**2 * u**3)**(1/8)

    # --- Geometry ---
    # A, dAdx, perimeter, radius = args.geometry(x, L) # When using symmetric geometry
    A, dAdx, perimeter, radius = args.geometry(x) # When using linear convergent divergent geometry
    diameter = 2 * radius

    # --- Wall heat transfer and friction ---
    Re_mix = u * rho_mix * diameter / mu_mix # jnp.maximum(mu_mix, 1e-12)
    f_D = get_friction_factor_haaland(Re_mix, eps_wall, diameter)
    tau_w_mix = get_wall_viscous_stress(f_D, rho_mix, u)
    Ai, d_2 = interfacial_area(alpha2)
    ht_1 = 6 # water
    Nu_2 = 12
    k_2 = state2["k"]
    ht_2 = (k_2 * Nu_2) / d_2 # nitrogen
    ht_21 = (1 / ((1/ht_1) + (1/ht_2)))

    a_mix = get_speed_of_sound_mixture(G1, alpha1, a1, rho1, G2, alpha2, a2, rho2)

    # Mask with booleans (convert to 0.0 if disabled)
    # tau_w_mix = wall_friction * tau_w_mix
    # f_D = wall_friction * f_D

    # a_mix_stadke = get_speed_of_sound_mixture_stadke(alpha1, a1, rho1, alpha2, a2, rho2)

    

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
    
    A_mat = A_mat / S
    det_D = det(A_mat)
    # det_D=1.0*alpha1*alpha2*rho1*rho2*u**3*(G1*a2**2*alpha1*rho1*rho2*u**2 - G1*a2**2*alpha1*rho2*rho_mix*u**2 + G2*a1**2*alpha2*rho1*rho2*u**2 - G2*a1**2*alpha2*rho1*rho_mix*u**2 + a1**2*a2**2*alpha1*rho1*rho2 + a1**2*a2**2*alpha2*rho1*rho2 - a1**2*alpha2*rho1*rho_mix*u**2 - a2**2*alpha1*rho2*rho_mix*u**2)/(a1**2*a2**2)


    b_vec = jnp.asarray(
                [
                0.0,
                -(dAdx/A) * alpha1 * rho1 *  u,
                -(dAdx/A) * alpha2 * rho2 *  u,
                -(tau_w_mix * perimeter) / A,
                ht_21 * Ai * (T2 - T1),
                ht_21 * Ai * (T1 - T2),
                0.0,
                0.0
                ]
            )
    
    b_vec = b_vec / S

    # --- Determinants ---
    
    rhs = jnp.linalg.solve(A_mat, b_vec)
    # jax.debug.print("x={x}, det={det_D}, Ma={Ma_mix}, delta_det={dd}", x=x, det_D=det_D, Ma_mix=u/a_mix, dd=det_D-det_sympy_actual)

    # jax.debug.print(
    #     "x={x}, Ma={Ma},det={D}", x=x, Ma=u/a_mix,D=det_D
    # )
    # jax.debug.print(
    #     "x={x}, Ma={Ma}, Ma_st={Ma_st}, det={D}", x=x, Ma=u/a_mix, Ma_st=u/a_mix_stadke,D=det_D
    # )

    # jax.debug.print("x={x} | determinant={D} |  Ma={Ma_mix}", Ma_mix=u/a_mix, D=det_D, x=x)

    
    # --- Autonomous system: dx/dτ = D, dy/dτ = N_i ---
    # rhs = jnp.array([det_N1 / det_D, det_N2 / det_D, det_N3 / det_D, det_N4 / det_D, det_N5 / det_D, det_N6 / det_D, det_N7 / det_D, det_N8 / det_D])
    # rhs_autonomous = jnp.array([det_D, det_N1, det_N2, det_N3, det_N4, det_N5, det_N6, det_N7, det_N8])

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
        # "rhs_autonomous": rhs_autonomous,
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
        "a_mix":a_mix,
        "Ma2": u / a2,
        "Ma_mix":u / a_mix,
        "Re": Re_mix,
        "f_D": f_D,
        "tau_w": tau_w_mix,
        "A_mat": A_mat,
        "b_vec": b_vec,
        "D": det_D,
        # "N": N,
        "m_dot": u * rho_mix * A,
    }

    return {**out, **state1.to_dict(include_aliases=True),  **state2.to_dict(include_aliases=True)}
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


# ------------------------------------------------------------------
# Describe the geometry of the converging diverging nozzle
# ------------------------------------------------------------------
def symmetric_nozzle_geometry(x, L=0.2, A_inlet=0.30, A_throat=0.15):
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

    return A, dAdx, perimeter, diameter

# # Nakagawa featurs now
# def linear_convergent_divergent_nozzle_old(
#     x,
#     convergent_length=(83.50e-3 - 56.15e-3),
#     divergent_length=56.15e-3,
#     radius_in=5e-3,
#     radius_throat=0.12e-3,
#     radius_out=0.72e-3,
#     axisymmetric=False,
#     width=3e-3,
# ):
#     """
#     JAX-safe linear convergent-divergent nozzle.
#     - If axisymmetric=True: A = pi r^2
#     - Else (planar):       A = 2 r * width
#     Uses JAX control flow to avoid TracerBoolConversionError.
#     Returns: (A, dAdx, perimeter, radius) with shapes matching x.
#     """
#     x = jnp.asarray(x)

#     if axisymmetric:  
#         area_in     = jnp.pi * radius_in**2
#         area_throat = jnp.pi * radius_throat**2
#         area_out    = jnp.pi * radius_out**2
#     else:
#         area_in     = 2.0 * radius_in * width
#         area_throat = 2.0 * radius_throat * width
#         area_out    = 2.0 * radius_out * width

#     dAdx_conv = (area_throat - area_in) / convergent_length
#     dAdx_div  = (area_out    - area_throat) / divergent_length

#     # JAX piecewise selection based on x
#     cond = x <= convergent_length
#     A    = jnp.where(cond,
#                      area_in + dAdx_conv * x,
#                      area_throat + dAdx_div * (x - convergent_length))
#     dAdx = jnp.where(cond, dAdx_conv, dAdx_div)

#     if axisymmetric:
#         radius    = jnp.sqrt(A / jnp.pi)
#         perimeter = 2.0 * jnp.pi * radius
#     else:
#         radius    = A / (2.0 * width)
#         perimeter = 2.0 * (width + 2.0 * radius)

#     return A, dAdx, perimeter, radius

def get_nozzle_elliot_old(
    length,
    total_length=150.93*1e-3,
    convergent_length=99.40*1e-3,
    divergent_length=97.18*1e-3,
    radius_in=50.96/2*1e-3,
    radius_throat=13.12/2*1e-3,
    radius_out=22.21*1e-3,
):

    # Dimensions comes for Amit's 3D scketch for turbo expo 2024
    throat_second_lenght = 11.51 * 1e-3
    throat_first_length = 42.23 * 1e-3
    radius_throat_in = 15.90/2 * 1e-3
    radius_throat_out = 13.72/2 * 1e-3
    area_in = jnp.pi * radius_in ** 2
    area_throat_in = jnp.pi * radius_throat_in ** 2
    area_throat_out = jnp.pi * radius_throat_out ** 2
    area_throat = jnp.pi * radius_throat ** 2
    area_out = jnp.pi * radius_out ** 2


    if length <= convergent_length:
        # area_slope = 2*np.pi*((radius_in+((radius_throat_in-radius_in)/convergent_length)*length)*((radius_throat_in-radius_in)/convergent_length))
        area = jnp.pi*((radius_in+((radius_throat_in-radius_in)/convergent_length)*length)**2)
        radius = jnp.sqrt(area / jnp.pi)
        perimeter = 2 * jnp.pi * radius
        area_slope = perimeter * (radius_throat_in-radius_in)/convergent_length

    # Throat section convergent
    elif convergent_length < length <= convergent_length + throat_first_length:
        length = length-convergent_length
        area = jnp.pi*((radius_throat_in+((radius_throat-radius_throat_in)/throat_first_length)*length)**2)
        radius = jnp.sqrt(area / jnp.pi)
        perimeter = 2 * jnp.pi * radius
        area_slope = perimeter * (radius_throat-radius_throat_in)/throat_first_length

    # Throat section divergent
    elif convergent_length + throat_first_length < length <= convergent_length + throat_first_length + throat_second_lenght:
        length = length - convergent_length - throat_first_length
        # area_slope = 2*np.pi*((radius_throat+((radius_throat_out-radius_throat)/throat_second_lenght))*length*((radius_throat_out-radius_throat)/throat_second_lenght))
        area = jnp.pi*((radius_throat+((radius_throat_out-radius_throat)/throat_second_lenght)*length)**2)
        radius = jnp.sqrt(area / jnp.pi)
        perimeter = 2 * jnp.pi * radius
        area_slope = perimeter * ((radius_throat_out-radius_throat)/throat_second_lenght)

    # Throat section divergent
    # elif length > convergent_length + throat_first_length + throat_second_lenght:
    else:
        length = length - convergent_length - throat_first_length - throat_second_lenght
        # area_slope = 2*np.pi*((radius_throat_out+((radius_out-radius_throat_out)/divergent_length))*length)*((radius_out-radius_throat_out)/divergent_length)
        area = jnp.pi*((radius_throat_out+((radius_out-radius_throat_out)/divergent_length)*length)**2)
        radius = jnp.sqrt(area / jnp.pi)
        perimeter = 2 * jnp.pi * radius
        area_slope = perimeter * ((radius_out-radius_throat_out)/divergent_length)           

    # radius = jnp.sqrt(area / jnp.pi)
    # perimeter = 2 * jnp.pi * radius

    return area, area_slope, perimeter, radius


def get_nozzle_elliot(
    length,
    total_length=150.93e-3,
    convergent_length=99.40e-3,
    divergent_length=97.18e-3,
    radius_in=50.96/2*1e-3,
    radius_throat=13.12/2*1e-3,
    radius_out=22.21e-3,
):
    """
    JAX-compatible version of get_nozzle_elliot.
    Uses lax.cond for control flow so it's JIT/vectorization safe.
    """

    # Geometry constants
    throat_second_length = 11.51e-3
    throat_first_length = 42.23e-3
    radius_throat_in = 15.90/2 * 1e-3
    radius_throat_out = 13.72/2 * 1e-3

    def region_convergent(length):
        # area = jnp.pi * (radius_in + ((radius_throat_in - radius_in) / convergent_length) * length) ** 2
        # radius = jnp.sqrt(area / jnp.pi)
        # jax.debug.print("{r}",r = length)
        radius = radius_in + (((radius_throat_in - radius_in) / convergent_length) * length)
        area = jnp.pi * radius**2
        perimeter = 2 * jnp.pi * radius
        area_slope = perimeter *( (radius_throat_in - radius_in) / convergent_length)
        return area, area_slope, perimeter, radius

    def region_throat_1(length):
        l = length - convergent_length
        # print((radius_throat - radius_throat_in) )
        # area = jnp.pi * (radius_throat_in + ((radius_throat - radius_throat_in) / throat_first_length) * l) ** 2
        # radius = jnp.sqrt(area / jnp.pi)
        radius = radius_throat_in + (((radius_throat - radius_throat_in) / throat_first_length) * l)
        area = jnp.pi * radius ** 2
        perimeter = 2 * jnp.pi * radius
        area_slope = perimeter * ((radius_throat - radius_throat_in) / throat_first_length)
        return area, area_slope, perimeter, radius

    def region_throat_2(length):
        l = length - convergent_length - throat_first_length
        # print((radius_throat_out - radius_throat))
        # area = jnp.pi * (radius_throat + ((radius_throat_out - radius_throat) / throat_second_length) * l) ** 2
        # radius = jnp.sqrt(area / jnp.pi)
        radius = radius_throat + (((radius_throat_out - radius_throat) / throat_second_length) * l)
        area = jnp.pi * radius ** 2
        perimeter = 2 * jnp.pi * radius
        area_slope = perimeter * ((radius_throat_out - radius_throat) / throat_second_length)
        return area, area_slope, perimeter, radius

    def region_divergent(length):
        # print((radius_out - radius_throat_out))
        l = length - convergent_length - throat_first_length - throat_second_length
        area = jnp.pi * (radius_throat_out + ((radius_out - radius_throat_out) / divergent_length) * l) ** 2
        radius = jnp.sqrt(area / jnp.pi)
        radius = radius_throat_out + (((radius_out - radius_throat_out) / divergent_length) * l)
        perimeter = 2 * jnp.pi * radius
        area = jnp.pi * radius ** 2
        area_slope = perimeter * ((radius_out - radius_throat_out) / divergent_length)
        return area, area_slope, perimeter, radius

    # Select the correct region via lax.switch
    conds = [
        length >= convergent_length,
        length >= convergent_length + throat_first_length,
        length >= convergent_length + throat_first_length + throat_second_length,
    ]

    idx = jnp.sum(jnp.array(conds))  # 0→convergent, 1→throat_1, 2→throat_2, 3→divergent
    funcs = [region_convergent, region_throat_1, region_throat_2, region_divergent]

    return jax.lax.switch(idx, funcs, length)


def get_nozzle_elliot_wrong(
    length,
    total_length=150.93e-3,
    convergent_length=99.40e-3,
    divergent_length=97.18e-3,
    radius_in=50.96/2*1e-3,
    radius_throat=13.12/2*1e-3,
    radius_out=22.21e-3,
):
    """
    JAX-compatible version of get_nozzle_elliot.
    Uses lax.cond for control flow so it's JIT/vectorization safe.
    """

    # Geometry constants
    throat_second_length = 11.51e-3
    throat_first_length = 42.23e-3
    radius_throat_in = 15.90/2 * 1e-3
    radius_throat_out = 13.72/2 * 1e-3

    def region_convergent(length):
        # area = jnp.pi * (radius_in + ((radius_throat_in - radius_in) / convergent_length) * length) ** 2
        # radius = jnp.sqrt(area / jnp.pi)
        # jax.debug.print("{r}",r = length)
        radius = radius_in + (((radius_throat - radius_in) / convergent_length) * length)
        area = jnp.pi * radius**2
        perimeter = 2 * jnp.pi * radius
        area_slope = perimeter *( (radius_throat - radius_in) / convergent_length)
        return area, area_slope, perimeter, radius

    # def region_throat_1(length):
    #     l = length - convergent_length
    #     # print((radius_throat - radius_throat_in) )
    #     # area = jnp.pi * (radius_throat_in + ((radius_throat - radius_throat_in) / throat_first_length) * l) ** 2
    #     # radius = jnp.sqrt(area / jnp.pi)
    #     radius = radius_throat_in + (((radius_throat - radius_throat_in) / throat_first_length) * l)
    #     area = jnp.pi * radius ** 2
    #     perimeter = 2 * jnp.pi * radius
    #     area_slope = perimeter * ((radius_throat - radius_throat_in) / throat_first_length)
    #     return area, area_slope, perimeter, radius

    # def region_throat_2(length):
    #     l = length - convergent_length - throat_first_length
    #     # print((radius_throat_out - radius_throat))
    #     # area = jnp.pi * (radius_throat + ((radius_throat_out - radius_throat) / throat_second_length) * l) ** 2
    #     # radius = jnp.sqrt(area / jnp.pi)
    #     radius = radius_throat + (((radius_throat_out - radius_throat) / throat_second_length) * l)
    #     area = jnp.pi * radius ** 2
    #     perimeter = 2 * jnp.pi * radius
    #     area_slope = perimeter * ((radius_throat_out - radius_throat) / throat_second_length)
    #     return area, area_slope, perimeter, radius

    def region_divergent(length):
        # print((radius_out - radius_throat_out))
        l = length - convergent_length
        # area = jnp.pi * (radius_throat + ((radius_out - radius_throat) / divergent_length) * l) ** 2
        radius = radius_throat + (((radius_out - radius_throat) / divergent_length) * l)
        perimeter = 2 * jnp.pi * radius
        area = jnp.pi * radius ** 2
        area_slope = perimeter * ((radius_out - radius_throat) / divergent_length)
        return area, area_slope, perimeter, radius

    # Select the correct region via lax.switch
    conds = [
        length >= convergent_length,
        # length >= convergent_length + throat_first_length,
        # length >= convergent_length + throat_first_length + throat_second_length,
    ]

    idx = jnp.sum(jnp.array(conds))  # 0→convergent, 1→throat_1, 2→throat_2, 3→divergent
    funcs = [region_convergent, region_divergent]

    return jax.lax.switch(idx, funcs, length)



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

def interfacial_area(alpha, alpha_b=0.3, alpha_d=0.7, Nb=1e10, Nd=1e8):
    """
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

    d = jnp.sqrt(Ai / (N * jnp.pi))
    return Ai, d



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
