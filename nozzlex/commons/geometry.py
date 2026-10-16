import jax
import jax.numpy as jnp
import nurbspy.jax as nrb

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

# Nakagawa geometry
def get_nozzle_nakagawa(
    x,
    L=83.50e-3,
    L_convergent=15.0e-3,
    height_in=5e-3,
    height_throat=0.2e-3,
    height_out=0.72e-3,
    width=1e-3,
    theta=0.48,
):
    """
    JAX-safe linear convergent–divergent nozzle (planar geometry).
    Uses jax.lax.cond (JAX's version of 'if') so only the active branch runs.
    Returns (A, dAdx, perimeter, height).
    """
    L_divergent = L - L_convergent
    theta_rad = jnp.deg2rad(theta)

    def convergent(x_):
        h = height_in + (height_throat - height_in) * x_ / L_convergent
        A = 2.0 * h * width
        dAdx = 2.0 * width * (height_throat - height_in) / L_convergent
        return A, dAdx, h

    def divergent(x_):
        l = x_ - L_convergent
        h = height_throat + l * jnp.tan(theta_rad)
        A = 2.0 * h * width
        dAdx = 2.0 * width * jnp.tan(theta_rad)
        jax.debug.print("h={h}", h=h)
        return A, dAdx, h

    A, dAdx, h = jax.lax.cond(x <= L_convergent, convergent, divergent, operand=x)
    perimeter = 2.0 * (width + 2.0 * h)
    D_h = 4.0 * A / perimeter
    radius = D_h * 0.5

    return A, dAdx, perimeter, h

def get_nozzle_elliot(
    length,
    convergent_length=0.09940,
    divergent_length=0.16652,
    radius_in=0.0254778,
    radius_throat=0.00655,
    radius_out=0.01413,
    throat_first_length = 0.04223,
    radius_throat_in = 0.007952,
    throat_second_length = 0.011512,
    radius_throat_out = 0.0068617,
):
    """
    JAX-compatible version of get_nozzle_elliot.
    Uses lax.cond for control flow so it's JIT/vectorization safe.
    """

    # Geometry constants


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
        points2 = jnp.array([
            [0.0, 0.005, 0.04, 0.04223],
            [radius_throat_in, 0.006995, radius_throat, radius_throat]
        ])

        l = jnp.atleast_1d(l)
        nurbs2 = nrb.NurbsCurve(control_points=points2)
        u = nurbs2.reparametrize_by_coordinate()

        _, radius  = nurbs2.get_value(u(l))
        dxdu, drdu  = nurbs2.get_derivative(u(l), 1)

        radius = radius[0]

        area = jnp.pi * radius ** 2
        perimeter = 2 * jnp.pi * radius

        drdx = drdu[0] / dxdu[0]
        area_slope = perimeter * drdx
        
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

def get_nozzle_elliot_fast(
    length,
    convergent_length=0.09940,
    divergent_length=0.16652,
    radius_in=0.0254778,
    radius_throat=0.00655,
    radius_out=0.01413,
    throat_first_length = 0.04223,
    radius_throat_in = 0.007952,
    throat_second_length = 0.011512,
    radius_throat_out = 0.0068617,
):
    """
    JAX-compatible version of get_nozzle_elliot.
    Uses lax.cond for control flow so it's JIT/vectorization safe.
    """

    # Geometry constants

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

def get_pipe(x, radius = 0.005):
    radius = jnp.float64(radius)
    perimeter = 2 * radius * jnp.pi
    area = jnp.pi * radius ** 2
    area_slope = jnp.float64(0.0)

    return area, area_slope, perimeter, radius

def get_nozzle_simoneau_fast(
    length,
    convergent_length=79.65e-3,
    divergent_length=220e-3,
    radius_in=11.26e-3,
    radius_throat=1.78e-3,
    radius_out=16.31e-3,
    throat_length = 11.35e-3,):
    """
    JAX-compatible version of get_nozzle_simoneau.
    Uses lax.cond for control flow so it's JIT/vectorization safe.
    """

    # Geometry constants

    def region_convergent(length):
        # area = jnp.pi * (radius_in + ((radius_throat_in - radius_in) / convergent_length) * length) ** 2
        # radius = jnp.sqrt(area / jnp.pi)
        # jax.debug.print("{r}",r = length)
        radius = radius_in + (((radius_throat - radius_in) / convergent_length) * length)
        area = jnp.pi * radius**2
        perimeter = 2 * jnp.pi * radius
        area_slope = perimeter *( (radius_throat - radius_in) / convergent_length)
        return area, area_slope, perimeter, radius

    def region_throat(length):
        l = length - convergent_length
        # print((radius_throat - radius_throat_in) )
        # area = jnp.pi * (radius_throat_in + ((radius_throat - radius_throat_in) / throat_first_length) * l) ** 2
        # radius = jnp.sqrt(area / jnp.pi)
        radius = radius_throat
        area = jnp.pi * radius ** 2
        perimeter = 2 * jnp.pi * radius
        area_slope = 0.0
        return area, area_slope, perimeter, radius

    def region_divergent(length):
        # print((radius_out - radius_throat_out))
        l = length - convergent_length - throat_length
        area = jnp.pi * (radius_throat + ((radius_out - radius_throat) / divergent_length) * l) ** 2
        radius = jnp.sqrt(area / jnp.pi)
        radius = radius_throat + (((radius_out - radius_throat) / divergent_length) * l)
        perimeter = 2 * jnp.pi * radius
        area = jnp.pi * radius ** 2
        area_slope = perimeter * ((radius_out - radius_throat) / divergent_length)
        return area, area_slope, perimeter, radius

    # Select the correct region via lax.switch
    conds = [
        length >= convergent_length,
        length >= convergent_length + throat_length
    ]

    idx = jnp.sum(jnp.array(conds))  # 0→convergent, 1→throat_1, 2→throat_2, 3→divergent
    funcs = [region_convergent, region_throat, region_divergent]

    return jax.lax.switch(idx, funcs, length)

def get_nozzle_simoneau_bspline(
    length,
    convergent_length=79.65e-3,
    divergent_length=220e-3,
    radius_in=11.26e-3,
    radius_throat=1.78e-3,
    radius_out=16.31e-3,
    throat_length=11.35e-3,
    bspline_in_convergent_length=30e-3,
):
    """
    Simoneau nozzle with a cubic B-spline raccordatura between the convergent
    section and the constant-area throat region.

    The linear convergent runs from x=0 to x = convergent_length - bspline_in_convergent_length.
    A cubic NURBS then transitions smoothly (C1) from that junction point to
    x = convergent_length + throat_length (throat exit), where radius = radius_throat
    and slope = 0.  The throat (minimum area, sonic point) is at the B-spline exit.

    Sections:
        0: Linear convergent,   x in [0, x_blend_start]
        1: Cubic B-spline,      x in [x_blend_start, x_throat_exit]
        2: Linear divergent,    x in [x_throat_exit, total_length]

    Note: bspline_in_convergent_length must be large enough so that the first
    B-spline control point (P1) stays above radius_throat.  With the default
    geometry, values >= ~10 mm are safe; 30 mm gives a comfortable margin.
    """

    x_blend_start = convergent_length - bspline_in_convergent_length
    x_throat_exit = convergent_length + throat_length

    # Convergent slope and radius at the blend start
    m0 = (radius_throat - radius_in) / convergent_length          # < 0  (converging)
    r_blend = radius_in + m0 * x_blend_start                      # r at x_blend_start

    # B-spline span in local coordinates (0 → delta_x)
    delta_x = x_throat_exit - x_blend_start

    # Cubic Bézier control-point radii for C1 continuity at both ends:
    #   At P0: tangent matches convergent slope  → P1 = P0 + (delta_x/3) * (1, m0)
    #   At P3: tangent is horizontal (slope = 0) → P2 has same r as P3
    r_P1 = r_blend + m0 * (delta_x / 3.0)   # always > radius_throat when bspline_in_convergent_length is large enough
    r_P2 = radius_throat                     # horizontal tangent into P3

    def region_convergent(length):
        radius = radius_in + m0 * length
        area = jnp.pi * radius ** 2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = perimeter * m0
        return area, area_slope, perimeter, radius

    def region_bspline(length):
        l = jnp.atleast_1d(length - x_blend_start)
        points = jnp.array([
            [0.0,     delta_x / 3.0,  2.0 * delta_x / 3.0,  delta_x],
            [r_blend, r_P1,           r_P2,                  radius_throat],
        ])
        nurbs = nrb.NurbsCurve(control_points=points)
        u = nurbs.reparametrize_by_coordinate()
        _, radius_arr = nurbs.get_value(u(l))
        dxdu, drdu = nurbs.get_derivative(u(l), 1)
        radius = radius_arr[0]
        drdx = drdu[0] / dxdu[0]
        area = jnp.pi * radius ** 2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = perimeter * drdx
        return area, area_slope, perimeter, radius

    def region_divergent(length):
        l = length - x_throat_exit
        dr_dx = (radius_out - radius_throat) / divergent_length
        radius = radius_throat + dr_dx * l
        area = jnp.pi * radius ** 2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = perimeter * dr_dx
        return area, area_slope, perimeter, radius

    conds = [
        length >= x_blend_start,
        length >= x_throat_exit,
    ]
    idx = jnp.sum(jnp.array(conds))   # 0 → convergent, 1 → bspline, 2 → divergent
    funcs = [region_convergent, region_bspline, region_divergent]
    return jax.lax.switch(idx, funcs, length)

def get_nozzle_elliot_big(
    length,
    r_inlet          = 7.25   * 0.0254,   # 7.250 in → 0.18415 m
    r_conv_end       = 2.10   * 0.0254,   # 2.100 in → 0.05334 m  (diam 4.2 in)
    convergent_len   = 13.46   * 0.0254,   # 18.5  in → 0.46990 m
    r_div_start      = 1.992  * 0.0254,   # 1.992 in → 0.05060 m  (table x=24 in)
    spline_len       = 24.0   * 0.0254,   # 24.0  in → 0.60960 m
    half_angle_div   = 2.5,               # divergent half-angle [deg]
):
    """
    JAX-compatible geometry for the large Elliot (1979) nozzle.

    Coordinate system: length=0 at nozzle inlet.

    Three regions:
      1. Convergent  [0,  convergent_len]:          linear cone, r_inlet → r_conv_end
      2. Throat      [convergent_len, convergent_len+spline_len]:
                                                    cubic spline through 49 tabulated
                                                    points (x=0..24 in from convergent
                                                    end, y=radius in inches).
                                                    Slopes clamped to match both linear
                                                    sections (C1 continuous overall).
      3. Divergent   [convergent_len+spline_len, ∞): linear cone at half_angle_div,
                                                    open-ended (caller decides exit).

    Source geometry (all dimensions in inches, from Elliot 1979):
      Inlet diam      = 14.50 in (r = 7.25 in)
      Convergent exit = 4.20 in  (r = 2.10 in)  at 18.5 in from inlet
      Throat diam     = 3.18 in  (r = 1.591 in) at ~7.5 in into spline
      Spline end      = r≈1.992 in               at 24 in into spline
      Divergent half-angle = 2.5 deg (open-ended)
    """
    # Precomputed cubic spline for throat section.
    # Breakpoints in metres (x_table * 0.0254), 49 points → 48 segments.
    # r(dx) = c[0]*dx^3 + c[1]*dx^2 + c[2]*dx + c[3], dx = length - break_i
    _breaks = jnp.array([
        0.0000000000, 0.0127000000, 0.0254000000, 0.0381000000,
        0.0508000000, 0.0635000000, 0.0762000000, 0.0889000000,
        0.1016000000, 0.1143000000, 0.1270000000, 0.1397000000,
        0.1524000000, 0.1651000000, 0.1778000000, 0.1905000000,
        0.2032000000, 0.2159000000, 0.2286000000, 0.2413000000,
        0.2540000000, 0.2667000000, 0.2794000000, 0.2921000000,
        0.3048000000, 0.3175000000, 0.3302000000, 0.3429000000,
        0.3556000000, 0.3683000000, 0.3810000000, 0.3937000000,
        0.4064000000, 0.4191000000, 0.4318000000, 0.4445000000,
        0.4572000000, 0.4699000000, 0.4826000000, 0.4953000000,
        0.5080000000, 0.5207000000, 0.5334000000, 0.5461000000,
        0.5588000000, 0.5715000000, 0.5842000000, 0.5969000000,
        0.6096000000,
    ])
    _coeffs = jnp.array([
        # c0 (cubic)
        [ 3.1191509097e+02, -2.0883571099e+02,  3.9826785786e+01, -3.7271605756e+01,
          2.2459463637e+01, -1.5366174393e+01,  1.8051595340e+00, -4.2544885433e+00,
          2.8127698392e+00, -6.9965908134e+00,  3.7354381445e-01,  5.5024155556e+00,
         -9.9831812370e+00,  9.6302597921e+00, -3.7378083314e+00, -7.0790512667e+00,
          7.2539637981e+00,  2.8632456743e+00, -6.3069216954e+00, -2.4356084929e+00,
          3.6493308668e+00,  2.3830982572e-01, -4.6025701697e+00,  5.7719460530e+00,
         -6.0851892421e+00,  6.1687861153e+00, -6.1899304193e+00,  6.1909107617e+00,
         -6.1736878273e+00,  6.1038157475e+00, -5.8415503628e+00,  4.8623609035e+00,
         -1.2078684511e+00, -3.0887098954e-02,  1.3314168469e+00, -5.2947802888e+00,
          7.4476795082e+00, -1.2095912944e+01,  1.6135922668e+01, -1.5247703326e+01,
          7.6548162372e+00, -2.9715368226e+00,  4.2313310531e+00, -1.5537625896e+00,
          1.9837193053e+00, -6.3811146318e+00,  1.1140714422e+01, -1.3381693455e+01],
        # c1 (quadratic)
        [-3.9315280822e+00,  7.9524368838e+00, -4.2037049257e-03,  1.5131968335e+00,
          9.3148654235e-02,  9.4885421881e-01,  3.6340297445e-01,  4.3217955269e-01,
          2.7008353919e-01,  3.7725007007e-01,  1.1067996007e-01,  1.2491197940e-01,
          3.3455401207e-01, -4.5805193054e-02,  3.2110770502e-01,  1.7869720760e-01,
         -9.1014645662e-02,  1.8536137505e-01,  2.9445103524e-01,  5.4157318644e-02,
         -3.8639364934e-02,  1.0040014109e-01,  1.0947974545e-01, -6.5878178014e-02,
          1.5403296660e-01, -7.7812743520e-02,  1.5721800748e-01, -7.8618341499e-02,
          1.5725535852e-01, -7.7962147700e-02,  1.5459323228e-01, -6.7969836541e-02,
          1.1728611388e-01,  7.1266325894e-02,  7.0089527424e-02,  1.2081650929e-01,
         -8.0914619711e-02,  2.0284196955e-01, -2.5801231361e-01,  3.5676634002e-01,
         -2.2417115671e-01,  6.7477341930e-02, -4.5738211010e-02,  1.1547550211e-01,
          5.6277147447e-02,  1.3185685298e-01, -1.1126361449e-01,  3.1319760498e-01],
        # c2 (linear)
        [-2.7837837838e-01, -2.2731283660e-01, -1.2637027523e-01, -1.0720606249e-01,
         -8.6805474799e-02, -7.3572038311e-02, -5.6906371957e-02, -4.6802473862e-02,
         -3.7883732595e-02, -2.9662595758e-02, -2.3465884375e-02, -2.0473866743e-02,
         -1.4638648652e-02, -1.0971538650e-02, -7.4751967481e-03, -1.1276743577e-03,
         -1.4105821095e-05,  1.1840976421e-03,  7.2777152527e-03,  1.1705041347e-02,
          1.1902119359e-02,  1.2686481216e-02,  1.5351955775e-02,  1.5905695682e-02,
          1.7025261497e-02,  1.7993258330e-02,  1.9001705182e-02,  1.9999920940e-02,
          2.0998611056e-02,  2.2005634834e-02,  2.2978849608e-02,  2.4078966734e-02,
          2.4705283456e-02,  2.7099899441e-02,  2.8895118778e-02,  3.1319625445e-02,
          3.1826379442e-02,  3.3374856785e-02,  3.2674193416e-02,  3.3928369551e-02,
          3.5612328379e-02,  3.3622316932e-02,  3.3898403894e-02,  3.4784067491e-02,
          3.6965326141e-02,  3.9354627946e-02,  3.9616162075e-02,  4.2180723754e-02],
        # c3 (constant = radius at break point)
        [ 5.3340000000e-02,  4.9809400000e-02,  4.7777400000e-02,  4.6253400000e-02,
          4.5059600000e-02,  4.4018200000e-02,  4.3205400000e-02,  4.2545000000e-02,
          4.2011600000e-02,  4.1579800000e-02,  4.1249600000e-02,  4.0970200000e-02,
          4.0741600000e-02,  4.0589200000e-02,  4.0462200000e-02,  4.0411400000e-02,
          4.0411400000e-02,  4.0411400000e-02,  4.0462200000e-02,  4.0589200000e-02,
          4.0741600000e-02,  4.0894000000e-02,  4.1071800000e-02,  4.1275000000e-02,
          4.1478200000e-02,  4.1706800000e-02,  4.1935400000e-02,  4.2189400000e-02,
          4.2443400000e-02,  4.2722800000e-02,  4.3002200000e-02,  4.3307000000e-02,
          4.3611800000e-02,  4.3942000000e-02,  4.4297600000e-02,  4.4678600000e-02,
          4.5085000000e-02,  4.5491400000e-02,  4.5923200000e-02,  4.6329600000e-02,
          4.6786800000e-02,  4.7218600000e-02,  4.7650400000e-02,  4.8082200000e-02,
          4.8539400000e-02,  4.9022000000e-02,  4.9530000000e-02,  5.0038000000e-02],
    ])

    n_seg       = _coeffs.shape[1]          # 48
    spline_x0   = convergent_len            # absolute start of spline section
    div_x0      = convergent_len + spline_len  # absolute start of divergent section
    slope_div   = jnp.tan(jnp.deg2rad(half_angle_div))

    # ── region 1: convergent (linear) ────────────────────────────────────────
    def region_convergent(length):
        dr_dx  = (r_conv_end - r_inlet) / convergent_len
        radius = r_inlet + dr_dx * length
        area      = jnp.pi * radius**2
        perimeter = 2.0 * jnp.pi * radius
        dAdx      = 2.0 * jnp.pi * radius * dr_dx
        return area, dAdx, perimeter, radius

    # ── region 2: throat (cubic spline) ──────────────────────────────────────
    def region_throat(length):
        xl    = length - spline_x0               # local coordinate [0, spline_len]
        idx   = jnp.searchsorted(_breaks, xl, side="right") - 1
        idx   = jnp.clip(idx, 0, n_seg - 1)
        dx    = xl - _breaks[idx]
        c0    = _coeffs[0, idx]
        c1    = _coeffs[1, idx]
        c2    = _coeffs[2, idx]
        c3    = _coeffs[3, idx]
        radius = c0 * dx**3 + c1 * dx**2 + c2 * dx + c3
        drdx   = 3.0 * c0 * dx**2 + 2.0 * c1 * dx + c2
        area      = jnp.pi * radius**2
        perimeter = 2.0 * jnp.pi * radius
        dAdx      = 2.0 * jnp.pi * radius * drdx
        return area, dAdx, perimeter, radius

    # ── region 3: divergent (linear, open-ended) ─────────────────────────────
    def region_divergent(length):
        xl     = length - div_x0
        radius = r_div_start + slope_div * xl
        area      = jnp.pi * radius**2
        perimeter = 2.0 * jnp.pi * radius
        dAdx      = 2.0 * jnp.pi * radius * slope_div
        return area, dAdx, perimeter, radius

    idx = jnp.sum(jnp.array([
        length >= spline_x0,
        length >= div_x0,
    ]))  # 0→convergent, 1→throat, 2→divergent

    return jax.lax.switch(idx, [region_convergent, region_throat, region_divergent], length)

def get_nozzle_smd(
    length,
    z_transition=0.089805825,
    r_const=0.010065,
    z_end=0.463106796,
):
    """
    JAX-compatible Supermobidick nozzle geometry.
    Cubic spline in the converging section, constant radius after.
    Uses lax.cond for control flow so it's JIT/vectorization safe.

    The spline coefficients are precomputed from nozzle_coordinates_smd.csv
    using scipy.interpolate.CubicSpline with bc_type=((2, 0), (1, 0))
    (natural left BC, clamped right BC with dr/dx=0 at the transition).
    """
    # Precomputed cubic spline breakpoints and coefficients for the converging section
    # Coefficients layout: for segment i, r(dx) = c0*dx^3 + c1*dx^2 + c2*dx + c3
    # where dx = length - breaks[i]
    _breaks = jnp.array([
        0.0, 0.00970874, 0.01941748, 0.02961165, 0.04029126,
        0.04951456, 0.05970874, 0.07038835, 0.08058252, 0.08980583,
    ])
    _coeffs = jnp.array([
        [-5.15687535e+02,  2.36106419e+02,  2.13628410e+01,
          3.45126602e+02,  4.88402424e+02, -4.79989669e+02,
          3.25965960e+01, -2.52971271e+02,  7.58975837e+02],
        [-1.77635684e-15, -1.50200255e+01, -8.14313939e+00,
         -7.48980984e+00,  3.56764475e+00,  1.70816925e+01,
          2.40239639e+00,  3.44675339e+00, -4.28974607e+00],
        [ 1.44451489e-02, -1.31380343e-01, -3.56265442e-01,
         -5.15630447e-01, -5.57517648e-01, -3.67062596e-01,
         -1.68438384e-01, -1.05971734e-01, -1.14565348e-01],
        [ 3.35000000e-02,  3.31683170e-02,  3.06930690e-02,
          2.62376240e-02,  2.02970300e-02,  1.58415840e-02,
          1.33663370e-02,  1.18811880e-02,  1.08910890e-02],
    ])

    def converging(length):
        n_seg = _coeffs.shape[1]
        idx = jnp.searchsorted(_breaks, length, side="right") - 1
        idx = jnp.clip(idx, 0, n_seg - 1)
        dx = length - _breaks[idx]
        c0 = _coeffs[0, idx]
        c1 = _coeffs[1, idx]
        c2 = _coeffs[2, idx]
        c3 = _coeffs[3, idx]
        radius = c0 * dx**3 + c1 * dx**2 + c2 * dx + c3
        drdx = 3.0 * c0 * dx**2 + 2.0 * c1 * dx + c2
        area = jnp.pi * radius**2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = 2.0 * jnp.pi * radius * drdx
        return area, area_slope, perimeter, radius

    def constant(length):
        radius = jnp.float64(r_const)
        area = jnp.pi * radius**2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = jnp.float64(0.0)
        return area, area_slope, perimeter, radius

    return jax.lax.cond(length < z_transition, converging, constant, length)

def get_nozzle_smd_divergent(
    length,
    z_transition=0.089805825,
    r_const=0.010065,
    z_const_end=0.463106796,
    z_div_end=0.900106796,
    r_outlet=0.03685,
):
    """
    JAX-compatible Supermobidick nozzle geometry with divergent section.
    Converging spline -> constant radius throat -> diverging (linear) to outlet.

    Geometry from drawing (all converted to metres):
      - Converging section: 0 to z_transition (~89.8 mm)
      - Constant throat:    z_transition to z_const_end (~463.1 mm)
      - Diverging section:  z_const_end to z_div_end (~900.1 mm)
        Throat radius = 10.065 mm, outlet radius = 36.85 mm
        Linear radius profile in the divergent: r(x) = r_const + (r_outlet - r_const) * (x - z_const_end) / (z_div_end - z_const_end)
    """
    # Precomputed cubic spline breakpoints and coefficients for the converging section
    _breaks = jnp.array([
        0.0, 0.00970874, 0.01941748, 0.02961165, 0.04029126,
        0.04951456, 0.05970874, 0.07038835, 0.08058252, 0.08980583,
    ])
    _coeffs = jnp.array([
        [-5.15687535e+02,  2.36106419e+02,  2.13628410e+01,
          3.45126602e+02,  4.88402424e+02, -4.79989669e+02,
          3.25965960e+01, -2.52971271e+02,  7.58975837e+02],
        [-1.77635684e-15, -1.50200255e+01, -8.14313939e+00,
         -7.48980984e+00,  3.56764475e+00,  1.70816925e+01,
          2.40239639e+00,  3.44675339e+00, -4.28974607e+00],
        [ 1.44451489e-02, -1.31380343e-01, -3.56265442e-01,
         -5.15630447e-01, -5.57517648e-01, -3.67062596e-01,
         -1.68438384e-01, -1.05971734e-01, -1.14565348e-01],
        [ 3.35000000e-02,  3.31683170e-02,  3.06930690e-02,
          2.62376240e-02,  2.02970300e-02,  1.58415840e-02,
          1.33663370e-02,  1.18811880e-02,  1.08910890e-02],
    ])

    def converging(length):
        n_seg = _coeffs.shape[1]
        idx = jnp.searchsorted(_breaks, length, side="right") - 1
        idx = jnp.clip(idx, 0, n_seg - 1)
        dx = length - _breaks[idx]
        c0 = _coeffs[0, idx]
        c1 = _coeffs[1, idx]
        c2 = _coeffs[2, idx]
        c3 = _coeffs[3, idx]
        radius = c0 * dx**3 + c1 * dx**2 + c2 * dx + c3
        drdx = 3.0 * c0 * dx**2 + 2.0 * c1 * dx + c2
        area = jnp.pi * radius**2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = 2.0 * jnp.pi * radius * drdx
        return area, area_slope, perimeter, radius

    def constant(length):
        radius = jnp.float64(r_const)
        area = jnp.pi * radius**2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = jnp.float64(0.0)
        return area, area_slope, perimeter, radius

    def diverging(length):
        L_div = z_div_end - z_const_end
        drdx = (r_outlet - r_const) / L_div
        radius = r_const + drdx * (length - z_const_end)
        area = jnp.pi * radius**2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = 2.0 * jnp.pi * radius * drdx
        return area, area_slope, perimeter, radius

    def const_or_div(length):
        return jax.lax.cond(length < z_const_end, constant, diverging, length)

    return jax.lax.cond(length < z_transition, converging, const_or_div, length)


def get_nozzle_smd_divergent_smooth(
    length,
    z_transition=0.089805825,
    r_const=0.010065,
    z_const_end=0.463106796,
    z_div_end=0.900106796,
    r_outlet=0.03685,
    blend_width=0.02,
):
    """
    Same geometry as get_nozzle_smd_divergent but with a C1-smooth cubic
    Hermite blend over [z_const_end, z_const_end + blend_width] so that
    dA/dx ramps smoothly from 0 to the divergent slope after the throat.
    The constant-radius section is preserved up to exactly z_const_end.
    """
    _breaks = jnp.array([
        0.0, 0.00970874, 0.01941748, 0.02961165, 0.04029126,
        0.04951456, 0.05970874, 0.07038835, 0.08058252, 0.08980583,
    ])
    _coeffs = jnp.array([
        [-5.15687535e+02,  2.36106419e+02,  2.13628410e+01,
          3.45126602e+02,  4.88402424e+02, -4.79989669e+02,
          3.25965960e+01, -2.52971271e+02,  7.58975837e+02],
        [-1.77635684e-15, -1.50200255e+01, -8.14313939e+00,
         -7.48980984e+00,  3.56764475e+00,  1.70816925e+01,
          2.40239639e+00,  3.44675339e+00, -4.28974607e+00],
        [ 1.44451489e-02, -1.31380343e-01, -3.56265442e-01,
         -5.15630447e-01, -5.57517648e-01, -3.67062596e-01,
         -1.68438384e-01, -1.05971734e-01, -1.14565348e-01],
        [ 3.35000000e-02,  3.31683170e-02,  3.06930690e-02,
          2.62376240e-02,  2.02970300e-02,  1.58415840e-02,
          1.33663370e-02,  1.18811880e-02,  1.08910890e-02],
    ])

    drdx_div = (r_outlet - r_const) / (z_div_end - z_const_end)
    x1_blend = z_const_end + blend_width

    def converging(length):
        n_seg = _coeffs.shape[1]
        idx = jnp.searchsorted(_breaks, length, side="right") - 1
        idx = jnp.clip(idx, 0, n_seg - 1)
        dx = length - _breaks[idx]
        c0, c1, c2, c3 = _coeffs[0, idx], _coeffs[1, idx], _coeffs[2, idx], _coeffs[3, idx]
        radius = c0 * dx**3 + c1 * dx**2 + c2 * dx + c3
        drdx = 3.0 * c0 * dx**2 + 2.0 * c1 * dx + c2
        area = jnp.pi * radius**2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = 2.0 * jnp.pi * radius * drdx
        return area, area_slope, perimeter, radius

    def constant(length):
        radius = jnp.float64(r_const)
        area = jnp.pi * radius**2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = jnp.float64(0.0)
        return area, area_slope, perimeter, radius

    def blend(length):
        # Cubic Hermite on [z_const_end, x1_blend]:
        #   at z_const_end: r = r_const, dr/dx = 0
        #   at x1_blend:    r = r_const + drdx_div * blend_width, dr/dx = drdx_div
        h = blend_width
        r0 = r_const
        r1 = r_const + drdx_div * blend_width
        t = (length - z_const_end) / h
        H00 =  2*t**3 - 3*t**2 + 1
        H01 = -2*t**3 + 3*t**2
        H11 =    t**3 -   t**2
        # dr0=0 so the H10 term vanishes
        radius = r0 * H00 + r1 * H01 + (drdx_div * h) * H11
        dH00 =  6*t**2 - 6*t
        dH01 = -6*t**2 + 6*t
        dH11 =  3*t**2 - 2*t
        drdx = (r0 * dH00 + r1 * dH01 + (drdx_div * h) * dH11) / h
        area = jnp.pi * radius**2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = 2.0 * jnp.pi * radius * drdx
        return area, area_slope, perimeter, radius

    def diverging(length):
        radius = r_const + drdx_div * (length - z_const_end)
        area = jnp.pi * radius**2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = 2.0 * jnp.pi * radius * drdx_div
        return area, area_slope, perimeter, radius

    conds = jnp.array([
        length >= z_transition,
        length >= z_const_end,
        length >= x1_blend,
    ], dtype=jnp.int32)
    idx = jnp.sum(conds)  # 0=converging, 1=constant, 2=blend, 3=diverging
    return jax.lax.switch(idx, [converging, constant, blend, diverging], length)


def get_pipe(x, radius = 0.005):
    radius = jnp.float64(radius)
    perimeter = 2 * radius * jnp.pi
    area = jnp.pi * radius ** 2
    area_slope = jnp.float64(0.0)

    return area, area_slope, perimeter, radius
