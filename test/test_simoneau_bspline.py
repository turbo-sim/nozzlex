import jax.numpy as jnp
import nurbspy.jax as nrb
import matplotlib.pyplot as plt
import jax
import numpy as np


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


# Example usage
if __name__ == "__main__":
    # ctrl_pts = get_nozzle_control_points()
    # print(ctrl_pts)


    # nozzle = nrb.NurbsCurve(control_points=ctrl_pts)
    # nozzle.plot()
    # plt.show()

    X = np.linspace(0, 0.311,2000)

    RADIUS_fast = []
    dAdx_fast = []
    RADIUS_bspline = []
    dAdx_bspline = []

    for x in X:
        _, area_slope, _, radius = get_nozzle_simoneau_fast(x)
        RADIUS_fast.append(radius)
        dAdx_fast.append(area_slope)

        _, area_slope, _, radius = get_nozzle_simoneau_bspline(x)
        RADIUS_bspline.append(radius)
        dAdx_bspline.append(area_slope)

    convergent_length = 79.65e-3
    throat_length = 11.35e-3
    bspline_in_convergent = 30e-3

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    ax1.plot(X * 1e3, RADIUS_fast,    label="linear (original)", color="gray", linestyle="--")
    ax1.plot(X * 1e3, RADIUS_bspline, label="B-spline raccordatura", color="royalblue")
    for ax in (ax1, ax2):
        ax.axvline((convergent_length - bspline_in_convergent) * 1e3,
                   color="green", linestyle=":", linewidth=1)
        ax.axvline(convergent_length * 1e3,
                   color="orange", linestyle=":", linewidth=1)
        ax.axvline((convergent_length + throat_length) * 1e3,
                   color="red", linestyle=":", linewidth=1)
    ax1.set_ylabel("Radius (m)")
    ax1.legend(fontsize=8)
    ax1.grid()

    ax2.plot(X * 1e3, dAdx_fast,    color="gray",      linestyle="--", label="dA/dx linear")
    ax2.plot(X * 1e3, dAdx_bspline, color="royalblue",               label="dA/dx B-spline")
    ax2.set_ylabel("dA/dx (m)")
    ax2.set_xlabel("Axial position x (mm)")
    ax2.legend(fontsize=8)
    ax2.grid()

    fig.tight_layout()
    plt.show()
