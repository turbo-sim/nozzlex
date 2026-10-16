import jax.numpy as jnp
import nurbspy.jax as nrb
import matplotlib.pyplot as plt
import jax
import numpy as np

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

    # Geometry control points



    def region_convergent(length):
        # area = jnp.pi * (radius_in + ((radius_throat_in - radius_in) / convergent_length) * length) ** 2
        # radius = jnp.sqrt(area / jnp.pi)
        # jax.debug.print("{r}",r = length)
        points1 = jnp.array([
            [0.0, convergent_length],
            [radius_in, radius_throat_in]
        ])
        nurbs1 = nrb.NurbsCurve(control_points=points1)
        nurbs1.reparametrize_by_coordinate()
        # print(u)
        radius, _= nurbs1.get_value(length)
        # jax.debug.print("val1={r}", r=radius)
        radius = radius[0]
        # jax.debug.print("val2={r}", r=radius)
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
            [0.0, 0.005, 0.04223],
            [0.0079561, 0.006995, 0.00655],
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
        
        area_slope_1 = perimeter * ((radius_throat - radius_throat_in) / throat_first_length)
        # jax.debug.print("dNurb:{dn} | dForm:{df}", dn=area_slope, df=area_slope_1)
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





# Example usage
if __name__ == "__main__":
    # ctrl_pts = get_nozzle_control_points()
    # print(ctrl_pts)


    # nozzle = nrb.NurbsCurve(control_points=ctrl_pts)
    # nozzle.plot()
    # plt.show()

    X = np.linspace(0, 80e-3, 500)
    AREA = []
    RADIUS = []
    PERIMETER = []
    dAdx = []

    for x in X:
        area, area_slope, perimeter, radius = get_nozzle_nakagawa(x,)
        AREA.append(area)
        RADIUS.append(radius)
        PERIMETER.append(perimeter)
        dAdx.append(area_slope)


    # plt.plot(X, AREA, label="area")
    # plt.plot(X, dAdx, label="dAdx")
    # plt.plot([0.1192587, 0.1192587], [-0.05, 0.05])
    plt.plot(X, RADIUS, label="radius")
    # plt.plot(X, PERIMETER, label="perimeter")
    # plt.plot(X, PERIMETER, label="perimeter")
    plt.legend()
    # plt.ylim([0.0, 0.026])
    plt.grid()

    plt.show()
