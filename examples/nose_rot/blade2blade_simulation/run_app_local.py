# import turbodash
# turbodash.launch_app()

import matplotlib.pyplot as plt
import numpy as np
import turbodash as td

from turbodash.geom_blade import (
    compute_blade_coordinates_radial,
    compute_blade_coordinates_cartesian,
)


def get_rotor_coordinates(
    out,
    ax=None,
    N_points=300,
):
    """
    Plot stator and rotor radial blade cascades (blade-to-blade view)
    using meanline data.

    Parameters
    ----------
    out : dict
        Output of compute_stage_meanline
    ax : matplotlib axis, optional
    N_points : int
        Points per blade camberline
    """

    geom = out["geometry"]["rotor"]
    # --- Blade parameters
    N_blades = geom["blade_count"]
    r_in = geom["radius_in"]
    r_out = geom["radius_out"]
    angle_in = geom["metal_angle_in"]
    angle_out = geom["metal_angle_out"]
    angle_0 = 0.0 
    loc_max = geom["maximum_thickness_location"]
    t_max = geom["maximum_thickness"]
    r_le = geom["leading_edge_radius"]
    t_te = geom["trailing_edge_thickness"]
    wedge_angle = geom["maximum_thickness_location"]

    x_b, y_b, *_ = compute_blade_coordinates_radial(
        "linear_angle_change",
        r_in,
        r_out,
        np.deg2rad(angle_in),
        np.deg2rad(angle_out),
        np.deg2rad(angle_0),
        loc_max,
        t_max,
        t_te,
        np.deg2rad(wedge_angle),
        r_le,
        N_points,
    )


    return x_b, y_b

def get_stator_coordinates(
    out,
    ax=None,
    N_points=300,
):
    """
    Plot stator and rotor radial blade cascades (blade-to-blade view)
    using meanline data.

    Parameters
    ----------
    out : dict
        Output of compute_stage_meanline
    ax : matplotlib axis, optional
    N_points : int
        Points per blade camberline
    """

    geom = out["geometry"]["stator"]
    # --- Blade parameters
    N_blades = geom["blade_count"]
    r_in = geom["radius_in"]
    r_out = geom["radius_out"]
    angle_in = geom["metal_angle_in"]
    angle_out = geom["metal_angle_out"]
    angle_0 = 0.0
    loc_max = geom["maximum_thickness_location"]
    t_max = geom["maximum_thickness"]
    r_le = geom["leading_edge_radius"]
    t_te = geom["trailing_edge_thickness"]
    wedge_angle = geom["maximum_thickness_location"]

    x_b, y_b, *_ = compute_blade_coordinates_radial(
        "linear_angle_change",
        r_in,
        r_out,
        np.deg2rad(angle_in),
        np.deg2rad(angle_out),
        np.deg2rad(angle_0),
        loc_max,
        t_max,
        t_te,
        np.deg2rad(wedge_angle),
        r_le,
        N_points,
    )


    return x_b, y_b




import yaml
from turbodash import compute_stage_meanline

with open("meanline_stage_2026-08-28_15-03-16.yaml") as f:
    cfg = yaml.safe_load(f)

result = compute_stage_meanline(**cfg["inputs"])


ax = td.mpl.plot_blades_radial(result)

xa, ya = get_stator_coordinates(result)
xb, yb = get_rotor_coordinates(result)
ax.plot(xb, yb, "ko")
ax.plot(xa, ya, "ko")

za = np.zeros_like(xa)
zb = np.zeros_like(xb)
np.savetxt("stator_blade.csv",
           np.column_stack([xa, ya, za]),
            delimiter=",",
              header="x,y,z", 
              comments="")
np.savetxt("rotor_blade.csv",
           np.column_stack([xb, yb, zb]),
            delimiter=",",
              header="x,y,z", 
              comments="")


plt.show()

