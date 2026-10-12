"""
Axial-turbine equivalent of run_app_local.py.

Same compute_stage_meanline() API as the radial case, just with
stage_type="axial" and radius_ratio_12/23/34 ~ 1 (constant mean radius
row to row). Unlike the radial-outflow case, blade span here stays
essentially constant (only compressibility-driven variation), so there
is no pitch-vs-span area conflict to worry about for the blade-to-blade
mesh/solve.
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
import yaml
import turbodash as td
from turbodash import compute_stage_meanline
from turbodash.geom_blade import compute_blade_coordinates_cartesian


def to_plain(obj):
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def get_blade_coordinates(geom, N_points=300):
    x_b, y_b, *_ = compute_blade_coordinates_cartesian(
        "linear_angle_change",
        0.0,
        0.0,
        np.deg2rad(geom["metal_angle_in"]),
        np.deg2rad(geom["metal_angle_out"]),
        geom["chord"],
        geom["maximum_thickness_location"],
        geom["maximum_thickness"],
        geom["trailing_edge_thickness"],
        np.deg2rad(geom["trailing_edge_wedge_angle"]),
        geom["leading_edge_radius"],
        N_points,
    )
    return x_b, y_b


if __name__ == "__main__":
    cfg = dict(
        fluid_name="Air",
        inlet_property_pair_string="PT_INPUTS",
        inlet_property_1=100000,
        inlet_property_2=300,
        exit_pressure=50000,
        mass_flow_rate=5,
        stator_inlet_angle=0,
        stator_exit_angle=70,
        blade_velocity_ratio=0.7,
        degree_reaction=0.5,
        meridional_velocity_ratio_12=1,
        meridional_velocity_ratio_23=1,
        meridional_velocity_ratio_34=1,
        radius_ratio_12=1.0,
        radius_ratio_23=1.0,
        radius_ratio_34=1.0,
        height_radius_ratio=0.25,
        zweiffel_stator=0.7,
        zweiffel_rotor=0.7,
        loss_coeff_stator=0.05,
        loss_coeff_rotor=0.06,
        stage_type="axial",
    )

    result = compute_stage_meanline(**cfg)

    print("Flow station Mach numbers:")
    for i, fs in enumerate(result["flow_stations"]):
        print(f"  {i}: Ma={fs['Ma']:.3f}  r={fs['r']:.4f}")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_yaml = f"meanline_stage_axial_{timestamp}.yaml"
    with open(out_yaml, "w") as f:
        yaml.safe_dump(to_plain({"inputs": cfg, **{k: result[k] for k in
                        ("stage_performance", "geometry", "flow_stations")}}), f)
    print(f"saved {out_yaml}")

    ax = td.mpl.plot_blades_axial(result)
    fig = ax.get_figure()

    xa, ya = get_blade_coordinates(result["geometry"]["stator"])
    xb, yb = get_blade_coordinates(result["geometry"]["rotor"])

    za = np.zeros_like(xa)
    zb = np.zeros_like(xb)
    np.savetxt("stator_blade_axial.csv", np.column_stack([xa, ya, za]),
               delimiter=",", header="x,y,z", comments="")
    np.savetxt("rotor_blade_axial.csv", np.column_stack([xb, yb, zb]),
               delimiter=",", header="x,y,z", comments="")
    print("saved stator_blade_axial.csv, rotor_blade_axial.csv")

    fig.savefig("axial_blades.png", dpi=150)
    print("saved axial_blades.png")
