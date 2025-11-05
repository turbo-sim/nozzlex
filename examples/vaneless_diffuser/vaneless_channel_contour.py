import jax.numpy as jnp
import equinox as eqx
import jaxprop as jxp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import vaneless_channel_model_v5 as vcm

jxp.set_plot_options(grid=False)


if __name__ == "__main__":

    # Define the working fluid
    # fluid = jxp.FluidPerfectGas("air", T_ref=300, p_ref=101325)
    fluid = jxp.FluidBicubic(
        fluid_name="air",
        backend="HEOS",
        h_min=100 * 1e3,
        h_max=500 * 1e3,
        p_min=0.1e5,
        p_max=10e5,
        N_h=50,
        N_p=50,
    )

    # Compute the inlet state
    p0_in = 101325.0
    T0_in = 25.0 + 273.15
    Ma_in = 0.75
    p_in, h_in, v_in = vcm.compute_static_state(p0_in, T0_in, Ma_in, fluid)

    # Define model parameters
    params = {
        "p_in": p_in,
        "h_in": h_in,
        "v_in": v_in,
        "alpha_in": 65 * jnp.pi / 180,
        "C_f": 0.010,
        "q_w": 0.0,
        "geometry": {
            "z_in": 0.0,
            "z_out": 2.0,
            "r_in": 1.0,
            "r_out": 3.0,
            "b_in": 0.20,
            "b_out": 0.30,
            "phi_in": jnp.deg2rad(90.0),
            "phi_out": jnp.deg2rad(00.0),
            "td_in": 1.5,
            "td_out": 1.0,
        },
    }

    # Create the geometry of the channel
    geom_handle = vcm.make_vaneless_channel_geometry(params["geometry"])

    # Define ODE solver settings
    solver_params = vcm.SolverParams(
        solver_name="Dopri5",
        adjoint_name="DirectAdjoint",
        rtol=1e-6,
        atol=1e-6,
        n_points=200,
    )

    # Calculate the flow field
    out = vcm.solve_vaneless_channel_model(params, fluid, geom_handle, solver_params)

    # Base colormaps
    cmap_blue = LinearSegmentedColormap.from_list(
        "soft_Blues", plt.cm.Blues(jnp.linspace(0.3, 0.9, 256))
    )
    cmap_inferno = LinearSegmentedColormap.from_list(
        "soft_magma", plt.cm.Reds(jnp.linspace(0.05, 0.7, 256))
    )

    # Variables to plot: (name, label, cmap)
    plot_vars = [
        ("Ma", "Mach number [-]", cmap_blue),
        ("p", "Static pressure [Pa]", cmap_blue),
        ("alpha_deg", "Flow angle [deg]", cmap_blue),
        # ("s", "Entropy [J/(kg·K)]", cmap_inferno),
    ]

    for var_name, label, cmap in plot_vars:
        # Create base geometry plot
        fig, ax = vcm.plot_vaneless_channel(geom_handle, plot_control_points=True)

        # Overlay the contour field
        vcm.plot_vaneless_channel_contour(
            geom_handle=geom_handle,
            solution=out,
            var_name=var_name,
            fig=fig,
            ax=ax,
            cmap=cmap,
            label=label,
        )

        fig.tight_layout(pad=1.0)

    # Show figures
    plt.show()
