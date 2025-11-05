import jax
import jax.numpy as jnp
import equinox as eqx
import jaxprop as jxp
import matplotlib.pyplot as plt

from time import perf_counter

import vaneless_channel_model_v5 as vcm

jxp.set_plot_options(grid=False)


# TODO, plot the loss breakdown for astraight and a curved diffuser channel
# Plot both in the same figure, including efficiency plot, flow distribution with contour of Mach, 
if __name__ == "__main__":

    # Define the working fluid
    fluid = jxp.FluidPerfectGas("air", T_ref=300, p_ref=101325)

    # Compute the inlet state
    p0_in = 101325.0
    T0_in = 25.0 + 273.15
    Ma_in = 0.25
    p_in, h_in, v_in = vcm.compute_static_state(p0_in, T0_in, Ma_in, fluid)

    # Define model parameters
    # params = {
    #     "p_in": p_in,
    #     "h_in": h_in,
    #     "v_in": v_in,
    #     "alpha_in": 60 * jnp.pi / 180,
    #     "C_f": 0.0,
    #     "q_w": 0.0,
    #     "roughness": 0.00,
    #     "geometry": {
    #         "z_in": 0.0,
    #         "z_out": 0.0,
    #         "r_in": 0.10,
    #         "r_out": 0.20,
    #         "b_in": 0.01,
    #         "b_out": 0.01,
    #         "phi_in": jnp.deg2rad(90.0),
    #         "phi_out": jnp.deg2rad(90.0),
    #         "td_in": 0.0,
    #         "td_out": 0.0,
    #     },
    # }

    params = {
        "p_in": p_in,
        "h_in": h_in,
        "v_in": v_in,
        "alpha_in": 60 * jnp.pi / 180,
        "C_f": 0.0,
        "q_w": 0.0,
        "roughness": 0.00,
        "geometry": {
            "z_in": 0.0,
            "z_out": 0.05,
            "r_in": 0.10,
            "r_out": 0.20,
            "b_in": 0.01,
            "b_out": 0.012,
            "phi_in": jnp.deg2rad(90.0),
            "phi_out": jnp.deg2rad(0.0),
            "td_in": 0.1,
            "td_out": 0.04,
        },
    }

    # Create the geometry of the channel
    geom_handle = vcm.make_vaneless_channel_geometry(params["geometry"])
    # # vcm.plot_vaneless_channel(geom_handle)
    # # fig, ax = plt.subplots()
    # # u = jnp.linspace(0, 1, 100)
    # # x, y = geom_handle.nurbs_midline.get_value(u)
    # # ax.plot(x, y)
    # geom_handle.nurbs_midline.plot(rescale=False)
    # plt.show()


    # Compute diffuser performance
    solver_params = vcm.SolverParams(
        solver_name="Dopri5",
        adjoint_name="DirectAdjoint",
        rtol=1e-6,
        atol=1e-6,
        n_points=50,
        max_steps=100,
        throw=True,
    )
    out = vcm.solve_vaneless_channel_model(params, fluid, geom_handle, solver_params)
    vcm.plot_vaneless_channel(geom_handle)

    # # vcm.plot_vaneless_channel_contour(geom_handle=geom_handle, solution=out, var_name="Ma")


    # -----------------------------------------------------------------------------
    # Plot the results
    # -----------------------------------------------------------------------------

    # Plot the pressure recovery coefficient distribution
    fig_1, ax_1 = plt.subplots(figsize=(6, 5))
    ax_1.grid(True)
    ax_1.set_xlabel("Radius ratio")
    ax_1.set_ylabel("Reynolds number\n")
    ax_1.plot(out["m"], out["Re"])
    # ax_1.legend(loc="lower right")

    # # Plot the Mach number distribution
    # fig_2, ax_2 = plt.subplots()
    # ax_2.grid(True)
    # ax_2.set_xlabel("Radius ratio")
    # ax_2.set_ylabel("Mach number\n")
    # ax_2.plot(out["m"], out["Ma"])
    # # ax_2.legend(loc="upper right")

    # Plot the Mach number distribution
    fig_2, ax_2 = plt.subplots()
    ax_2.grid(True)
    ax_2.set_xlabel("Radius ratio")
    ax_2.set_ylabel("Mach number\n")
    ax_2.plot(out["m"], out["p0"], "r-")
    ax_2.plot(out["m"], out["p0_bis"], "b--")
    # ax_2.legend(loc="upper right")


    # # Plot the skin friction factor distribution
    # fig_3, ax_3 = plt.subplots()
    # ax_3.grid(True)
    # ax_3.set_xlabel("Meridional length")
    # ax_3.set_ylabel("Fanning friction coefficient $c_f$\n")
    # ax_3.plot(out["m"], out["cf_total"], label="Total $c_f$", color="k", linewidth=2.0)
    # ax_3.plot(out["m"], out["cf_wall"], label="Wall $c_{f,W}$", linestyle="--", color="tab:blue")
    # ax_3.plot(out["m"], out["cf_diffusion"], label="Diffusion $c_{f,D}$", linestyle=":", color="tab:orange")
    # ax_3.plot(out["m"], out["cf_curvature"], label="Curvature $c_{f,C}$", linestyle="-.", color="tab:green")
    # ax_3.legend(loc="upper right", frameon=False)


    # # -------------------------------------------------------------------------
    # # Plot pressure recovery and loss breakdown
    # # -------------------------------------------------------------------------
    # fig_5, ax_5 = plt.subplots(figsize=(6, 5))
    # ax_5.grid(True, zorder=0)
    # ax_5.set_xlabel("Meridional coordinate $m$")
    # ax_5.set_ylabel("Pressure recovery and losses\n")
    # ax_5.set_title("Pressure recovery breakdown")

    # # Extract components
    # m = out["m"]
    # Cp = out["Cp"]
    # dCp_kinetic = out["dCp_kinetic"]
    # dCp_loss_wall = out["dCp_loss_wall"]
    # dCp_loss_diff = out["dCp_loss_diffusion"]
    # dCp_loss_curv = out["dCp_loss_curvature"]

    # # Colors and hatches for each component
    # colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:gray"]
    # hatches = [None, None, "//", "\\\\", "xx"]

    # # Base cumulative stacking
    # bottom = jnp.zeros_like(m)

    # # Cp (static pressure recovery)
    # ax_5.fill_between(
    #     m, bottom, Cp, color="tab:blue", alpha=0.4, label="$C_p$ (pressure recovery)", hatch=None, zorder=3
    # )
    # bottom = Cp

    # # Wall loss
    # ax_5.fill_between(
    #     m, bottom, bottom + dCp_loss_wall, color="tab:green", alpha=0.5,
    #     label="Wall loss", hatch="\\\\", edgecolor="k", linewidth=0.4, zorder=3
    # )
    # bottom += dCp_loss_wall

    # # Diffusion loss
    # ax_5.fill_between(
    #     m, bottom, bottom + dCp_loss_diff, color="tab:red", alpha=0.5,
    #     label="Diffusion loss", hatch="xx", edgecolor="k", linewidth=0.4, zorder=3
    # )
    # bottom += dCp_loss_diff

    # # Curvature loss
    # ax_5.fill_between(
    #     m, bottom, bottom + dCp_loss_curv, color="tab:gray", alpha=0.5,
    #     label="Curvature loss", hatch="--", edgecolor="k", linewidth=0.4, zorder=3
    # )
    # bottom += dCp_loss_curv


    # # Kinetic contribution
    # ax_5.fill_between(
    #     m, bottom, bottom + dCp_kinetic, color="tab:orange", alpha=0.4,
    #     label="Kinetic loss", hatch="//", edgecolor="k", linewidth=0.5, zorder=3
    # )
    # bottom += dCp_kinetic

    # # Axis limits and legend
    # ax_5.set_ylim(0, 1.05)
    # ax_5.legend(loc="lower right", ncol=1, fontsize=9)
    # fig_5.tight_layout(pad=1)

    # ax_5.set_xlim([0, out["m"][-1]])
    # ax_5.set_ylim([0, 1])
    # plt.show()


    # # # Plot streamlines
    # # number_of_streamlines = 5
    # # fig_4, ax_4 = plt.subplots()
    # # ax_4.set_aspect("equal", adjustable="box")
    # # ax_4.grid(False)
    # # ax_4.set_xlabel("x coordinate")
    # # ax_4.set_ylabel("y coordinate\n")
    # # ax_4.set_title("Diffuser streamlines\n")
    # # ax_4.axis(1.1 * params["geometry"]["r_out"] * jnp.array([-1, 1, -1, 1]))

    # # # Inlet and outlet circular boundaries
    # # theta = jnp.linspace(0, 2 * jnp.pi, 100)
    # # x_in = params["geometry"]["r_in"] * jnp.cos(theta)
    # # y_in = params["geometry"]["r_in"] * jnp.sin(theta)
    # # x_out = params["geometry"]["r_out"] * jnp.cos(theta)
    # # y_out = params["geometry"]["r_out"] * jnp.sin(theta)
    # # ax_4.plot(x_in, y_in, "k")
    # # ax_4.plot(x_out, y_out, "k")

    # # # Plot streamlines for visualization
    # # theta_stream = jnp.linspace(0, 2 * jnp.pi, number_of_streamlines + 1)
    # # for j in range(len(theta_stream)):
    # #     x = out["r"] * jnp.cos(out["theta"] + theta_stream[j])
    # #     y = out["r"] * jnp.sin(out["theta"] + theta_stream[j])
    # #     ax_4.plot(x, y, color="tab:blue")

    # # Adjust layout and show plots
    # for fig in [fig_1, fig_2, fig_3]:
    #     fig.tight_layout(pad=1)

    plt.show()


