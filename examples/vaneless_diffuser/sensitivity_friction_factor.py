import jax
import jax.numpy as jnp
import equinox as eqx
import jaxprop as jxp
import matplotlib.pyplot as plt

from time import perf_counter

import vaneless_channel_model_v5 as vcm

jxp.set_plot_options(grid=False)


if __name__ == "__main__":

    # Define the working fluid
    fluid = jxp.FluidPerfectGas("air", T_ref=300, p_ref=101325)
    # fluid = jxp.FluidBicubic(
    #     fluid_name="air",
    #     backend="HEOS",
    #     h_min=100 * 1e3,
    #     h_max=500 * 1e3,
    #     p_min=0.1e5,
    #     p_max=10e5,
    #     N_h=50,
    #     N_p=50,
    # )

    # Compute the inlet state
    p0_in = 101325.0
    T0_in = 25.0 + 273.15
    Ma_in = 0.25
    p_in, h_in, v_in = vcm.compute_static_state(p0_in, T0_in, Ma_in, fluid)

    # Define model parameters
    params = {
        "p_in": p_in,
        "h_in": h_in,
        "v_in": v_in,
        "alpha_in": 65 * jnp.pi / 180,
        "C_f": 0.0,
        "q_w": 0.0,
        "roughness": 0.0,
        "geometry": {
            "z_in": 0.0,
            "z_out": 0.0,
            "r_in": 1.0,
            "r_out": 3.0,
            "b_in": 0.25,
            "b_out": 0.25,
            "phi_in": jnp.deg2rad(90.0),
            "phi_out": jnp.deg2rad(90.0),
            "td_in": 0.10,
            "td_out": 0.10,
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
        n_points=25,
    )

    # Plot the pressure recovery coefficient distribution
    fig_1, ax_1 = plt.subplots(figsize=(6, 5))
    ax_1.grid(True)
    ax_1.set_xlabel("Radius ratio")
    ax_1.set_ylabel("Pressure recovery coefficient\n")

    # Plot the Mach number distribution
    fig_2, ax_2 = plt.subplots()
    ax_2.grid(True)
    ax_2.set_xlabel("Radius ratio")
    ax_2.set_ylabel("Mach number\n")

    # Plot streamlines
    number_of_streamlines = 5
    fig_3, ax_3 = plt.subplots()
    ax_3.set_aspect("equal", adjustable="box")
    ax_3.grid(False)
    ax_3.set_xlabel("x coordinate")
    ax_3.set_ylabel("y coordinate\n")
    ax_3.set_title("Diffuser streamlines\n")
    ax_3.axis(1.1 * params["geometry"]["r_out"] * jnp.array([-1, 1, -1, 1]))
    theta = jnp.linspace(0, 2 * jnp.pi, 100)
    x_in = params["geometry"]["r_in"] * jnp.cos(theta)
    y_in = params["geometry"]["r_in"] * jnp.sin(theta)
    x_out = params["geometry"]["r_out"] * jnp.cos(theta)
    y_out = params["geometry"]["r_out"] * jnp.sin(theta)
    ax_3.plot(x_in, y_in, "k", label=None)  # HandleVisibility='off'
    ax_3.plot(x_out, y_out, "k", label=None)  # HandleVisibility='off'
    theta = jnp.linspace(0, 2 * jnp.pi, number_of_streamlines + 1)

    # Compute diffuser performance for different friction factors
    # Cf_array = jnp.asarray([0.0, 0.01, 0.02, 0.03])
    Cf_array = jnp.asarray([0e-6, 2e-6, 4e-6, 6e-6, 8e-6, 10e-6])
    colors = plt.cm.magma(jnp.linspace(0.2, 0.8, len(Cf_array)))  # Generate colors
    print()
    print("-" * 42)
    print("Friction factor sensitivity analysis")
    print("-" * 42)
    for i, Cf in enumerate(Cf_array):
        params["roughness"] = Cf
        t0 = perf_counter()
        out = vcm.solve_vaneless_channel_model(
            params, fluid, geom_handle, solver_params
        )
        print(f"Call {i+1}: Model evaluation time: {(perf_counter()-t0)*1e3:.4f} ms")

        # Plot the pressure recovery coefficient distribution
        ax_1.plot(
            out["m"],
            out["Cp"],
            label=f"$C_f = {Cf:0.3f}$",
            color=colors[i],
        )
        ax_1.legend(loc="lower right")

        # Plot the Mach number distribution
        ax_2.plot(
            out["m"],
            out["Ma"],
            label=f"$C_f = {Cf:0.3f}$",
            color=colors[i],
        )
        ax_2.legend(loc="upper right")

        # Plot streamlines
        for j in range(len(theta)):
            x = out["r"] * jnp.cos(out["theta"] + theta[j])
            y = out["r"] * jnp.sin(out["theta"] + theta[j])
            if j == 0:
                ax_3.plot(x, y, label=f"$C_f = {Cf:0.3f}$", color=colors[i])
            else:
                ax_3.plot(x, y, color=colors[i])

    # Adjust pad
    for fig in [fig_1, fig_2, fig_3]:
        fig.tight_layout(pad=1)

    # # --------------------------------------------------------------------- #
    # # ----------  Compute the gradients of an objective function ---------- #
    # # --------------------------------------------------------------------- #
    # def objective(params, fluid):
    #     sol = vcm.solve_vaneless_channel_model(
    #         params,
    #         fluid,
    #         geom_handle,
    #         number_of_points=None,
    #         solver_name="Dopri5",
    #         adjoint_name="DirectAdjoint",
    #     )
    #     return jnp.squeeze(sol.ys["Cp"])  # scalar

    # # JIT the scalar oand gradients
    # objective_jit = eqx.filter_jit(objective)
    # jac_fwd = eqx.filter_jit(jax.jacfwd(objective))
    # jac_rev = eqx.filter_jit(jax.jacrev(objective))
    # hess_fn = eqx.filter_jit(jax.jacfwd(jax.jacrev(objective)))

    # # Print the results
    # print()
    # print("-" * 42)
    # print("Function timings (Cp only):")
    # print("-" * 42)
    # for i in range(5):
    #     t0 = perf_counter()
    #     v = objective_jit(params, fluid)
    #     print(f"Call {i+1}: {(perf_counter()-t0)*1e3:.4f} ms")

    # print()
    # print("-" * 42)
    # print("Gradient timings (forward-mode, Cp only):")
    # print("-" * 42)
    # for i in range(5):
    #     t0 = perf_counter()
    #     g_ad = jac_fwd(params, fluid)
    #     print(f"Call {i+1}: {(perf_counter()-t0)*1e3:.4f} ms")

    # print()
    # print("-" * 42)
    # print("Gradient timings (reverse-mode, Cp only):")
    # print("-" * 42)
    # for i in range(5):
    #     t0 = perf_counter()
    #     g_ad = jac_rev(params, fluid)
    #     print(f"Call {i+1}: {(perf_counter()-t0)*1e3:.4f} ms")

    # print()
    # print("-" * 42)
    # print("Gradient of Cp_out w.r.t. parameters")
    # print("-" * 42)

    # rel_eps = 1e-5

    # for k, base in params.items():
    #     # Skip nested dictionaries or non-numeric entries
    #     if isinstance(base, dict) or not jnp.issubdtype(jnp.array(base).dtype, jnp.floating):
    #         continue

    #     eps = rel_eps * (jnp.abs(base) + 1.0)
    #     p_plus  = dict(params); p_plus[k]  = base + eps
    #     p_minus = dict(params); p_minus[k] = base - eps

    #     f_plus  = objective(p_plus,  fluid)
    #     f_minus = objective(p_minus, fluid)
    #     g_fd = (f_plus - f_minus) / (2.0 * eps)
    #     g_ad_val = g_ad.get(k, jnp.nan)

    #     err_abs = jnp.abs(g_ad_val - g_fd)
    #     err_rel = err_abs / (jnp.abs(g_ad_val) + 1e-16)

    #     print(f" {k:<10}  AD: {g_ad_val: .6e}   FD: {g_fd: .6e}   abs.err: {err_abs: .3e}   rel.err: {err_rel: .3e}")

    # Show plots
    plt.show()
