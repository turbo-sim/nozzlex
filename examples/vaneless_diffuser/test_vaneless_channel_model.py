import jax.numpy as jnp
import jaxprop as jxp
import matplotlib.pyplot as plt

import vaneless_channel_model_v5 as vcm


if __name__ == "__main__":

    # Define the working fluid
    fluid = jxp.FluidPerfectGas("air", T_ref=300, p_ref=101325)

    # Compute the inlet state
    p0_in = 101325.0
    T0_in = 25.0 + 273.15
    Ma_in = 0.05
    p_in, h_in, v_in = vcm.compute_static_state(p0_in, T0_in, Ma_in, fluid)

    # Define model parameters
    params = {
        "p_in": p_in,
        "h_in": h_in,
        "v_in": v_in,
        "alpha_in": 65 * jnp.pi / 180,
        "C_f": 0.0,
        "q_w": 0.0,
        "geometry": {
            "z_in": 0.0,
            "z_out": 0.0,
            "r_in": 1.0,
            "r_out": 3.0,
            "b_in": 0.20,
            "b_out": 0.30,
            "phi_in": jnp.deg2rad(90.0),
            "phi_out": jnp.deg2rad(90.0),
            "td_in": 0.10,
            "td_out": 0.10,
        },
    }

    # Create the geometry of the channel
    geom_handle = vcm.make_vaneless_channel_geometry(params["geometry"])
    vcm.plot_vaneless_channel(geom_handle)

    # Define ODE solver settings
    solver_params = vcm.SolverParams(
        solver_name="Dopri5",
        adjoint_name="DirectAdjoint",
        rtol=1e-6,
        atol=1e-6,
        n_points=25,
    )

    # --- Range of inlet angles ---
    alpha_array = jnp.deg2rad(jnp.linspace(0, 80, 9))
    Cp_num = []
    Cp_theory = []

    # --- Run cases ---
    for alpha_in in alpha_array:
        params["alpha_in"] = alpha_in
        out = vcm.solve_vaneless_channel_model(
            params, fluid, geom_handle, solver_params
        )
        Cp_num.append(out["Cp"][-1])
        Cp_theory.append(vcm.get_analytical_Cp(alpha_in, geom_handle))

    Cp_num = jnp.array(Cp_num)
    Cp_theory = jnp.array(Cp_theory)

    # --- Compute and print results table ---
    print("\n" + "-" * 90)
    print("Comparison of analytical and numerical pressure recovery (Cp)")
    print("-" * 90)
    print(
        f"{'alpha_in [deg]':>14} | {'Cp (analytical)':>17} | {'Cp (numerical)':>17} | {'abs. error':>12} | {'rel. error [%]':>15}"
    )
    print("-" * 90)
    abs_err = jnp.abs(Cp_num - Cp_theory)
    rel_err = 100.0 * abs_err / (jnp.abs(Cp_theory) + 1e-16)

    for a_deg, cp_t, cp_n, e_abs, e_rel in zip(
        jnp.rad2deg(alpha_array), Cp_theory, Cp_num, abs_err, rel_err
    ):
        print(
            f"{a_deg:14.2f} | {cp_t:17.6f} | {cp_n:17.6f} | {e_abs:12.3e} | {e_rel:15.3e}"
        )

    print("-" * 90)
    max_err = jnp.max(rel_err)
    print(f"Maximum relative error: {max_err:.3e} %")
    print("-" * 90)

    # --- Assertion check ---
    assert (
        max_err < 1e-1
    ), f"Relative error too high ({max_err:.2e} %): numerical solution deviates from analytical."

    # --- Plot comparison ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(jnp.rad2deg(alpha_array), Cp_theory, "k-", label="Analytical")
    ax.plot(jnp.rad2deg(alpha_array), Cp_num, "o", label="Numerical")
    ax.set_xlabel(r"Inlet flow angle $\alpha_1$ [deg]")
    ax.set_ylabel(r"Pressure recovery coefficient $C_p$")
    ax.grid(True)
    ax.legend()
    fig.tight_layout(pad=1)
    plt.show()
