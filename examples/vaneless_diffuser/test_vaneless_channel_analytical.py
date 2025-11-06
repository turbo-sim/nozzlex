import os
import time
import yaml
import jax.numpy as jnp
import jaxprop as jxp
import equinox as eqx
import matplotlib.pyplot as plt

from nozzlex import vaneless_channel as vcm

if __name__ == "__main__":

    # Load configuration file
    config_file = "case_radial_diffuser.yaml"

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Manually set friction model
    config["model_options"]["friction_model"] = {
        "type": "constant_friction_factor",
        "Cf": 0.0,
    }

    # Define the working fluid
    fluid = jxp.FluidPerfectGas("air", T_ref=300, p_ref=101325)

    # Create base model
    diffuser = vcm.VanelessChannel.from_dict(config, fluid)

    # --- Range of inlet angles ---
    alpha_array = jnp.linspace(0, 80, 9)
    Cp_num = []
    Cp_theory = []

    # --- Run cases ---
    for alpha_in in alpha_array:
        print(f"Running alpha_in = {alpha_in:.1f} deg ... ", end="", flush=True)

        # Update only inlet flow angle (no rebuild)
        t0 = time.perf_counter()
        diffuser_i = eqx.tree_at(
            lambda d: d.operating_conditions.alpha_in,
            diffuser,
            jnp.array(alpha_in),
        )
        t1 = time.perf_counter()

        # Solve model
        out = diffuser_i.solve()
        t2 = time.perf_counter()

        dt_update = (t1 - t0) * 1000
        dt_solve = (t2 - t1) * 1000
        print(f"done (update: {dt_update:.3f} ms, solve: {dt_solve:.3f} ms)")

        # Store pressure recovery results
        Cp_num.append(out["Cp"][-1])
        print(out["area_ratio"][-1], out["radius_ratio"][-1], alpha_in)
        Cp_theory.append(vcm.get_analytical_Cp(alpha_in, out["area_ratio"][-1], out["radius_ratio"][-1]))

    # Convert to arrays
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
        alpha_array, Cp_theory, Cp_num, abs_err, rel_err
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
