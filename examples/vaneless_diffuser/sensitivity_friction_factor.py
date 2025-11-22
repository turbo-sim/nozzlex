import os
import time
import yaml
import jax.numpy as jnp
import equinox as eqx
import jaxprop as jxp
import matplotlib.pyplot as plt

from time import perf_counter

from nozzlex import vaneless_channel as vcm

jxp.set_plot_options(grid=False)

OUTDIR = "output"
os.makedirs(OUTDIR, exist_ok=True)


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

    # Create base model
    diffuser = vcm.VanelessChannel.from_dict(config, fluid)

    # Friction values to test
    Cf_values = jnp.asarray([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    colors = plt.cm.magma(jnp.linspace(0.1, 0.9, len(Cf_values)))

    # Create figure once
    fig0, ax0 = plt.subplots(figsize=(6, 5))

    # Run each case
    print("Parameter sweep on Cf:\n")
    for i, (Cf, color) in enumerate(zip(Cf_values, colors)):

        print(f"Running Cf = {Cf:.3f} ... ", end="", flush=True)

        # --- Update only Cf (no rebuild) ---
        t0 = time.perf_counter()
        diffuser = eqx.tree_at(
            lambda d: d.model_options.friction.Cf,  # path to field
            diffuser,
            jnp.array(Cf),
        )
        t1 = time.perf_counter()

        # --- Solve model ---
        out = diffuser.solve()
        t2 = time.perf_counter()

        dt_update = (t1 - t0) * 1000  # ms
        dt_solve = (t2 - t1) * 1000  # ms
        print(f"done (update: {dt_update:.3f} ms, solve: {dt_solve:.3f} ms)")

        # --- Plot pressure recovery ---
        ax0.plot(out["m"], out["Cp"], color=color, label=f"$C_f={Cf:.2f}$")

        # --- Plot streamlines ---
        if i == 0:
            fig1, ax1 = diffuser.plot_streamlines(out, color=color, label=f"$C_f={Cf:.2f}$")
        else:
            diffuser.plot_streamlines(out, fig=fig1, ax=ax1, color=color, label=f"$C_f={Cf:.2f}$")


    # Finalize figure 
    ax0.legend(loc="lower right", fontsize=10)
    ax1.legend(loc="lower right", fontsize=8)
    fig1.tight_layout(pad=1)
    fig1.tight_layout(pad=1)

    # Show figures
    plt.show()


