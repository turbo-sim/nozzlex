import os
import time
import yaml
import jax.numpy as jnp
import jaxprop as jxp
import equinox as eqx
import matplotlib.pyplot as plt

from nozzlex import vaneless_channel as vcm

# --- Configuration ---
OUTDIR = "output"
os.makedirs(OUTDIR, exist_ok=True)

# Sweep inlet flow angle (deg)
ALPHA_VALUES = jnp.asarray([40.0, 45.0, 50.0, 55.0, 60.0,])
colors = plt.cm.magma(jnp.linspace(0.1, 0.9, len(ALPHA_VALUES)))

jxp.set_plot_options(grid=False)


if __name__ == "__main__":

    # Load configuration file
    config_file = "case_radial_diffuser.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Define the working fluid
    fluid = jxp.FluidPerfectGas("air", T_ref=300, p_ref=101325)

    # Build the base model ONCE
    diffuser = vcm.VanelessChannel.from_dict(config, fluid)

    # Create figure
    fig0, ax0 = plt.subplots(figsize=(6, 5))
    print("Parameter sweep on inlet flow angle (alpha_in):\n")

    # Run each case
    for i, (alpha_deg, color) in enumerate(zip(ALPHA_VALUES, colors)):

        print(f"Running alpha_in = {alpha_deg:.1f} deg ... ", end="", flush=True)

        # --- Update only inlet angle (no rebuild) ---
        t0 = time.perf_counter()
        diffuser_i = eqx.tree_at(
            lambda d: d.operating_conditions.alpha_in,
            diffuser,
            jnp.array(alpha_deg),
        )
        t1 = time.perf_counter()

        # --- Solve model ---
        out = diffuser_i.solve()
        t2 = time.perf_counter()

        dt_update = (t1 - t0) * 1000
        dt_solve = (t2 - t1) * 1000
        print(f"done (update: {dt_update:.3f} ms, solve: {dt_solve:.3f} ms)")

        # --- Plot pressure recovery ---
        ax0.plot(out["m"], out["Cp"], color=color, label=f"$\\alpha_{{in}}={alpha_deg:.0f}°$")

        # --- Plot streamlines ---
        if i == 0:
            fig1, ax1 = diffuser_i.plot_streamlines(out, color=color, label=f"$\\alpha_{{in}}={alpha_deg:.0f}°$")
        else:
            diffuser_i.plot_streamlines(out, fig=fig1, ax=ax1, color=color, label=f"$\\alpha_{{in}}={alpha_deg:.0f}°$")

    # --- Finalize figures ---
    ax0.set_xlabel("Meridional coordinate [m]")
    ax0.set_ylabel("Pressure recovery coefficient $C_p$")
    ax0.legend(loc="lower right", fontsize=10)
    fig0.tight_layout(pad=1)

    ax1.legend(loc="lower right", fontsize=8)
    fig1.tight_layout(pad=1)

    plt.show()
