import os
import yaml
import jax.numpy as jnp
import jaxprop as jxp
import equinox as eqx
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from nozzlex import vaneless_channel as vcm

# --- Configuration ---
OUTDIR = "output"
os.makedirs(OUTDIR, exist_ok=True)
jxp.set_plot_options(grid=False)


if __name__ == "__main__":

    # --- Load configuration ---
    config_file = "./case_axial_diffuser.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # --- Define fluid and base model ---
    fluid = jxp.FluidPerfectGas("air", T_ref=300, p_ref=101325)
    diffuser = vcm.VanelessChannel.from_dict(config, fluid)

    # --- Parameter sweeps ---
    theta_array = jnp.logspace(jnp.log10(0.2), jnp.log10(20.0), 51)
    AR_values = jnp.array([1.2, 1.6, 2.0, 2.5, 3.0])
    colors = plt.cm.magma(jnp.linspace(0.1, 0.9, len(AR_values)))

    b_in = config["geometry"]["b_in"]

    # Storage for global data
    results = {}

    for AR in AR_values:
        eta_loss_total = []
        eta_loss_wall = []
        eta_loss_diff = []
        eta = []

        print(f"\n--- Running AR = {AR:.1f} ---")

        for theta in theta_array:

            print(f"      Divergence semiangle = {theta:.3f} deg")
            # Update geometry
            z_out = b_in * (AR - 1.0) / (2.0 * jnp.tan(jnp.deg2rad(theta)))
            b_out = b_in * AR
            diffuser_i = eqx.tree_at(
                lambda d: (d.geometry.z_out, d.geometry.b_out),
                diffuser,
                (jnp.array(z_out), jnp.array(b_out)),
            )

            # Solve the model
            out = diffuser_i.solve()

            # Store quantities
            eta.append(out["efficiency"][-1])
            eta_loss_total.append(out["efficiency_loss"][-1])
            eta_loss_wall.append(out["efficiency_loss_wall"][-1])
            eta_loss_diff.append(out["efficiency_loss_diffusion"][-1])

        results[float(AR)] = {
            "theta": theta_array,
            "eta": jnp.array(eta),
            "eta_loss_total": jnp.array(eta_loss_total),
            "eta_loss_wall": jnp.array(eta_loss_wall),
            "eta_loss_diff": jnp.array(eta_loss_diff),
        }

    # ============================================================
    # 1 Plot pressure recovery vs θ for different AR
    # ============================================================
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    for color, AR in zip(colors, AR_values):
        ax1.plot(results[float(AR)]["theta"], results[float(AR)]["eta"],
                 color=color, label=f"AR = {AR:.1f}")
    ax1.set_xscale("log")
    ax1.set_xlim(theta_array[0], theta_array[-1])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlabel(r"$\theta$ $-$ Divergence semi-angle")
    ax1.set_ylabel(r"$C_p$ $-$ Pressure recovery coefficient")
    ax1.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: rf"{x:.1f}°"))
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, which="both", ls=":")
    fig1.tight_layout(pad=1)

    # ============================================================
    # 2 Plot Y (total loss) vs θ for different AR
    # ============================================================
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    for color, AR in zip(colors, AR_values):
        ax2.plot(results[float(AR)]["theta"], results[float(AR)]["eta_loss_total"],
                 color=color, label=f"AR = {AR:.1f}")
    ax2.set_xscale("log")
    ax2.set_xlim(theta_array[0], theta_array[-1])
    ax2.set_ylim([0.0, 0.6])
    ax2.set_xlabel(r"$\theta$ $-$ Divergence semi-angle")
    ax2.set_ylabel(r"$Y$ $-$ Pressure loss coefficient")
    ax2.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: rf"{x:.1f}°"))
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, which="both", ls=":")
    fig2.tight_layout()

    # ============================================================
    # 3 For AR = 2, plot total efficiency and component breakdown
    # ============================================================
    AR_ref = 2.0
    data = results[AR_ref]
    eta_total = data["eta_loss_total"]
    eta_wall = data["eta_loss_wall"]
    eta_diff = data["eta_loss_diff"]
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    ax3.plot(data["theta"], eta_total, "-", color="k", label="Total loss")
    ax3.plot(data["theta"], eta_wall, "--", color="k", label="Wall loss")
    ax3.plot(data["theta"], eta_diff, ":", color="k", label="Diffusion loss")
    ax3.set_xscale("log")
    ax3.set_xlim(theta_array[0], theta_array[-1])
    ax3.set_ylim([0.0, 0.6])
    ax3.set_xlabel(r"$\theta$ $-$ Divergence semi-angle")
    ax3.set_ylabel(r"$Y$ $-$ Pressure loss coefficient")
    ax3.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: rf"{x:.1f}°"))
    ax3.legend(loc="upper right", fontsize=10)
    ax3.grid(True, which="both", ls=":", alpha=0.6)
    fig3.tight_layout()

    # Save figures
    jxp.savefig_in_formats(fig1, os.path.join(OUTDIR, "sensitivity_divergence_pressure_recovery"))
    jxp.savefig_in_formats(fig2, os.path.join(OUTDIR, "sensitivity_divergence_pressure_loss"))
    jxp.savefig_in_formats(fig3, os.path.join(OUTDIR, "sensitivity_divergence_pressure_loss_distribution"))

    # Show plots
    plt.show()
