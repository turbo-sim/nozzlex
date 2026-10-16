"""Test that get_nozzle_smd from nozzle_model_core matches the CSV points."""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os


def get_pipe(x, radius = 0.01):
    perimeter = 2 * radius * jnp.pi
    area = jnp.pi * radius ** 2
    area_slope = 0.0

    return area, area_slope, perimeter, radius


if __name__ == "__main__":
    z_end = 0.463106796
    x_eval = np.linspace(0.0, z_end, 500)

    get_nozzle_jit = jax.jit(get_pipe)

    A_arr, dAdx_arr, r_arr = [], [], []
    for x in x_eval:
        A, dAdx, perimeter, radius = get_nozzle_jit(x)
        A_arr.append(float(A))
        dAdx_arr.append(float(dAdx))
        r_arr.append(float(radius))

    # Original CSV points
    csv_path = os.path.join(os.path.dirname(__file__), "nozzle_coordinates_smd.csv")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    z_orig, r_orig = data[:, 0], data[:, 1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(x_eval * 1e3, np.array(r_arr) * 1e3, label="Spline + constant")
    axes[0].plot(z_orig * 1e3, r_orig * 1e3, "ro", markersize=4, label="CSV points")
    axes[0].axvline(0.089805825 * 1e3, color="gray", ls="--", label="Transition")
    axes[0].set_ylabel("Radius [mm]")
    axes[0].legend()
    axes[0].grid()
    axes[0].set_title("Supermobidick nozzle geometry (from nozzle_model_core)")

    axes[1].plot(x_eval * 1e3, dAdx_arr)
    axes[1].axvline(0.089805825 * 1e3, color="gray", ls="--")
    axes[1].set_ylabel("dA/dx [m]")
    axes[1].set_xlabel("Axial position [mm]")
    axes[1].grid()

    plt.tight_layout()
    plt.show()
