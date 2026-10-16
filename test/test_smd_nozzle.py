"""Test that get_nozzle_smd from nozzle_model_core matches the CSV points."""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os

from nozzle_model_core import get_nozzle_smd_divergent_smooth

def get_nozzle_smd(
    length,
    z_transition=0.089805825,
    r_const=0.010065,
    z_end=0.463106796,
):
    """Copy of get_nozzle_smd for standalone testing (no nurbspy dependency)."""
    _breaks = jnp.array([
        0.0, 0.00970874, 0.01941748, 0.02961165, 0.04029126,
        0.04951456, 0.05970874, 0.07038835, 0.08058252, 0.08980583,
    ])
    _coeffs = jnp.array([
        [-5.15687535e+02,  2.36106419e+02,  2.13628410e+01,
          3.45126602e+02,  4.88402424e+02, -4.79989669e+02,
          3.25965960e+01, -2.52971271e+02,  7.58975837e+02],
        [-1.77635684e-15, -1.50200255e+01, -8.14313939e+00,
         -7.48980984e+00,  3.56764475e+00,  1.70816925e+01,
          2.40239639e+00,  3.44675339e+00, -4.28974607e+00],
        [ 1.44451489e-02, -1.31380343e-01, -3.56265442e-01,
         -5.15630447e-01, -5.57517648e-01, -3.67062596e-01,
         -1.68438384e-01, -1.05971734e-01, -1.14565348e-01],
        [ 3.35000000e-02,  3.31683170e-02,  3.06930690e-02,
          2.62376240e-02,  2.02970300e-02,  1.58415840e-02,
          1.33663370e-02,  1.18811880e-02,  1.08910890e-02],
    ])

    def converging(length):
        n_seg = _coeffs.shape[1]
        idx = jnp.searchsorted(_breaks, length, side="right") - 1
        idx = jnp.clip(idx, 0, n_seg - 1)
        dx = length - _breaks[idx]
        c0 = _coeffs[0, idx]
        c1 = _coeffs[1, idx]
        c2 = _coeffs[2, idx]
        c3 = _coeffs[3, idx]
        radius = c0 * dx**3 + c1 * dx**2 + c2 * dx + c3
        drdx = 3.0 * c0 * dx**2 + 2.0 * c1 * dx + c2
        area = jnp.pi * radius**2
        perimeter = 2.0 * jnp.pi * radius
        area_slope = 2.0 * jnp.pi * radius * drdx
        return area, area_slope, perimeter, radius

    def constant(length):
        area = jnp.pi * r_const**2
        perimeter = 2.0 * jnp.pi * r_const
        area_slope = 0.0
        return area, area_slope, perimeter, r_const

    return jax.lax.cond(length < z_transition, converging, constant, length)


if __name__ == "__main__":
    z_end = 0.6
    x_eval = np.linspace(0.0, z_end, 500)

    get_nozzle_jit = jax.jit(get_nozzle_smd_divergent_smooth)

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
