"""
JAX implementation of Stadtke's alpha_b_star (Fig. 8.2) — hyperbolic form.

Piecewise definition:
    alpha_b*(alpha_g) = alpha_g                                            if alpha_g <= alpha_trans
                     = alpha_max - k^2 / (alpha_g - alpha_trans + k)       if alpha_g > alpha_trans
where k = alpha_max - alpha_trans.

Properties:
  - C^1 continuous at alpha_trans (slope = 1 from both sides)
  - Asymptote: alpha_b* -> alpha_max as alpha_g -> infinity (never reaches)
  - Two physically-interpretable parameters (linear-region length and asymptote)
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from functools import partial


@partial(jax.jit, static_argnames=())
def alpha_b_star(alpha_g, alpha_trans=0.2, alpha_max=0.35):
    """Stadtke's effective bubble void fraction (hyperbolic saturation form).

    Parameters
    ----------
    alpha_g : array-like or scalar
        Gas volume fraction, in [0, 1].
    alpha_trans : float
        End of the linear region (where alpha_b* = alpha_g exactly). Default 0.15.
    alpha_max : float
        Asymptotic upper bound (regime-transition ceiling). Default 0.35.

    Returns
    -------
    alpha_b_star : same shape as alpha_g
    """
    k = alpha_max - alpha_trans
    hyperbolic = alpha_max - (k * k) / (alpha_g - alpha_trans + k)
    return jnp.where(alpha_g <= alpha_trans, alpha_g, hyperbolic)


# Stadtke (2006) Fig. 8.2 — digitized data
STADTKE_DATA = jnp.array([
    [0.000000, 0.000000], [0.150055, 0.150291], [0.170318, 0.168193],
    [0.193319, 0.186095], [0.213582, 0.198585], [0.241512, 0.212739],
    [0.273275, 0.226062], [0.305038, 0.237302], [0.338992, 0.247294],
    [0.374042, 0.255620], [0.413472, 0.263530], [0.456188, 0.271024],
    [0.492333, 0.276436], [0.532859, 0.281432], [0.578861, 0.286428],
    [0.631435, 0.291424], [0.690581, 0.296003], [0.748631, 0.299750],
    [0.807777, 0.303497], [0.862541, 0.306411], [0.900876, 0.308077],
    [0.953450, 0.310158], [0.990142, 0.311823],
])


def fit_check(alpha_trans=0.15, alpha_max=0.35):
    """Verify the formula reproduces Stadtke's data."""
    x_data, y_data = STADTKE_DATA[:, 0], STADTKE_DATA[:, 1]
    y_pred = alpha_b_star(x_data, alpha_trans, alpha_max)
    rmse = float(jnp.sqrt(jnp.mean((y_pred - y_data) ** 2)))
    max_abs = float(jnp.max(jnp.abs(y_pred - y_data)))
    return rmse, max_abs


def plot(alpha_trans=0.15, alpha_max=0.35, save_path="alpha_b_star.png"):
    """Plot the function, overlay Stadtke data, mark reference lines."""
    alpha_g = jnp.linspace(0.0, 1.0, 401)
    y = alpha_b_star(alpha_g, alpha_trans, alpha_max)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)

    # Hyperbolic alpha_b*
    ax.plot(alpha_g, y, color="#1D9E75", linewidth=2.5,
            label=r"$\alpha_b^*$ (hyperbolic)")
    # y = alpha_g reference
    ax.plot([0, 1], [0, 1], color="#888780", linewidth=1, linestyle="--",
            label=r"$y = \alpha_g$")
    # asymptote
    ax.axhline(alpha_max, color="#BA7517", linewidth=1, linestyle=":",
               label=rf"$\alpha_{{b,\max}} = {alpha_max}$")
    # Stadtke data
    ax.scatter(np.array(STADTKE_DATA[:, 0]), np.array(STADTKE_DATA[:, 1]),
               s=28, color="#D85A30", zorder=5, label="Stadtke (2006) Fig. 8.2")
    # mark the join
    ax.axvline(alpha_trans, color="#888780", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.text(alpha_trans, 0.46, rf"$\alpha_{{b,\mathrm{{trans}}}}={alpha_trans}$",
            ha="center", fontsize=9, color="#5F5E5A")

    ax.set_xlabel(r"$\alpha_g$  (gas void fraction)")
    ax.set_ylabel(r"$\alpha_b^*$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=10, frameon=False)

    rmse, max_abs = fit_check(alpha_trans, alpha_max)
    ax.set_title(
        rf"Stadtke $\alpha_b^*(\alpha_g)$ — hyperbolic form    "
        rf"RMSE vs data = {rmse:.2e},   max|res| = {max_abs:.2e}",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved plot to {save_path}")


    plt.show()
    return fig


if __name__ == "__main__":
    # 1. Verify the function works on scalar, vector, and jit-compiles
    print("Scalar test:  alpha_b_star(0.5) =", float(alpha_b_star(0.5)))
    print("Vector test:  alpha_b_star([0.0, 0.1, 0.5, 1.0]) =",
          np.asarray(alpha_b_star(jnp.array([0.0, 0.1, 0.5, 1.0]))))

    # 2. Confirm derivative continuity at the join (autodiff)
    g = jax.grad(lambda a: alpha_b_star(a, 0.15, 0.35).sum())
    eps = 1e-5
    slope_left  = float(g(jnp.array(0.15 - eps)))
    slope_right = float(g(jnp.array(0.15 + eps)))
    print(f"Slope at alpha_trans:  left = {slope_left:.6f},  right = {slope_right:.6f}")
    print(f"  -> slope jump = {abs(slope_left - slope_right):.2e}")

    # 3. Fit quality vs Stadtke data
    rmse, max_abs = fit_check()
    print(f"\nFit vs Stadtke data (alpha_trans=0.15, alpha_max=0.35):")
    print(f"  RMSE        = {rmse:.6e}")
    print(f"  max |resid| = {max_abs:.6e}")

    # 4. Draw the plot
    plot()
