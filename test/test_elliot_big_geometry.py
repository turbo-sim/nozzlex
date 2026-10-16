"""
Plot the get_nozzle_elliot_big geometry:
  - radius and diameter vs axial position
  - two unit sets: metres (left axis) and inches (right axis)
  - tabulated data points overlaid as scatter
  - region boundaries marked
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from nozzlex.homogeneous_nonequilibrium.nozzle_model_core import get_nozzle_elliot_big

IN2M = 0.0254

# ── Geometry parameters (same defaults as the function) ───────────────────────
conv_len   = 13.46 * IN2M
spline_len = 24.0  * IN2M
div_len    = 13.95 * IN2M   # extra divergent length to show

x_total = conv_len + spline_len + div_len

print(x_total)
xs = np.linspace(0.0, x_total, 2000)

radius_m = np.array([float(get_nozzle_elliot_big(jnp.float64(x))[3]) for x in xs])

# ── Tabulated throat points (x from convergent end, y = radius, both inches) ─
x_tab_in = np.array([
     0.0,  0.5,  1.0,  1.5,  2.0,  2.5,  3.0,  3.5,  4.0,  4.5,
     5.0,  5.5,  6.0,  6.5,  7.0,  7.5,  8.0,  8.5,  9.0,  9.5,
    10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5,
    15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5,
    20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0,
])
y_tab_in = np.array([
    2.100, 1.961, 1.881, 1.821, 1.774, 1.733, 1.701, 1.675, 1.654, 1.637,
    1.624, 1.613, 1.604, 1.598, 1.593, 1.591, 1.591, 1.591, 1.593, 1.598,
    1.604, 1.610, 1.617, 1.625, 1.633, 1.642, 1.651, 1.661, 1.671, 1.682,
    1.693, 1.705, 1.717, 1.730, 1.744, 1.759, 1.775, 1.791, 1.808, 1.824,
    1.842, 1.859, 1.876, 1.893, 1.911, 1.930, 1.950, 1.970, 1.992,
])
x_tab_m = conv_len + x_tab_in * IN2M   # absolute axial position [m]
y_tab_m = y_tab_in * IN2M              # radius [m]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, unit, scale in zip(axes, ["metres", "inches"], [1.0, 1.0 / IN2M]):
    r = radius_m * scale
    xt = xs * scale
    r_tab = y_tab_m * scale
    xt_tab = x_tab_m * scale

    u = "m" if unit == "metres" else "in"

    # Nozzle wall (upper half)
    ax.plot(xt,  r,  "b-",  lw=2,   label="Spline profile")
    ax.plot(xt, -r,  "b-",  lw=2)

    # Fill cross-section
    ax.fill_between(xt,  r, -r, alpha=0.08, color="blue")

    # Tabulated points
    ax.scatter(xt_tab,  r_tab, s=20, color="red",  zorder=5, label="Table points")
    ax.scatter(xt_tab, -r_tab, s=20, color="red",  zorder=5)

    # Region boundary lines
    for xb, lbl in [
        (conv_len * scale,                "conv / throat"),
        ((conv_len + spline_len) * scale, "throat / div"),
    ]:
        ax.axvline(xb, color="gray", ls="--", lw=1, label=lbl)

    ax.set_xlabel(f"Axial position  [{u}]")
    ax.set_ylabel(f"Radius  [{u}]")
    ax.set_title(f"Elliot big nozzle — {unit}")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, ls=":", alpha=0.4)

fig.suptitle("Elliot (1979) large nozzle geometry", fontsize=13)
plt.tight_layout()
plt.show()
