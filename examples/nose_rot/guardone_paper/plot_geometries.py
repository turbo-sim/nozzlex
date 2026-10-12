"""
Plot the domain geometry (inlet straight -> 180 deg bend -> outlet
straight) for each case in figure10_slc.CASES, to scale, so the actual
channel widths being simulated can be compared directly.

Run directly: python plot_geometries.py
"""
import numpy as np
import matplotlib.pyplot as plt

import figure10_slc as f10

r_i = f10.R_I
theta_max = np.deg2rad(f10.THETA_MAX_DEG)
theta_bend = np.linspace(0.0, theta_max, 200)

fig, axes = plt.subplots(1, len(f10.CASES), figsize=(4.2 * len(f10.CASES), 5),
                          sharey=True)

for ax, (r_e_over_ri, M_in) in zip(axes, f10.CASES):
    r_e = r_e_over_ri * r_i
    L = f10.L_FACTOR * r_e
    x_straight = np.linspace(-L, 0.0, 60)
    s_out = np.linspace(0.0, L, 60)
    normal_exit = np.array([np.sin(theta_max), np.cos(theta_max)])
    tangent_exit = np.array([np.cos(theta_max), -np.sin(theta_max)])

    for r_wall, lw in [(r_i, 2.2), (r_e, 2.2)]:
        ax.plot(x_straight, np.full_like(x_straight, r_wall), "k-", lw=lw)
        ax.plot(r_wall * np.sin(theta_bend), r_wall * np.cos(theta_bend), "k-", lw=lw)
        xo = r_wall * normal_exit[0] + s_out * tangent_exit[0]
        yo = r_wall * normal_exit[1] + s_out * tangent_exit[1]
        ax.plot(xo, yo, "k-", lw=lw)

    # initial-guess streamlines: equispaced, same at every station
    # (Sec. 3.3.1 first guess), not yet solved -- capped for readability
    n_show = min(f10.N, 21)
    for r_stream in np.linspace(r_i, r_e, n_show)[1:-1]:
        ax.plot(x_straight, np.full_like(x_straight, r_stream), "b-", lw=0.6, alpha=0.7)
        ax.plot(r_stream * np.sin(theta_bend), r_stream * np.cos(theta_bend), "b-", lw=0.6, alpha=0.7)
        xo = r_stream * normal_exit[0] + s_out * tangent_exit[0]
        yo = r_stream * normal_exit[1] + s_out * tangent_exit[1]
        ax.plot(xo, yo, "b-", lw=0.6, alpha=0.7)

    ax.plot(0, 0, "r+", ms=10, mew=1.5)
    ax.set_aspect("equal")
    ax.set_title(f"r_e/r_i={r_e_over_ri}, M_in={M_in}\nL={L:.2f}", fontsize=11)
    ax.set_xlabel("x")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("y")
fig.suptitle("Domain geometry for each figure10_slc.CASES entry (to scale)")
plt.tight_layout()
fig.savefig("geometries.png", dpi=150)
plt.show()
print("saved geometries.png")
