import glob
import os
import pickle as pkl
import re

import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaxprop as jxp

from matplotlib import gridspec

jxp.set_plot_options(grid=False)


def load_pickle(path):
    with open(path, "rb") as f:
        return pkl.load(f)


def parse_label(path):
    fname = os.path.basename(path)
    m = re.search(r"solution_(.+)\.pkl$", fname)
    return m.group(1) if m else fname


def plot_solutions(paths, labels=None):
    """
    Plot one or more TFM solutions on the same axes.

    Parameters
    ----------
    paths  : list of str — paths to .pkl files produced by diffrax_nozzle_two_fluid_jump.py
    labels : list of str, optional — legend labels (defaults to parsed filename)
    """
    if labels is None:
        labels = [parse_label(p) for p in paths]

    colors = plt.cm.tab10.colors

    fig = plt.figure(figsize=(5, 10))
    gs  = gridspec.GridSpec(5, 1, height_ratios=[3, 3, 3, 3, 1])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    axs = [ax0, ax1, ax2, ax3, ax4]

    axs[0].set_ylabel("Mach number")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[2].set_ylabel("Pressure [Pa]")
    axs[3].set_ylabel("Determinant")
    axs[4].set_xlabel("Axial coordinate x (m)")

    axs[0].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    axs[3].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)

    geometry_drawn = False

    for i, (path, label) in enumerate(zip(paths, labels)):
        data  = load_pickle(path)
        out1  = data["pass1"]   # Pass 1 (autonomous, inlet → Ma_low)
        out   = data["pass2"]   # Pass 2 (non-autonomous, throat → outlet)
        color = colors[i % len(colors)]

        # Concatenate pass 1 + pass 2 for a continuous line
        x_full    = jnp.concatenate([out1["x"],     out["x"]])
        Ma_full   = jnp.concatenate([out1["Ma_mix"], out["Ma_mix"]])
        v1_full   = jnp.concatenate([out1["v1"],     out["v1"]])
        v2_full   = jnp.concatenate([out1["v2"],     out["v2"]])
        p_full    = jnp.concatenate([out1["p"],      out["p"]])
        D_full    = jnp.concatenate([out1["D"],      out["D"]])

        axs[0].plot(x_full, Ma_full, color=color, label=label)
        axs[1].plot(x_full, v1_full, color=color, label=f"{label} – fluid 1")
        axs[1].plot(x_full, v2_full, color=color, linestyle="--", label=f"{label} – fluid 2")
        axs[2].plot(x_full, p_full,  color=color, label=label)
        axs[3].plot(x_full, D_full,  color=color, label=label)

        if not geometry_drawn:
            xg = out["x"]
            rg = out["diameter"] / 2.0
            axs[4].fill_between(xg, -rg, +rg, color="lightgrey")
            x_closed = jnp.concatenate([xg, xg[::-1], xg[:1]])
            y_closed = jnp.concatenate([rg, -rg[::-1], rg[:1]])
            axs[4].plot(x_closed, y_closed, "k", linewidth=1.2)
            geometry_drawn = True

    axs[0].legend(fontsize=7)
    axs[1].legend(fontsize=7)

    fig.tight_layout(pad=1)
    return fig, axs


# -----------------------------------------------------------------------------
# Edit the list below to choose which solutions to plot.
# Glob patterns are supported, e.g. "tfm_solutions/solution_*.pkl"
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    solution_folder = "tfm_solutions"

    # Collect all pkl files in the folder, sorted by name
    paths = sorted(glob.glob(os.path.join(solution_folder, "solution_*.pkl")))

    if not paths:
        print(f"No solution files found in '{solution_folder}/'.")
    else:
        print(f"Plotting {len(paths)} solution(s):")
        for p in paths:
            print(f"  {p}")

        fig, axs = plot_solutions(paths)
        plt.show()
